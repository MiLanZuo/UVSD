"""Sequence and image leakage checks; perceptual matches are review candidates."""
from collections import defaultdict
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from .data import read_csv, image_path

def hashes(path):
    with Image.open(path) as im:
        rgb=im.convert('RGB'); a=np.asarray(rgb)
        exact=hashlib.sha256(np.asarray(a.shape,dtype=np.int64).tobytes()+a.tobytes()).hexdigest()
        small=np.asarray(rgb.convert('L').resize((9,8),Image.Resampling.BILINEAR))
        bits=(small[:,1:]>small[:,:-1]).ravel()
        perceptual=sum(int(b)<<i for i,b in enumerate(bits))
        return exact,perceptual

def audit_splits(root, rows, split_path='annotations/sequence_splits.csv', near_hamming=4):
    root=Path(root); path=root/split_path
    if not path.exists():
        return {'status':'blocked','reason':'Missing sequence_splits.csv: sequence_id,recording_group,split,review_status',
                'sequences':sorted({r['sequence_id'] for r in rows}), 'formal_eligible':False}
    splits=read_csv(path); mapping={r['sequence_id']:r for r in splits}
    problems=[]; groups=defaultdict(set)
    if len(mapping)!=len(splits): problems.append('Repeated sequence_id in split manifest')
    for r in splits:
        if r.get('split') not in ('train','val','test'): problems.append('Invalid split name')
        if not r.get('recording_group'): problems.append('Missing continuous-recording group')
        if r.get('review_status')!='confirmed': problems.append('Unconfirmed sequence/recording identity')
        groups[r.get('recording_group')].add(r.get('split'))
    problems.extend(f'Continuous recording crosses subsets: {g}' for g,s in groups.items() if len(s)>1)
    missing={r['sequence_id'] for r in rows}-set(mapping)
    if missing: problems.append(f'Missing sequence assignments: {sorted(missing)}')
    present={mapping[r['sequence_id']]['split'] for r in rows if r['sequence_id'] in mapping}
    if present!={'train','val','test'}: problems.append('Formal protocol requires nonempty train/val/test sequences')
    # A reviewed distinct scene can resolve a perceptual-hash false positive;
    # exact decoded-pixel duplicates cannot be overridden by this file.
    review_path=root/'annotations/duplicate_review.csv'; reviewed_distinct=set()
    if review_path.exists():
        for review in read_csv(review_path):
            if review.get('review_status')=='confirmed_distinct' and review.get('reviewer') and review.get('review_note'):
                reviewed_distinct.add((frozenset((str(review['pair_id_a']),str(review['pair_id_b']))),review['modality']))
    entries=[]
    for r in rows:
        if r['sequence_id'] not in mapping: continue
        hs={m:hashes(image_path(root,r,m)) for m in ('optical','sonar')}
        entries.append((r,mapping[r['sequence_id']]['split'],hs))
    duplicates=[]
    for i,(r,s,h) in enumerate(entries):
        for rr,ss,hh in entries[:i]:
            if s==ss: continue
            for m in ('optical','sonar'):
                exact=h[m][0]==hh[m][0]; distance=(h[m][1]^hh[m][1]).bit_count()
                if exact or distance<=near_hamming:
                    resolved=not exact and (frozenset((str(r['pair_id']),str(rr['pair_id']))),m) in reviewed_distinct
                    duplicates.append({'pair_id_a':r['pair_id'],'pair_id_b':rr['pair_id'],'modality':m,
                                       'exact':exact,'dhash_distance':distance,'status':'leakage' if exact else ('confirmed_distinct' if resolved else 'needs_review')})
    if any(r['status']!='confirmed_distinct' for r in duplicates): problems.append('Cross-subset exact/near image matches require resolution')
    return {'status':'passed' if not problems else 'blocked','formal_eligible':not problems,'problems':problems,
            'duplicates':duplicates,'perceptual_method':'64-bit dHash','near_hamming_threshold':near_hamming,
            'notes':'Near duplicates are candidates, not confirmed identity; review before formal evaluation.',
            'sequences':len(mapping),'recording_groups':len(groups),'pairs_checked':len(entries)}
