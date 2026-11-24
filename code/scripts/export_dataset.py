"""Export complete reviewed optical GT for shared COCO/YOLO baseline interfaces."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
import sys
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from uvsd.data import PairedDataset,export_coco,read_csv,image_path,ResizeTransform,CLASSES
from uvsd.audit import audit_splits

def source_signature(dataset,sonar_variant):
    paths=[dataset.root/p for p in ['data/processed/pairs.csv','annotations/detection_annotations.csv','annotations/image_review.csv','annotations/sequence_splits.csv']]
    for row in dataset.rows:
        paths.extend([image_path(dataset.root,row,'optical'),image_path(dataset.root,row,'sonar_denoised' if sonar_variant=='denoised' else 'sonar')])
    h=hashlib.sha256(('square640_export_v1:'+sonar_variant).encode())
    for path in sorted(paths,key=str):h.update(str(path).encode());h.update(path.read_bytes())
    return h.hexdigest()

def verify_export(output,signature):
    receipt=json.loads((output/'export_receipt.json').read_text(encoding='utf-8'))
    if receipt['source_signature']!=signature:raise ValueError('Export belongs to different source images, annotations or splits; regenerate to a fresh destination')
    expected=set(receipt['files'])
    actual={str(p.relative_to(output)).replace('\\','/') for p in output.rglob('*') if p.is_file() and p.name!='export_receipt.json' and p.suffix!='.cache'}
    if expected!=actual:raise ValueError('Export file set changed; stale or missing files detected')
    for rel,sha in receipt['files'].items():
        if hashlib.sha256((output/rel).read_bytes()).hexdigest()!=sha:raise ValueError('Export file content changed: '+rel)
    return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='data/exports');p.add_argument('--sonar-variant',choices=['raw','denoised'],default='denoised')
    args=p.parse_args();out=ROOT/args.output
    ds=PairedDataset(ROOT,require_gt=True,sonar_variant=args.sonar_variant)
    audit=audit_splits(ROOT,ds.rows)
    if not audit['formal_eligible']: raise ValueError('Formal export blocked by split audit: '+json.dumps(audit))
    signature=source_signature(ds,args.sonar_variant)
    if out.exists() and any(out.iterdir()):
        verify_export(out,signature)
        print('Reused verified immutable export:',out)
        return
    splits={r['sequence_id']:r['split'] for r in read_csv(ROOT/'annotations/sequence_splits.csv')}
    out.mkdir(parents=True,exist_ok=True); manifest=[]
    for split in ('train','val','test'):
        indices=[i for i,r in enumerate(ds.rows) if splits[r['sequence_id']]==split]
        coco=export_coco(ds,indices)
        (out/f'{split}_optical_coco.json').write_text(json.dumps(coco,ensure_ascii=False,indent=2),encoding='utf-8')
        labels=out/'yolo'/'labels'/split;images=out/'yolo'/'images'/split
        labels.mkdir(parents=True,exist_ok=True);images.mkdir(parents=True,exist_ok=True)
        from PIL import Image
        for i in indices:
            r=ds.rows[i];pid=r['pair_id'];sample=ds[i]
            optical=image_path(ROOT,r,'optical');sonar=image_path(ROOT,r,'sonar_denoised' if args.sonar_variant=='denoised' else 'sonar')
            # Square stretch export avoids native YOLO letterbox changing the paper's input transform.
            with Image.open(optical) as im:im.convert('RGB').resize((640,640),Image.Resampling.BILINEAR).save(images/f'{int(pid):06d}.png')
            b=sample['targets']['optical']['boxes'];c=sample['targets']['optical']['labels']
            lines=[' '.join([str(int(cid))]+[format(float(x),'.9g') for x in box]) for box,cid in zip(b,c)]
            (labels/f'{int(pid):06d}.txt').write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
            manifest.append({'pair_id':pid,'sequence_id':r['sequence_id'],'split':split,'optical_path':str(optical),
                             'sonar_path':str(sonar),'sonar_variant':args.sonar_variant,'optical_coco':str(out/f'{split}_optical_coco.json'),
                             'optical_width':sample['transforms']['optical'].width,'optical_height':sample['transforms']['optical'].height,'yolo_input_size':640})
    with (out/'paired_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
    (out/'yolo'/'uvsd.yaml').write_text(yaml.safe_dump({'path':str(out/'yolo'),'train':'images/train','val':'images/val','test':'images/test','names':dict(enumerate(CLASSES))},allow_unicode=True),encoding='utf-8')
    files={str(p.relative_to(out)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    (out/'export_receipt.json').write_text(json.dumps({'source_signature':signature,'files':files,'resize':'independent square640 stretch'},indent=2),encoding='utf-8')
    verify_export(out,signature)
    print('Exported reviewed COCO, optical YOLO and paired-manifest interfaces:',len(manifest),'pairs')

if __name__=='__main__':main()
