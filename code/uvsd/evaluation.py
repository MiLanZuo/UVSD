"""COCO bbox AP and separately reported fixed-threshold counts.

Global score truncation to 100 detections/image occurs BEFORE COCO evaluation.
COCO itself also uses maxDets=[1,10,100]. AP never applies the .5 score threshold.
"""
from collections import defaultdict
from copy import deepcopy
import contextlib
import io
import numpy as np
from .data import category_id, class_id

def iou_xywh(a,b):
    ax,ay,aw,ah = a; bx,by,bw,bh = b
    inter = max(0,min(ax+aw,bx+bw)-max(ax,bx)) * max(0,min(ay+ah,by+bh)-max(ay,by))
    return inter / max(aw*ah+bw*bh-inter, np.finfo(float).eps)

def validate_and_truncate(gt,predictions,max_detections=100):
    image_ids = {i['id'] for i in gt['images']}
    cat_ids = {i['id'] for i in gt['categories']}
    grouped = defaultdict(list)
    for p in predictions:
        if p['image_id'] not in image_ids or p['category_id'] not in cat_ids:
            raise ValueError('Prediction image/category ID is absent from GT')
        b = np.asarray(p['bbox'],dtype=float)
        if b.shape != (4,) or not np.isfinite(b).all() or np.any(b[2:] <= 0):
            raise ValueError('Prediction bbox must be finite xywh with positive size')
        if not np.isfinite(p['score']) or not 0 <= p['score'] <= 1:
            raise ValueError('Prediction scores must be probabilities in [0,1]')
        grouped[p['image_id']].append(dict(p))
    out = []
    for image_id in sorted(grouped):
        out.extend(sorted(grouped[image_id],key=lambda p:-p['score'])[:max_detections])
    return out

def fixed_threshold_counts(gt,predictions,confidence=.5,iou_threshold=.5):
    """Descending-score, same-class greedy one-to-one matching; no crowd GT accepted."""
    if any(a.get('iscrowd',0) or a.get('ignore',0) for a in gt['annotations']):
        raise ValueError('UVSD fixed counts require explicit ordinary instance GT, not crowd/ignored GT')
    truth = defaultdict(list)
    pred = defaultdict(list)
    for a in gt['annotations']:
        truth[a['image_id'],a['category_id']].append(a['bbox'])
    for p in predictions:
        if p['score'] >= confidence:
            pred[p['image_id'],p['category_id']].append(p)
    per_class = {}
    for cat in gt['categories']:
        tp=fp=fn=0
        for im in gt['images']:
            boxes = truth[im['id'],cat['id']]
            matched = set()
            for p in sorted(pred[im['id'],cat['id']],key=lambda r:-r['score']):
                eligible = [(iou_xywh(p['bbox'],b),i) for i,b in enumerate(boxes) if i not in matched]
                overlap, index = max(eligible,default=(-1,-1))
                if overlap >= iou_threshold:
                    tp += 1; matched.add(index)
                else:
                    fp += 1
            fn += len(boxes)-len(matched)
        per_class[cat['name']] = _counts(tp,fp,fn)
    totals = [sum(c[k] for c in per_class.values()) for k in ('TP','FP','FN')]
    return {'overall':_counts(*totals),'per_class':per_class,'confidence':confidence,'iou_threshold':iou_threshold,
            'matching':'score-descending same-class one-to-one'}

def _counts(tp,fp,fn):
    return {'TP':tp,'FP':fp,'FN':fn,'GT':tp+fn,'P':tp/(tp+fp) if tp+fp else 0.,
            'R':tp/(tp+fn) if tp+fn else 0.,'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}

def evaluate_coco(gt,predictions,image_ids=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = deepcopy(gt)
    # Validate against the full dataset before optional condition subsetting.
    predictions = validate_and_truncate(gt,predictions)
    if image_ids is not None:
        ids = set(image_ids)
        if not ids <= {i['id'] for i in gt['images']}:
            raise ValueError('Unknown image IDs in condition subset')
        gt['images'] = [i for i in gt['images'] if i['id'] in ids]
        gt['annotations'] = [a for a in gt['annotations'] if a['image_id'] in ids]
        predictions = [p for p in predictions if p['image_id'] in ids]
    if not gt['images']:
        return {'status':'empty_subset','AP50':None,'AP75':None,'AP50_95':None,'images':0}
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        coco = COCO(); coco.dataset=gt; coco.createIndex()
        if predictions:
            dt = coco.loadRes(predictions)
        else:
            dt = COCO(); dt.dataset={'images':deepcopy(gt['images']),'categories':deepcopy(gt['categories']),'annotations':[]}; dt.createIndex()
        ev=COCOeval(coco,dt,'bbox'); ev.params.imgIds=[i['id'] for i in gt['images']]
        ev.params.maxDets=[1,10,100]; ev.evaluate(); ev.accumulate(); ev.summarize()
    prec=ev.eval['precision'] # T,R,K,A,M
    by_class={}
    for k,cat_id in enumerate(ev.params.catIds):
        name=next(c['name'] for c in gt['categories'] if c['id']==cat_id)
        values=prec[:,:,k,0,-1]; valid=values[values>=0]
        at50=prec[np.isclose(ev.params.iouThrs,.5),:,k,0,-1]; valid50=at50[at50>=0]
        at75=prec[np.isclose(ev.params.iouThrs,.75),:,k,0,-1]; valid75=at75[at75>=0]
        by_class[name]={'AP50_95':float(valid.mean()) if len(valid) else None,'AP50':float(valid50.mean()) if len(valid50) else None,'AP75':float(valid75.mean()) if len(valid75) else None}
    return {'status':'evaluated','AP50_95':float(ev.stats[0]) if ev.stats[0]>=0 else None,
            'AP50':float(ev.stats[1]) if ev.stats[1]>=0 else None,'AP75':float(ev.stats[2]) if ev.stats[2]>=0 else None,'per_class':by_class,
            'fixed_threshold':fixed_threshold_counts(gt,predictions),'images':len(gt['images']),
            'predictions_after_global_top100':len(predictions),'max_detections_policy':'global score top100 per image before COCO; COCO maxDets=[1,10,100]',
            'coco_stdout':capture.getvalue(),'coordinate_system':'original_optical_crop','AP_units':'fraction'}

def predictions_to_coco(pair_id, logits, normalized_boxes, transform):
    """RT-DETR style flattened class-score top100; separate from variant-specific NMS."""
    logits=np.asarray(logits,dtype=np.float64); boxes=np.asarray(normalized_boxes,dtype=np.float64)
    scores=np.exp(-np.logaddexp(0,-logits))
    flat=scores.ravel(); inds=np.argsort(-flat,kind='stable')[:100]
    original=transform.from_normalized(boxes)
    original[:,[0,2]]=np.clip(original[:,[0,2]],0,transform.width)
    original[:,[1,3]]=np.clip(original[:,[1,3]],0,transform.height)
    result=[]
    for idx in inds:
        q,c=divmod(int(idx),logits.shape[-1]); x1,y1,x2,y2=original[q]
        if x2>x1 and y2>y1:
            result.append({'image_id':int(pair_id),'category_id':category_id(c),'score':float(flat[idx]),'bbox':[float(x1),float(y1),float(x2-x1),float(y2-y1)]})
    return result
