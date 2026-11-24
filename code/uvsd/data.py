"""Paired image IO, explicit label completeness and reversible box transforms.

Detection boxes are continuous xyxy coordinates in each original crop.
Evidence regions are never used as detection GT. Image-level review is required
to distinguish complete labels, confirmed negatives, and incomplete annotations.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

CLASSES = ('sphere', 'cube', 'cone', 'wrench')

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def category_id(class_id):
    value = int(class_id)
    if value not in range(4):
        raise ValueError(f'Invalid UVSD class ID {value}')
    return value + 1

def class_id(category):
    value = int(category) - 1
    if value not in range(4):
        raise ValueError(f'Invalid COCO category ID {category}')
    return value

def xyxy_to_cxcywh(boxes):
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.concatenate(((b[:, :2] + b[:, 2:]) / 2, b[:, 2:] - b[:, :2]), axis=-1)

def cxcywh_to_xyxy(boxes):
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.concatenate((b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2), axis=-1)

def flip_boxes(boxes, width):
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    b[:, [0, 2]] = width - b[:, [2, 0]]
    return b

class ResizeTransform:
    def __init__(self, width, height, size=640, flipped=False):
        self.width, self.height, self.size = int(width), int(height), int(size)
        if min(self.width, self.height, self.size) <= 0:
            raise ValueError('Image sizes must be positive')
        self.sx, self.sy = size / width, size / height
        self.flipped = bool(flipped)

    def to_input(self, boxes):
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        if self.flipped:
            b = flip_boxes(b, self.width)
        return b * [self.sx, self.sy, self.sx, self.sy]

    def from_input(self, boxes):
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4) / [self.sx, self.sy, self.sx, self.sy]
        return flip_boxes(b, self.width) if self.flipped else b

    def to_normalized(self, boxes):
        return xyxy_to_cxcywh(self.to_input(boxes)) / self.size

    def from_normalized(self, boxes):
        return self.from_input(cxcywh_to_xyxy(boxes) * self.size)

    def matrix(self):
        resize = np.diag([self.sx, self.sy, 1.])
        flip = np.array([[-1., 0., self.width], [0., 1., 0.], [0., 0., 1.]])
        return resize @ flip if self.flipped else resize

    def as_dict(self):
        return vars(self).copy()

def augmented_mapping(mapping, source_transform, target_transform):
    """Conjugate a pixel mapping by resize/flip; G_aug = T_t G T_s^-1."""
    return target_transform.matrix() @ np.asarray(mapping) @ np.linalg.inv(source_transform.matrix())

def image_path(root, row, modality):
    for key in (f'{modality}_path', f'{modality}_png', f'{modality}_relpath'):
        if row.get(key):
            p = Path(row[key])
            return p if p.is_absolute() else Path(root) / p
    raise KeyError(f'Missing {modality}_path for pair {row.get("pair_id")}')

class PairedDataset:
    def __init__(self, root, pairs='data/processed/pairs.csv', annotations='annotations/detection_annotations.csv',
                 review='annotations/image_review.csv', training=False, optical_only=False, seed=0, size=640,
                 mean=(.485,.456,.406), std=(.229,.224,.225), require_gt=False, sonar_variant='denoised'):
        self.root, self.size = Path(root), size
        self.rows = read_csv(self.root / pairs)
        self.training, self.optical_only = training, optical_only
        if sonar_variant not in ('raw','denoised'):
            raise ValueError('sonar_variant must be raw or denoised')
        self.sonar_variant = sonar_variant
        self.rng = np.random.default_rng(seed)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3,1,1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3,1,1)
        self.annotations = read_csv(self.root / annotations) if (self.root / annotations).exists() else []
        self.reviews = {(str(r['pair_id']), r['modality']): r for r in read_csv(self.root / review)} if (self.root / review).exists() else {}
        self.require_gt = require_gt
        ids = [r['pair_id'] for r in self.rows]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate pair IDs')
        if require_gt:
            for row in self.rows:
                self.labels(row, 'optical', required=True)

    def __len__(self):
        return len(self.rows)

    def labels(self, row, modality, required=False):
        key = (str(row['pair_id']), modality)
        review = self.reviews.get(key, {})
        state = review.get('annotation_status', 'unannotated')
        records = [r for r in self.annotations if str(r['pair_id']) == key[0] and r['modality'] == modality]
        confirmed = [r for r in records if r.get('review_status') == 'confirmed']
        complete = state in ('complete', 'confirmed_negative') and review.get('review_status') == 'confirmed'
        if required and not complete:
            raise ValueError(f'Pair {key[0]} {modality} has no confirmed complete image-level annotation review')
        if state == 'confirmed_negative' and records:
            raise ValueError(f'Pair {key[0]} {modality} negative review contradicts object annotations')
        if complete and len(confirmed) != len(records):
            raise ValueError(f'Pair {key[0]} {modality} complete review contains unresolved objects')
        if state == 'complete' and not confirmed:
            raise ValueError('An empty complete annotation must instead be reviewed as confirmed_negative')
        boxes, labels = [], []
        for r in confirmed:
            if r.get('coordinate_system') != f'{modality}_crop_xyxy':
                raise ValueError('Detection box coordinate system must be explicit and modality-specific')
            if r.get('frame_index', row['frame_index']) != row['frame_index']:
                raise ValueError('Detection annotation frame mismatch')
            if r.get('sequence_id', row['sequence_id']) != row['sequence_id']:
                raise ValueError('Detection annotation sequence mismatch')
            boxes.append([float(r[k]) for k in ('x1','y1','x2','y2')])
            c = int(r['class_id']); category_id(c)
            if r.get('class_name', CLASSES[c]).lower() != CLASSES[c]:
                raise ValueError('Detection class ID/name mismatch')
            labels.append(c)
        return np.asarray(boxes, dtype=np.float64).reshape(-1,4), np.asarray(labels, dtype=np.int64), complete, state

    def __getitem__(self, index):
        row = self.rows[index]
        flipped = bool(self.training and self.rng.random() < .5)
        result = {'metadata':dict(row), 'flipped':flipped, 'transforms':{}, 'targets':{}}
        result['metadata']['sonar_variant'] = self.sonar_variant
        for modality in ('optical',) if self.optical_only else ('optical','sonar'):
            path_modality = 'sonar_denoised' if modality == 'sonar' and self.sonar_variant == 'denoised' else modality
            with Image.open(image_path(self.root, row, path_modality)) as opened:
                im = opened.convert('RGB')
                w,h = im.size
                t = ResizeTransform(w,h,self.size,flipped)
                if flipped:
                    im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                arr = np.asarray(im.resize((self.size,self.size),Image.Resampling.BILINEAR), dtype=np.float32)
            boxes, labels, complete, state = self.labels(row, modality, required=self.require_gt and modality=='optical')
            if len(boxes) and (np.any(boxes[:,:2] < 0) or np.any(boxes[:,2:] > [w,h]) or np.any(boxes[:,2:] <= boxes[:,:2])):
                raise ValueError(f'Out-of-domain or empty {modality} GT box')
            result['image' if modality=='optical' else 'sonar'] = np.ascontiguousarray((arr.transpose(2,0,1)/255-self.mean)/self.std)
            result['transforms'][modality] = t
            result['targets'][modality] = {'boxes':t.to_normalized(boxes).astype('float32'), 'labels':labels,
                                         'boxes_original':boxes, 'complete':complete, 'annotation_status':state}
        return result

def export_coco(dataset, indices=None):
    """Only reliable optical GT, never degradation evidence, enters detection evaluation."""
    images, annotations = [], []
    for idx in range(len(dataset)) if indices is None else indices:
        row = dataset.rows[idx]
        boxes, labels, _, _ = dataset.labels(row,'optical',required=True)
        with Image.open(image_path(dataset.root,row,'optical')) as im:
            w,h = im.size
        pid = int(row['pair_id'])
        images.append({'id':pid, 'width':w, 'height':h, 'file_name':str(image_path(dataset.root,row,'optical')),
                       'sequence_id':row['sequence_id'], 'frame_index':int(row['frame_index'])})
        for b,c in zip(boxes,labels):
            x1,y1,x2,y2 = b.tolist()
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError('Invalid optical GT bounds')
            annotations.append({'id':len(annotations)+1,'image_id':pid,'category_id':category_id(c),
                                'bbox':[x1,y1,x2-x1,y2-y1], 'area':(x2-x1)*(y2-y1),'iscrowd':0})
    return {'info':{'coordinate_system':'original_optical_crop','class_mapping':'UVSD 0..3 -> COCO 1..4'},
            'images':images,'annotations':annotations,'categories':[{'id':i+1,'name':c} for i,c in enumerate(CLASSES)]}
