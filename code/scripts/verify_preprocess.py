"""Reopen every exported PNG and validate geometry, provenance and review status."""
import argparse,csv,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np,yaml
from PIL import Image
LABELS=['overexposure','blur','low_contrast','weak_echo','clutter']
def readcsv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def verify(config):
 cfg=yaml.safe_load(Path(config).read_text(encoding='utf-8-sig'));root=Path(cfg['output_root']);rows=readcsv(root/'data/processed/pairs.csv');anns=readcsv(root/'annotations/degradation_annotations.csv');errors=[];scales=[];last=-1;seen={'optical':{},'sonar':{},'pair':{}};dup=Counter();maxerr=0
 def check(v,msg):
  if not v:errors.append(msg)
 bypair=defaultdict(list)
 for a in anns:bypair[int(a['pair_id'])].append(a)
 check(set(bypair).issubset({int(r['pair_id']) for r in rows}),'orphan degradation annotations')
 timestamps={}
 if (root/'reports/source_frame_timestamps.csv').exists():timestamps={int(r['frame_index']):r for r in readcsv(root/'reports/source_frame_timestamps.csv')}
 for pid,r in enumerate(rows):
  check(int(r['pair_id'])==pid,'nonconsecutive pair ID');frame=int(r['frame_index']);check(frame>last,'nonincreasing source frame');last=frame;hs=[]
  if timestamps:
   ts=timestamps.get(frame)
   check(ts is not None and abs(float(r['timestamp_sec'])-float(ts['timestamp_sec']))<1e-9,f'{pid} source PTS mismatch')
   check(r['timestamp_source']=='pyav_frame_pts',f'{pid} source timestamp provenance')
  for mod,wh in [('optical',(1695,951)),('sonar',(1155,797))]:
   p=root/r[mod+'_path'];im=Image.open(p).convert('RGB');im.load();ar=np.array(im);check(im.size==wh,f'{pid} {mod} size');h=hashlib.sha256(ar.tobytes()).hexdigest();hs.append(h);check(h==r[mod+'_sha256'],f'{pid} pixel hash mismatch');expected=seen[mod].get(h,'');check(str(expected)==r[mod+'_duplicate_of_frame'],f'{pid} duplicate mismatch');seen[mod].setdefault(h,frame);dup[mod]+=expected!=''
   tensor=np.array(im.resize((640,640)),dtype=np.float32).transpose(2,0,1)/255.;check(tensor.shape==(3,640,640),'input shape');sx,sy=640/wh[0],640/wh[1];scales.append({'pair_id':pid,'modality':mod,'sx':sx,'sy':sy,'shape':list(tensor.shape)})
   b=np.array([13.25,22.75,wh[0]-17.125,wh[1]-19.625],dtype=np.float64);factor=np.array([sx,sy,sx,sy]);maxerr=max(maxerr,float(np.abs(b*factor/factor-b).max()))
  key=':'.join(hs);expected=seen['pair'].get(key,'');check(str(expected)==r['pair_duplicate_of_frame'],'pair duplicate mismatch');seen['pair'].setdefault(key,frame);dup['pair']+=expected!=''
  raw=np.array(Image.open(root/r['sonar_path']).convert('RGB'));clean=np.array(Image.open(root/r['sonar_denoised_path']).convert('RGB'));mask=np.array(Image.open(root/f'data/processed/sonar_masks/{pid:04d}.png'))>0;sup=np.array(Image.open(root/f'data/processed/sonar_support_masks/{pid:04d}.png'))>0
  check(raw.shape==clean.shape,'clean dimensions');check(np.array_equal(raw[~sup],clean[~sup]),'changed outside stripe support');check(np.array_equal(np.any(raw!=clean,axis=2),mask),'mask mismatch');check(hashlib.sha256(clean.tobytes()).hexdigest()==r['sonar_denoised_sha256'],'clean hash')
  aa=bypair[pid];check(any(a['review_status']=='confirmed' for a in aa),f'{pid} no confirmed evidence')
  for a in aa:
   check(a['modality'] in ['optical','sonar'],'unknown evidence modality')
   check(a['class_id'] in ['0','1','2','3'],'unknown target class')
   check(a['class_name']==['sphere','cube','cone','wrench'][int(a['class_id'])] if a['class_id'] in ['0','1','2','3'] else False,'class identity mismatch')
   check(int(a['frame_index'])==frame,f'{pid} evidence frame');check(a['sequence_id']==r['sequence_id'],'sequence mismatch');check(a['degradation_label'] in LABELS,'unknown label');check((root/a['evidence_path']).is_file(),'missing evidence');box=json.loads(a['evidence_region']);w,h=(1695,951) if a['modality']=='optical' else (1155,797);check(0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h,'evidence out of bounds')
 for mod in ['optical','sonar','sonar_denoised']:
  check({p.name for p in (root/f'data/processed/{mod}').glob('*.png')}=={Path(r[mod+'_path']).name for r in rows},'manifest/files mismatch '+mod)
 counts={lab:{'pair_count':len({a['pair_id'] for a in anns if a['degradation_label']==lab and a['review_status']=='confirmed'}),'object_label_count':sum(a['degradation_label']==lab and a['review_status']=='confirmed' for a in anns)} for lab in LABELS}
 extraction=json.loads((root/'reports/extraction_status.json').read_text());missing=[lab for lab in LABELS if counts[lab]['pair_count']==0]
 result={'file_integrity_pass':not errors,'protocol_complete':not errors and len(rows)==cfg['target_pairs'] and not missing and extraction['machine_only_frames']==0,'errors':errors,'pair_count':len(rows),'raw_png_count':2*len(rows),'denoised_sonar_png_count':len(rows),'dimensions':{'optical':[1695,951],'sonar':[1155,797]},'roi':cfg['rois'],'first_frame':int(rows[0]['frame_index']) if rows else None,'last_frame':int(rows[-1]['frame_index']) if rows else None,'time_span_sec':float(rows[-1]['timestamp_sec'])-float(rows[0]['timestamp_sec']) if rows else 0,'duplicates':dict(dup),'input_shapes':{'optical':[3,640,640],'sonar':[3,640,640]},'bbox_roundtrip_max_error_pixels':maxerr,'deficit':max(0,cfg['target_pairs']-len(rows)),'degradation_counts':counts,'missing_degradation_categories':missing,'detection_annotation_status':'unannotated; evidence regions are NOT complete detection GT','extraction':extraction,'limitations':['Sequential machine screen covered every frame, but only confirmed evidence frames received target-level visual adjudication. Remaining frames stay needs_review; no claim that absent labels do not occur.','Sonar echo class identity and cross-modal correspondences are unresolved; no weak_echo/clutter labels promoted from brightness alone.','A3/A4 formal experiments require measured calibration including per-frame sonar range mapping.','Stripe suppression is a derived input, not raw acoustic data or experimentally proven target-preserving restoration.']}
 (root/'reports/preprocess_validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');(root/'reports/input_scales.json').write_text(json.dumps(scales,indent=2))
 lines=['# 数据制备验收报告','',f'文件完整性：{"通过" if not errors else "失败"}；完整协议验收：{"通过" if result["protocol_complete"] else "尚未通过"}。',f'已导出 {len(rows)} 对，100 张原始裁剪 PNG（实际为 {2*len(rows)} 张），另有 {len(rows)} 张三柱噪声抑制声呐派生图。',f'源视频元数据 {extraction["metadata_frame_count"]} 帧；本轮顺序解码 {extraction["decoded_frame_count"]} 帧；有目标级确认记录 {extraction["target_visual_reviewed_frames"]} 帧。',f'首末采纳帧 {result["first_frame"]}–{result["last_frame"]}；时间跨度 {result["time_span_sec"]:.3f} 秒。',f'配对数量缺额 {result["deficit"]}；尚无确认记录的类别：{", ".join(missing)}。','', '| 退化 | 涉及配对 | 对象级标签次数 |','|---|---:|---:|']
 lines += [f'| {lab} | {v["pair_count"]} | {v["object_label_count"]} |' for lab,v in counts.items()]
 lines += ['',f'原像素重复统计：{dict(dup)}。框往返最大误差：{maxerr:.3g} px。','ROI：光学 (113,77,1695,951)；声呐 (2200,170,1155,797)。','正式检测框未标注；退化证据框不能冒充完整检测 GT。','确认来自本轮原尺寸局部视觉复核；未冒称人工标注员签核。','未确认帧保留在 annotations/pending_review.csv；不能把机器扫描等同于逐帧目标复核。','三柱处理保留 raw sonar，派生图及掩膜独立保存；不把抑制后的结果计作原始退化证据。','尚无足够可核验的声呐目标类别/跨模态对应，弱回波和目标级杂波维持待复核。','下一步：补充 review_decisions.json 中的逐帧证据，再运行 `python .\\scripts\\extract_pairs.py --config .\\configs\\extract_config.yaml`。选择集合改变时使用新 output_root，避免混入旧批次。']
 (root/'reports/preprocess_report.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(result,ensure_ascii=False));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();r=verify(a.config);raise SystemExit(0 if r['file_integrity_pass'] else 1)
