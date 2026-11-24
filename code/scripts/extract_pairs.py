"""Sequential read-only screening and reviewed-pair export. No seek in this pipeline."""
import argparse,csv,hashlib,json,sys,subprocess,time
from pathlib import Path
from collections import Counter
import cv2,numpy as np,yaml
from PIL import Image,ImageDraw
CLASSES=['sphere','cube','cone','wrench']
LABELS=['overexposure','blur','low_contrast','weak_echo','clutter']
def writecsv(p,rows,fields):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def save(p,a):
 p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(a).save(p)
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
def run(config,stage='all'):
 cfg=yaml.safe_load(Path(config).read_text(encoding='utf-8-sig'));root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
 for d in ['reports','work/candidates','annotations','data/processed/previews','data/processed/optical','data/processed/sonar','data/processed/sonar_denoised','data/processed/sonar_masks','data/processed/sonar_support_masks']:(root/d).mkdir(parents=True,exist_ok=True)
 src=Path(cfg['source_video']); stat={'source_exists':src.exists(),'stage':stage,'status':'pending','target_pairs':cfg['target_pairs']}
 if not src.exists():
  stat['missing_input']=str(src);(root/'reports/extraction_status.json').write_text(json.dumps(stat,indent=2));return
 rois=cfg['rois'];ready=all(rois[m]['status']=='confirmed' for m in ['optical','sonar'])
 decisions=json.loads((root/cfg['review_file']).read_text(encoding='utf-8')) if (root/cfg['review_file']).exists() else []
 accepted={}
 for a in decisions:
  if a['review_status']=='confirmed':
   assert a['degradation_label'] in LABELS and a['modality'] in ['optical','sonar'] and 0<=a['class_id']<4
   accepted.setdefault(int(a['frame_index']),[]).append(a)
 selected=[];covered=set()
 for i in sorted(accepted):
  labs={a['degradation_label'] for a in accepted[i]}
  if labs-covered:selected.append(i);covered|=labs
 for i in sorted(accepted):
  if len(selected)>=cfg['target_pairs']:break
  if i not in selected:selected.append(i)
 selected=sorted(selected[:cfg['target_pairs']]);pair_ids={f:i for i,f in enumerate(selected)}
 if stage!='scan' and not ready:raise ValueError('Both ROI statuses must be confirmed')
 existing=list((root/'data/processed/optical').glob('*.png'))
 if existing:
  old=json.loads((root/'reports/export_identity.json').read_text())
  if old!={'source_video':str(src),'selected_frames':selected,'roi_version':cfg['roi_version']}:raise RuntimeError('Output batch differs; use a fresh output root')
 if stage!='scan':(root/'reports/export_identity.json').write_text(json.dumps({'source_video':str(src),'selected_frames':selected,'roi_version':cfg['roi_version']}))
 timestamps={};timestamp_audit_path=root/'reports/source_timestamp_audit.json'
 if timestamp_audit_path.exists():
  ta=json.loads(timestamp_audit_path.read_text(encoding='utf-8'))
  if (ta['source_video'],ta['source_size_bytes'],ta['source_mtime_ns'])!=(str(src),src.stat().st_size,src.stat().st_mtime_ns):raise ValueError('Timestamp sidecar belongs to a different source version')
  with (root/ta['sidecar']).open(encoding='utf-8-sig',newline='') as f:timestamps={int(r['frame_index']):r for r in csv.DictReader(f)}
 cap=cv2.VideoCapture(str(src))
 if not cap.isOpened():raise ValueError('Source video cannot be opened')
 fps=cap.get(cv2.CAP_PROP_FPS);expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));size=[int(cap.get(3)),int(cap.get(4))]
 for m in rois:
  x,y,w,h=rois[m]['xywh'];assert 0<=x and 0<=y and x+w<=size[0] and y+h<=size[1]
 rows=[];anns=[];review=[];denoise_stats=[];seen={'optical':{},'sonar':{},'pair':{}}
 i=0;start=time.time();before={'size':src.stat().st_size,'mtime_ns':src.stat().st_mtime_ns}
 while True:
  ok,frame=cap.read()
  if not ok:break
  # Each pair comes from this single decoded frame. PTS queried immediately after read.
  pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000;pts_source='opencv_pos_msec' if np.isfinite(pts) and (i==0 or pts>0) else 'estimated'
  if pts_source=='estimated':pts=i/fps
  if i in timestamps:pts=float(timestamps[i]['timestamp_sec']);pts_source='pyav_frame_pts'
  crops={m:cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2RGB) for m,rr in rois.items() for x,y,w,h in [rr['xywh']]}
  empty=all(float(a[::12,::12].std())<=cfg['empty_thresholds']['std_max'] and int(np.ptp(a[::12,::12]))<=cfg['empty_thresholds']['range_max'] for a in crops.values())
  excluded=next((e for e in cfg['confirmed_non_acquisition_intervals'] if e['start']<=i<=e['end']),None)
  reason='target-level evidence not yet visually adjudicated';status='needs_review';visual=False
  if empty:status='empty_candidate';reason='Both crops near constant; machine flag, not target absence confirmation'
  if excluded:status='excluded_non_acquisition';reason=excluded['reason']
  if i<cfg['earliest_accepted_frame']:status='excluded_before_minimum';reason='Configured earliest acceptance boundary'
  if stage!='export' and any(a<=i<=b for a,b in cfg['candidate_cache_intervals']):
   save(root/f'work/candidates/f{i:06d}_optical.png',crops['optical'])
  if i in accepted:status='confirmed_not_selected';reason='Confirmed target degradation; coverage/order selection';visual=True
  if i in pair_ids and stage!='scan':
   if empty or excluded or i<cfg['earliest_accepted_frame']:raise ValueError('Reviewed selected frame violates exclusion')
   pid=pair_ids[i];status='accepted';reason=';'.join(sorted({a['degradation_label'] for a in accepted[i]}))
   row={'pair_id':pid,'sequence_id':cfg['sequence_id'],'source_video':str(src),'frame_index':i,'timestamp_sec':pts,'timestamp_source':pts_source,'pair_semantics':'same_screen_frame','sensor_sync_status':'sensor_acq_time_unverified','roi_version':cfg['roi_version'],'pair_degradation_status':'optical_degraded_sonar_needs_review','acceptance_basis':reason,'detection_annotation_status':'unannotated','sonar_review_status':'needs_review'}
   degraded={a['modality'] for a in accepted[i]}
   row['pair_degradation_status']='both_degraded' if len(degraded)==2 else ('optical_degraded_sonar_needs_review' if 'optical' in degraded else 'sonar_degraded_optical_needs_review')
   row['optical_review_status']='degraded_confirmed' if 'optical' in degraded else 'needs_review'
   row['sonar_review_status']='degraded_confirmed' if 'sonar' in degraded else 'needs_review'
   row.update(source_pts=timestamps.get(i,{}).get('source_pts',''),source_time_base=timestamps.get(i,{}).get('source_time_base',''))
   hs=[]
   for m,a in crops.items():
    rel=f'data/processed/{m}/{pid:04d}.png';save(root/rel,a);sha=digest(a);hs.append(sha)
    row.update({m+'_path':rel,m+'_roi':json.dumps(rois[m]['xywh']),m+'_width':a.shape[1],m+'_height':a.shape[0],m+'_sha256':sha,m+'_duplicate_of_frame':seen[m].get(sha,'')});seen[m].setdefault(sha,i)
   key=':'.join(hs);row['pair_duplicate_of_frame']=seen['pair'].get(key,'');seen['pair'].setdefault(key,i)
   from denoise_sonar import process,PARAMS
   clean,mask,support,delta,stats=process(crops['sonar']);rel=f'data/processed/sonar_denoised/{pid:04d}.png';save(root/rel,clean)
   save(root/f'data/processed/sonar_masks/{pid:04d}.png',mask.astype('uint8')*255);save(root/f'data/processed/sonar_support_masks/{pid:04d}.png',support.astype('uint8')*255)
   row['sonar_denoised_path']=rel;row['sonar_preprocessing']=cfg['sonar_preprocessing'];row['sonar_denoised_sha256']=digest(clean)
   denoise_stats.append({'pair_id':pid,'frame_index':i,**stats})
   for n,annotation in enumerate(accepted[i]):
    a=dict(annotation);a.update(pair_id=pid,sequence_id=cfg['sequence_id'],class_name=CLASSES[a['class_id']],evidence_path=f'data/processed/previews/evidence_{pid:04d}_{n:02d}.png',detection_annotation_status='unannotated',sonar_visibility='uncertain',cross_modal_obj_id='')
    xy=a['evidence_region'];x1,y1,x2,y2=xy;im=Image.fromarray(crops[a['modality']]);assert 0<=x1<x2<=im.width and 0<=y1<y2<=im.height
    patch=im.crop((max(0,x1-20),max(0,y1-20),min(im.width,x2+20),min(im.height,y2+20)))
    canvas=Image.new('RGB',(max(420,patch.width),patch.height+55),'#111111');canvas.paste(patch,(0,55));dr=ImageDraw.Draw(canvas);dr.text((5,3),f'pair={pid} frame={i} {a["modality"]} {a["class_name"]}',fill='white');dr.text((5,22),a['degradation_label']+' xyxy='+str(xy),fill='yellow');canvas.save(root/a['evidence_path']);a['evidence_region']=json.dumps(xy);anns.append(a)
   if pid in [0,len(selected)//2,len(selected)-1]:
    canvas=Image.new('RGB',(1280,420),'#111111');dr=ImageDraw.Draw(canvas);dr.text((10,5),f'pair={pid} frame={i}; raw optical / stripe-suppressed sonar',fill='white')
    for m,ar,x in [('optical',crops['optical'],0),('sonar',clean,640)]:
     im=Image.fromarray(ar);im.thumbnail((640,380));canvas.paste(im,(x,30))
    canvas.save(root/f'data/processed/previews/pair_{pid:04d}.png')
   rows.append(row)
  review.append({'frame_index':i,'timestamp_sec':pts,'timestamp_source':pts_source,'status':status,'reason':reason,'machine_screened':True,'target_visual_review':visual})
  review[-1].update(source_pts=timestamps.get(i,{}).get('source_pts',''),source_time_base=timestamps.get(i,{}).get('source_time_base',''))
  i+=1
  if i%500==0:print(json.dumps({'decoded':i,'total':expected,'exported':len(rows),'elapsed_sec':round(time.time()-start)}),flush=True)
 cap.release()
 if timestamps and i!=len(timestamps):raise ValueError('Sequential decoded count does not match PTS audit')
 assert before=={'size':src.stat().st_size,'mtime_ns':src.stat().st_mtime_ns},'source changed during read'
 writecsv(root/'reports/frame_review.csv',review,list(review[0]) if review else ['frame_index','status','reason'])
 fields=['pair_id','sequence_id','frame_index','modality','instance_id','class_id','class_name','degradation_label','evidence_region','sonar_visibility','cross_modal_obj_id','review_status','evidence_path','reason','detection_annotation_status','reviewer','identity_basis']
 if stage!='scan':
  writecsv(root/'data/processed/pairs.csv',rows,list(rows[0]) if rows else ['pair_id','frame_index','optical_path','sonar_path'])
  writecsv(root/'annotations/degradation_annotations.csv',anns,fields)
  template=[dict(pair_id=r['pair_id'],sequence_id=r['sequence_id'],frame_index=r['frame_index'],modality=m,detection_annotation_status='unannotated',review_status='needs_review') for r in rows for m in ['optical','sonar']]
  writecsv(root/'annotations/annotation_template.csv',template,fields+['x1','y1','x2','y2'])
  writecsv(root/'annotations/pending_review.csv',[r for r in review if not r['target_visual_review']],list(review[0]))
  (root/'reports/sonar_preprocess.json').write_text(json.dumps({'parameters':PARAMS if rows else {},'samples':denoise_stats},indent=2))
 stat.update(status='executed',metadata_frame_count=expected,decoded_frame_count=i,end_of_stream_reached=i==expected,source_size=size,fps=fps,exported_pairs=len(rows),confirmed_frames=len(accepted),target_visual_reviewed_frames=sum(r['target_visual_review'] for r in review),machine_only_frames=sum(not r['target_visual_review'] for r in review),missing_labels=sorted(set(LABELS)-covered),source_read_only_verified=True,elapsed_sec=time.time()-start)
 (root/'reports/extraction_status.json').write_text(json.dumps(stat,indent=2))
 print(json.dumps(stat),flush=True)
 if stage!='scan':subprocess.run([sys.executable,str(root/'scripts/verify_preprocess.py'),'--config',str(Path(config).resolve())],check=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stage',choices=['all','scan','export'],default='all');a=p.parse_args();run(a.config,a.stage)
