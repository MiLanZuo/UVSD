"""Deterministic fixed radial-stripe suppression on rendered sonar images."""
from pathlib import Path
import json, cv2, numpy as np
from scipy.ndimage import median_filter, gaussian_filter, gaussian_filter1d
from PIL import Image,ImageDraw,ImageFont

PARAMS={
 'method':'polar_robust_radial_background_subtraction_v1',
 'apex_xy':[575.0,786.0], 'max_radius_px':780,
 'angle_step_degrees':0.1,'angle_limits_degrees':[-40,40],
 'stripe_centers_degrees':[-23.2,0.0,23.1],
 'support_halfwidth_degrees':3.2,'sideband_offset_degrees':[3.6,5.0],
 'radial_median_window_px':101,'radial_gaussian_sigma_px':8,
 'angular_gaussian_sigma_samples':1,'positive_excess_margin_8bit':4,
 'activation_excess_8bit':10,'preserved_gray_max_chroma_ratio':0.30,
 'preserved_gray_min_channel':35,'background_floor_fraction':0.55,
 'near_apex_fade_radius_px':[10,55], 'outer_fade_radius_px':[775,779]
}

def process(rgb):
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.shape!=(797,1155,3):
        raise ValueError('process expects uint8 RGB with fixed 1155x797 crop geometry')
    h,w=rgb.shape[:2]; cx,cy=PARAMS['apex_xy']; step=PARAMS['angle_step_degrees']
    angles=np.arange(-40,40+step/2,step,dtype=np.float32)
    radius=np.arange(781,dtype=np.float32)
    a=np.deg2rad(angles)[None,:];r=radius[:,None]
    mx=(cx+r*np.sin(a)).astype('float32');my=(cy-r*np.cos(a)).astype('float32')
    value=rgb.max(axis=2).astype('float32')
    polar=cv2.remap(value,mx,my,cv2.INTER_LINEAR)
    continuous=gaussian_filter(median_filter(polar,size=(101,1),mode='nearest'),sigma=(8,1))
    correction=np.zeros_like(polar);background=np.zeros_like(polar)
    strengths=[]
    for center in PARAMS['stripe_centers_degrees']:
        side_l=(angles>=center-5)&(angles<=center-3.6)
        side_r=(angles>=center+3.6)&(angles<=center+5)
        left=gaussian_filter1d(median_filter(np.median(polar[:,side_l],axis=1),size=101,mode='nearest'),8)
        right=gaussian_filter1d(median_filter(np.median(polar[:,side_r],axis=1),size=101,mode='nearest'),8)
        band=np.abs(angles-center)<=3.2
        weight=np.clip((angles[band]-(center-4.3))/8.6,0,1)
        baseline=left[:,None]*(1-weight)+right[:,None]*weight
        extra=np.maximum(continuous[:,band]-baseline-4,0)
        core=(np.abs(angles[band]-center)<=1.0)&(np.abs(angles[band])>.25)
        strength=float(np.median(extra[150:750,core]))
        enabled=strength>=10
        taper=np.clip((3.2-np.abs(angles[band]-center))/.45,0,1)
        if enabled:correction[:,band]=extra*taper
        background[:,band]=baseline
        strengths.append({'angle_degrees':center,'estimated_excess':round(strength,3),'active':bool(enabled)})
    yy,xx=np.indices((h,w),dtype='float32')
    rr=np.hypot(xx-cx,cy-yy);aa=np.degrees(np.arctan2(xx-cx,cy-yy))
    xa=((aa+40)/step).astype('float32');ya=rr.astype('float32')
    corr=cv2.remap(correction,xa,ya,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    bg=cv2.remap(background,xa,ya,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    in_sector=(rr<=779)&(rr>=10)&(np.abs(aa)<=40)&(yy<=cy)
    support=np.zeros((h,w),dtype=bool)
    for s in strengths:
        if s['active']:support|=(np.abs(aa-s['angle_degrees'])<=3.2)
    support&=in_sector
    chroma=rgb.max(axis=2).astype('float32')-rgb.min(axis=2).astype('float32')
    protected=(chroma<=value*.30)&(rgb.min(axis=2)>=35)
    fade=np.clip((rr-10)/45,0,1)*np.clip((779-rr)/4,0,1)
    corr*=fade
    corr=np.minimum(corr,np.maximum(value-.55*bg,0))
    corr[~support|protected]=0
    target=np.maximum(value-corr,0)
    ratio=np.divide(target,value,out=np.ones_like(value),where=value>0)
    output=np.rint(rgb.astype('float32')*ratio[...,None]).clip(0,255).astype('uint8')
    changed=np.any(output!=rgb,axis=2)
    assert np.array_equal(output[~support],rgb[~support])
    assert np.array_equal(output[protected],rgb[protected])
    delta=rgb.astype('int16')-output.astype('int16')
    stats={'bands':strengths,'changed_pixel_count':int(changed.sum()),'changed_fraction_total':round(float(changed.mean()),6),'max_channel_reduction':int(delta.max()),'mean_reduction_changed':round(float(delta.max(axis=2)[changed].mean()),3) if changed.any() else 0,'outside_support_pixels_unchanged':True,'gray_overlay_pixels_unchanged':True,'max_absolute_channel_increase':int((-delta).clip(0).max())}
    return output,changed,support,delta.max(axis=2).astype('uint8'),stats

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Suppress the three fixed radial interference bands. Input images must use the supplied 1155x797 crop geometry.')
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ['sonar','masks','support_masks','difference']:(args.output/name).mkdir()
    reports=[]
    for source in sorted(args.input.glob('*.png')):
        raw=np.array(Image.open(source).convert('RGB'))
        if raw.shape[:2]!=(797,1155):raise ValueError('Expected 1155x797 cropped sonar image: '+str(source))
        clean,mask,support,diff,stats=process(raw)
        for folder,array in [('sonar',clean),('masks',mask.astype('uint8')*255),('support_masks',support.astype('uint8')*255),('difference',diff)]:Image.fromarray(array).save(args.output/folder/source.name)
        reports.append({'pair_id':source.stem,**stats})
        print(source.stem,flush=True)
    (args.output/'parameters.json').write_text(json.dumps(PARAMS,indent=2),encoding='utf-8')
    (args.output/'report.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
