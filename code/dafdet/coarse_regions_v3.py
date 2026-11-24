"""Unverified finite-cell acoustic/optical search regions for the 50-pair pilot.

The user accepted original optical pixels and coarse parallel-axis geometry.
This adapter never labels itself measured calibration and never reads GT.
It samples the prior fixed screen15_pad16 hypothesis on an 80x56 grid.
"""
from pathlib import Path
import json
import numpy as np

class CoarseRegionRegistry:
    def __init__(self,root):
        base=Path(root)/'calibration_materials/04_reports/coarse_projection_50_20261113'
        self.protocol=json.loads((base/'protocol.json').read_text(encoding='utf-8'))
        self.audit=json.loads((base/'grid_geometry.json').read_text(encoding='utf-8'))
        self.objects={}
        self.paths=[base/'protocol.json',base/'grid_geometry.json']
        for i,rec in enumerate(self.audit['records']):
            pid=int(Path(rec['image_path']).stem)
            self.objects[pid]=CoarseRegions(self.protocol,rec)
    def __getitem__(self,pid):return self.objects[int(pid)]

class CoarseRegions:
    def __init__(self,protocol,record,cols=80,rows=56):
        self.config={'status':'approximate_unverified','method':'finite_cell_lookup',
            'camera_model':'parallel axes, uncorrected optical crop','scenario':'screen15_pad16',
            'grid':[cols,rows],'calibration_verified':False}
        sw,sh=protocol['sonar_size_wh'];ow,oh=protocol['optical_size_wh']
        self.optical_size=(ow,oh);self.sonar_size=(sw,sh)
        k=protocol['K_crop'];baseline=protocol['baseline_camera_below_sonar_m']
        cx,cy=record['common_circle_center_xy'];radius=record['outer_radius_px']
        amin,amax=record['display_boundary_angles_deg'];max_range=record['displayed_range_m']
        xx,yy=np.meshgrid(np.arange(cols),np.arange(rows))
        x1=xx.ravel()*sw/cols;y1=yy.ravel()*sh/rows
        x2=x1+sw/cols;y2=y1+sh/rows
        source=np.stack([x1,y1,x2,y2],-1)
        # 4 corners plus center. Approximate raster support, not calibrated uncertainty.
        x=np.stack([x1,x1,x2,x2,(x1+x2)/2],1)
        y=np.stack([y1,y2,y1,y2,(y1+y2)/2],1)
        rho=np.hypot(x-cx,cy-y);ang=np.rad2deg(np.arctan2(x-cx,cy-y))
        ranges=max_range*rho/radius
        valid=(rho>1)&(rho<=radius)&(y<=cy)&(ang>=amin)&(ang<=amax)&(ranges>0.1)
        theta=np.deg2rad(ang-(amin+amax)/2);phi=np.deg2rad(15.)
        u=k['fx']*np.tan(theta)+k['cx']
        def vertical(ph):return k['fy']*(np.tan(ph)-baseline/(np.maximum(ranges,.1)*np.cos(ph)))/np.cos(theta)+k['cy']
        vlow=vertical(-phi);vhigh=vertical(phi)
        valid &= np.isfinite(u)&np.isfinite(vlow)&np.isfinite(vhigh)&(np.cos(theta)>0)
        lo=np.stack([np.min(np.where(valid,u,np.inf),1),np.min(np.where(valid,vlow,np.inf),1)],1)-16
        hi=np.stack([np.max(np.where(valid,u,-np.inf),1),np.max(np.where(valid,vhigh,-np.inf),1)],1)+16
        projected=np.clip(np.concatenate([lo,hi],1),[0,0,0,0],[ow,oh,ow,oh])
        good=valid.any(1)&np.isfinite(projected).all(1)&(projected[:,2]>projected[:,0])&(projected[:,3]>projected[:,1])
        self.source=(source[good]/[sw,sh,sw,sh]).astype('float32')
        self.projected=(projected[good]/[ow,oh,ow,oh]).astype('float32')
        self.config.update(display_range_m=max_range,valid_lookup_cells=int(good.sum()),pair_id=int(Path(record['image_path']).stem))

    def normalized(self,boxes,direction,flipped=False):
        a=np.asarray(boxes,dtype='float32');shape=a.shape
        if a.shape[-1]!=4 or not np.isfinite(a).all():raise ValueError('Invalid normalized boxes')
        b=a.reshape(-1,4)
        xy=np.concatenate([b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2],1)
        if flipped:xy[:,[0,2]]=1-xy[:,[2,0]]
        if direction=='sonar_to_optical':src,dst=self.source,self.projected
        elif direction=='optical_to_sonar':src,dst=self.projected,self.source
        else:raise ValueError('Unknown projection direction')
        result=np.zeros_like(xy);valid=np.zeros(len(xy),bool)
        for start in range(0,len(xy),256):
            q=xy[start:start+256]
            hit=(q[:,None,0]<src[None,:,2])&(q[:,None,2]>src[None,:,0])&(q[:,None,1]<src[None,:,3])&(q[:,None,3]>src[None,:,1])
            ok=hit.any(1)&(q[:,2]>q[:,0])&(q[:,3]>q[:,1])
            # Reduce each float32 coordinate without allocating hit x cells x 2 float temporaries.
            lo=np.stack([np.min(np.broadcast_to(dst[None,:,k],hit.shape),axis=1,where=hit,initial=np.inf) for k in (0,1)],1)
            hi=np.stack([np.max(np.broadcast_to(dst[None,:,k],hit.shape),axis=1,where=hit,initial=-np.inf) for k in (2,3)],1)
            out=np.concatenate([lo,hi],1);out[~ok]=0
            result[start:start+len(q)]=out;valid[start:start+len(q)]=ok
        if flipped:result[:,[0,2]]=1-result[:,[2,0]]
        out=np.concatenate([(result[:,:2]+result[:,2:])/2,result[:,2:]-result[:,:2]],1)
        out[~valid]=0
        return out.reshape(shape).astype('float32'),valid.reshape(shape[:-1])
