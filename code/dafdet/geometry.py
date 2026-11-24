"""Measured-calibration acoustic/optical geometry. All pixel boxes use xyxy.

No camera, beam or rendering parameter is inferred from the video. The inverse
map is a finite lookup intersection, not an analytic inverse or exact registration.
"""
from pathlib import Path
import json
import numpy as np

class CalibrationError(ValueError): pass

def project_points(r, theta, phi, K, R_sc, t_sc):
    r, theta, phi = np.broadcast_arrays(r, theta, phi)
    ps = np.stack([r*np.cos(phi)*np.cos(theta),r*np.cos(phi)*np.sin(theta),r*np.sin(phi)],-1)
    pc = ps @ np.asarray(R_sc,dtype=float).T + np.asarray(t_sc,dtype=float)
    valid = np.isfinite(pc).all(-1) & (pc[...,0] > 0)
    den=np.where(valid,pc[...,0],1)
    uv=np.stack([K['fx']*pc[...,1]/den+K['cx'], K['fy']*pc[...,2]/den+K['cy']],-1)
    uv[~valid]=np.nan
    return uv,valid

def flip_box(box,width):
    b=np.asarray(box,dtype=float).copy(); b[...,0],b[...,2]=width-b[...,2].copy(),width-b[...,0].copy(); return b

class AcousticOpticalGeometry:
    def __init__(self, config, allow_test_calibration=False):
        if isinstance(config,(str,Path)):
            p=Path(config); import yaml; cfg=yaml.safe_load(p.read_text(encoding='utf-8-sig')); base=p.parent
        else: cfg=dict(config);base=Path.cwd()
        self.config=cfg
        if cfg.get('status')!='verified' and not (allow_test_calibration and cfg.get('status')=='synthetic_test_only'):
            raise CalibrationError('Measured calibration must have status=verified; synthetic_test_only is allowed only in tests.')
        required=['sequence_id','distance_unit','angle_unit','coordinate_convention','intrinsics_coordinate_system','K','R_sc','t_sc','optical_size','sonar_size','vertical_beam_rad','epsilon_px']
        for field in required:
            if field not in cfg: raise CalibrationError('Missing calibration field: '+field)
        if cfg['distance_unit']!='metre' or cfg['angle_unit']!='radian' or cfg['coordinate_convention']!='forward_right_down' or cfg['intrinsics_coordinate_system']!='optical_crop':
            raise CalibrationError('Explicit conversion required to metres, radians, forward_right_down and optical_crop intrinsics')
        if cfg.get('status')=='verified':
            if cfg.get('range_setting_status')!='verified_constant' or not cfg.get('range_setting_source'):
                raise CalibrationError('A measured mapping needs verified constant sonar range/render settings for its segment')
            interval=cfg.get('valid_frame_interval')
            if not isinstance(interval,list) or len(interval)!=2 or interval[0]<0 or interval[1]<interval[0]:
                raise CalibrationError('Missing valid_frame_interval [first,last] for this calibration/range segment')
        self.ow,self.oh=cfg['optical_size'];self.sw,self.sh=cfg['sonar_size']
        R=np.array(cfg['R_sc']);
        if not np.allclose(R.T@R,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(R),1,atol=1e-5):raise CalibrationError('R_sc is not a proper rotation')
        if 'mapping_npz' in cfg:
            maps=np.load(base/cfg['mapping_npz']); self.r=maps['range_m'];self.theta=maps['bearing_rad'];self.mask=maps['valid'].astype(bool)
        elif allow_test_calibration and cfg.get('status')=='synthetic_test_only':
            self.r=np.asarray(cfg['range_m']);self.theta=np.asarray(cfg['bearing_rad']);self.mask=np.asarray(cfg['valid'],bool)
        else: raise CalibrationError('Missing measured pixel range/bearing/valid mapping_npz')
        if self.r.shape!=(self.sh,self.sw) or self.theta.shape!=self.r.shape or self.mask.shape!=self.r.shape: raise CalibrationError('Mapping size differs from sonar crop')
        self.phi=np.linspace(*cfg['vertical_beam_rad'],cfg.get('elevation_samples',33))
        self.cells=None;self.projected=None
    def _regions_for_pixels(self,xy):
        xy=np.asarray(xy,int).reshape(-1,2)
        if not len(xy):return np.empty((0,4)),np.zeros(0,bool)
        r=self.r[xy[:,1],xy[:,0],None];th=self.theta[xy[:,1],xy[:,0],None]
        uv,good=project_points(r,th,self.phi[None],self.config['K'],self.config['R_sc'],self.config['t_sc'])
        valid=good.any(1);eps=self.config['epsilon_px']
        lo=np.min(np.where(good[...,None],uv,np.inf),axis=1)-eps
        hi=np.max(np.where(good[...,None],uv,-np.inf),axis=1)+eps
        box=np.concatenate([lo,hi],1);box=np.clip(box,[0,0,0,0],[self.ow,self.oh,self.ow,self.oh])
        valid &= (box[:,2]>box[:,0])&(box[:,3]>box[:,1]);box[~valid]=0
        return box,valid
    def sonar_to_optical_region(self,box,flipped=False):
        if flipped:box=flip_box(box,self.sw)
        bounds=np.asarray(box,dtype=float)
        # Remove round-trip arithmetic noise at integer cell boundaries only.
        # 1e-6 pixel is far below one pixel; genuine partial-cell overlap remains.
        nearest=np.rint(bounds)
        bounds=np.where(np.abs(bounds-nearest)<=1e-6,nearest,bounds)
        x1,y1,x2,y2=bounds
        x1,y1=np.maximum(np.floor([x1,y1]),0).astype(int);x2,y2=np.minimum(np.ceil([x2,y2]),[self.sw,self.sh]).astype(int)
        if x2<=x1 or y2<=y1:return np.zeros(4),False
        yy,xx=np.nonzero(self.mask[y1:y2,x1:x2]);xy=np.stack([xx+x1,yy+y1],1)
        # All valid pixels within proposal, no subsampling of forward mapping.
        boxes,valid=self._regions_for_pixels(xy)
        if not valid.any():return np.zeros(4),False
        b=boxes[valid]; out=np.r_[b[:,:2].min(0),b[:,2:].max(0)]
        return (flip_box(out,self.ow) if flipped else out),True
    def build_lookup(self):
        # Every valid sonar cell retained; chunking bounds temporary RAM.
        yy,xx=np.nonzero(self.mask); cells=np.stack([xx,yy],1);regs=[];valid=[]
        for start in range(0,len(cells),4096):
            b,v=self._regions_for_pixels(cells[start:start+4096]);regs.append(b);valid.append(v)
        v=np.concatenate(valid) if valid else np.zeros(0,bool)
        self.cells=cells[v];self.projected=np.concatenate(regs)[v] if regs else np.empty((0,4))
    def optical_to_sonar_region(self,box,flipped=False):
        if self.cells is None:self.build_lookup()
        if flipped:box=flip_box(box,self.ow)
        b=np.asarray(box);p=self.projected
        hit=(p[:,0]<=b[2])&(p[:,2]>=b[0])&(p[:,1]<=b[3])&(p[:,3]>=b[1])
        if not hit.any():return np.zeros(4),False
        xy=self.cells[hit];out=np.r_[xy.min(0),xy.max(0)+1].astype(float)
        return (flip_box(out,self.sw) if flipped else out),True
    def normalized(self,boxes,direction,flipped=False):
        arr=np.asarray(boxes);out=[];valid=[]
        # Input and output cxcywh normalized, geometry always in original crops.
        si=np.array([self.sw,self.sh]) if direction=='sonar_to_optical' else np.array([self.ow,self.oh])
        so=np.array([self.ow,self.oh]) if direction=='sonar_to_optical' else np.array([self.sw,self.sh])
        fn=self.sonar_to_optical_region if direction=='sonar_to_optical' else self.optical_to_sonar_region
        for b in arr.reshape(-1,4):
            xyxy=np.r_[(b[:2]-b[2:]/2)*si,(b[:2]+b[2:]/2)*si]
            r,v=fn(xyxy,flipped=flipped);out.append(np.r_[(r[:2]+r[2:])/2/so,(r[2:]-r[:2])/so]);valid.append(v)
        return np.asarray(out,dtype='float32').reshape(arr.shape),np.array(valid).reshape(arr.shape[:-1])
