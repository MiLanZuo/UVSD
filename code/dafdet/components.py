"""Shared ResNet/HybridEncoder, deformable attention and loss components.

Extracted without the superseded A-series experiment-policy implementation.
The revised six-row model owns input modality, query selection and geometry.
"""
import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F
from .upstream import load_upstream
PPDET_PATH=load_upstream()
from ppdet.modeling.backbones.resnet import ResNet
from ppdet.modeling.transformers.hybrid_encoder import HybridEncoder,TransformerLayer
from ppdet.modeling.transformers.utils import get_contrastive_denoising_training_group
from ppdet.modeling.transformers.matchers import HungarianMatcher
import ppdet.modeling.transformers as _transforms
from ppdet.modeling.transformers import utils as _utils
for _name in ["bbox_cxcywh_to_xyxy","sigmoid_focal_loss","varifocal_loss_with_logits"]:
    setattr(_transforms,_name,getattr(_utils,_name))
from ppdet.modeling.losses.detr_loss import DINOLoss,DETRLoss

def inverse_sigmoid(x):
    x=paddle.clip(x,1e-5,1-1e-5);return paddle.log(x/(1-x))

def gather(x,idx):
    return paddle.take_along_axis(x,idx.unsqueeze(-1).expand([-1,-1,x.shape[-1]]),axis=1)

def select_queries(logits, n_visual, k, strategy='competitive',valid=None):
    scores=F.sigmoid(logits).max(-1)
    if valid is not None:scores=paddle.where(valid,scores,paddle.full_like(scores,-1e9))
    if strategy!='competitive':raise ValueError('Only shared top-k selection belongs to the revised six-row implementation')
    _,idx=paddle.topk(scores,k,axis=1)
    if valid is not None and not bool(paddle.take_along_axis(valid.astype("int32"),idx,axis=1).all()):raise ValueError('Insufficient valid candidates for requested query count')
    return idx,(idx>=n_visual).astype('int64')

class MLP(nn.Layer):
    def __init__(self,inp,hid,out):
        super().__init__();self.layers=nn.Sequential(nn.Linear(inp,hid),nn.ReLU(),nn.Linear(hid,hid),nn.ReLU(),nn.Linear(hid,out))
    def forward(self,x):return self.layers(x)

class RegionAttention(nn.Layer):
    def __init__(self,dim=256,heads=8,modalities=2,levels=3,points=4):
        super().__init__();self.h=heads;self.m=modalities;self.l=levels;self.p=points;self.d=dim//heads
        self.offsets=nn.Linear(dim,heads*modalities*levels*points*2)
        self.weights=nn.Linear(dim,heads*modalities*levels*points)
        self.value=nn.Linear(dim,dim);self.output=nn.Linear(dim,dim)
        nn.initializer.Constant(0)(self.offsets.weight);nn.initializer.Constant(0)(self.weights.weight);nn.initializer.Constant(0)(self.weights.bias)
        # Effective box-relative offsets absorb upstream 0.5/P exactly once.
        angle=np.arange(heads)*2*np.pi/heads;grid=np.stack([np.cos(angle),np.sin(angle)],-1)
        grid=grid/np.abs(grid).max(-1,keepdims=True)
        grid=np.tile(grid[:,None,None,None,:],(1,modalities,levels,points,1))
        grid*=np.arange(1,points+1)[None,None,None,:,None]*(0.5/points)
        self.offsets.bias.set_value(grid.astype('float32').reshape(-1))
    def forward(self,q,features,regions,valid,diagnostics=False):
        b,n,c=q.shape
        off=self.offsets(q).reshape([b,n,self.h,self.m,self.l,self.p,2])
        logits=self.weights(q).reshape([b,n,self.h,self.m,self.l,self.p])
        mask=valid[:,:,None,:,None,None]
        logits=paddle.where(mask,logits,paddle.full_like(logits,-1e9))
        weights=F.softmax(logits.reshape([b,n,self.h,-1]),axis=-1).reshape(logits.shape)
        weights=weights*mask.astype(weights.dtype)
        weights=weights/paddle.clip(weights.sum([3,4,5],keepdim=True),min=1e-12)
        xy=regions[:,:,None,:,None,None,:2]+regions[:,:,None,:,None,None,2:]*off
        total=paddle.zeros([b,n,self.h,self.d],dtype=q.dtype)
        for m in range(self.m):
            for l in range(self.l):
                feat=features[m][l];hh,ww=feat.shape[-2:]
                v=self.value(feat.flatten(2).transpose([0,2,1])).reshape([b,hh,ww,self.h,self.d]).transpose([0,3,4,1,2]).reshape([b*self.h,self.d,hh,ww])
                grid=(2*xy.reshape([b,n,self.h,self.m*self.l,self.p,2])[:,:,:,m*self.l+l]-1).transpose([0,2,1,3,4]).reshape([b*self.h,n,self.p,2])
                sample=F.grid_sample(v,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
                sample=sample.reshape([b,self.h,self.d,n,self.p]).transpose([0,3,1,4,2])
                total+=(sample*weights[:,:,:,m,l,:,None]).sum(3)
        out=self.output(total.reshape([b,n,c]))
        diag={'sampling_locations':xy,'attention_weights':weights,'effective_offsets':off,'reference_regions':regions,'region_valid':valid} if diagnostics else None
        return out,diag

class DecoderLayer(nn.Layer):
    def __init__(self,dim=256,heads=8,modalities=2,ff=1024):
        super().__init__();self.self_attn=nn.MultiHeadAttention(dim,heads,dropout=0)
        self.cross=RegionAttention(dim,heads,modalities)
        self.norm1=nn.LayerNorm(dim);self.norm2=nn.LayerNorm(dim);self.norm3=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.Linear(dim,ff),nn.ReLU(),nn.Linear(ff,dim))
    def forward(self,q,pos,features,regions,valid,mask=None,diagnostics=False):
        attn_mask=None if mask is None else paddle.where(mask,paddle.zeros(mask.shape),paddle.full(mask.shape,float('-inf')))
        q=self.norm1(q+self.self_attn(q+pos,q+pos,q,attn_mask=attn_mask))
        # Paper offsets and weights are predicted from post-self-attention q.
        fused,diag=self.cross(q,features,regions,valid,diagnostics);q=self.norm2(q+fused);q=self.norm3(q+self.ff(q));return q,diag

class SharedDetectionScaffold(nn.Layer):
    def __init__(self, config, modalities, decoder_modalities=None):
        super().__init__()
        self.config = dict(config)
        cfg = self.config
        self.k = cfg.get('num_queries', 300)
        self.depth = cfg.get('decoder_layers', 6)
        self.modalities = modalities
        dim=cfg.get('hidden_dim',256);self.dim=dim
        self.backbones=nn.LayerList([ResNet(depth=50,variant='d',return_idx=[1,2,3],freeze_at=-1,freeze_norm=True,lr_mult_list=[.1]*4) for _ in range(self.modalities)])
        self.encoders=nn.LayerList([HybridEncoder(hidden_dim=dim,encoder_layer=TransformerLayer(d_model=dim,nhead=8,dim_feedforward=1024,dropout=0.,activation='gelu')) for _ in range(self.modalities)])
        self.memory_proj=nn.Sequential(nn.Linear(dim,dim),nn.LayerNorm(dim))
        self.enc_score_head=nn.Linear(dim,4);self.enc_box_head=MLP(dim,dim,4)
        nn.initializer.Constant(-4.595)(self.enc_score_head.bias)
        nn.initializer.Constant(0)(self.enc_box_head.layers[-1].weight);nn.initializer.Constant(0)(self.enc_box_head.layers[-1].bias)
        self.pos=MLP(4,2*dim,dim)
        self.decoder=nn.LayerList([DecoderLayer(dim,8,decoder_modalities or self.modalities) for _ in range(self.depth)])
        self.cls_heads=nn.LayerList([nn.Linear(dim,4) for _ in range(self.depth)])
        self.box_heads=nn.LayerList([MLP(dim,dim,4) for _ in range(self.depth)])
        for h in self.box_heads:
            nn.initializer.Constant(0)(h.layers[-1].weight);nn.initializer.Constant(0)(h.layers[-1].bias)
        for h in self.cls_heads:nn.initializer.Constant(-4.595)(h.bias)
        self.dn_embed=nn.Embedding(4,dim)
        self.initialization={'policy':'random_no_pretrained_loaded','loaded':[],'skipped':[],'reinitialized':list(self.state_dict())}
    def load_pretrained(self,path):
        state=paddle.load(path);current=self.state_dict();mapped={};skipped=[]
        for key,value in state.items():
            candidates=[key] if key.startswith(('backbones.','encoders.')) else []
            for m in range(self.modalities):
                if key.startswith('backbone.'):candidates.append(f'backbones.{m}.'+key[len('backbone.'):])
                if key.startswith('neck.'):candidates.append(f'encoders.{m}.'+key[len('neck.'):])
            used=False
            for target in candidates:
                if target in current and list(current[target].shape)==list(value.shape):mapped[target]=value;used=True
            if not used:skipped.append(key)
        self.set_state_dict({**current,**mapped});self.initialization={'policy':'compatible_backbone_encoder_only','loaded':sorted(mapped),'skipped':skipped,'reinitialized':sorted(set(current)-set(mapped))}
        return self.initialization
    def _anchors(self,feats):
        arr=[]
        for l,f in enumerate(feats):
            h,w=f.shape[-2:];y,x=paddle.meshgrid(paddle.arange(h,dtype='float32'),paddle.arange(w,dtype='float32'))
            xy=paddle.stack([(x+.5)/w,(y+.5)/h],-1).reshape([-1,2]);wh=paddle.full(xy.shape,.05*(2**l));arr.append(paddle.concat([xy,wh],-1))
        return paddle.concat(arr).unsqueeze(0)

class CommonDetectionCriterion(nn.Layer):
    def __init__(self):
        super().__init__();matcher=HungarianMatcher(matcher_coeff={'class':2,'bbox':5,'giou':2},use_focal_loss=True)
        kwargs=dict(num_classes=4,matcher=matcher,loss_coeff={'class':1,'bbox':5,'giou':2},use_focal_loss=True,use_vfl=True)
        self.main=DINOLoss(aux_loss=True,**kwargs);self.encoder=DETRLoss(aux_loss=False,**kwargs)
    def forward(self,out,targets):
        self.main.aux_loss=out['all_boxes'].shape[0]>1
        loss=self.main(out['all_boxes'],out['all_logits'],targets['gt_bbox'],targets['gt_class'],dn_out_bboxes=out['dn_boxes'],dn_out_logits=out['dn_logits'],dn_meta=out['dn_meta'])
        # An unannotated modality is absent, never an empty negative target.
        for m,prefix in enumerate(['','sonar_']):
            if m>=len(out['encoder_outputs']) or prefix+'gt_bbox' not in targets:continue
            e=out['encoder_outputs'][m]
            # Candidate validity is identical across this fixed input-size batch.
            ids=paddle.nonzero(e['valid'][0]).flatten()
            eb=paddle.index_select(e['pred_boxes'],ids,axis=1).unsqueeze(0);el=paddle.index_select(e['pred_logits'],ids,axis=1).unsqueeze(0)
            loss.update(self.encoder(eb,el,targets[prefix+'gt_bbox'],targets[prefix+'gt_class'],postfix='_enc_'+str(m)))
        return loss

def postprocess(out,original_size,max_detections=100):
    # Upstream focal postprocessing ranks flattened query/class scores.
    probs=F.sigmoid(out['pred_logits']);flat=probs.flatten(1)
    score,flat_idx=paddle.topk(flat,min(max_detections,flat.shape[1]),axis=1)
    idx=flat_idx//probs.shape[-1];classes=flat_idx%probs.shape[-1]
    b=gather(out['pred_boxes'],idx);xy1=b[:,:,:2]-b[:,:,2:]/2;xy2=b[:,:,:2]+b[:,:,2:]/2
    w,h=original_size;scale=paddle.to_tensor([w,h,w,h],dtype='float32')
    xyxy=paddle.clip(paddle.concat([xy1,xy2],-1),0,1)*scale
    return {'boxes_xyxy':xyxy,'scores':score,'classes':classes}
