"""Load only needed files from the single pinned DAMSDet checkout.

Package shells avoid eager importing unrelated MOT/deployment dependencies;
all actual model, workspace registry, loss and operators are upstream modules.
Reject any ppdet package already imported from another checkout.
"""
from pathlib import Path
import sys,types
ROOT=Path(__file__).resolve().parents[1]
PPDET=ROOT/'third_party/DAMSDet/ppdet'
def load_upstream():
    for name in ['ppdet','ppdet.core','ppdet.modeling','ppdet.modeling.backbones','ppdet.modeling.transformers','ppdet.modeling.heads','ppdet.modeling.losses','ppdet.utils']:
        path=PPDET.joinpath(*name.split('.')[1:])
        if name in sys.modules:
            f=getattr(sys.modules[name],'__file__','')
            if f and not Path(f).resolve().is_relative_to(PPDET.resolve()):raise RuntimeError('Mixed ppdet checkout: '+f)
        else:
            m=types.ModuleType(name);m.__path__=[str(path)];m.__file__=str(path/'__init__.py');sys.modules[name]=m
    return str(PPDET)
