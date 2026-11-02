import json, os, pathlib, torch
import torch.nn.functional as F
from torch._inductor import config
config.triton.mix_order_reduction=os.environ.get('TORCHINDUCTOR_MIX_ORDER_REDUCTION','1')=='1'
torch.manual_seed(0)

def fn(x,b,g,v):
    y=x+b.to(x.dtype)
    return (F.rms_norm(y,(y.shape[-1],),weight=g.to(x.dtype))*v).sum()

x=torch.randn(8,1024,768,device='cuda',dtype=torch.bfloat16,requires_grad=True)
b=torch.randn(768,device='cuda',requires_grad=True)
g=torch.ones(768,device='cuda',requires_grad=True)
v=torch.randn_like(x)
ref=torch.autograd.grad(fn(x,b,g,v),(x,b,g))
compiled=torch.compile(fn,dynamic=False)
actual=torch.autograd.grad(compiled(x,b,g,v),(x,b,g))
rows=[]
for name,a,c in zip(['input','bias','gain'],ref,actual):
    rows.append(dict(name=name,finite=bool(c.isfinite().all()),max_error=float((c.float()-a.float()).abs().max()),relative_l2=float((c.float()-a.float()).norm()/a.float().norm()),max_actual=float(c.abs().max())))
print(json.dumps(dict(mix_order=config.triton.mix_order_reduction,rows=rows),indent=2),flush=True)
