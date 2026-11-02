import json,pathlib,torch

torch.manual_seed(0)
x=torch.randn(8,1024,768,device='cuda',dtype=torch.bfloat16)
def fn(x):
    return x.sum((0,1)).float(), x.float().square().sum(-1)
eager=fn(x)
compiled=torch.compile(fn,dynamic=False)(x)
rows=[]
for name,a,b in zip(['column_sum','row_sum'],eager,compiled):
    rows.append(dict(name=name,finite=bool(b.isfinite().all()),max_actual=float(b.abs().max()),max_error=float((b-a).abs().max())))
print(json.dumps(rows,indent=2))
