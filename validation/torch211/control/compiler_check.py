import json,pathlib,torch
from torch._inductor import config
root=pathlib.Path('/workspace/validation211')
assert torch.__version__.startswith('2.11.0')
assert config.triton.mix_order_reduction
torch.manual_seed(0)
x=torch.randn(8,1024,768,device='cuda',dtype=torch.bfloat16)
def fn(x):
    return x.sum((0,1)).float(),x.float().square().sum(-1)
expected=fn(x);actual=torch.compile(fn,dynamic=False)(x)
rows=[]
for name,a,b in zip(('column_sum','row_sum'),expected,actual):
    assert b.isfinite().all()
    torch.testing.assert_close(b,a,atol=1e-3,rtol=0.01)
    rows.append(dict(name=name,finite=True,max_error=float((b-a).abs().max()),relative_l2=float((b-a).norm()/a.norm())))
(root/'compiler-check.json').write_text(json.dumps(dict(torch=torch.__version__,mix_order_reduction=config.triton.mix_order_reduction,checks=rows),indent=2))
print(json.dumps(rows,indent=2))
