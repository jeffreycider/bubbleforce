import json,pathlib,torch,sys
root=pathlib.Path('/workspace/diagnosis');ref,other=sys.argv[1:]
a=torch.load(root/f'{ref}-gradients.pt',weights_only=True,mmap=True)
b=torch.load(root/f'{other}-gradients.pt',weights_only=True,mmap=True)
rows=[]
for name in a:
    x=a[name].float();y=b[name].float();d=y-x
    rows.append(dict(name=name,finite=bool(y.isfinite().all()),reference_norm=float(x.norm()),test_norm=float(y.norm()),relative_l2=float(d.norm()/x.norm().clamp_min(1e-30)),max_absolute_error=float(d.abs().max())))
report=dict(reference=ref,test=other,rows=rows)
(root/f'comparison-{other}.json').write_text(json.dumps(report,indent=2))
print(json.dumps(sorted(rows,key=lambda r:(not r['finite'],r['relative_l2']),reverse=True)[:10],indent=2))
