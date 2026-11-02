import hashlib,json,math,pathlib,platform,xml.etree.ElementTree as ET
import torch
from torch._inductor import config
r=pathlib.Path('/workspace/validation211')
assert (r/'complete').exists()
assert int((r/'pytest.exit').read_text())==0
xml=ET.parse(r/'pytest.xml').getroot()
suites=[xml] if xml.tag=='testsuite' else list(xml.iter('testsuite'))
counts={k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
assert counts['failures']==counts['errors']==0
compiler=json.loads((r/'compiler-check.json').read_text());assert compiler['mix_order_reduction']
restart=[]
for rank in range(8):
 rows=[json.loads(s) for s in (r/f'restart/checks-rank{rank}.jsonl').read_text().splitlines()]
 assert [x['update'] for x in rows]==list(range(2201,2211))
 assert all(x['gradients_finite'] and x['parameters_finite'] for x in rows)
 restart.append(dict(rank=rank,completed_updates=10,passed=True))
def measurement(folder,tokens,world):
    report=json.loads((folder/'measurement_results.json').read_text());assert report['stage']=='complete'
    readout=report['measurements'][0]['readouts'][-1]
    assert all(math.isfinite(v) for v in readout['eigenvalues']+readout['residuals'])
    norm2=0
    for rank in range(world):
        v=torch.load(folder/f'eigenvectors_N{tokens}_rank{rank}.pt',weights_only=True,map_location='cpu')
        assert v.isfinite().all();norm2+=float(v[0].double().square().sum())
    assert abs(norm2-1)<1e-5
    return dict(stage=report['stage'],update=report['update'],tokens=tokens,world_size=world,readout=readout,eigenvector_norm_squared=norm2)
measurements={'hf_restart':measurement(r/'restart/spectra/update_002210',8192,8)}
for name,update in [('prefix',2),('resumed',5),('uninterrupted',5),('snapshot_only',5)]:
    measurements[name]=measurement(r/f'fresh/{name}/spectra/update_{update:06d}',2048,2)
def compare(a,b):
    count=0;differences=[]
    def visit(x,y,name):
        nonlocal count
        if isinstance(x,torch.Tensor):
            count+=1;assert x.shape==y.shape and x.dtype==y.dtype,name
            assert x.isfinite().all() and y.isfinite().all(),name
            if not torch.equal(x,y):
                d=x.double()-y.double()
                differences.append(dict(name=name,max_absolute=float(d.abs().max()),relative_l2=float(d.norm()/x.double().norm().clamp_min(1e-30))))
        elif isinstance(x,dict):
            assert x.keys()==y.keys(),name
            for k in x:visit(x[k],y[k],name+'/'+str(k))
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y),name
            for i,(v,w) in enumerate(zip(x,y)):visit(v,w,name+'/'+str(i))
        else:assert x==y,(name,x,y)
    visit(a,b,'');return dict(tensors=count,exact_equal=not differences,differences=differences)
comparisons=[]
for rank in range(2):
    p={name:torch.load(r/f'fresh/{name}/captures/rank{rank}/checkpoint_000006.pt',weights_only=True,map_location='cpu',mmap=True) for name in ('resumed','uninterrupted')}
    comparisons.append(dict(rank=rank,model=compare(p['resumed']['model'],p['uninterrupted']['model']),optimizers=compare(p['resumed']['optimizers'],p['uninterrupted']['optimizers'])))
manifest=json.loads(pathlib.Path('/workspace/source-manifest.json').read_text())
for filename,digest in manifest['files'].items():
    data=(pathlib.Path('/workspace/bubbleforce')/filename).read_bytes()
    if filename=='requirements.txt':
        assert b'torch==2.11.0\n' in data
        data=data.replace(b'torch==2.11.0\n',b'torch==2.10\n')
    assert hashlib.sha256(data).hexdigest()==digest,filename
summary=dict(environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,gpus=[torch.cuda.get_device_name(i) for i in range(8)],mix_order_reduction=config.triton.mix_order_reduction),pytest=counts,compiler_regression=compiler,restart=restart,measurements=measurements,fresh_resume_comparison=comparisons,tested_source_verified=True,source_revision=manifest['git_head'])
(r/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
