import hashlib, json, math, pathlib, platform, re
import torch

root=pathlib.Path('/workspace/bubbleforce')
output=root/'validation/hf-restart-002201/results'
run=output/'run'
load=lambda p:torch.load(p,map_location='cpu',weights_only=True,mmap=True)
command=json.loads((output/'command.json').read_text())
assert command['returncode']==0
log=(output/'run.log').read_text()
assert 'step:2202/6475' in log and 'step:2203/6475' in log
source=load('/workspace/restart/rank0/checkpoint_002201.pt')
first=load(run/'captures/rank0/metrics_snapshot_002201.pt')
assert all(torch.equal(value,source['model'][name]) for name,value in first['model_before'].items())
second=load(run/'captures/rank0/metrics_snapshot_002202.pt')
changed=[name for name,value in second['model_before'].items() if not torch.equal(value,first['model_before'][name])]
assert changed
summary=dict(environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]),command=command,initial_weights_exact=True,parameter_tensors_changed_after_first_update=len(changed),validation_losses=re.findall(r'step:(\d+)/6475 val_loss:([0-9.]+)',log),snapshots={})
for update in [2201,2202]:
    folder=run/'spectra'/f'update_{update:06d}'
    report=json.loads((folder/'measurement_results.json').read_text())
    assert report['stage']=='complete' and report['world_size']==8
    names=[]
    norm2=0.
    for rank in range(8):
        snapshot=load(run/f'captures/rank{rank}/metrics_snapshot_{update:06d}.pt')
        assert snapshot['update']==update and snapshot['rank']==rank
        for name,value in snapshot['post_insertion_prepolar_momentum'].items():
            assert torch.isfinite(value).all()
            names.append(name)
        vectors=load(folder/f'eigenvectors_N8192_rank{rank}.pt')
        assert len(vectors)==1 and torch.isfinite(vectors).all()
        norm2+=vectors[0].double().square().sum().item()
    assert len(names)==len(set(names))==72
    assert abs(norm2-1)<1e-5
    readout=report['measurements'][0]['readouts'][-1]
    assert all(math.isfinite(v) for v in readout['eigenvalues']+readout['residuals'])
    summary['snapshots'][str(update)]=dict(matrix_count=len(names),eigenvector_norm_squared=norm2,readout=readout)
manifest=json.loads(pathlib.Path('/workspace/source-manifest.json').read_text())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in manifest.items())
summary['source_hashes_verified']=True
(output/'verification.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
