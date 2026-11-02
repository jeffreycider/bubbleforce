import json,math,pathlib,platform,torch
root=pathlib.Path('/workspace/bubbleforce/validation/hf-restart-002201/results')
load=lambda p:torch.load(p,map_location='cpu',weights_only=True,mmap=True)
compiled=json.loads((root/'command.json').read_text())
assert compiled['returncode']!=0
assert 'step:2202/6475' in (root/'run.log').read_text()
assert 'non-finite training loss' in (root/'run.log').read_text()
state=json.loads((root/'state-diagnostics.json').read_text())
updates=json.loads((root/'update-diagnostics.json').read_text())
assert state[0]['weights_match']
assert not any(r['optimizer_issues'] or r['bad_momenta'] for r in state)
assert updates[0]['fields']['aux_gradients']['nonfinite']==['blocks.0.attn.proj.bias']
assert updates[0]['fields']['executed_update']['nonfinite']==['blocks.0.attn.proj.bias']
eager_log=pathlib.Path('/workspace/eager-control.log').read_text()
assert 'step:2202/6475' in eager_log and 'step:2203/6475' in eager_log
assert 'EAGER_CONTROL_TRAINING_COMPLETE' in eager_log
p=load(root/'eager_control/captures/rank0/metrics_snapshot_002202.pt')
assert all(torch.isfinite(v).all() for v in p['model_before'].values())
assert all(torch.isfinite(v).all() for v in p['executed_update'].values())
assert all(torch.isfinite(p['model_before'][k]-v).all() for k,v in p['executed_update'].items())
for rank in range(8):
    s=load(root/f'eager_control/captures/rank{rank}/metrics_snapshot_002202.pt')
    assert all(torch.isfinite(v).all() for v in s['post_insertion_prepolar_momentum'].values())
folder=root/'diagnostic/spectra/update_002201'
report=json.loads((folder/'measurement_results.json').read_text())
assert report['stage']=='complete' and report['world_size']==8
readout=report['measurements'][0]['readouts'][-1]
assert all(math.isfinite(v) for v in readout['eigenvalues']+readout['residuals'])
norm2=0.
for rank in range(8):
    vectors=load(folder/f'eigenvectors_N8192_rank{rank}.pt')
    assert len(vectors)==1 and torch.isfinite(vectors).all()
    norm2+=vectors[0].double().square().sum().item()
assert abs(norm2-1)<1e-5
summary=dict(checkpoint_hashes_verified=8,initial_weights_exact=True,source_optimizer_tensors_finite=True,compiled_restart=dict(passed=False,completed_updates=1,failure='non-finite training loss on second update',first_nonfinite_gradient='blocks.0.attn.proj.bias'),eager_control=dict(passed=True,completed_updates=2,parameters_finite_after_both_updates=True,change='Disable GPT.compile only in the diagnostic worker'),snapshot_measurement=dict(passed=True,update=2201,tokens=8192,world_size=8,readout=readout,eigenvector_norm_squared=norm2),environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
(root/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
