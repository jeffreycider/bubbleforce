import json,pathlib,subprocess,sys,time
root=pathlib.Path('/workspace/validation211/fresh');root.mkdir(parents=True,exist_ok=True)
common=['world_size=2','training.train_steps=6','training.batch_tokens=4096',
 'training.microbatch_sequences=1','training.val_tokens=2048',
 'training.model.num_layers=2','training.model.model_dim=128',
 'training.checkpoint.every=3','measurement.token_counts=[2048]',
 'measurement.microbatch_sequences=1','measurement.cpu_threads=4',
 'measurement.eigensolver.krylov_basis_vector_counts=[4,8,16]']
measurement=['world_size=2','training=null','measurement.updates=[5]',
 'measurement.token_counts=[2048]','measurement.microbatch_sequences=1',
 'measurement.cpu_threads=4','measurement.eigensolver.krylov_basis_vector_counts=[4,8,16]']
phases=[('prefix',common+['training.stop_step=3','measurement.updates=[2]']),
 ('resumed',common+[f'training.resume_checkpoint={root}/prefix/captures/checkpoint_000003','measurement.updates=[5]']),
 ('uninterrupted',common+['measurement.updates=[5]']),
 ('snapshot_only',measurement+[f'measurement.capture_dir={root}/resumed/captures'])]
rows=[]
for name,overrides in phases:
    args=[sys.executable,'run.py','original_implementation',*overrides,f'measurement.output_dir={root}/{name}']
    row=dict(name=name,command=args,started=time.time());rows.append(row)
    (root/'commands.json').write_text(json.dumps(rows,indent=2))
    with (root/f'{name}.log').open('w') as log:
        result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=360)
    row.update(returncode=result.returncode,seconds=time.time()-row['started'])
    (root/'commands.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
    assert result.returncode==0,name
print('FRESH_TRAINING_RESUME_MEASUREMENT_COMPLETE',flush=True)
