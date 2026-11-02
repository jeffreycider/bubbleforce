import hashlib, json, os, pathlib, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

root=pathlib.Path('/workspace/bubbleforce')
os.chdir(root)
output=root/'validation/hf-restart-002201/results'
output.mkdir(parents=True,exist_ok=True)
checkpoint=pathlib.Path('/workspace/restart/checkpoint_002201')
complete=json.loads((checkpoint/'COMPLETE.json').read_text())
assert complete['format']==3 and complete['boundary']==2201 and complete['world_size']==8
def verify_shard(shard):
    path=checkpoint.parent/shard['file']
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    assert digest==shard['sha256'],path
    print('Verified',shard['file'],flush=True)
    return dict(file=shard['file'],bytes=path.stat().st_size,sha256=digest)

with ThreadPoolExecutor(max_workers=4) as pool:
    checks=list(pool.map(verify_shard,complete['shards']))
(output/'checkpoint-verification.json').write_text(json.dumps(checks,indent=2))

args=[sys.executable,'run.py','decreased_momentum',
      'training.train_steps=6475','training.microbatch_sequences=8',
      'training.resume_checkpoint=/workspace/restart/checkpoint_002201',
      'measurement.updates=[2201,2202]',
      'measurement.output_dir=validation/hf-restart-002201/results/run',
      'measurement.token_counts=[8192]','measurement.microbatch_sequences=1',
      'measurement.cpu_threads=4',
      'measurement.eigensolver.krylov_basis_vector_counts=[4,8]']
record=dict(command=args,started=time.time())
(output/'command.json').write_text(json.dumps(record,indent=2))
with (output/'run.log').open('w') as log:
    result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=900)
record.update(returncode=result.returncode,seconds=time.time()-record['started'])
(output/'command.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
raise SystemExit(result.returncode)
