import json,pathlib,shutil
r=pathlib.Path('/workspace/validation211');out=pathlib.Path('/workspace/torch211-compact');out.mkdir(exist_ok=True)
for p in r.rglob('*'):
    if not p.is_file() or 'wandb' in p.parts:continue
    if p.suffix not in ('.json','.jsonl','.log','.xml','.csv','.exit'):continue
    target=out/p.relative_to(r);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
shutil.copy2('/workspace/wheels/verification.json',out/'package-hashes.json')
for name in ('torch-install.log','train-install.log','bootstrap.log','wheels.log'):
    shutil.copy2('/workspace/'+name,out/name)
