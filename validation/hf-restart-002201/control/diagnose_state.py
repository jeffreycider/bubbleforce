import json,pathlib,torch
from concurrent.futures import ThreadPoolExecutor
root=pathlib.Path('/workspace/bubbleforce/validation/hf-restart-002201/results')
def inspect(rank):
    checkpoint=torch.load(f'/workspace/restart/rank{rank}/checkpoint_002201.pt',map_location='cpu',weights_only=True,mmap=True)
    snapshot=torch.load(root/f'run/captures/rank{rank}/metrics_snapshot_002201.pt',map_location='cpu',weights_only=True,mmap=True)
    row=dict(rank=rank,bad_momenta=[],optimizer_issues=[],optimizer_summaries=[])
    for name,value in snapshot['post_insertion_prepolar_momentum'].items():
        if not torch.isfinite(value).all(): row['bad_momenta'].append(name)
    if rank==0:
        row['weights_match']=all(torch.equal(v,checkpoint['model'][k]) for k,v in snapshot['model_before'].items())
        row['bad_weights']=[k for k,v in snapshot['model_before'].items() if not torch.isfinite(v).all()]
    for i,optimizer in enumerate(checkpoint['optimizers']):
        keys=set()
        for parameter,state in optimizer['state'].items():
            keys.update(state)
            for field,value in state.items():
                if isinstance(value,torch.Tensor):
                    finite=bool(torch.isfinite(value).all())
                    negative=field=='exp_avg_sq' and bool((value<0).any())
                    if not finite or negative:
                        row['optimizer_issues'].append(dict(group=i,parameter=parameter,field=field,shape=list(value.shape),dtype=str(value.dtype),finite=finite,negative_second_moment=negative))
        row['optimizer_summaries'].append(dict(group=i,keys=sorted(keys),param_groups=optimizer['param_groups']))
    print('Inspected',rank,'bad momentum',len(row['bad_momenta']),'optimizer issues',len(row['optimizer_issues']),flush=True)
    return row
with ThreadPoolExecutor(max_workers=4) as pool:
    rows=list(pool.map(inspect,range(8)))
(root/'state-diagnostics.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2),flush=True)
