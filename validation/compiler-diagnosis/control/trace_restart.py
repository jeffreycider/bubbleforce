import json, os, pathlib, time
import torch
import torch.distributed as dist
from torch.distributed.launcher.api import LaunchConfig, elastic_launch
from run import prepare_experiment, run_training
from harness import loop

ROOT=pathlib.Path('/workspace/diagnosis')
def inspect_gradients(model):
    named=list(model.named_parameters())
    finite=torch.stack([torch.isfinite(p.grad).all() for _,p in named]).tolist()
    bad=[n for (n,p),f in zip(named,finite) if not f]
    bias=model.blocks[0].attn.proj.bias.grad
    return dict(nonfinite=bad,bias_nan=int(bias.isnan().sum()),bias_inf=int(bias.isinf().sum()),bias_max=float(bias.abs().max()))

def accumulate(model,batch,microbatch_sequences):
    rank=dist.get_rank()
    torch.save(tuple(x.cpu() for x in batch),ROOT/f'batch-rank{rank}.pt')
    if rank==0:
        torch.save({k:v.cpu() for k,v in model.state_dict().items()},ROOT/'model.pt')
    rows=[]
    for i in range(len(batch[0])//microbatch_sequences):
        sl=slice(i*microbatch_sequences,(i+1)*microbatch_sequences)
        loss=model(batch[0][sl],batch[1][sl]);loss.backward()
        row=dict(microbatch=i,loss=float(loss),**inspect_gradients(model))
        rows.append(row)
        print('GRADIENT',rank,json.dumps(row),flush=True)
        (ROOT/f'trace-rank{rank}.json').write_text(json.dumps(rows,indent=2))
    for _,p in model.named_parameters():
        dist.all_reduce(p.grad)
    row=dict(after_all_reduce=True,**inspect_gradients(model));rows.append(row)
    (ROOT/f'trace-rank{rank}.json').write_text(json.dumps(rows,indent=2))
    print('REDUCED',rank,json.dumps(row),flush=True)

def worker(training,capture,world):
    loop.accumulate_global_gradient=accumulate
    run_training(training,capture,world)

if __name__=='__main__':
    ROOT.mkdir(exist_ok=True)
    overrides=['training.train_steps=6475','training.microbatch_sequences=8',
      'training.resume_checkpoint=/workspace/restart/checkpoint_002201',
      'measurement.updates=[2201]','measurement.output_dir=/workspace/diagnosis/trace',
      'training.metrics_snapshot.quantities=[weights,momentum]',
      'measurement.token_counts=[8192]','measurement.microbatch_sequences=1']
    training,measurement,world=prepare_experiment('decreased_momentum',overrides)
    launch=LaunchConfig(min_nodes=1,max_nodes=1,nproc_per_node=world,run_id='compiler-trace',rdzv_backend='c10d',rdzv_endpoint='localhost:0',local_addr='127.0.0.1',max_restarts=0)
    elastic_launch(launch,worker)(training,measurement.capture_dir,world)
