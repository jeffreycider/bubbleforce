import json,os,pathlib
import torch
import torch.distributed as dist
from torch.distributed.launcher.api import LaunchConfig,elastic_launch
from run import prepare_experiment,run_training
from bubbleforce.run_analysis import run_measurements
from harness import loop
from harness.optimizers import GroupedOptimizers

ROOT=pathlib.Path('/workspace/diagnosis/fixed_restart')

def worker(training,capture,world):
    original_accumulate=loop.accumulate_global_gradient
    original_step=GroupedOptimizers.step
    state=dict(update=2201)
    def accumulate(model,batch,microbatch):
        state['model']=model
        original_accumulate(model,batch,microbatch)
        names=list(model.named_parameters())
        flags=torch.stack([torch.isfinite(p.grad).all() for _,p in names]).tolist()
        bad=[name for (name,_),ok in zip(names,flags) if not ok]
        assert not bad,bad
    def step(optimizer):
        original_step(optimizer)
        names=list(state['model'].named_parameters())
        flags=torch.stack([torch.isfinite(p).all() for _,p in names]).tolist()
        bad=[name for (name,_),ok in zip(names,flags) if not ok]
        row=dict(update=state['update'],gradients_finite=True,parameters_finite=not bad,nonfinite=bad)
        with (ROOT/f'checks-rank{dist.get_rank()}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        assert not bad,bad
        state['update']+=1
    loop.accumulate_global_gradient=accumulate
    GroupedOptimizers.step=step
    run_training(training,capture,world)

if __name__=='__main__':
    assert os.environ['TORCHINDUCTOR_MIX_ORDER_REDUCTION']=='0'
    ROOT.mkdir(parents=True,exist_ok=True)
    overrides=['training.train_steps=6475','training.microbatch_sequences=8',
      'training.resume_checkpoint=/workspace/restart/checkpoint_002201',
      'measurement.updates=[2210]',f'measurement.output_dir={ROOT}',
      'training.metrics_snapshot.quantities=[weights,momentum]',
      'measurement.token_counts=[8192]','measurement.microbatch_sequences=1',
      'measurement.cpu_threads=4','measurement.eigensolver.krylov_basis_vector_counts=[4,8]']
    training,measurement,world=prepare_experiment('decreased_momentum',overrides)
    (ROOT/'settings.json').write_text(json.dumps(dict(overrides=overrides,change='TORCHINDUCTOR_MIX_ORDER_REDUCTION=0',torch=torch.__version__),indent=2))
    launch=LaunchConfig(min_nodes=1,max_nodes=1,nproc_per_node=world,run_id='compiler-workaround',rdzv_backend='c10d',rdzv_endpoint='localhost:0',local_addr='127.0.0.1',max_restarts=0)
    elastic_launch(launch,worker)(training,measurement.capture_dir,world)
    print('FIXED_TRAINING_COMPLETE',flush=True)
    elastic_launch(launch,run_measurements)(measurement)
    print('FIXED_MEASUREMENTS_COMPLETE',flush=True)
