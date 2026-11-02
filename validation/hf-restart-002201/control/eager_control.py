import json,os,pathlib
from torch.distributed.launcher.api import LaunchConfig,elastic_launch
from run import prepare_experiment,run_training
from harness.model_gpt import GPT

def eager_worker(training,capture,world):
    GPT.compile=lambda self,**kwargs: None
    run_training(training,capture,world)

if __name__=='__main__':
    os.chdir('/workspace/bubbleforce')
    output=pathlib.Path('validation/hf-restart-002201/results/eager_control')
    overrides=['training.train_steps=6475','training.microbatch_sequences=8',
      'training.resume_checkpoint=/workspace/restart/checkpoint_002201',
      'measurement.updates=[2202]',f'measurement.output_dir={output}',
      'training.metrics_snapshot.quantities=[weights,momentum,executed_update]',
      'measurement.token_counts=[8192]','measurement.microbatch_sequences=1',
      'measurement.cpu_threads=4','measurement.eigensolver.krylov_basis_vector_counts=[4,8]']
    training,measurement,world=prepare_experiment('decreased_momentum',overrides)
    output.mkdir(parents=True)
    (output/'settings.json').write_text(json.dumps(dict(overrides=overrides,change='Disable GPT.compile in the diagnostic worker; retain optimizer execution and checkpoint settings.'),indent=2))
    launch=LaunchConfig(min_nodes=1,max_nodes=1,nproc_per_node=world,run_id='hf-restart-eager-control',rdzv_backend='c10d',rdzv_endpoint='localhost:0',local_addr='127.0.0.1',max_restarts=0)
    elastic_launch(launch,eager_worker)(training,measurement.capture_dir,world)
    print('EAGER_CONTROL_TRAINING_COMPLETE',flush=True)
