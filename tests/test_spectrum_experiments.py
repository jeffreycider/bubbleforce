from copy import deepcopy
import csv
import json

from omegaconf import OmegaConf
import pytest
import torch
import yaml

import run
from configs.measurement_config import MeasurementConfig
from bubbleforce.run_analysis import predict_equilibrium
from harness.checkpoint import training_config_fingerprint
from harness.schedules import learning_rate_factor
from harness.model_gpt import GPT
from harness.optimizers import GroupedOptimizers
from configs.train_config import resolve_optimizer_groups
from optimizers.ringed.momentum.config import momentum_scheduler


def experiment_for(name, *overrides):
    return run.prepare_experiment(name, list(overrides))


def launch(config):
    run.run_experiment(*config)


def worker_settings(config):
    training, measurement, world_size = config
    return dict(training.to_training_dict(), capture_dir=str(measurement.capture_dir),
                require_world_size=world_size)


@pytest.mark.parametrize('name', run.Experiment)
def test_named_recipes_resolve_training_and_measurements(name):
    config = experiment_for(name)
    training = worker_settings(config)
    assert training['seed'] == 0
    assert training['batch_tokens'] == 524288
    assert config[2] == training['require_world_size'] == 8
    assert training['metrics_snapshot']['steps'] == list(config[1].updates)
    assert training['metrics_snapshot']['quantities'] == ['weights', 'momentum']
    assert 'stop_step' not in training


def test_nested_overrides_control_training_and_measurement_frequency(tmp_path):
    config = experiment_for('original_implementation', 'measurement.measure_every=300',
                            'training.train_steps=1000', f'measurement.output_dir={tmp_path / "custom_results"}')
    assert config[1].updates == (0, 300, 600, 900, 999)
    assert config[0].metrics_snapshot['steps'] == [0, 300, 600, 900, 999]
    assert config[1].output_dir == tmp_path / 'custom_results'
    assert config[1].capture_dir == config[1].output_dir / 'captures'
    assert config[0].output_dir == str(config[1].output_dir / 'training')


@pytest.mark.parametrize('name,after', [('lr_half', .5), ('lr_double', 2.)])
def test_lr_interventions_change_every_group_at_the_saved_iteration(name, after):
    config = experiment_for(name)[0].to_training_dict()
    for group in config['optimizer_groups']:
        schedule = group['lr_schedule']
        assert learning_rate_factor(schedule, 2008, 3250) == 1.
        assert learning_rate_factor(schedule, 2009, 3250) == after
        assert learning_rate_factor(schedule, 3000, 3250) == after
    muon = config['optimizer_groups'][2]
    assert muon['hyperparams']['momentum'] == dict(kind='ema', decay=.95, nesterov_mix=.95)


@pytest.mark.parametrize('name,beta', [
    ('increased_momentum', .9750961447125188),
    ('decreased_momentum', .8990666066208468),
])
@pytest.mark.parametrize('step', [2008, 2009, 2010, 2020, 2265])
def test_momentum_intervention_retains_the_pre_switch_decay_history(name, beta, step):
    config = experiment_for(name)[0].to_training_dict()
    scheduler = momentum_scheduler(config['optimizer_groups'][2]['hyperparams']['momentum'])
    mix = .95 if step < 2009 else beta
    product, expected = 1., []
    for age in range(256):
        decay = .95 if step - age < 2009 else beta
        expected.append(mix * (1 - decay) * product)
        product *= decay
    expected[0] += 1 - mix
    torch.testing.assert_close(scheduler(step).weights, torch.tensor(expected))
    for group in config['optimizer_groups']:
        assert learning_rate_factor(group['lr_schedule'], step, 3250) == 1.


def test_original_and_maxent_keep_cooldown():
    for name in ['original_implementation', 'maxentslop']:
        config = experiment_for(name)[0].to_training_dict()
        for group in config['optimizer_groups']:
            assert learning_rate_factor(group['lr_schedule'], 2009, 3250) == pytest.approx((1 - 2009 / 3250) / .7)
    schedule = momentum_scheduler(experiment_for('maxentslop')[0].to_training_dict()['optimizer_groups'][2]['hyperparams']['momentum'])
    assert not torch.equal(schedule(999).weights, schedule(1000).weights)


def test_measurement_only_overrides_paths_and_process_count(tmp_path):
    config = experiment_for('original_implementation', 'training=null',
                            f'measurement.capture_dir={tmp_path / "captures"}',
                            'measurement.updates=[975,1000]', 'world_size=2')
    assert config[0] is None
    assert config[2] == 2
    assert config[1].updates == (975, 1000)
    assert config[1].capture_dir == tmp_path / 'captures'


def test_measurement_only_accepts_updates_beyond_the_training_recipe():
    config = experiment_for('original_implementation', 'training=null', 'measurement.updates=[4800]')
    assert config[1].updates == (4800,)


@pytest.mark.parametrize('override', ['training.capture_dir=other', 'measurement.capture=other'])
def test_capture_directory_is_only_a_measurement_setting(override):
    with pytest.raises(TypeError, match='unexpected keyword'):
        experiment_for('original_implementation', override)


@pytest.mark.parametrize('reuse_previous_vector', [False, True])
def test_shared_measurement_config_round_trips_paths_and_updates(tmp_path, reuse_previous_vector):
    initial_vector = tmp_path / 'initial_vector'
    config = experiment_for('original_implementation', 'measurement.updates=[20,10]',
                            f'measurement.warm_start={initial_vector}', f'measurement.output_dir={tmp_path / "result"}',
                            f'measurement.reuse_previous_vector={str(reuse_previous_vector).lower()}')
    measurement = config[1]
    assert measurement.updates == (10, 20)
    assert measurement.warm_start == initial_vector
    assert measurement.output_dir == tmp_path / 'result'
    assert measurement.reuse_previous_vector == reuse_previous_vector
    restored = MeasurementConfig.from_dict(yaml.safe_load(yaml.safe_dump(measurement.to_dict())))
    assert restored == measurement
    path = tmp_path / 'measurement.yaml'
    OmegaConf.save(measurement.to_dict(), path)
    assert MeasurementConfig.from_yaml(path) == measurement


def test_restart_preserves_training_recipe_and_stops_after_requested_update(tmp_path):
    bundle = tmp_path / 'checkpoint_002009'
    bundle.mkdir()
    original = worker_settings(experiment_for('increased_momentum'))
    (bundle / 'COMPLETE.json').write_text(json.dumps(dict(
        format=3, boundary=2009, world_size=8,
        protocol_sha256=training_config_fingerprint(original),
    )))
    config = experiment_for('increased_momentum', f'training.resume_checkpoint={bundle}',
                            'measurement.updates=[2009]')
    assert config[0].stop_step == 2010
    assert config[0].train_steps == 3250
    assert config[0].resume_checkpoint == str(bundle)
    with pytest.raises(ValueError, match='precede'):
        launch(experiment_for('increased_momentum', f'training.resume_checkpoint={bundle}'))
    with pytest.raises(ValueError, match='recipe differs'):
        launch(experiment_for('decreased_momentum', f'training.resume_checkpoint={bundle}',
                                         'measurement.updates=[2009]'))


@pytest.mark.parametrize('arguments', [('world_size=0',), ('measurement.updates=[3250]',), ('world_size=3',)])
def test_invalid_training_requests_fail_before_launch(arguments):
    with pytest.raises(ValueError):
        experiment_for('original_implementation', *arguments)


@pytest.mark.parametrize('override', [
    'unknown=1', 'training.unknown=1', 'measurement.unknown=1',
    'measurement.eigensolver.unknown=1',
])
def test_structured_plan_rejects_unknown_settings(override):
    with pytest.raises((TypeError, ValueError)):
        experiment_for('original_implementation', override)


@pytest.mark.parametrize('override', [
    'world_size=invalid', 'train=invalid', 'training.train_steps=invalid',
    'measurement.updates=[invalid]', 'experiment=invalid',
])
def test_structured_plan_rejects_invalid_types_and_enum_values(override):
    with pytest.raises(ValueError):
        experiment_for('original_implementation', override)


@pytest.mark.parametrize('overrides,message', [
    (['measurement.token_counts=[7]'], 'whole validation sequences'),
    (['measurement.sequence_length=0'], 'sequence_length'),
    (['measurement.measure_every=0'], 'measure_every'),
    (['measurement.updates=[]'], 'updates'),
    (['measurement.updates=[2,2]'], 'updates'),
    (['measurement.updates=[-1]'], 'updates'),
    (['training.train_steps=0'], 'train_steps'),
    (['training=null'], 'explicit measurement.updates'),
    (['experiment=lr_half'], 'unknown experiment'),
    (['measurement.updates=[10]', 'training.stop_step=10'], 'precede stop_step'),
])
def test_plan_checks_related_settings_before_execution(overrides, message):
    with pytest.raises(ValueError, match=message):
        experiment_for('original_implementation', *overrides)


def test_stop_step_selects_available_measurements_without_changing_schedule():
    config = experiment_for('original_implementation', 'training.stop_step=600')
    assert config[0].train_steps == 3250
    assert config[0].stop_step == 600
    assert config[1].updates == (*range(0, 600, 50), 599)
    config = experiment_for('original_implementation', 'measurement.updates=[10]',
                            'training.stop_step=20')
    assert config[0].stop_step == 20


def test_preview_does_not_require_checkpoint_or_data_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run.sys, 'argv', [
        'run', 'original_implementation', 'measurement.updates=[2009]',
        f'training.resume_checkpoint={tmp_path / "missing_checkpoint"}',
        f'measurement.validation={tmp_path / "missing_validation.bin"}',
        f'measurement.output_dir={tmp_path / "unused"}', '--print-settings',
    ])
    run.main()
    settings = yaml.safe_load(capsys.readouterr().out)
    assert settings['training']['stop_step'] == 2010
    assert settings['world_size'] == 8
    assert not (tmp_path / 'unused').exists()


def test_run_passes_measurement_config_without_writing_settings(tmp_path, monkeypatch):
    capture = tmp_path / 'captures' / 'rank0'
    capture.mkdir(parents=True)
    (capture / 'metrics_snapshot_000975.pt').touch()
    (capture / 'metrics_snapshot_001000.pt').touch()
    validation = tmp_path / 'validation.bin'
    validation.touch()
    config = experiment_for('original_implementation', 'training=null', f'measurement.capture_dir={capture.parent}',
                            'measurement.updates=[975,1000]', 'world_size=1',
                            f'measurement.validation={validation}', f'measurement.output_dir={tmp_path / "result"}')
    calls = []
    def record_worker(launcher, worker):
        assert launcher.nproc_per_node == 1
        return lambda *args: calls.append((worker, args))
    monkeypatch.setattr(run, 'elastic_launch', record_worker)
    launch(config)
    assert calls == [(run.run_measurements, (config[1],))]
    assert list(config[1].output_dir.iterdir()) == []
    with pytest.raises(FileExistsError):
        launch(config)


def test_missing_later_snapshot_fails_before_launch(tmp_path, monkeypatch):
    capture = tmp_path / 'captures' / 'rank0'
    capture.mkdir(parents=True)
    (capture / 'metrics_snapshot_000975.pt').touch()
    validation = tmp_path / 'validation.bin'
    validation.touch()
    config = experiment_for('original_implementation', 'training=null', f'measurement.capture_dir={capture.parent}',
                            'measurement.updates=[975,1000]', 'world_size=1',
                            f'measurement.validation={validation}', f'measurement.output_dir={tmp_path / "result"}')
    def reject_launch(*args, **kwargs):
        raise AssertionError('measurement started before checking every snapshot')
    monkeypatch.setattr(run, 'elastic_launch', reject_launch)
    with pytest.raises(FileNotFoundError, match='001000'):
        launch(config)
    assert not config[1].output_dir.exists()


def test_named_groups_select_the_expected_gpt_parameters():

    with torch.device('meta'):
        model = GPT(vocab_size=16, num_layers=1, model_dim=128)
    groups = GroupedOptimizers(model, experiment_for('original_implementation')[0].to_training_dict()['optimizer_groups'])
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    selected = [{names[id(parameter)] for group in optimizer.param_groups for parameter in group['params']}
                for _, optimizer in groups.groups]
    assert selected[0] == {'embed.weight'}
    assert selected[1] == {'proj.weight'}
    assert selected[2] == {
        'blocks.0.attn.q.weight', 'blocks.0.attn.k.weight', 'blocks.0.attn.v.weight',
        'blocks.0.attn.proj.weight', 'blocks.0.mlp.fc.weight', 'blocks.0.mlp.proj.weight',
    }
    assert selected[3] == {
        'proj.bias', 'norm1.gains', 'norm2.gains', 'blocks.0.norm1.gains', 'blocks.0.norm2.gains',
        'blocks.0.attn.q.bias', 'blocks.0.attn.k.bias', 'blocks.0.attn.v.bias',
        'blocks.0.attn.proj.bias', 'blocks.0.mlp.fc.bias', 'blocks.0.mlp.proj.bias',
    }
    assert [type(optimizer).__name__ for _, optimizer in groups.groups] == ['AdamW', 'AdamW', 'Muon', 'AdamW']


@pytest.mark.parametrize('change, message', [('unknown', 'unknown'), ('duplicate', 'duplicate'), ('missing', 'missing')])
def test_invalid_model_group_assignments_fail_before_training(change, message):

    groups = experiment_for('original_implementation')[0].optimizer_groups
    if change == 'unknown':
        groups[0]['block_type'] = 'embedding_typo'
    elif change == 'duplicate':
        groups[1]['block_type'] = groups[0]['block_type']
    else:
        groups.pop()
    with pytest.raises(ValueError, match=message):
        resolve_optimizer_groups(groups)


def test_training_default_lists_are_independent():

    first = experiment_for('original_implementation')[0]
    second = experiment_for('original_implementation')[0]
    first.setup[0]['name'] = 'changed'
    first.model['num_layers'] = 1
    assert second.setup[0]['name'] == 'open_rank_zero_log'
    assert second.model['num_layers'] == 12


def test_print_settings_resolves_defaults_and_keeps_named_groups(tmp_path, monkeypatch, capsys):
    output = tmp_path / 'unused'
    monkeypatch.setattr(run.sys, 'argv', [
        'run', 'original_implementation', 'measurement.updates=[2009]',
        f'measurement.output_dir={output}', '--print-settings',
    ])
    def reject_launch(*args, **kwargs):
        raise AssertionError('settings preview started a process')
    monkeypatch.setattr(run, 'elastic_launch', reject_launch)
    run.main()
    settings = yaml.safe_load(capsys.readouterr().out)
    assert settings['measurement']['updates'] == [2009]
    assert len(settings['training']['setup']) == 9
    assert settings['training']['optimizer_groups'][0]['block_type'] == 'embed'
    assert 'pattern' not in settings['training']['optimizer_groups'][0]
    assert settings['measurement']['token_counts'] == [16777216]
    assert not output.exists()


def test_maxentslop_predictions_recompute_both_schedules():

    training = experiment_for('maxentslop')[0].to_training_dict()
    updates = [999, 1000, 1500, 2500]
    rows = list(predict_equilibrium(training, updates))
    assert [row['update'] for row in rows] == updates
    assert rows[-1]['learning_rate'] < rows[1]['learning_rate']
    assert rows[-1]['current_gradient_coefficient'] > rows[1]['current_gradient_coefficient']
    fixed_lr = deepcopy(training)
    fixed_lr['optimizer_groups'][2]['lr_schedule'] = dict(kind='piecewise_constant', boundaries=[], factors=[1.])
    fixed_kernel = deepcopy(training)
    fixed_kernel['optimizer_groups'][2]['hyperparams']['momentum'] = dict(kind='ema', decay=.95, nesterov_mix=.95)
    assert rows[-1]['predicted_zeta'] != pytest.approx(list(predict_equilibrium(fixed_lr, [2500]))[0]['predicted_zeta'])
    assert rows[-1]['predicted_zeta'] != pytest.approx(list(predict_equilibrium(fixed_kernel, [2500]))[0]['predicted_zeta'])


def test_saved_capture_curve_uses_saved_recipe_and_all_updates(tmp_path):

    capture = tmp_path / 'capture'
    capture.mkdir()
    training = experiment_for('maxentslop')[0].to_training_dict()
    training['train_steps'] = 4
    (capture / 'capture_config.json').write_text(json.dumps(training))
    output = tmp_path / 'output'
    output.mkdir()
    run.write_prediction_curve(None, capture_dir=capture, output_dir=output)
    rows = list(csv.DictReader((output / 'predicted_equilibrium.csv').open()))
    assert [int(row['update']) for row in rows] == [0, 1, 2, 3]
    assert sorted(path.name for path in output.iterdir()) == ['predicted_equilibrium.csv']


def test_training_run_writes_dense_predictions_between_sparse_measurements(tmp_path, monkeypatch):
    validation = tmp_path / 'validation.bin'
    validation.touch()
    training_shard = tmp_path / 'train.bin'
    training_shard.touch()
    config = experiment_for('original_implementation', 'measurement.updates=[2]',
                            f'measurement.validation={validation}', f'measurement.output_dir={tmp_path / "result"}',
                            f'training.train_data={training_shard}')
    calls = []
    def record_worker(launcher, worker):
        assert launcher.nproc_per_node == config[2]
        return lambda *args: calls.append((worker, args))
    monkeypatch.setattr(run, 'elastic_launch', record_worker)
    launch(config)
    assert calls == [
        (run.run_training, (config[0], config[1].capture_dir, config[2])),
        (run.run_measurements, (config[1],)),
    ]
    rows = list(csv.DictReader((config[1].output_dir / 'predicted_equilibrium.csv').open()))
    assert [int(row['update']) for row in rows] == [0, 1, 2]
    assert sorted(path.name for path in config[1].output_dir.iterdir()) == ['predicted_equilibrium.csv']


def test_zero_learning_rate_records_an_unavailable_prediction():

    training = experiment_for('original_implementation')[0].to_training_dict()
    training['optimizer_groups'][2]['lr_schedule'] = dict(kind='piecewise_constant', boundaries=[1], factors=[1., 0.])
    rows = list(predict_equilibrium(training, [0, 1]))
    assert rows[0]['predicted_zeta'] > 0
    assert rows[1]['predicted_zeta'] is None
    assert 'positive learning rate' in rows[1]['reason']
