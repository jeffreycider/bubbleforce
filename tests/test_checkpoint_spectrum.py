from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import run
from bubbleforce import run_analysis as analysis
from bubbleforce.load_checkpoint import (
    load_checkpoint_for_analysis, load_validation_token_batch,
    load_model_from_checkpoint, load_initial_eigenvector,
)
from bubbleforce.run_analysis import run_analysis, compute_bubbleforce_eigenpairs
from configs.measurement_config import EigenSolveConfig, MeasurementConfig, ShardedVectorConfig
from harness.model_gpt import GPT


def write_tokens(path, count=17):
    header = np.zeros(256, dtype=np.int32)
    header[:3] = [20240520, 1, count]
    with path.open('wb') as stream:
        header.tofile(stream)
        (np.arange(count, dtype=np.uint16) % 16).tofile(stream)


def test_validation_partitions_match_the_global_prefix(tmp_path):
    path = tmp_path / 'tokens.bin'
    write_tokens(path)
    for rank in range(2):
        x, y = load_validation_token_batch(path, 8, rank=rank, world_size=2,
                                     device=torch.device('cpu'), sequence_length=2)
        assert x.tolist() == [[rank * 4, rank * 4 + 1], [rank * 4 + 2, rank * 4 + 3]]
        assert torch.equal(y, x + 1)
    with pytest.raises(ValueError, match='whole validation sequences'):
        load_validation_token_batch(path, 7, rank=0, world_size=2, device=torch.device('cpu'), sequence_length=2)
    with pytest.raises(ValueError, match='too few tokens'):
        load_validation_token_batch(path, 18, rank=0, world_size=1, device=torch.device('cpu'), sequence_length=2)


def make_gpt_capture(tmp_path):
    torch.manual_seed(7)
    model = GPT(vocab_size=16, num_layers=1, model_dim=128)
    weights = {name: value.detach().clone() for name, value in model.named_parameters()}
    momenta = {name: torch.randn_like(value) for name, value in weights.items()
               if name.startswith('blocks.') and value.ndim == 2}
    pattern = r'^blocks\..*\.weight$'
    snapshot = dict(format=3, update=2, rank=0, model_before=weights,
                    buffers=dict(model.named_buffers()), post_insertion_prepolar_momentum=momenta,
                    training_batch_tokens=8,
                    settings=[dict(pattern=pattern, optimizer='muon', groups=[dict(lr=.025, weight_decay=.05)])])
    recipe = dict(train_steps=4, optimizer_groups=[dict(
        pattern=pattern, optimizer='muon',
        lr_schedule=dict(kind='piecewise_constant', boundaries=[], factors=[1.]),
        hyperparams=dict(lr=.025, weight_decay=.05, momentum=dict(kind='ema', decay=.9)),
    )])
    folder = tmp_path / 'capture' / 'rank0'
    folder.mkdir(parents=True)
    (folder.parent / 'capture_config.json').write_text(json.dumps(recipe))
    torch.save(snapshot, folder / 'metrics_snapshot_000002.pt')
    validation = tmp_path / 'validation.bin'
    write_tokens(validation)
    return MeasurementConfig(
        validation=validation, output_dir=tmp_path / 'result', capture_dir=tmp_path / 'capture', measured_updates=(2,), measure_every=250,
        token_counts=(2,), microbatch_sequences=1, sequence_length=2,
        hvp_mode='trace', gemm_precision='fp32', matmul_precision='highest', cpu_threads=1,
        warm_start=None, reuse_previous_vector=False,
        eigensolver=EigenSolveConfig(num_requested_eigenpairs=1, krylov_basis_vector_counts=(2, 3), symmetric=False),
    )


def test_load_snapshot_recovers_optimizer_and_parameter_assignments(tmp_path):
    config = make_gpt_capture(tmp_path)
    checkpoint = load_checkpoint_for_analysis((tmp_path / 'capture'), config.measured_updates[0], 0, 1, torch.device('cpu'))
    assert len(checkpoint.momenta) == 6
    assert set(checkpoint.momenta) == set(checkpoint.vector_config.parameter_names)
    assert all(value.current_gradient_coefficient == pytest.approx(.1) for value in checkpoint.step_configs.values())
    with pytest.raises(ValueError, match='update differs'):
        replacement = checkpoint.paths[0].with_suffix('.replacement.pt')
        torch.save(dict(checkpoint.snapshot, update=3), replacement)
        replacement.replace(checkpoint.paths[0])
        load_checkpoint_for_analysis((tmp_path / 'capture'), config.measured_updates[0], 0, 1, torch.device('cpu'))


def test_tiny_gpt_checkpoint_to_spectrum_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    config = make_gpt_capture(tmp_path)
    result = run_analysis(config, update=2, output=config.output_dir / "spectra" / "update_000002")
    saved = json.loads((config.output_dir / 'spectra' / 'update_000002' / 'measurement_results.json').read_text())
    assert saved['stage'] == 'complete'
    assert saved['operator'] == 'restoring-response'
    assert 'config' not in saved
    assert {entry['path'] for entry in saved['inputs']} == {
        str(config.validation), str((tmp_path / 'capture') / 'capture_config.json'),
        str((tmp_path / 'capture') / 'rank0' / 'metrics_snapshot_000002.pt'),
    }
    assert saved['update'] == 2
    assert saved['output'] == str(config.output_dir / 'spectra' / 'update_000002')
    assert result[0]['readouts'][-1]['residuals']
    assert (config.output_dir / 'spectra' / 'update_000002' / 'eigenvalues.csv').read_text().count('\n') >= 2
    vectors = torch.load(config.output_dir / 'spectra' / 'update_000002' / 'eigenvectors_N2_rank0.pt', weights_only=True)
    assert vectors.ndim == 2 and len(vectors) == 1
    with pytest.raises(FileExistsError, match='new output directory'):
        run_analysis(config, update=2, output=config.output_dir / "spectra" / "update_000002")


def test_two_rank_checkpoint_measurement_writes_all_vector_shards(tmp_path):
    config = make_gpt_capture(tmp_path)
    config = replace(config, token_counts=(4,), measured_updates=(2, 3), reuse_previous_vector=True)
    path = (tmp_path / 'capture') / 'rank0' / 'metrics_snapshot_000002.pt'
    snapshot = torch.load(path, weights_only=True)
    names = sorted(snapshot['post_insertion_prepolar_momentum'])
    for rank in range(2):
        owned = dict(snapshot, rank=rank, post_insertion_prepolar_momentum={
            name: snapshot['post_insertion_prepolar_momentum'][name] for name in names[rank::2]
        })
        if rank:
            del owned['model_before'], owned['buffers']
        folder = (tmp_path / 'capture') / f'rank{rank}'
        folder.mkdir(exist_ok=True)
        torch.save(owned, folder / 'metrics_snapshot_000002.pt')
        torch.save(dict(owned, update=3), folder / 'metrics_snapshot_000003.pt')
    with pytest.raises(ValueError, match='every hidden matrix'):
        load_checkpoint_for_analysis((tmp_path / 'capture'), config.measured_updates[0], 0, 1, torch.device('cpu'))
    run.run_experiment(None, config, 2)
    assert not list(config.output_dir.rglob('*.yaml'))
    for update in config.measured_updates:
        output = config.output_dir / 'spectra' / f'update_{update:06d}'
        saved = json.loads((output / 'measurement_results.json').read_text())
        assert saved['stage'] == 'complete'
        assert saved['world_size'] == 2
        assert saved['update'] == update
        assert 'config' not in saved
        assert len(saved['inputs']) == (4 if update == 2 else 6)
        previous = config.output_dir / 'spectra' / 'update_000002'
        assert saved['warm_start'] == (None if update == 2 else str(previous))
        if update == 3:
            assert {str(previous / f'current_vector_rank{rank}.pt') for rank in range(2)} <= {
                entry['path'] for entry in saved['inputs']
            }
        norms = []
        for rank in range(2):
            vectors = torch.load(output / f'eigenvectors_N4_rank{rank}.pt', weights_only=True)
            assert len(vectors) == 1
            norms.append(vectors[0].square().sum())
        assert sum(norms).item() == pytest.approx(1., abs=1e-5)


@pytest.mark.parametrize('updates', [(), (-1,), (True,), (1.5,), ('2',), (2, 2)])
def test_measurement_config_rejects_invalid_updates(tmp_path, updates):
    config = make_gpt_capture(tmp_path)
    with pytest.raises(ValueError, match='updates must be'):
        replace(config, measured_updates=updates)


def test_measurement_post_init_selects_updates_and_resolves_paths(tmp_path, monkeypatch):
    config = make_gpt_capture(tmp_path)
    monkeypatch.chdir(tmp_path)
    resolved = replace(config, measured_updates=None, measure_every=3, end_step=8,
                       capture_dir='captures', output_dir='results', warm_start='vectors')
    repository = Path(analysis.__file__).resolve().parents[1]
    assert resolved.measured_updates == (0, 3, 6, 7)
    assert resolved.capture_dir == repository / 'captures'
    assert resolved.output_dir == repository / 'results'
    assert resolved.warm_start == repository / 'vectors'
    assert replace(config, measured_updates=(3, 2)).measured_updates == (2, 3)


@pytest.mark.parametrize('interval', [0, -1, True, 1.5])
def test_measurement_post_init_rejects_invalid_frequency(tmp_path, interval):
    config = make_gpt_capture(tmp_path)
    with pytest.raises(ValueError, match='measure_every'):
        replace(config, measure_every=interval)


@pytest.mark.parametrize('reuse_previous_vector', [False, True])
def test_worker_reuses_shared_config_and_selects_warm_start(tmp_path, monkeypatch, reuse_previous_vector):
    config = replace(
        make_gpt_capture(tmp_path), measured_updates=(2, 3),
        warm_start=tmp_path / 'initial_vector', reuse_previous_vector=reuse_previous_vector,
    )
    calls = []
    def record_measurement(shared_config, **arguments):
        calls.append((shared_config, arguments))
    monkeypatch.setattr(analysis, 'run_analysis', record_measurement)
    analysis.run_measurements(config)
    assert len(calls) == 2
    assert calls[0][0] is calls[1][0]
    assert calls[0][0].to_dict() == config.to_dict()
    for index, update in enumerate(config.measured_updates):
        assert calls[index][1] == dict(
            update=update, output=config.output_dir / 'spectra' / f'update_{update:06d}',
            warm_start=(config.output_dir / 'spectra' / 'update_000002'
                        if index and reuse_previous_vector else config.warm_start),
        )


@pytest.mark.parametrize('symmetric', [False, True])
def test_measurement_config_accepts_both_solver_modes(tmp_path, symmetric):
    config = make_gpt_capture(tmp_path)
    config = replace(config, eigensolver=replace(config.eigensolver, symmetric=symmetric))
    assert MeasurementConfig.from_dict(config.to_dict()).eigensolver.symmetric is symmetric


@pytest.mark.parametrize('symmetric', [False, True])
def test_compute_bubbleforce_without_logging(tmp_path, symmetric):

    config = make_gpt_capture(tmp_path)
    config = replace(config, eigensolver=replace(config.eigensolver, symmetric=symmetric))
    checkpoint = load_checkpoint_for_analysis((tmp_path / 'capture'), config.measured_updates[0], 0, 1, torch.device('cpu'))
    model = load_model_from_checkpoint(checkpoint)
    validation_batch = load_validation_token_batch(
        config.validation, 2, rank=0, world_size=1, device=torch.device('cpu'), sequence_length=2,
    )
    measurement, eigenvectors = compute_bubbleforce_eigenpairs(config, checkpoint, model, validation_batch, 2)
    assert measurement['readouts'][-1]['eigenvalues']
    assert len(eigenvectors) == 1
    assert not config.output_dir.exists()


def test_load_initial_eigenvector_checks_saved_layout(tmp_path):

    layout = ShardedVectorConfig([[('weight', (2, 3))]])
    assert load_initial_eigenvector(None, layout) is None
    vector = torch.arange(6, dtype=torch.float32)
    torch.save(vector, tmp_path / 'current_vector_rank0.pt')
    torch.testing.assert_close(load_initial_eigenvector(tmp_path, layout), vector)
    torch.save(vector[:3], tmp_path / 'current_vector_rank0.pt')
    with pytest.raises(ValueError, match='vector size'):
        load_initial_eigenvector(tmp_path, layout)
