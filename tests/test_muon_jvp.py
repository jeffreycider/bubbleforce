from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch.func import jvp

import bubbleforce.analysis.muon_jvp as muon_jvp
from bubbleforce.analysis.sharded_vector import pack_tensor
from configs.measurement_config import MuonStepConfig, ShardedVectorConfig
from optimizers.ringed.muon import Muon, compute_weight_after_muon_step


def compute_one_matrix_jvp(monkeypatch, momentum, config, direction, gradient_response):
    """Run compute_optimizer_jvp on one matrix, with H v replaced by gradient_response."""
    vector_config = ShardedVectorConfig([[("weight", tuple(momentum.shape))]], dtype=momentum.dtype)
    monkeypatch.setattr(
        muon_jvp, "compute_distributed_hvp",
        lambda *_: pack_tensor({"weight": gradient_response}, config=vector_config),
    )
    checkpoint = SimpleNamespace(
        vector_config=vector_config, momenta={"weight": momentum}, step_configs={"weight": config},
    )
    packed = muon_jvp.compute_optimizer_jvp(
        checkpoint, None, pack_tensor({"weight": direction}, config=vector_config),
    )
    return packed.view(momentum.shape)


@pytest.mark.parametrize("shape", [(4, 4), (3, 7), (7, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("learning_rate", [1e-4, .02])
@pytest.mark.parametrize("weight_decay", [0., .1])
def test_pathspring_response_converts_to_differentiated_weight_update(
    monkeypatch, shape, dtype, learning_rate, weight_decay,
):
    generator = torch.Generator().manual_seed(42)
    weight, momentum, direction, gradient_response = [
        torch.randn(shape, dtype=dtype, generator=generator) for _ in range(4)
    ]
    config = MuonStepConfig(
        learning_rate=learning_rate, weight_decay=weight_decay,
        current_gradient_coefficient=.2, training_batch_tokens=8,
    )
    polar, polar_response = muon_jvp.compute_newton_schulz_jvp(
        momentum,
        config.current_gradient_coefficient * config.training_batch_tokens * gradient_response,
    )
    scale = Muon.shape_scale(weight)
    _, next_weight_response = jvp(
        partial(
            compute_weight_after_muon_step,
            learning_rate=learning_rate, weight_decay=weight_decay,
        ),
        (weight.double(), (scale * polar).double()),
        (direction.double(), (scale * polar_response).double()),
    )
    expected = ((direction.double() - next_weight_response) / learning_rate).to(dtype)
    actual = compute_one_matrix_jvp(monkeypatch, momentum, config, direction, gradient_response)
    converted = (
        config.current_gradient_coefficient * actual.double()
        + weight_decay * direction.double()
    ).to(dtype)
    # Scaling tangents before or after the iterations changes FP32 rounding.
    relative_error = (converted - expected).norm() / expected.norm()
    assert relative_error < (1e-6 if dtype == torch.float32 else 1e-10)


@pytest.mark.parametrize("training_tokens", [8, 524288])
def test_pathspring_excludes_weight_decay_at_tiny_learning_rate(monkeypatch, training_tokens):
    weight = torch.eye(3, dtype=torch.float64)
    direction = torch.ones_like(weight)
    config = MuonStepConfig(
        learning_rate=1e-20, weight_decay=.1,
        current_gradient_coefficient=.2, training_batch_tokens=training_tokens,
    )
    actual = compute_one_matrix_jvp(
        monkeypatch, weight, config, direction, torch.zeros_like(weight),
    )
    expected = torch.zeros_like(direction)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_pathspring_is_independent_of_current_coefficient_and_decay(monkeypatch):
    generator = torch.Generator().manual_seed(19)
    momentum, direction, gradient_response = [
        torch.randn((3, 7), dtype=torch.float64, generator=generator) for _ in range(3)
    ]
    responses = [
        compute_one_matrix_jvp(
            monkeypatch, momentum,
            MuonStepConfig(learning_rate=.02, weight_decay=decay,
                           current_gradient_coefficient=c0, training_batch_tokens=8),
            direction, gradient_response,
        ) for c0, decay in [(.2, .1), (.7, .3)]
    ]
    torch.testing.assert_close(responses[0], responses[1], rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(4, 4), (3, 7), (7, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pathspring_is_invariant_to_training_token_duplication(monkeypatch, shape, dtype):
    generator = torch.Generator().manual_seed(77)
    mean_momentum, direction, mean_hvp = [
        torch.randn(shape, dtype=dtype, generator=generator) for _ in range(3)
    ]
    responses = [
        compute_one_matrix_jvp(
            monkeypatch, tokens * mean_momentum,
            MuonStepConfig(learning_rate=.025, weight_decay=.05,
                           current_gradient_coefficient=.0975, training_batch_tokens=tokens),
            direction, mean_hvp,
        ) for tokens in [524288, 1048576]
    ]
    # Duplicating every training token doubles summed momentum, but not pathspring.
    torch.testing.assert_close(responses[0], responses[1], rtol=0, atol=0)
