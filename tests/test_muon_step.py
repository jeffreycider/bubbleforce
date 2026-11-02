from functools import partial

import pytest
import torch
from torch.func import jvp

from bubbleforce.analysis.muon_jvp import compute_newton_schulz_jvp
from bubbleforce.analysis.muon_step import compute_negative_update_jvp
from configs.measurement_config import MuonStepConfig
from optimizers.ringed.muon import Muon, compute_weight_after_muon_step


@pytest.mark.parametrize("shape", [(4, 4), (3, 7), (7, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("learning_rate", [1e-4, .02])
@pytest.mark.parametrize("weight_decay", [0., .1])
def test_muon_response_matches_differentiated_weight_update(
    shape, dtype, learning_rate, weight_decay,
):
    generator = torch.Generator().manual_seed(42)
    weight, momentum, direction, gradient_response = [
        torch.randn(shape, dtype=dtype, generator=generator) for _ in range(4)
    ]
    config = MuonStepConfig(
        learning_rate=learning_rate, weight_decay=weight_decay,
        current_gradient_coefficient=.2, training_batch_tokens=8,
    )
    polar, polar_response = compute_newton_schulz_jvp(
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
    actual = compute_negative_update_jvp(weight, momentum, config, direction, gradient_response)
    torch.testing.assert_close(actual, expected, rtol=1e-6 if dtype == torch.float32 else 1e-10, atol=1e-10)


def test_muon_weight_decay_response_at_tiny_learning_rate():
    weight = torch.eye(3, dtype=torch.float64)
    direction = torch.ones_like(weight)
    config = MuonStepConfig(
        learning_rate=1e-20, weight_decay=.1,
        current_gradient_coefficient=.2, training_batch_tokens=8,
    )
    actual = compute_negative_update_jvp(
        weight, weight, config, direction, torch.zeros_like(weight),
    )
    torch.testing.assert_close(actual, .1 * direction, rtol=0, atol=0)
