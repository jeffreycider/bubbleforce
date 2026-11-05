import time

import pytest
import torch
from torch import nn

from harness.hooks import print_training_progress
from harness.loop import accumulate_global_gradient
from harness.run_state import NanoGPTRunState


class SquaredError(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, inputs, targets):
        return ((self.weight * inputs - targets) ** 2).sum()


@pytest.mark.parametrize("microbatch_sequences", [1, 2, 4])
def test_loss_averages_tokens_across_ranks_and_preserves_gradients(monkeypatch, microbatch_sequences):
    inputs = torch.arange(1.0, 9.0).reshape(4, 2)
    targets = torch.ones_like(inputs)
    model = SquaredError()
    reference = SquaredError()
    reference_loss = reference(inputs, targets)
    reference_loss.backward()
    remote_loss, remote_tokens, remote_gradient = 17.0, 3, 11.0

    def reduce(value, op):
        if value.shape == (2,):
            value.add_(torch.tensor([remote_loss, remote_tokens]))
        else:
            value.add_(remote_gradient)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    loss = accumulate_global_gradient(model, (inputs, targets), microbatch_sequences)
    assert loss.item() == pytest.approx(
        (reference_loss.item() + remote_loss) / (targets.numel() + remote_tokens),
    )
    torch.testing.assert_close(model.weight.grad, reference.weight.grad + remote_gradient)
    assert not loss.requires_grad


def test_training_progress_saves_loss_with_completed_update_count():
    rows = []
    state = NanoGPTRunState(rank=0, world_size=1, step=7, train_loss=2.125)
    state.clock_start = time.perf_counter()
    state.print_log = lambda line, **kwargs: rows.append((line, kwargs))
    print_training_progress()(dict(train_steps=10), state)
    line, options = rows[0]
    assert line.startswith("step:8/10 train_loss:2.12500 ")
    assert options.get("log", True)
