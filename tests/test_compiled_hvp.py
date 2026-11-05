from configs.measurement_config import LossHessianConfig
from tests.reference_hessian import FrozenMuonHessian
import pytest
import torch
from torch import nn

from bubbleforce.analysis.loss_hessian import LossHessian


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(3, 4, dtype=torch.float64))
        self.b = nn.Parameter(torch.randn(4, 2, dtype=torch.float64))
        self.adam = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))

    def forward(self, x, y):
        return 0.5 * ((x @ self.a).tanh() @ self.b * self.adam - y).square().sum()


@pytest.mark.parametrize("mode", ["trace", "compile"])
@pytest.mark.parametrize("microbatch", [1, 3, 8])
def test_staged_whole_hvp_matches_autodiff_runtime_tokens_and_directions(
    mode, microbatch
):
    torch.manual_seed(8)
    m = Network()
    batch = (
        torch.randn(5, 3, dtype=torch.float64),
        torch.randn(5, 2, dtype=torch.float64),
    )
    args = (m, dict(m.named_parameters()), {}, ["a", "b"], batch, microbatch)
    ref = FrozenMuonHessian(*args, global_tokens=17)
    staged = LossHessian(m, ["a", "b"], batch, microbatch, global_tokens=17, config=LossHessianConfig(mode=mode))
    original = {n: p.clone() for n, p in m.named_parameters()}
    for _ in range(2):
        direction = torch.randn_like(m.a), torch.randn_like(m.b)
        for a, e in zip(staged(direction), ref(direction)):
            torch.testing.assert_close(a, e, rtol=1e-10, atol=1e-10)
        batch[0].add_(
            0.2
        )  # Token/data tensors are runtime inputs, not frozen constants.
    assert all(
        p.grad is None and torch.equal(p, original[n]) for n, p in m.named_parameters()
    )
    assert staged._programs
    staged.clear_program_cache()


@pytest.mark.parametrize("microbatch", [1, 3, 8])
def test_mean_hvp_is_invariant_to_validation_token_duplication(microbatch):
    torch.manual_seed(8)
    model = Network()
    batch = (torch.randn(5, 3, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64))
    direction = (torch.randn_like(model.a), torch.randn_like(model.b))
    products = []
    for repeat in (1, 2):
        repeated = tuple(value.repeat(repeat, 1) for value in batch)
        hessian = LossHessian(model, ["a", "b"], repeated, microbatch,
                              global_tokens=repeated[1].numel(), config=LossHessianConfig(mode="trace"))
        products.append(hessian(direction))
    for original, duplicated in zip(*products):
        torch.testing.assert_close(original, duplicated, rtol=1e-12, atol=1e-12)


def test_hessian_freezes_adam_and_promotes_saved_bf16_parameters():
    m = Network().bfloat16()
    batch = (torch.randn(5, 3), torch.randn(5, 2))
    h = LossHessian(m, ["a", "b"], batch, 3, global_tokens=batch[1].numel(), config=LossHessianConfig(mode="trace"))
    assert all(p.dtype == torch.float32 for p in h.params.values())
    assert not h.params["adam"].requires_grad
    assert all(h.params[n].requires_grad for n in h.parameter_names)
    original = {n: p.clone() for n, p in m.named_parameters()}
    h(tuple(torch.ones_like(p) for p in h.selected_parameters))
    assert all(
        torch.equal(p, original[n]) and p.grad is None for n, p in m.named_parameters()
    )


def test_linear_and_unused_parameters_have_zero_hessian():
    class LinearLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.used = nn.Parameter(torch.ones(2, 2))
            self.unused = nn.Parameter(torch.ones(2, 2))

        def forward(self, x, y):
            return (x @ self.used).sum()

    m = LinearLoss()
    h = LossHessian(
        m, ["used", "unused"], (torch.ones(3, 2), torch.zeros(3, 2)), 2, global_tokens=6, config=LossHessianConfig(mode="trace")
    )
    assert all(
        torch.count_nonzero(v) == 0
        for v in h(tuple(torch.ones_like(p) for p in h.selected_parameters))
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_hessian_owns_checkpoint_even_when_source_model_changes(dtype):
    class QuarticLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0, dtype=dtype))
            self.frozen = nn.Parameter(torch.tensor(3.0, dtype=dtype))
            self.register_buffer("scale", torch.tensor(2.0, dtype=dtype))

        def forward(self, x, y):
            return (self.scale * self.frozen * self.weight.pow(4) * x).sum()

    model = QuarticLoss()
    hessian = LossHessian(
        model, ["weight"], (torch.ones(2), torch.zeros(2)), 1,
        global_tokens=2, config=LossHessianConfig(mode="trace"),
    )
    direction = (torch.ones_like(hessian.selected_parameters[0]),)
    expected = torch.tensor(288.0, dtype=hessian.selected_parameters[0].dtype)
    torch.testing.assert_close(hessian(direction)[0], expected)
    with torch.no_grad():
        model.weight.fill_(5)
        model.frozen.fill_(7)
        model.scale.fill_(11)
    torch.testing.assert_close(hessian(direction)[0], expected)
    # Also test a new traced shape after mutation, not just the cached program.
    hessian.microbatch_size = 2
    torch.testing.assert_close(hessian(direction)[0], expected)
