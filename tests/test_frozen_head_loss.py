from configs.measurement_config import FixedLMHeadConfig, LossHessianConfig
from tests.reference_hessian import FrozenMuonHessian
import pytest
import torch
import torch.nn.functional as F
from bubbleforce.analysis.fixed_lm_head_loss import (
    fixed_lm_head_cross_entropy,
    GPTWithFixedHeadLoss,
)
from harness.model_gpt import GPT
from bubbleforce.analysis.loss_hessian import LossHessian


def reference_loss(x, weight, bias, targets):
    z = F.linear(x, weight, bias)
    return F.cross_entropy(
        15 * z * (z.square() + 225).rsqrt(), targets, reduction="sum"
    )


@pytest.mark.parametrize("chunk", [1, 3, 20])
def test_float64_loss_gradient_hvp_and_chunk_invariance(chunk):
    torch.manual_seed(321)
    x = (torch.randn(5, 4, dtype=torch.float64) * 10).requires_grad_()
    w = torch.randn(13, 4, dtype=x.dtype) * 3
    b = torch.randn(13, dtype=x.dtype)
    y = torch.tensor([0, 3, 12, 7, 2])
    v = torch.randn_like(x)

    def f(z):
        return fixed_lm_head_cross_entropy(z, w, b, y, config=FixedLMHeadConfig(chunk_size=chunk))

    assert torch.autograd.gradcheck(f, (x,))
    assert torch.autograd.gradgradcheck(f, (x,))
    expected = reference_loss(x, w, b, y)
    actual = f(x)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    g = torch.autograd.grad(actual, x, create_graph=True)[0]
    e = torch.autograd.grad(expected, x, create_graph=True)[0]
    torch.testing.assert_close(g, e, rtol=1e-11, atol=1e-12)
    h = torch.autograd.grad(g, x, v)[0]
    eh = torch.autograd.grad(e, x, v)[0]
    torch.testing.assert_close(h, eh, rtol=1e-10, atol=1e-12)


def test_nonlinear_upstream_loss_chain_rule():
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    w = torch.randn(7, 4, dtype=x.dtype)
    b = torch.randn(7, dtype=x.dtype)
    y = torch.tensor([1, 6, 2])
    assert torch.autograd.gradgradcheck(
        lambda z: fixed_lm_head_cross_entropy(z, w, b, y).square(), (x,)
    )


@pytest.mark.parametrize("parameter", ["weight", "bias"])
def test_trainable_adam_head_refuses(parameter):
    x = torch.randn(3, 4, requires_grad=True)
    w = torch.randn(7, 4)
    b = torch.randn(7)
    (w if parameter == "weight" else b).requires_grad_()
    with pytest.raises(ValueError, match="trainable head"):
        fixed_lm_head_cross_entropy(x, w, b, torch.tensor([0, 1, 2]))


@pytest.mark.parametrize("seed,mode", [(0, "trace"), (7, "trace"), (7, "compile")])
def test_full_network_cross_matrix_hvp(seed, mode):
    torch.manual_seed(seed)
    model = GPT(31, 1, 128).float()
    fast = GPTWithFixedHeadLoss(31, 1, 128, config=FixedLMHeadConfig(chunk_size=3)).float()
    fast.load_state_dict(model.state_dict())
    names = [
        n
        for n, p in model.named_parameters()
        if "blocks." in n and n.endswith("weight")
    ]
    batch = torch.randint(0, 31, (3, 4)), torch.randint(0, 31, (3, 4))
    ref = FrozenMuonHessian(
        model,
        dict(model.named_parameters()),
        dict(model.named_buffers()),
        names,
        batch,
        2,
    )
    opt = LossHessian(
        fast, names, batch, 2, global_tokens=batch[1].numel(), config=LossHessianConfig(mode=mode)
    )
    for _ in range(2):
        v = tuple(torch.randn_like(p) for p in ref.p)
        for a, e in zip(opt(v), ref(v), strict=True):
            torch.testing.assert_close(a, e, rtol=5e-4, atol=3e-6)
        batch[0].copy_(torch.randint(0, 31, batch[0].shape))
    # Both full and tail microbatches reuse programs with new tokens/directions.
    assert len(opt._programs) == 2


def test_analytic_head_can_be_staged_as_full_hvp():
    torch.manual_seed(7)
    model = GPTWithFixedHeadLoss(31, 1, 128, config=FixedLMHeadConfig(chunk_size=3)).float()
    names = [
        n
        for n, p in model.named_parameters()
        if "blocks." in n and n.endswith("weight")
    ]
    batch = torch.randint(0, 31, (2, 4)), torch.randint(0, 31, (2, 4))
    args = (
        model,
        dict(model.named_parameters()),
        dict(model.named_buffers()),
        names,
        batch,
        1,
    )
    reference = FrozenMuonHessian(*args)
    staged = LossHessian(model, names, batch, 1, global_tokens=batch[1].numel(), config=LossHessianConfig(mode="trace"))
    assert {name for name, value in staged.params.items() if value.requires_grad} == {
        name for name, value in model.named_parameters()
        if name.startswith("blocks.") and value.ndim == 2
    }
    assert not staged.params["embed.weight"].requires_grad
    assert not staged.params["proj.weight"].requires_grad
    assert all(not value.requires_grad for value in staged.params.values() if value.ndim < 2)
    v = tuple(torch.randn_like(p) for p in reference.p)
    for a, e in zip(staged(v), reference(v)):
        torch.testing.assert_close(a, e, rtol=5e-4, atol=3e-6)


def test_cached_head_hvp_has_only_four_projections():
    from torch.fx.experimental.proxy_tensor import make_fx

    torch.manual_seed(91)
    x = torch.randn(5, 4, requires_grad=True)
    w = torch.randn(13, 4)
    b = torch.randn(13)
    y = torch.tensor([0, 3, 12, 7, 2])

    def hvp(z, v):
        loss = fixed_lm_head_cross_entropy(z, w, b, y, config=FixedLMHeadConfig(chunk_size=20))
        gradient = torch.autograd.grad(loss, z, create_graph=True)[0]
        return torch.autograd.grad(gradient, z, v)[0]

    graph = make_fx(hvp)(x, torch.randn_like(x))
    multiplications = [
        node
        for node in graph.graph.nodes
        if node.op == "call_function"
        and node.target in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default)
    ]
    assert len(multiplications) == 4


def test_cached_default_uses_one_projection_for_small_batches():
    from torch.fx.experimental.proxy_tensor import make_fx

    x = torch.randn(513, 4, requires_grad=True)
    w = torch.randn(13, 4)
    b = torch.randn(13)
    y = torch.zeros(513, dtype=torch.long)
    graph = make_fx(lambda z: fixed_lm_head_cross_entropy(z, w, b, y))(x)
    assert (
        sum(
            node.op == "call_function" and node.target == torch.ops.aten.addmm.default
            for node in graph.graph.nodes
        )
        == 1
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("vocab,dim,tokens", [(131, 32, 17), (50304, 768, 33)])
def test_cuda_projection_loss_gradient_and_hvp(vocab, dim, tokens):
    torch.manual_seed(33)
    x = (torch.randn(tokens, dim, device="cuda") * 0.1).requires_grad_()
    w = torch.randn(vocab, dim, device="cuda") * 0.1
    b = torch.randn(vocab, device="cuda") * 0.01
    y = torch.randint(0, vocab, (tokens,), device="cuda")
    v = torch.randn_like(x) * 0.1
    expected = reference_loss(x, w, b, y)
    actual = fixed_lm_head_cross_entropy(x, w, b, y, config=FixedLMHeadConfig(chunk_size=17))
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-5)
    g = torch.autograd.grad(actual, x, create_graph=True)[0]
    eg = torch.autograd.grad(expected, x, create_graph=True)[0]
    torch.testing.assert_close(g, eg, rtol=2e-3, atol=5e-6)
    h = torch.autograd.grad(g, x, v)[0]
    eh = torch.autograd.grad(eg, x, v)[0]
    torch.testing.assert_close(h, eh, rtol=3e-3, atol=5e-6)
