from records.track_3_optimization.configs.measurement_config import AddmmConfig, LossHessianConfig
import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from records.track_3_optimization.bubbleforce.analysis.bf16_gemm import (
    _fp32_mm_via_bf16,
    _fp32_bmm_via_bf16,
    _fp32_addmm_via_bf16,
    rewrite_bf16_gemms,
)
from records.track_3_optimization.bubbleforce.analysis.loss_hessian import LossHessian


@pytest.mark.parametrize("batched", [False, True])
def test_fp32_accumulation_rounds_only_operands(batched):
    torch.manual_seed(7)
    a = torch.randn((2, 7, 9) if batched else (7, 9))
    b = torch.randn((2, 9, 5) if batched else (9, 5))
    actual = (_fp32_bmm_via_bf16 if batched else _fp32_mm_via_bf16)(a, b)
    expected = a.bfloat16().float() @ b.bfloat16().float()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, expected.bfloat16().float())


@pytest.mark.parametrize("beta,alpha", [(0, 1), (2, 0.5), (-1, -2)])
def test_addmm_fp32_epilogue(beta, alpha):
    a, b = torch.randn(3, 4), torch.randn(4, 5)
    bias = torch.full((5,), float("nan")) if beta == 0 else torch.randn(5)
    expected = (a.bfloat16().float() @ b.bfloat16().float()) * alpha
    if beta:
        expected += bias * beta
    torch.testing.assert_close(
        _fp32_addmm_via_bf16(bias, a, b, config=AddmmConfig(beta=beta, alpha=alpha)), expected
    )


@pytest.mark.parametrize("scales", [{}, {"beta": 0}, {"beta": 2, "alpha": 0.5}])
def test_rewritten_addmm_passes_config_through_tracing_and_compilation(scales):
    a, b = torch.randn(3, 4), torch.randn(4, 5)
    bias = torch.full((5,), float("nan")) if scales.get("beta") == 0 else torch.randn(5)
    graph = make_fx(lambda bias, a, b: torch.addmm(bias, a, b, **scales))(bias, a, b)
    assert rewrite_bf16_gemms(graph) == {"aten.addmm.default": 1}
    expected = _fp32_addmm_via_bf16(bias, a, b, config=AddmmConfig(**scales))
    torch.testing.assert_close(graph(bias, a, b), expected)
    compiled = torch.compile(graph, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(bias, a, b), expected)


def test_rewrite_preserves_float64_and_fp32_pointwise():
    a, b = torch.randn(3, 4), torch.randn(4, 5)
    fn = lambda a, b: (a @ b).square() + (a.double() @ b.double()).float()
    graph = make_fx(fn, tracing_mode="fake")(a, b)
    assert rewrite_bf16_gemms(graph) == {"aten.mm.default": 1}
    torch.testing.assert_close(
        graph(a, b),
        _fp32_mm_via_bf16(a, b).square() + (a.double() @ b.double()).float(),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float64])
def test_bf16_substitution_refuses_unknown_input_precision(dtype):
    with pytest.raises(TypeError, match="FP32 inputs"):
        _fp32_mm_via_bf16(torch.ones(2, 3, dtype=dtype), torch.ones(3, 2, dtype=dtype))


def test_staging_reuses_program_across_global_token_normalization():
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(3, 4))

        def forward(self, x, y):
            return ((x @ self.w).tanh() - y).square().sum()

    m = Net()
    batch = torch.randn(2, 3), torch.randn(2, 4)
    h = LossHessian(
        m, ["w"], batch, 2, global_tokens=8, config=LossHessianConfig(mode="trace", gemm_precision="bf16_fp32")
    )
    v = (torch.randn_like(m.w),)
    original = h(v)[0]
    graph = next(iter(h._programs.values())).graph
    assert any(node.target is _fp32_mm_via_bf16 for node in graph.nodes)
    programs = dict(h._programs)
    h.global_tokens = 16
    torch.testing.assert_close(h(v)[0], original / 2, rtol=0, atol=0)
    assert programs == h._programs
    h.clear_program_cache()


def test_hessian_construction_does_not_change_process_precision():
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        LossHessian(torch.nn.Linear(3, 2), ["weight"],
                    (torch.randn(2, 3), torch.randn(2, 2)), 2,
                    global_tokens=4, config=LossHessianConfig(mode="trace", gemm_precision="bf16_fp32"))
        assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="actual CUDA BF16 GEMM required"
)
@pytest.mark.parametrize("batched", [False, True])
def test_cuda_fp32_output_accumulation_and_compile(batched):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    a = torch.randn((2, 17, 129) if batched else (17, 129), device="cuda")
    b = torch.randn((2, 129, 33) if batched else (129, 33), device="cuda")
    fn = _fp32_bmm_via_bf16 if batched else _fp32_mm_via_bf16
    expected = a.bfloat16().double() @ b.bfloat16().double()
    for actual in (fn(a, b), torch.compile(fn, fullgraph=True)(a, b)):
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual.double(), expected, rtol=3e-5, atol=3e-5)
