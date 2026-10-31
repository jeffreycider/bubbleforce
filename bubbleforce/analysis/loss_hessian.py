from collections.abc import Callable, Sequence
import torch
from torch.func import functional_call
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.attention import SDPBackend, sdpa_kernel

from bubbleforce.analysis.bf16_gemm import rewrite_bf16_gemms
from configs.measurement_config import LossHessianConfig


class LossHessian:
    """Apply the token-mean loss Hessian to directions for selected parameters.

    Weights and buffers are copied; only ``parameter_names`` are differentiation
    variables. FP16/BF16 weights are promoted to FP32; other dtypes are preserved.
    Each microbatch program expands both derivatives before optional compilation.
    Tokens and directions remain runtime inputs, not cached activations.
    The model returns a summed loss and must be deterministic (GPT has no dropout).
    Unselected parameters remain fixed, including Adam parameters in Muon probes.
    The object stores no explicit Hessian matrix: each call computes H[v] over
    the supplied token batch, reusing its checkpoint copies and executable cache.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        parameter_names: Sequence[str],
        batch: tuple[torch.Tensor, torch.Tensor],
        microbatch_size: int,
        *,
        global_tokens: int,
        config: LossHessianConfig = LossHessianConfig(),
    ) -> None:
        """Capture checkpoint weights and configure reusable HVP programs.

        Args:
            model: Deterministic module returning scalar summed loss from
                ``model(inputs, targets)``.
            parameter_names: Nonempty, unique parameter names defining the order
                of direction and result tensors. All other weights stay fixed.
            batch: Local (inputs, targets), each with B leading rows; for GPT,
                both are token-index tensors [B, T]. Tensors remain live inputs.
            microbatch_size: Positive number of leading batch rows per HVP.
            global_tokens: Loss normalization denominator, at least the local
                target count. Distributed callers sum the returned local products.
            config: Select traced or compiled execution and GEMM precision.
                With bf16_fp32, round FP32 matrix operands to BF16 and accumulate
                in FP32. Disable allow_bf16_reduced_precision_reduction in the
                CUDA caller before execution.
        """
        self.parameter_names = tuple(parameter_names)
        if not self.parameter_names or len(set(self.parameter_names)) != len(
            self.parameter_names
        ):
            raise ValueError("nonempty unique parameter names required")
        self.model = model
        self.params = {
            parameter_name: (
                parameter.detach().float()
                if parameter.dtype in {torch.bfloat16, torch.float16}
                else parameter.detach().clone()
            ).requires_grad_(parameter_name in self.parameter_names)
            for parameter_name, parameter in model.named_parameters()
        }
        self.selected_parameters = tuple(
            self.params[parameter_name] for parameter_name in self.parameter_names
        )
        self.buffers = {
            buffer_name: buffer.detach().clone()
            for buffer_name, buffer in model.named_buffers()
        }
        self.batch, self.microbatch_size = batch, microbatch_size
        self.global_tokens = global_tokens
        if microbatch_size < 1 or self.global_tokens < batch[1].numel():
            raise ValueError("invalid microbatch/global normalization")
        self.config = config
        self._programs = {}

    def _microbatch_hvp(
        self, model_inputs: torch.Tensor, targets: torch.Tensor, *parameter_directions: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """
        Compute a microbatch's summed-loss Hessian-vector product.

        Args:
            model_inputs: Input tokens for this microbatch.
            targets: Target tokens for this microbatch.
            *parameter_directions: Direction tensors matching the selected
                parameters' shapes and order.

        Returns:
            Detached HVP tensors in selected-parameter order, including
            cross-parameter contributions. Global token normalization is
            applied by the caller.
        """
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            summed_loss = functional_call(self.model, (self.params, self.buffers), (model_inputs, targets))
            parameter_gradients = torch.autograd.grad(
                summed_loss, self.selected_parameters, create_graph=True, allow_unused=True
            )
            differentiable_gradient_direction_pairs = [
                (parameter_gradient, parameter_direction)
                for parameter_gradient, parameter_direction in zip(parameter_gradients, parameter_directions)
                if parameter_gradient is not None and parameter_gradient.requires_grad
            ]

            per_parameter_gradient = tuple(gradient for gradient, _ in differentiable_gradient_direction_pairs)
            per_parameter_direction_perturbation = tuple(direction for _, direction in differentiable_gradient_direction_pairs)
            if differentiable_gradient_direction_pairs:
                hessian_vector_products = (
                    torch.autograd.grad(
                        per_parameter_gradient,
                        self.selected_parameters,
                        grad_outputs=per_parameter_direction_perturbation,
                        allow_unused=True,
                    )
                )
            else:
                hessian_vector_products = (None,) * len(self.selected_parameters)

            return tuple(
                torch.zeros_like(parameter) if hessian_vector_product is None else hessian_vector_product.detach()
                for parameter, hessian_vector_product in zip(self.selected_parameters, hessian_vector_products)
            )

    def _get_or_build_microbatch_hvp(
        self,
        model_inputs: torch.Tensor,
        targets: torch.Tensor,
        *parameter_directions: torch.Tensor,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Prepare reusable instructions for computing a microbatch HVP.

        For new input shapes, dtypes, or device types, trace the forward and
        both derivative calculations using fake tensors. Optionally prepare
        compiled execution, then cache the resulting callable for reuse.

        Each call to the returned function uses real tokens and directions
        to recompute the required forward intermediates and numerical
        derivative values using the saved checkpoint weights.

        Args:
            model_inputs: Input tokens defining the microbatch input format.
            targets: Target tokens defining the microbatch target format.
            *parameter_directions: Direction tensors matching the selected
                parameters' shapes and order.

        Returns:
            A callable accepting tokens, targets, and direction tensors and
            computing a fresh summed-loss HVP. The caller applies global
            token normalization after executing it.
        """

        input_descriptions = []
        for tensor in (model_inputs, targets, *parameter_directions):
            input_descriptions.append(
                (tuple(tensor.shape), tensor.dtype, tensor.device.type)
            )
        input_signature = tuple(input_descriptions)

        if input_signature in self._programs:
            return self._programs[input_signature]

        trace_hvp = make_fx(
            self._microbatch_hvp,
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
        )
        traced_hvp = trace_hvp(
            model_inputs, targets, *parameter_directions,
        )

        traced_hvp.graph.eliminate_dead_code()
        traced_hvp.recompile()

        if self.config.gemm_precision == "bf16_fp32":
            rewrite_bf16_gemms(traced_hvp)

        hvp_function = traced_hvp
        if self.config.mode == "compile":
            hvp_function = torch.compile(
                traced_hvp.forward,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )

        self._programs[input_signature] = hvp_function
        return hvp_function

    @torch.no_grad()
    def __call__(self, direction: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Apply the selected-parameter Hessian, including cross-parameter terms.

        Args:
            direction: One tensor per ``parameter_names`` entry, in that order,
                matching the shape, dtype and device of ``selected_parameters``.

        Returns:
            Detached HVP tensors with the selected parameters' shapes, dtypes
            and order. The local summed-loss product is divided by
            ``global_tokens``; no distributed reduction is performed.
        """
        if len(direction) != len(self.selected_parameters):
            raise ValueError("one direction per selected parameter required")
        accumulated_hessian_vector_products = [torch.zeros_like(parameter) for parameter in self.selected_parameters]
        for microbatch_start in range(0, len(self.batch[0]), self.microbatch_size):
            microbatch_end = microbatch_start + self.microbatch_size
            model_inputs = self.batch[0][microbatch_start:microbatch_end]
            targets = self.batch[1][microbatch_start:microbatch_end]
            microbatch_hvp = self._get_or_build_microbatch_hvp(
                model_inputs, targets, *direction,
            )
            microbatch_hessian_vector_products = microbatch_hvp(model_inputs, targets, *direction)
            for accumulated_product, microbatch_product in zip(
                accumulated_hessian_vector_products, microbatch_hessian_vector_products,
            ):
                accumulated_product.add_(microbatch_product / self.global_tokens)
        return tuple(accumulated_hessian_vector_products)

    def clear_program_cache(self) -> None:
        """Discard cached programs; later calls rebuild them as needed.

        Returns:
            None. Checkpoint tensors are retained.
        """
        self._programs.clear()
