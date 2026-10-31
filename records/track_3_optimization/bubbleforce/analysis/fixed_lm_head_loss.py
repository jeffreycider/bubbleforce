import torch
import torch.nn.functional as F

from records.track_3_optimization.harness.model_gpt import GPT
from records.track_3_optimization.configs.measurement_config import FixedLMHeadConfig


def _softcapped_cross_entropy_terms(
    logits: torch.Tensor, targets: torch.Tensor, log_normalizers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute shared terms for the hidden-state loss gradient and HVP.

    Args:
        logits: Cached vocabulary logits [N, V], before softcapping.
        targets: Integer target vocabulary indices [N].
        log_normalizers: Cached log-sum-exp of softcapped logits [N].

    Returns:
        Four tensors [N, V]: probabilities, cross-entropy gradient with respect
        to softcapped logits, softcap derivative, and 1 / sqrt(logits**2 + 225).
    """
    inverse_softcap_denominator = (logits.square() + 225).rsqrt()
    probabilities = (15 * logits * inverse_softcap_denominator - log_normalizers[:, None]).exp()
    softcap_derivative = 3375 * inverse_softcap_denominator.pow(3)
    # Scatter a single target per row rather than materializing one_hot.
    cross_entropy_gradient = probabilities.scatter_add(
        1, targets[:, None], -torch.ones_like(targets[:, None], dtype=logits.dtype)
    )
    return probabilities, cross_entropy_gradient, softcap_derivative, inverse_softcap_denominator


def _hidden_state_loss_gradient(
    logits: torch.Tensor, head_weight: torch.Tensor, targets: torch.Tensor,
    log_normalizers: torch.Tensor,
) -> torch.Tensor:
    """Compute the summed-loss gradient with respect to final hidden states.

    N is the token count, V the vocabulary size and D the hidden dimension.

    Args:
        logits: Cached vocabulary logits [N, V], before softcapping.
        head_weight: Frozen output projection [V, D].
        targets: Integer target vocabulary indices [N].
        log_normalizers: Cached log-sum-exp of softcapped logits [N].

    Returns:
        Hidden-state loss gradient [N, D].
    """
    _, cross_entropy_gradient, softcap_derivative, _ = _softcapped_cross_entropy_terms(
        logits, targets, log_normalizers
    )
    return (cross_entropy_gradient * softcap_derivative) @ head_weight


def _hidden_state_loss_hvp(
    logits: torch.Tensor, head_weight: torch.Tensor, targets: torch.Tensor,
    log_normalizers: torch.Tensor, hidden_state_direction: torch.Tensor,
) -> torch.Tensor:
    """Apply the hidden-state loss Hessian using cached forward-pass values.

    N is the token count, V the vocabulary size and D the hidden dimension.

    Args:
        logits: Cached vocabulary logits [N, V], before softcapping.
        head_weight: Frozen output projection [V, D].
        targets: Integer target vocabulary indices [N].
        log_normalizers: Cached log-sum-exp of softcapped logits [N].
        hidden_state_direction: Perturbation of final hidden states [N, D].

    Returns:
        Hidden-state Hessian-vector product [N, D] for the summed loss.
    """
    probabilities, cross_entropy_gradient, softcap_derivative, inverse_softcap_denominator = (
        _softcapped_cross_entropy_terms(logits, targets, log_normalizers)
    )
    logits_direction = F.linear(hidden_state_direction, head_weight)
    softcapped_logits_direction = softcap_derivative * logits_direction
    mean_softcapped_logits_direction = (probabilities * softcapped_logits_direction).sum(-1, keepdim=True)
    loss_gradient_change = probabilities * (
        softcapped_logits_direction - mean_softcapped_logits_direction
    ) * softcap_derivative
    softcap_second_derivative = -10125 * logits * inverse_softcap_denominator.pow(5)
    loss_gradient_change = (
        loss_gradient_change + cross_entropy_gradient * softcap_second_derivative * logits_direction
    )
    return loss_gradient_change @ head_weight


class _FixedLMHeadGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z, weight, targets, lse, upstream):
        ctx.save_for_backward(z, weight, targets, lse, upstream)
        return _hidden_state_loss_gradient(z, weight, targets, lse) * upstream

    @staticmethod
    def backward(ctx, direction):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Frozen head supports second derivatives, not third derivatives"
            )
        z, weight, targets, lse, upstream = ctx.saved_tensors
        dx = _hidden_state_loss_hvp(z, weight, targets, lse, direction) * upstream
        d_upstream = None
        if ctx.needs_input_grad[5]:
            gradient = _hidden_state_loss_gradient(z, weight, targets, lse)
            d_upstream = (direction * gradient).sum().reshape_as(upstream)
        # z and lse are frozen intermediates. The complete analytic Hessian
        # returns through x, not through a second path via cached logits.
        return dx, None, None, None, None, d_upstream


class _FixedLMHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, targets):
        z = F.linear(x, weight, bias)
        u = 15 * z * (z.square() + 225).rsqrt()
        lse = u.logsumexp(-1)
        ctx.save_for_backward(x, z, weight, targets, lse)
        return (lse - u.gather(1, targets[:, None]).squeeze(1)).sum()

    @staticmethod
    def backward(ctx, upstream):
        x, z, weight, targets, lse = ctx.saved_tensors
        dx = _FixedLMHeadGradient.apply(x, z, weight, targets, lse, upstream)
        return dx, None, None, None


def fixed_lm_head_cross_entropy(
    hidden_states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
    targets: torch.Tensor, *, config: FixedLMHeadConfig = FixedLMHeadConfig(),
) -> torch.Tensor:
    """Compute cross-entropy through a fixed LM head with explicit derivatives.

    Logits use the softcap ``15 * z / sqrt(z**2 + 225)``. Cache the vocabulary
    projection, retaining every token and vocabulary entry. Hidden-state
    derivatives are supported through second order, but not third order.
    D is the hidden dimension and V the vocabulary size.

    Args:
        hidden_states: Hidden states [..., D].
        weight: Output projection [V, D], with requires_grad=False.
        bias: Output bias [V], with requires_grad=False.
        targets: Integer vocabulary indices with shape hidden_states.shape[:-1].
        config: Set the number of flattened tokens per projection chunk.

    Returns:
        Scalar tensor containing cross-entropy summed over all tokens.
    """
    if weight.requires_grad or bias.requires_grad:
        raise ValueError(
            "Frozen-head loss cannot differentiate trainable head parameters"
        )
    chunk_size = config.chunk_size
    if hidden_states.shape[:-1] != targets.shape or hidden_states.shape[-1] != weight.shape[1]:
        raise ValueError("hidden states, targets, and weight shapes disagree")
    x = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    targets = targets.reshape(-1).contiguous()
    return sum(
        (
            _FixedLMHeadLoss.apply(
                x[start : start + chunk_size],
                weight,
                bias,
                targets[start : start + chunk_size],
            )
            for start in range(0, len(x), chunk_size)
        ),
        x.new_zeros(()),
    )


class GPTWithFixedHeadLoss(GPT):
    """GPT using fixed-head cross-entropy with explicit gradient and HVP formulas.

    Construction does not freeze the head. Freeze proj.weight and proj.bias
    before forward, or pass fixed head parameters through LossHessian.
    """

    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, *,
                 config: FixedLMHeadConfig = FixedLMHeadConfig()) -> None:
        """Configure the GPT architecture and output-head token chunking.

        Args:
            vocab_size: Number of vocabulary entries.
            num_layers: Number of transformer blocks.
            model_dim: Hidden-state width.
            config: Set the number of flattened tokens per head chunk.
        """
        super().__init__(vocab_size, num_layers, model_dim)
        self.head_config = config

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Evaluate summed token loss with fixed output-head parameters.

        Args:
            inputs: Integer input token indices [B, T].
            targets: Integer target token indices [B, T].

        Returns:
            Scalar softcapped cross-entropy summed over all B * T tokens.
        """
        return fixed_lm_head_cross_entropy(
            self.compute_hidden_states(inputs),
            self.proj.weight,
            self.proj.bias,
            targets,
            config=self.head_config,
        )
