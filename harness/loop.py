from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor, nn

from harness.hooks import inject
from harness.recording.snapshots import Recording
from harness.run_state import Config, Hook, NanoGPTRunState


def accumulate_global_gradient(model: nn.Module, batch: tuple[Tensor, Tensor],
                               microbatch_sequences: int) -> Tensor:
    """Sum gradients across ranks and return the global mean loss per token."""
    inputs, targets = batch
    assert len(inputs) % microbatch_sequences == 0
    loss_and_tokens = torch.zeros(2, dtype=torch.float64, device=targets.device)
    loss_and_tokens[1] = targets.numel()
    for i in range(len(inputs) // microbatch_sequences):
        loss = model(inputs[i*microbatch_sequences:(i+1)*microbatch_sequences],
                     targets[i*microbatch_sequences:(i+1)*microbatch_sequences])
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss_and_tokens[0] += loss.detach().double()
        loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    dist.all_reduce(loss_and_tokens, op=dist.ReduceOp.SUM)
    return loss_and_tokens[0] / loss_and_tokens[1]


def run(config: Config, hooks: dict[str, list[Hook]], recording: Recording) -> None:
    """The shared Track-3 loop, with an optional per-update pre-forward site.

    Setup installs the model, optimizer and data in one NanoGPTRunState.
    Hooks mutate that object; experimental instrumentation lives in extras.

    Args:
        config: The fully resolved YAML config dict.
        hooks: The configured training callbacks, in YAML order.
        recording: The resolved snapshot/checkpoint policy for this run.

    Raises:
        ValueError: On a world size other than config["require_world_size"] --
            rank count sets the effective batch, so it is part of the protocol.
    """
    if dist.get_world_size() != config["require_world_size"]:
        raise ValueError(f"this config's reproduction protocol requires world_size"
                         f" {config['require_world_size']}, got {dist.get_world_size()}")
    run_state = NanoGPTRunState(dist.get_rank(), dist.get_world_size(),
                               step=config.get("start_step", 0) - 1)
    inject(hooks["setup"], config, run_state)
    recording.setup(run_state)
    run_state.setup_finished = True
    assert run_state.train_batches is not None
    # Early stopping does not shorten the learning-rate schedule.
    stop_step = min(config["train_steps"], int(config.get("stop_step", config["train_steps"])))
    for step in range(run_state.step + 1, stop_step):
        run_state.step = step
        inject(hooks["pre_forward"], config, run_state)
        batch = next(run_state.train_batches)
        run_state.train_loss = accumulate_global_gradient(
            run_state.model, batch, config["microbatch_sequences"],
        ).item()
        inject(hooks["pre_optimizer"], config, run_state)
        recording.before_optimizer(run_state)
        run_state.optimizer.step()
        run_state.model.zero_grad(set_to_none=True)
        recording.after_optimizer(run_state)
        inject(hooks["post_optimizer"], config, run_state)
        recording.after_step(run_state)
    inject(hooks["teardown"], config, run_state)
