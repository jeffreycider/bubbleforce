from __future__ import annotations
from collections.abc import Callable
import os
import time
import torch
import torch.distributed as dist
try:
    import wandb
except ModuleNotFoundError:
    wandb = None
from harness.schedules import learning_rate_factor
from harness.checkpoint import load_checkpoint
from harness.optimizers import GroupedOptimizers
from harness.data_fineweb import distributed_data_generator
from harness.model_gpt import GPT, initialize_parameters_like_record
from harness.run_state import Config, Hook, NanoGPTRunState
SITES = ("setup", "pre_forward", "pre_optimizer", "post_optimizer", "teardown")



def read_git_head() -> str:
    """The current commit hash by reading .git directly -- never a subprocess:
    forking a process that holds a live NCCL communicator (as every rank does by
    the time setup hooks run) can hang later collectives."""
    try:
        with open(".git/HEAD") as f:
            ref = f.read().strip()
        if not ref.startswith("ref: "):
            return ref
        ref_path = os.path.join(".git", ref[5:])
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                return f.read().strip()
        with open(".git/packed-refs") as f:
            for line in f:
                if line.strip().endswith(ref[5:]):
                    return line.split()[0]
        return "unknown"
    except OSError:
        return "unknown"


def open_rank_zero_log() -> Hook:
    """Install the rank-zero logger and record the Git commit and GPU environment."""
    def hook(config: Config, state: NanoGPTRunState) -> None:
        rank = state.rank
        logfile = None
        if rank == 0:
            os.makedirs(config["output_dir"], exist_ok=True)
            logfile = os.path.join(config["output_dir"], "train.log")
            print(logfile)

        def print_log(s: str, console: bool = False, log: bool = True) -> None:
            if rank == 0:
                if console:
                    print(s)
                if log:
                    with open(logfile, "a") as f:
                        print(s, file=f)

        state.print_log = print_log
        print_log("=" * 100)
        print_log(f"git HEAD: {read_git_head()}")
        print_log(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
                  + f" on {torch.cuda.get_device_name()} with world_size {state.world_size}")
        print_log("=" * 100)
        if rank == 0 and config.get("wandb_offline", False):
            if wandb is None:
                raise RuntimeError("wandb_offline requires the wandb package")
            state.extras['wandb'] = wandb.init(mode="offline", dir=config["output_dir"],
                                        project="track3-reproductions", name=config["run_id"])
    return hook


def load_validation_tokens() -> Hook:
    """Loads the fixed validation data into the run state."""
    def hook(config: Config, state: NanoGPTRunState) -> None:
        state.validation_data = next(
            distributed_data_generator(config["val_data"], config["val_tokens"],
                                       minibatch_seed=config["minibatch_seed"], shuffle=False))
    return hook


def build_compiled_gpt() -> Hook:
    """Construct and compile the configured GPT model on CUDA."""
    def hook(config: Config, state: NanoGPTRunState) -> None:
        model = GPT(**config["model"]).cuda()
        model.compile(dynamic=False)
        state.model = model
    return hook


def seed_then_initialize_parameters() -> Hook:
    """Optionally reseed the global RNG, then initialize the model parameters.

    No intervening hook may consume the RNG before initial parameters are broadcast.
    """
    def hook(config: Config, state: NanoGPTRunState) -> None:
        if config["reseed_before_initialization"]:
            torch.manual_seed(config["seed_offset"] + config["seed"])
        initialize_parameters_like_record(state.model, **config["initialization"])
    return hook


def seed_process(*, rank_offset: bool = True) -> Hook:
    def hook(config: Config, state: NanoGPTRunState) -> None:
        seed = config["seed"] + (dist.get_rank() if rank_offset else 0)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return hook


def schedule_optimizer_groups() -> Hook:
    def hook(config: Config, state: NanoGPTRunState) -> None:
        for row, (_, optimizer) in zip(config["optimizer_groups"], state.optimizer.groups):
            factor = learning_rate_factor(row["lr_schedule"], state.step, config["train_steps"])
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
    return hook


def assemble_grouped_optimizer() -> Hook:
    """Builds the regex-grouped hybrid optimizer for the run."""
    def hook(config: Config, state: NanoGPTRunState) -> None:
        state.optimizer = GroupedOptimizers(
            state.model, config["optimizer_groups"])
    return hook


def open_training_batches() -> Hook:
    """Opens the run's sharded training-token stream."""
    def hook(config: Config, state: NanoGPTRunState) -> None:
        state.train_batches = distributed_data_generator(
            config["train_data"], config["batch_tokens"], start_batch=state.step + 1,
            minibatch_seed=config["minibatch_seed"])
    return hook


def restore_checkpoint() -> Hook:
    """Restore model, optimizer and RNG before opening the training stream."""
    def hook(config: Config, state: NanoGPTRunState) -> None:
        if config.get("resume_checkpoint"):
            load_checkpoint(config, state)
    return hook


def broadcast_initial_parameters() -> Hook:
    """Broadcast rank zero's initialized parameters to all training ranks."""
    def hook(config: Config, state: NanoGPTRunState) -> None:
        for p in state.model.parameters():
            dist.broadcast(p.detach(), 0)
    return hook


def mark_log_finished() -> Hook:
    """Teardown: writes the closing line so finished logs are distinguishable."""
    def hook(config: Config, state: NanoGPTRunState) -> None:
        state.print_log(f"run finished: {state.step + 1} executed steps,"
                           + f" total train_time {state.training_time:.3f}s")
        if state.extras.get("wandb") is not None:
            state.extras['wandb'].finish()
    return hook


def boundary_is_due(boundary: int, train_steps: int, every: int, final_tenth_every: int,
                    dense_window: list[int] | None, dense_every: int) -> bool:
    """Select the final checkpoint and configured regular, late or dense validation steps."""
    if boundary == train_steps:
        return True
    if every:
        frequency = every if boundary / train_steps < 0.9 else (final_tenth_every or every)
        if boundary % frequency == 0:
            return True
    if dense_window is not None and dense_window[0] <= boundary <= dense_window[1] \
            and boundary % dense_every == 0:
        return True
    return False


def validate_at_step_boundaries(*, every: int = 0, final_tenth_every: int = 0,
                                dense_window: list[int] | None = None,
                                dense_every: int = 10) -> Hook:
    """Validate and log global token-mean loss at the configured checkpoints.

    Evaluate the initial/restored checkpoint during setup and exclude validation
    time from the training clock.
    """
    last_boundary = None

    def run_validation(config: Config, state: NanoGPTRunState, boundary: int) -> None:
        nonlocal last_boundary
        train_steps = config["train_steps"]
        mbs = config["microbatch_sequences"]
        model = state.model
        # stop the clock
        dist.barrier()
        elapsed = time.perf_counter() - state.clock_start
        previous = config.get("start_step", 0) if last_boundary is None else last_boundary
        completed = boundary - previous
        step_avg = elapsed / completed if completed > 0 else float("nan")
        last_boundary = boundary
        state.training_time += elapsed
        model.eval()
        val_inputs, val_targets = state.validation_data
        val_loss = 0
        with torch.no_grad():
            assert len(val_inputs) % mbs == 0
            for i in range(len(val_inputs) // mbs):
                val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss /= config["val_tokens"]
        if not torch.isfinite(val_loss):
            raise RuntimeError("non-finite validation loss")
        state.print_log(f"step:{boundary}/{train_steps} val_loss:{val_loss:.5f}"
                           + f" train_time:{state.training_time:.3f}s"
                           + f" step_avg:{1000*step_avg:.2f}ms", console=True)
        if state.extras.get("wandb") is not None:
            state.extras['wandb'].log({"step": boundary, "val_loss": val_loss.item(),
                                "objective": val_loss.item()})
        model.train()
        # start the clock again
        dist.barrier()
        state.clock_start = time.perf_counter()

    def hook(config: Config, state: NanoGPTRunState) -> None:
        if not state.setup_finished:
            # setup site: establish the clock, then validate boundary 0
            state.training_time = 0.0
            dist.barrier()
            state.clock_start = time.perf_counter()
            run_validation(config, state, state.step + 1)
            return
        boundary = state.step + 1
        if boundary_is_due(boundary, config["train_steps"],
                           every, final_tenth_every, dense_window, dense_every):
            run_validation(config, state, boundary)
    return hook


def print_training_progress(*, every: int = 1) -> Hook:
    """Log completed updates, mean training loss and elapsed training time."""
    def hook(config: Config, state: NanoGPTRunState) -> None:
        step = state.step
        if every and step % every == 0:
            approx_training_time = state.training_time + (time.perf_counter() - state.clock_start)
            state.print_log(f"step:{step+1}/{config['train_steps']}"
                               + f" train_loss:{state.train_loss:.5f}"
                               + f" train_time:{approx_training_time:.3f}s"
                               + f" step_avg:{1000*approx_training_time/(step + 1 - config.get('start_step', 0)):.2f}ms",
                               console=True)
    return hook


def inject(hooks: list[Hook], config: Config, state: NanoGPTRunState) -> None:
    """Call mutating hooks on one run state, in YAML list order."""
    for hook in hooks:
        hook(config, state)

def hook_factories() -> dict[str, Callable[..., Hook]]:
    """The supported YAML hook names and their concrete factories."""
    return {
        "open_rank_zero_log": open_rank_zero_log,
        "load_validation_tokens": load_validation_tokens,
        "build_compiled_gpt": build_compiled_gpt,
        "seed_process": seed_process,
        "seed_then_initialize_parameters": seed_then_initialize_parameters,
        "assemble_grouped_optimizer": assemble_grouped_optimizer,
        "restore_checkpoint": restore_checkpoint,
        "open_training_batches": open_training_batches,
        "broadcast_initial_parameters": broadcast_initial_parameters,
        "mark_log_finished": mark_log_finished,
        "schedule_optimizer_groups": schedule_optimizer_groups,
        "validate_at_step_boundaries": validate_at_step_boundaries,
        "print_training_progress": print_training_progress,
    }

def bind_sites(config: Config) -> dict[str, list[Hook]]:
    if config.get("resume_checkpoint"):
        setup = [entry["name"] for entry in config.get("setup", [])]
        required = ["assemble_grouped_optimizer", "restore_checkpoint", "open_training_batches"]
        if (any(name not in setup for name in required)
                or not setup.index(required[0]) < setup.index(required[1]) < setup.index(required[2])):
            raise ValueError("resume requires restoring the assembled optimizer before opening training data")
    factories = hook_factories()
    unknown = sorted({entry["name"] for site in SITES for entry in config.get(site, [])
                      if entry["name"] not in factories})
    if unknown:
        raise ValueError(f"unsupported selected lifecycle hooks: {unknown}")
    hooks = {site: [factories[entry["name"]](**entry.get("hyperparams", {}))
                    for entry in config.get(site, [])] for site in SITES}
    return hooks
