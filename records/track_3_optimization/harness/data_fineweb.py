from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
from torch import Tensor


def _load_data_shard(file: Path) -> Tensor:
    """Load one FineWeb binary shard into a pinned uint16 token tensor."""
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def distributed_data_generator(filename_pattern: str, batch_size: int,
                               seq_len: int = 1024, start_batch: int = 0, *,
                               minibatch_seed: int, shuffle: bool = True) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield CUDA input/target batches from this rank's contiguous slice of each batch.

    Shuffle complete global batches within each shard using identical private
    CPU RNG sequences on all ranks. Validation uses shuffle=False.
    batch_size counts global tokens; start_batch skips that many global batches.
    """
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % dist.get_world_size() == 0
    local_batch_size = batch_size // dist.get_world_size()
    generator = torch.Generator(device="cpu").manual_seed(minibatch_seed)
    for file in files:
        tokens = _load_data_shard(file)
        if dist.get_rank() == 0:
            print(f"data shard loaded: {file} ({len(tokens)} tokens; batch {batch_size} tokens)", flush=True)
        # Match the record's strict cutoff: pos + batch_size + 1 < len(tokens).
        batches = range((len(tokens) - 2) // batch_size)
        if shuffle:
            batches = torch.randperm(len(batches), generator=generator, device="cpu").tolist()
        for batch in batches:
            if start_batch > 0:
                start_batch -= 1
                continue
            pos = batch * batch_size + dist.get_rank() * local_batch_size
            buf = tokens[pos:pos + local_batch_size + 1]
            inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
            targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
            yield inputs.view(-1, seq_len), targets.view(-1, seq_len)
