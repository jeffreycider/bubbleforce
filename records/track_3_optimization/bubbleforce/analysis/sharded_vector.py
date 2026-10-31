"""Pack weight-shaped tensors into vectors split across GPUs; reduce dot products."""
from collections.abc import Mapping
from math import prod

import torch
import torch.distributed as dist

from records.track_3_optimization.configs.measurement_config import ShardedVectorConfig


def pack_tensor(
    tensors_by_parameter_name: Mapping[str, torch.Tensor],
    config: ShardedVectorConfig,
    *,
    all_ranks: bool = False,
) -> torch.Tensor:
    """Copy named tensors into a vector, preserving each rank's parameter order.

    Args:
        tensors_by_parameter_name: Read tensor values by parameter name.
        config: Select rank assignments, vector size, device, and precision.
        all_ranks: Include every rank's tensors when true. Otherwise,
            include only this rank. Read all values from the supplied mapping.

    Returns:
        Tensor [local_vector_size], or [world_size * local_vector_size]
        in rank order when all_ranks is true. Each rank's tail is zero padded.
    """
    ranks_to_pack = range(config.world_size) if all_ranks else [config.rank]
    packed_vectors_by_row = torch.zeros(
        (len(ranks_to_pack), config.local_vector_size), device=config.device, dtype=config.dtype,
    )
    for row_index, rank in enumerate(ranks_to_pack):
        next_tensor_start = 0
        for parameter_name, parameter_shape in config.parameter_names_and_shapes_by_rank[rank]:
            tensor_element_count = prod(parameter_shape)
            next_tensor_end = next_tensor_start + tensor_element_count
            flattened_tensor = tensors_by_parameter_name[parameter_name].reshape(-1)
            packed_vectors_by_row[row_index, next_tensor_start:next_tensor_end].copy_(flattened_tensor)
            next_tensor_start = next_tensor_end
    return packed_vectors_by_row.reshape(-1)


def unpack_vector(
    packed_vector: torch.Tensor,
    config: ShardedVectorConfig,
    *,
    all_ranks: bool = False,
) -> dict[str, torch.Tensor]:
    """Read sections of a packed vector as named tensors with their original shapes.

    Args:
        packed_vector: Read a tensor [local_vector_size], or
            [world_size * local_vector_size] when all_ranks is true.
        config: Use the same rank assignments and vector size as during packing.
        all_ranks: Unpack every rank's matrices instead of this rank's only.

    Returns:
        Read each tensor through a view into packed_vector, restoring its
        parameter shape and skipping each rank's padding.
    """
    ranks_to_unpack = range(config.world_size) if all_ranks else [config.rank]
    packed_vectors_by_row = packed_vector.reshape(len(ranks_to_unpack), config.local_vector_size)
    tensors_by_parameter_name = {}
    for row_index, rank in enumerate(ranks_to_unpack):
        next_tensor_start = 0
        for parameter_name, parameter_shape in config.parameter_names_and_shapes_by_rank[rank]:
            tensor_element_count = prod(parameter_shape)
            next_tensor_end = next_tensor_start + tensor_element_count
            flattened_tensor = packed_vectors_by_row[row_index, next_tensor_start:next_tensor_end]
            tensors_by_parameter_name[parameter_name] = flattened_tensor.view(parameter_shape)
            next_tensor_start = next_tensor_end
    return tensors_by_parameter_name


def compute_global_dot(
    first_vector: torch.Tensor,
    second_vector: torch.Tensor,
    config: ShardedVectorConfig,
) -> float:
    """Sum the coordinate dot products across ranks, excluding padding.

    All ranks must call this together. An all-reduce SUM produces the same
    scalar on every rank; padding contributes nothing.

    Args:
        first_vector: This rank's first vector shard [local_vector_size].
        second_vector: This rank's second vector shard [local_vector_size].
        config: Select this rank's unpadded coordinates and the number of ranks.

    Returns:
        Global scalar dot product, excluding padding, reduced in FP64.
    """
    local_parameter_count = config.parameter_counts_by_rank[config.rank]
    dot_product = torch.sum(
        first_vector[:local_parameter_count].double()
        * second_vector[:local_parameter_count].double()
    )
    if config.world_size > 1:
        dist.all_reduce(dot_product)
    return dot_product.item()


def compute_global_norm(vector: torch.Tensor, config: ShardedVectorConfig) -> float:
    """Compute a global Euclidean norm; all ranks must participate.

    Args:
        vector: This rank's vector shard [local_vector_size].
        config: Select this rank's unpadded coordinates and the number of ranks.

    Returns:
        Global scalar norm, excluding padding.
    """
    return compute_global_dot(vector, vector, config) ** 0.5


def get_random_sharded_unit_vector(seed: int, config: ShardedVectorConfig) -> torch.Tensor:
    """Draw a globally normalized direction; all ranks must participate.

    Args:
        seed: Shared seed, offset internally by rank.
        config: Select rank assignments, vector size, device, and precision.

    Returns:
        This rank's shard [local_vector_size], with zero padding. The
        concatenated unpadded direction has unit global norm.
    """
    generator = torch.Generator(device=config.device).manual_seed(seed + 104729 * config.rank)
    vector = torch.randn(config.local_vector_size, device=config.device, dtype=config.dtype, generator=generator)
    vector[config.parameter_counts_by_rank[config.rank]:] = 0
    return vector / compute_global_norm(vector, config)
