"""Gather v, sum each GPU's token-shard HVP, and return its parameter shard."""
import torch
import torch.distributed as dist

from configs.measurement_config import ShardedVectorConfig
from bubbleforce.analysis.loss_hessian import LossHessian
from bubbleforce.analysis.sharded_vector import (
    pack_tensor, unpack_vector,
)


def compute_distributed_hvp(
    vector_config: ShardedVectorConfig, hessian: LossHessian,
    direction_shard: torch.Tensor,
) -> torch.Tensor:
    """Gather a direction, sum token-sharded HVPs, return this rank's matrices.

    Each GPU has a copy of the model but different validation tokens. For its
    tokens it computes H_rank v for the entire direction v, not just its owned
    matrices. Summing these contributions preserves cross-matrix curvature:
    H v = sum_rank (H_rank v). Then each GPU keeps only the matrices it owns.
    LossHessian already divides by the GLOBAL token count, so the reduction
    sums rather than averages. No polar map or weight decay is applied.

    Args:
        vector_config: Assign named tensors to ranks and select vector storage.
            Run this call on every rank in the default process group.
        hessian: This rank's token-shard loss Hessian, normalized by the global
            token count and covering all matrices in vector_config.
        direction_shard: This rank's direction [vector_config.local_vector_size];
            padding is ignored.

    Returns:
        This rank's coupled, token-mean HVP shard [vector_config.local_vector_size],
        with zero padding and vector_config's device and dtype.
    """
    directions_by_parameter = gather_direction(vector_config, direction_shard)
    hessian_vector_products = hessian(tuple(
        directions_by_parameter[name] for name in hessian.parameter_names
    ))
    return emit_shard_of_all_reduced_matrix_vector_product(
        vector_config,
        dict(zip(hessian.parameter_names, hessian_vector_products, strict=True)),
    )


def gather_direction(
    vector_config: ShardedVectorConfig,
    direction_shard: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Gather direction shards and restore each parameter's shape."""
    all_direction_shards = torch.empty(
        vector_config.world_size * vector_config.local_vector_size,
        device=vector_config.device, dtype=vector_config.dtype,
    )
    if vector_config.world_size > 1:
        dist.all_gather_into_tensor(all_direction_shards, direction_shard.contiguous())
    else:
        all_direction_shards.copy_(direction_shard)
    return unpack_vector(
        all_direction_shards, all_ranks=True, config=vector_config,
    )


def emit_shard_of_all_reduced_matrix_vector_product(
    vector_config: ShardedVectorConfig,
    products_by_parameter: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Sum HVP contributions across ranks and return this rank's shard."""
    all_product_shards = pack_tensor(
        products_by_parameter, all_ranks=True, config=vector_config,
    )
    local_product = torch.empty(
        vector_config.local_vector_size,
        device=vector_config.device, dtype=vector_config.dtype,
    )
    if vector_config.world_size > 1 and dist.get_backend() == "nccl":
        dist.reduce_scatter_tensor(local_product, all_product_shards)
    elif vector_config.world_size > 1:
        dist.all_reduce(all_product_shards)
        local_product.copy_(all_product_shards.view(
            vector_config.world_size, vector_config.local_vector_size,
        )[vector_config.rank])
    else:
        local_product.copy_(all_product_shards)
    return local_product
