"""Build a Krylov basis from matrix-vector products, then solve its projection"""

from collections.abc import Callable

import numpy as np
import torch

from configs.measurement_config import ShardedVectorConfig
from bubbleforce.analysis.sharded_vector import (
    ShardedKrylovBasisVector, ShardedVector,
    compute_global_dot, compute_global_norm, get_random_sharded_unit_vector,
)


def _filter_and_sort_positive_real_hessenberg_eigenpairs(
    hessenberg_eigenvalues: np.ndarray,
    hessenberg_eigenvectors: np.ndarray,
    imaginary_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep each Hessenberg eigenvalue if its real part is positive and its
    imaginary part has magnitude at most imaginary_tolerance. Sort so that largest is first.

    Args:
        hessenberg_eigenvalues: Read the computed eigenvalues, shape (s,).
        hessenberg_eigenvectors: Read the corresponding columns, shape (m, s).
        imaginary_tolerance: Use this absolute threshold when checking
            whether a Hessenberg eigenvalue is approximately real.

    Returns:
        A tuple containing the retained real Hessenberg eigenvalues,
        shape (c,), and corresponding eigenvectors, shape (m, c).
        Preserve complex components in the eigenvectors for conversion.
    """
    candidate_indices = np.flatnonzero(
        (hessenberg_eigenvalues.real > 0)
        & (np.abs(hessenberg_eigenvalues.imag) <= imaginary_tolerance)
    )
    order = np.argsort(
        -hessenberg_eigenvalues[candidate_indices].real, kind="stable",
    )
    candidate_indices = candidate_indices[order]
    return (
        hessenberg_eigenvalues[candidate_indices].real,
        hessenberg_eigenvectors[:, candidate_indices],
    )


def _select_real_hessenberg_eigenvectors_by_rank(
    candidate_hessenberg_eigenvectors: np.ndarray,
    num_requested_eigenpairs: int,
) -> tuple[list[int], np.ndarray]:
    """Convert Hessenberg eigenvectors to real unit vectors,
    see if it increases the rank of the eigenvector matrix. Filter out if not.

    Args:
        candidate_hessenberg_eigenvectors: Read candidate columns, shape
            (m, c), in selection order.
        num_requested_eigenpairs: Select up to this nonnegative number
            of independent Hessenberg eigenvectors.

    Returns:
        A tuple containing the indices of the selected eigenvectors and the real
        unit Hessenberg eigenvectors, shape (m, k). Stop at the requested
        count or after evaluating every candidate.
    """
    selected_indices: list[int] = []
    selected_hessenberg_eigenvectors = np.empty((candidate_hessenberg_eigenvectors.shape[0], 0))

    for index, hessenberg_eigenvector in enumerate(candidate_hessenberg_eigenvectors.T):
        if len(selected_indices) == num_requested_eigenpairs:
            break

        largest_component = np.argmax(np.abs(hessenberg_eigenvector))
        phase = np.angle(hessenberg_eigenvector[largest_component])
        real_hessenberg_eigenvector = (hessenberg_eigenvector * np.exp(-1j * phase)).real
        norm = np.linalg.norm(real_hessenberg_eigenvector)
        if norm == 0:
            continue
        real_hessenberg_eigenvector /= norm

        trial_hessenberg_eigenvectors = np.column_stack(
            (selected_hessenberg_eigenvectors, real_hessenberg_eigenvector),
        )
        if np.linalg.matrix_rank(trial_hessenberg_eigenvectors) > len(selected_indices):
            selected_indices.append(index)
            selected_hessenberg_eigenvectors = trial_hessenberg_eigenvectors

    return selected_indices, selected_hessenberg_eigenvectors


def initialize_krylov_basis(
    sharded_vector_config: "ShardedVectorConfig",
    start_seed: int,
    *,
    initial_basis_vector: ShardedVector | None = None,
) -> tuple[list[ShardedKrylovBasisVector], np.ndarray]:
    """Normalize the starting vector and initialize the empty Hessenberg matrix.

    Args:
        sharded_vector_config: Supplies padded local size n_local_params,
            dtype, device, and global norm reduction.
        start_seed: Random seed used when initial_basis_vector is absent.
        initial_basis_vector: Optional local shard, shape (n_local_params,).
            Clone it before normalization.

    Returns:
        Return a list containing the local shard of q_1 = v / norm(v)
        and an empty float64 Hessenberg matrix, shape (1, 0).
        Compute the norm across ranks; reject a zero or nonfinite norm.

    Raises:
        ValueError: The starting vector has a zero or nonfinite global norm.
    """
    initial_krylov_basis_vector = (
        get_random_sharded_unit_vector(start_seed, config=sharded_vector_config)
        if initial_basis_vector is None
        else initial_basis_vector.clone()
    )
    initial_basis_vector_norm = compute_global_norm(initial_krylov_basis_vector, config=sharded_vector_config)
    if not np.isfinite(initial_basis_vector_norm) or initial_basis_vector_norm == 0:
        raise ValueError("initial_basis_vector must have a finite, positive norm")
    first_orthonormal_basis_vector = initial_krylov_basis_vector / initial_basis_vector_norm
    extended_hessenberg_matrix = np.zeros((1, 0), dtype=np.float64)
    return [first_orthonormal_basis_vector], extended_hessenberg_matrix


def extend_krylov_basis(
    matrix_vector_product: Callable[[ShardedVector], ShardedVector],
    sharded_vector_config: ShardedVectorConfig,
    krylov_basis_vectors: list[ShardedKrylovBasisVector],
    extended_hessenberg_matrix: np.ndarray,
    target_krylov_basis_size: int,
    *,
    span_exhaustion_tol: float,
    krylov_basis_exhausted: bool = False,
) -> tuple[list[ShardedKrylovBasisVector], np.ndarray, bool]:
    """Extend the Krylov basis and Hessenberg matrix to the requested size.

    For each product A @ q_m, remove its components along the basis in
    two passes. Append the coefficients and remaining norm to the matrix.
    Append the normalized remainder to the basis unless its norm is at
    most max(span_exhaustion_tol, m * eps) * norm(A @ q_m), or m equals the
    global unpadded dimension. Here eps is the dtype's machine precision.

    Args:
        matrix_vector_product: Apply the operator to a local vector shard.
        sharded_vector_config: Use the configured dtype and global reductions.
        krylov_basis_vectors: Copy this list, then append new local basis
            vectors to the copy. Read existing tensors without changing them.
        extended_hessenberg_matrix: Extend the existing matrix, shape (m + 1, m).
        target_krylov_basis_size: Process up to this total number of columns.
        span_exhaustion_tol: Set the relative threshold for the remaining norm.
        krylov_basis_exhausted: Skip extension if an earlier call stopped.

    Returns:
        Return the extended basis list, extended Hessenberg matrix, and
        whether extension stopped.
    """
    extended_krylov_basis_vectors = list(krylov_basis_vectors)
    while (
        extended_hessenberg_matrix.shape[1] < target_krylov_basis_size
        and not krylov_basis_exhausted
    ):
        matrix_product = matrix_vector_product(extended_krylov_basis_vectors[-1])
        matrix_product_norm = compute_global_norm(
            matrix_product, config=sharded_vector_config,
        )
        matrix_product_outside_krylov_span = matrix_product.clone()

        num_basis_vectors = len(extended_krylov_basis_vectors)
        hessenberg_column = np.zeros(num_basis_vectors + 1, dtype=np.float64)

        # Reorthogonalize twice to reduce residual components from roundoff.
        for _ in range(2):
            for index, basis_vector in enumerate(extended_krylov_basis_vectors):
                coefficient = compute_global_dot(
                    basis_vector,
                    matrix_product_outside_krylov_span,
                    config=sharded_vector_config,
                )
                hessenberg_column[index] += coefficient
                matrix_product_outside_krylov_span.add_(
                    basis_vector, alpha=-coefficient,
                )

        matrix_product_outside_krylov_span_norm = compute_global_norm(
            matrix_product_outside_krylov_span,
            config=sharded_vector_config,
        )
        hessenberg_column[-1] = matrix_product_outside_krylov_span_norm
        extended_hessenberg_matrix = np.pad(
            extended_hessenberg_matrix, ((0, 1), (0, 1)),
        )
        extended_hessenberg_matrix[:, -1] = hessenberg_column

        roundoff_rtol = (
            num_basis_vectors * torch.finfo(sharded_vector_config.dtype).eps
        )
        stopping_threshold = (
            max(span_exhaustion_tol, roundoff_rtol) * matrix_product_norm
        )
        num_parameters = sum(sharded_vector_config.parameter_counts_by_rank)
        krylov_basis_exhausted = (
            matrix_product_outside_krylov_span_norm <= stopping_threshold
            or num_basis_vectors >= num_parameters
        )

        if not krylov_basis_exhausted:
            extended_krylov_basis_vectors.append(
                matrix_product_outside_krylov_span
                / matrix_product_outside_krylov_span_norm
            )

    return extended_krylov_basis_vectors, extended_hessenberg_matrix, krylov_basis_exhausted


def solve_hessenberg_eigenpairs(
    extended_hessenberg_matrix: np.ndarray,
    num_requested_eigenpairs: int,
    *,
    symmetric: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the Hessenberg eigenproblem and select positive real eigenpairs.

    Sort selected eigenvalues from largest to smallest. For nonsymmetric
    matrices, filter approximately real eigenvalues, then select real
    eigenvectors by increasing numerical rank.

    Args:
        extended_hessenberg_matrix: Extract the leading square block H,
            shape (m, m), from this augmented matrix, shape (m + 1, m).
        num_requested_eigenpairs: Select up to this nonnegative count p.
        symmetric: Solve (H + H.T) / 2 with eigh when True. Otherwise,
            solve H with eig and use eps64 * m * norm(H) as the absolute
            imaginary-part tolerance. Use float64 machine precision for
            eps64 and the Frobenius norm for norm(H).

    Returns:
        A tuple containing selected real Hessenberg eigenvalues, shape
        (k,), their real eigenvectors, shape (m, k), and the full spectrum,
        shape (m,). Select at most p eigenpairs.

    Raises:
        ValueError: num_requested_eigenpairs is negative.
    """
    if num_requested_eigenpairs < 0:
        raise ValueError("num_requested_eigenpairs must be nonnegative")

    num_basis_vectors = extended_hessenberg_matrix.shape[1]
    hessenberg_matrix = extended_hessenberg_matrix[:num_basis_vectors]
    if symmetric:
        hessenberg_eigenvalues, hessenberg_eigenvectors = np.linalg.eigh(
            (hessenberg_matrix + hessenberg_matrix.T) / 2,
        )
        selected_indices = np.flatnonzero(hessenberg_eigenvalues > 0)[::-1]
        selected_indices = selected_indices[:num_requested_eigenpairs]
        return (
            hessenberg_eigenvalues[selected_indices],
            hessenberg_eigenvectors[:, selected_indices],
            hessenberg_eigenvalues,
        )

    hessenberg_eigenvalues, hessenberg_eigenvectors = np.linalg.eig(hessenberg_matrix)
    imaginary_tolerance = (
        np.finfo(np.float64).eps * num_basis_vectors * np.linalg.norm(hessenberg_matrix)
    )
    candidate_hessenberg_eigenvalues, candidate_hessenberg_eigenvectors = (
        _filter_and_sort_positive_real_hessenberg_eigenpairs(
            hessenberg_eigenvalues, hessenberg_eigenvectors, imaginary_tolerance,
        )
    )
    selected_indices, selected_hessenberg_eigenvectors = (
        _select_real_hessenberg_eigenvectors_by_rank(
            candidate_hessenberg_eigenvectors, num_requested_eigenpairs,
        )
    )
    return (
        candidate_hessenberg_eigenvalues[selected_indices],
        selected_hessenberg_eigenvectors,
        hessenberg_eigenvalues,
    )
