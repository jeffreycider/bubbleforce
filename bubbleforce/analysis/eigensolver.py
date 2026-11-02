"""Estimate eigenpairs by extending a Krylov basis and comparing successive estimates.

Use n_params for the total unpadded parameter count across ranks and
n_local_params for the padded parameter-shard length on each rank.
Represent the global operator A with shape (n_params, n_params); apply it
to local shards with shape (n_local_params,). Use m for the projected
Krylov dimension, k for the selected eigenpair count, p for the requested
eigenpair count.
"""

from collections.abc import Callable
from enum import Enum
from typing import NotRequired, TypedDict

import numpy as np
import torch

from configs.measurement_config import EigenSolveConfig, ShardedVectorConfig
from bubbleforce.analysis.krylov import (
    extend_krylov_basis,
    initialize_krylov_basis,
    solve_hessenberg_eigenpairs,
)
from bubbleforce.analysis.sharded_vector import compute_global_norm


class StoppingReason(Enum):
    """Identify the condition used to end Krylov basis extension."""

    SPAN_EXHAUSTED = "span_exhausted"
    AGREEMENT = "agreement"
    BUDGET = "budget"


class EigenpairReadout(TypedDict):
    num_krylov_basis_vectors: int
    eigenvalues: list[float]
    seed: int
    stopping_reason: StoppingReason | None
    residuals: NotRequired[list[float]]


def _are_hessenberg_eigenspaces_in_agreement(
    previous_hessenberg_eigenvectors: np.ndarray | None,
    current_hessenberg_eigenvectors: np.ndarray,
    eigensolve_config: EigenSolveConfig,
) -> bool:
    """Check agreement between successive Hessenberg eigenspaces.

    Pad the previous eigenvectors to the current dimension, construct
    orthonormal bases with SVD, then compare their largest principal angle.

    Args:
        previous_hessenberg_eigenvectors: Read previous coefficient columns,
            shape (m_previous, k), or None.
        current_hessenberg_eigenvectors: Read current coefficient columns,
            shape (m_current, k).
        eigensolve_config: Use the requested eigenpair count and the
            tolerance for the sine of the largest principal angle.

    Returns:
        True if the angle check passes. Return False for missing,
        incomplete, or rank-deficient spans. For one requested eigenpair,
        accept a single current column without comparing spans.
    """
    num_eigenvectors = eigensolve_config.num_requested_eigenpairs
    if current_hessenberg_eigenvectors.shape[1] != num_eigenvectors or previous_hessenberg_eigenvectors.shape[1] != num_eigenvectors:
        return False
    if previous_hessenberg_eigenvectors is None:
        return False
    if num_eigenvectors == 1:
        return True

    num_added_rows = (
        current_hessenberg_eigenvectors.shape[0]
        - previous_hessenberg_eigenvectors.shape[0]
    )
    if num_added_rows < 0:
        return False

    previous_hessenberg_eigenvectors = np.pad(
        previous_hessenberg_eigenvectors, ((0, num_added_rows), (0, 0)),
    )

    orthonormal_bases = []
    for eigenvectors in (
        previous_hessenberg_eigenvectors,
        current_hessenberg_eigenvectors,
    ):
        basis, singular_values, _ = np.linalg.svd(eigenvectors, full_matrices=False)
        if singular_values.size == 0:
            return False

        rank_tolerance = (
            np.finfo(np.float64).eps * max(eigenvectors.shape) * singular_values[0]
        )
        if np.count_nonzero(singular_values > rank_tolerance) != num_eigenvectors:
            return False
        orthonormal_bases.append(basis)

    previous_basis, current_basis = orthonormal_bases
    principal_angle_cosines = np.linalg.svd(
        previous_basis.T @ current_basis, compute_uv=False,
    )
    smallest_cosine = np.clip(principal_angle_cosines.min(), 0.0, 1.0)
    largest_angle_sine = np.sqrt(1.0 - smallest_cosine**2)
    return bool(largest_angle_sine <= eigensolve_config.subspace_tolerance)


def _are_hessenberg_eigenvalues_in_agreement(
    hessenberg_eigenvalue_history: list[np.ndarray],
    eigensolve_config: EigenSolveConfig,
) -> bool:
    """Check relative Hessenberg eigenvalue ranges over the latest window.

    Compare eigenvalues by sorted position across checkpoints. Divide
    each range by its largest absolute eigenvalue; use zero for an
    all-zero column.

    Args:
        hessenberg_eigenvalue_history: Read eigenvalues from successive
            checkpoints, ordered from largest to smallest.
        eigensolve_config: Use the requested eigenpair count, comparison
            window, and relative eigenvalue tolerance.

    Returns:
        True if every relative range is within tolerance. Return False
        if the window is incomplete or any checkpoint has a different
        number of eigenvalues from the requested count.
    """
    recent_eigenvalues = hessenberg_eigenvalue_history[-eigensolve_config.comparison_window:]
    if len(recent_eigenvalues) != eigensolve_config.comparison_window:
        return False
    if any(
        len(eigenvalues) != eigensolve_config.num_requested_eigenpairs
        for eigenvalues in recent_eigenvalues
    ):
        return False

    eigenvalues = np.array(recent_eigenvalues)
    max_absolute_eigenvalues = np.max(np.abs(eigenvalues), axis=0)
    relative_ranges = np.divide(
        np.ptp(eigenvalues, axis=0),
        max_absolute_eigenvalues,
        out=np.zeros_like(max_absolute_eigenvalues),
        where=max_absolute_eigenvalues != 0,
    )
    return bool(np.max(relative_ranges) <= eigensolve_config.eigenvalue_rtol)


def _evaluate_eigensolver_stopping_conditions(
    hessenberg_eigenvalue_history: list[np.ndarray],
    previous_hessenberg_eigenvectors: np.ndarray | None,
    current_hessenberg_eigenvectors: np.ndarray,
    eigensolve_config: EigenSolveConfig,
    *,
    num_krylov_basis_vectors: int,
    krylov_basis_exhausted: bool,
    final_configured_basis_size_checked: bool,
) -> EigenpairReadout:
    """Compare current eigenpair estimates and build the checkpoint readout.

    Args:
        hessenberg_eigenvalue_history: Nonempty history of arrays with shape (k_t,).
        previous_hessenberg_eigenvectors: Previous coefficient array, shape
            (m_old, k_old), or None.
        current_hessenberg_eigenvectors: Current coefficient array, shape (m, k).
        eigensolve_config: Requested count p, window length, comparison tolerances.
        num_krylov_basis_vectors: Record the current projected matrix dimension.
        krylov_basis_exhausted: Whether basis extension stopped at numerical
            breakdown or at the full global vector dimension.
        final_configured_basis_size_checked: Whether the last configured
            basis size was processed.

    Returns:
        Build a readout from the selected eigenvalues, seed, and stopping
        reason. Check span exhaustion first, then
        eigenvalue and eigenspace agreement, then the configured budget.
    """
    stopping_reason = None
    if krylov_basis_exhausted:
        stopping_reason = StoppingReason.SPAN_EXHAUSTED
    elif (
        _are_hessenberg_eigenvalues_in_agreement(
            hessenberg_eigenvalue_history, eigensolve_config,
        )
        and _are_hessenberg_eigenspaces_in_agreement(
            previous_hessenberg_eigenvectors,
            current_hessenberg_eigenvectors,
            eigensolve_config,
        )
    ):
        stopping_reason = StoppingReason.AGREEMENT
    elif final_configured_basis_size_checked:
        stopping_reason = StoppingReason.BUDGET

    readout: EigenpairReadout = {
        "num_krylov_basis_vectors": num_krylov_basis_vectors,
        "eigenvalues": hessenberg_eigenvalue_history[-1].tolist(),
        "seed": eigensolve_config.start_seed,
        "stopping_reason": stopping_reason,
    }
    return readout


def compute_relative_eigenpair_residuals(
    matrix_vector_product: Callable[[torch.Tensor], torch.Tensor],
    sharded_vector_config: ShardedVectorConfig,
    estimated_eigenvalues: np.ndarray,
    estimated_eigenvectors: torch.Tensor,
) -> list[float]:
    """Measure the relative error in each approximate eigenpair.

    Compute norm(A @ v_j - lambda_j * v_j) / (norm(A @ v_j) + abs(lambda_j)),
    using the Euclidean norm across ranks.

    Args:
        matrix_vector_product: Apply the global linear operator A to
            a local vector shard, mapping shape (n_local_params,)
            to the same shape.
        sharded_vector_config: Supplies shard layout and global norm reduction.
        estimated_eigenvalues: Real eigenvalues lambda_j, shape (k,).
        estimated_eigenvectors: Local shards of globally unit-norm vectors v_j,
            shape (k, n_local_params), with eigenvectors in rows.
            The local size includes padding from sharded_vector_config.

    Returns:
        A list of k relative residuals, computed using fresh products
        with A and norms across ranks. Return zero for a zero denominator.
        Returns a normalized number between 0 and 1.

    Raises:
        ValueError: The eigenvalue count differs from the number of
            eigenvector rows.
    """
    relative_eigenpair_residuals = []
    for estimated_eigenvalue, estimated_eigenvector in zip(
        estimated_eigenvalues, estimated_eigenvectors, strict=True,
    ):
        estimated_eigenvalue = float(estimated_eigenvalue)
        matrix_eigenvector_product = matrix_vector_product(estimated_eigenvector)
        residual_normalization_scale = (
            compute_global_norm(matrix_eigenvector_product, config=sharded_vector_config)
            + abs(estimated_eigenvalue)
        )
        if residual_normalization_scale == 0:
            relative_eigenpair_residuals.append(0.0)
            continue

        eigenpair_residual = (
            matrix_eigenvector_product - estimated_eigenvalue * estimated_eigenvector
        )
        relative_eigenpair_residuals.append(
            compute_global_norm(eigenpair_residual, config=sharded_vector_config)
            / residual_normalization_scale
        )
    return relative_eigenpair_residuals


def reconstruct_eigenvectors(
    sharded_vector_config: ShardedVectorConfig,
    krylov_basis_vectors: list[torch.Tensor],
    hessenberg_eigenvectors: np.ndarray,
) -> torch.Tensor:
    """Reconstruct eigenvector shards from the selected Hessenberg eigenvectors.

    Form Q = [q_1, ..., q_m] from the first m basis vectors. For each
    coefficient column y_j, reconstruct v_j = (Q @ y_j) / norm(Q @ y_j).
    Compute the normalization across ranks.

    Args:
        sharded_vector_config: Supplies padded local size, dtype, device,
            global norms.
        krylov_basis_vectors: At least m local basis shards, each shape
            (n_local_params,), from the growing orthonormal basis.
        hessenberg_eigenvectors: Selected projected eigenvectors, shape (m, k).

    Returns:
        Stack the local eigenvector shards into shape (k, n_local_params),
        using the Krylov basis dtype and device.
        Return shape (0, n_local_params) if no eigenvectors were selected.
    """
    if hessenberg_eigenvectors.shape[1] == 0:
        return torch.empty(
            (0, sharded_vector_config.local_vector_size),
            device=sharded_vector_config.device,
            dtype=sharded_vector_config.dtype,
        )
    krylov_basis_matrix = torch.stack(krylov_basis_vectors[:hessenberg_eigenvectors.shape[0]])
    coefficients = torch.as_tensor(
        hessenberg_eigenvectors,
        dtype=krylov_basis_matrix.dtype,
        device=krylov_basis_matrix.device,
    )
    eigenvectors = []
    for coefficient_column in coefficients.T:
        eigenvector = torch.einsum("m,mn->n", coefficient_column, krylov_basis_matrix)
        eigenvectors.append(
            eigenvector / compute_global_norm(eigenvector, config=sharded_vector_config)
        )
    return torch.stack(eigenvectors)


def solve_eigenpairs( # check other PRs
    matrix_vector_product: Callable[[torch.Tensor], torch.Tensor],
    sharded_vector_config: ShardedVectorConfig,
    eigensolve_config: EigenSolveConfig,
    *,
    initial_basis_vector: torch.Tensor | None = None,
) -> tuple[list[EigenpairReadout], torch.Tensor]:
    """Estimate the largest positive real eigenpairs of a linear operator.

    Extend an orthonormal basis for
    K_m(A, q_1) = span(q_1, A @ q_1, ..., matrix_power(A, m - 1) @ q_1).
    At each sorted, distinct checkpoint, solve the small projected problem
    H_m @ y = lambda * y, with H_m approximately equal to Q_m.T @ A @ Q_m.
    Select up to p independent positive real modes in descending
    eigenvalue order.

    Args:
        matrix_vector_product: Apply a real global operator
            A of shape (n_params, n_params), mapping local shards of shape
            (n_local_params,) to the same shape. Here n_params is the
            total unpadded parameter count across ranks.
        sharded_vector_config: Supplies padded local size, dtype, device, global dots.
        eigensolve_config: Supplies basis checkpoints, requested eigenvalue count 'p',
            tolerances for stopping, and whether to use Lanczos iterations.
        initial_basis_vector: Optional local starting shard, shape
            (n_local_params,). Normalize globally; otherwise draw
            a random start using eigensolve_config.start_seed.

    Returns:
        Return the checkpoint readouts in order and the final eigenvector
        shards, shape (k, n_local_params). Add measured residuals to the
        last readout after reconstructing its eigenvectors.
    """
    krylov_basis_vector_counts = sorted(set(eigensolve_config.krylov_basis_vector_counts))

    krylov_basis_vectors, extended_hessenberg_matrix = initialize_krylov_basis(
        sharded_vector_config, eigensolve_config.start_seed, initial_basis_vector=initial_basis_vector,
    )

    krylov_basis_exhausted = False
    previous_hessenberg_eigenvectors = None
    selected_eigenvalue_history: list[np.ndarray] = []
    eigenpair_readouts: list[EigenpairReadout] = []

    for target_krylov_basis_size in krylov_basis_vector_counts:
        (
            krylov_basis_vectors,
            extended_hessenberg_matrix,
            krylov_basis_exhausted,
        ) = extend_krylov_basis(
            matrix_vector_product,
            sharded_vector_config,
            krylov_basis_vectors,
            extended_hessenberg_matrix,
            target_krylov_basis_size,
            breakdown_rtol=eigensolve_config.breakdown_rtol,
            krylov_basis_exhausted=krylov_basis_exhausted,
        )

        (
            selected_eigenvalues,
            selected_hessenberg_eigenvectors,
            _,
        ) = solve_hessenberg_eigenpairs(
            extended_hessenberg_matrix,
            eigensolve_config.num_requested_eigenpairs,
            symmetric=eigensolve_config.symmetric,
        )
        selected_eigenvalue_history.append(selected_eigenvalues)

        eigenpair_readout = _evaluate_eigensolver_stopping_conditions(
            selected_eigenvalue_history,
            previous_hessenberg_eigenvectors,
            selected_hessenberg_eigenvectors,
            eigensolve_config,
            num_krylov_basis_vectors=extended_hessenberg_matrix.shape[1],
            krylov_basis_exhausted=krylov_basis_exhausted,
            final_configured_basis_size_checked=(
                target_krylov_basis_size == krylov_basis_vector_counts[-1]
            ),
        )
        eigenpair_readouts.append(eigenpair_readout)
        previous_hessenberg_eigenvectors = selected_hessenberg_eigenvectors
        if eigenpair_readout["stopping_reason"] is not None:
            break

    estimated_eigenvectors = reconstruct_eigenvectors(
        sharded_vector_config, krylov_basis_vectors, selected_hessenberg_eigenvectors,
    )
    eigenpair_readouts[-1]["residuals"] = compute_relative_eigenpair_residuals(
        matrix_vector_product, sharded_vector_config, selected_eigenvalues, estimated_eigenvectors,
    )
    return eigenpair_readouts, estimated_eigenvectors
