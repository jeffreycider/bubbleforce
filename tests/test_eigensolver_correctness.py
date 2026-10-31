"""Compare Krylov calculations with explicit matrices and known eigensystems."""

import numpy as np
import pytest
import torch

from configs.measurement_config import EigenSolveConfig, ShardedVectorConfig
from bubbleforce.analysis.eigensolver import (
    StoppingReason,
    _evaluate_eigensolver_stopping_conditions,
    solve_eigenpairs,
)
from bubbleforce.analysis.krylov import (
    extend_krylov_basis,
    initialize_krylov_basis,
    solve_hessenberg_eigenpairs,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("scale", [1e-15, 1., 1e15])
def test_projection_matches_explicit_matrix_products(dtype, symmetric, scale):
    generator = torch.Generator().manual_seed(41)
    rotation = torch.linalg.qr(torch.randn(12, 12, dtype=dtype, generator=generator)).Q
    triangular = torch.diag(torch.linspace(-4., 9., 12, dtype=dtype))
    if not symmetric:
        triangular += torch.diag(torch.full((11,), .75, dtype=dtype), diagonal=1)
    matrix = scale * (rotation @ triangular @ rotation.T)
    vector_config = ShardedVectorConfig(
        [[("weight", (3, 4))]], dtype=dtype,
    )
    basis_vectors, projected_matrix = initialize_krylov_basis(vector_config, start_seed=17)
    basis_extension_complete = False
    tolerance = 3e-5 if dtype == torch.float32 else 1e-11

    for num_krylov_basis_vectors in (3, 7, 12):
        basis_vectors, projected_matrix, basis_extension_complete = extend_krylov_basis(
            lambda vector: matrix @ vector,
            vector_config,
            basis_vectors,
            projected_matrix,
            num_krylov_basis_vectors,
            breakdown_rtol=1e-12,
            krylov_basis_exhausted=basis_extension_complete,
        )
        basis = torch.stack(basis_vectors).T
        active_basis = basis[:, :num_krylov_basis_vectors]
        projection = torch.as_tensor(projected_matrix, dtype=dtype)
        assert projection.shape == (num_krylov_basis_vectors + 1, num_krylov_basis_vectors)
        assert basis_extension_complete == (num_krylov_basis_vectors == 12)
        assert basis.shape[1] == min(num_krylov_basis_vectors + 1, 12)
        torch.testing.assert_close(
            basis.T @ basis, torch.eye(basis.shape[1], dtype=dtype),
            atol=tolerance, rtol=tolerance,
        )
        torch.testing.assert_close(
            active_basis.T @ matrix @ active_basis / scale,
            projection[:num_krylov_basis_vectors] / scale,
            atol=tolerance, rtol=tolerance,
        )
        # Include the extra basis direction when reconstructing partial products.
        torch.testing.assert_close(
            matrix @ active_basis / scale,
            basis @ projection[:basis.shape[1]] / scale,
            atol=tolerance, rtol=tolerance,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("scale", [1e-15, 1., 1e15])
def test_eigenpairs_match_known_spectrum_and_direct_residuals(dtype, symmetric, scale):
    generator = torch.Generator().manual_seed(73)
    rotation = torch.linalg.qr(torch.randn(9, 9, dtype=dtype, generator=generator)).Q
    known_eigenvalues = torch.tensor([-6., -3., -1., 1., 2., 4., 6., 8., 10.], dtype=dtype)
    triangular = torch.diag(known_eigenvalues)
    if not symmetric:
        triangular += torch.diag(torch.full((8,), .8, dtype=dtype), diagonal=1)
    matrix = scale * (rotation @ triangular @ rotation.T)
    vector_config = ShardedVectorConfig(
        [[("weight", (3, 3))]], dtype=dtype,
    )
    result, eigenvectors = solve_eigenpairs(
        lambda vector: matrix @ vector,
        vector_config,
        EigenSolveConfig(
            num_requested_eigenpairs=3,
            krylov_basis_vector_counts=(3, 6, 9),
            comparison_window=4,
            symmetric=symmetric,
        ),
    )
    final = result[-1]
    eigenvalues = torch.tensor(final["eigenvalues"], dtype=dtype)
    tolerance = 3e-5 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(
        eigenvalues / scale, known_eigenvalues.flip(0)[:3], atol=tolerance, rtol=tolerance,
    )
    torch.testing.assert_close(
        eigenvectors.norm(dim=1), torch.ones(3, dtype=dtype), atol=tolerance, rtol=tolerance,
    )
    direct_residuals = []
    for eigenvalue, eigenvector in zip(final["eigenvalues"], eigenvectors, strict=True):
        product = matrix @ eigenvector
        residual = product - eigenvalue * eigenvector
        direct_residuals.append(
            (residual.double().norm() / (product.double().norm() + abs(eigenvalue))).item()
        )
        torch.testing.assert_close(
            product / scale, (eigenvalue / scale) * eigenvector,
            atol=tolerance, rtol=tolerance,
        )
    np.testing.assert_allclose(final["residuals"], direct_residuals, rtol=1e-12, atol=1e-16)
    assert max(final["residuals"]) < tolerance
    assert final["num_krylov_basis_vectors"] == 9
    assert final["stopping_reason"] == StoppingReason.SPAN_EXHAUSTED


@pytest.mark.parametrize("seed", [7, 31, 89])
def test_dense_mixed_spectrum_selects_available_positive_real_modes(seed):
    generator = torch.Generator().manual_seed(seed)
    rotation = torch.linalg.qr(torch.randn(6, 6, dtype=torch.float64, generator=generator)).Q
    triangular = torch.diag(torch.tensor([9., 5., 2., -20., 12., 12.], dtype=torch.float64))
    triangular[0, 1] = 1.1
    triangular[4, 5], triangular[5, 4] = -3., 3.
    matrix = rotation @ triangular @ rotation.T
    vector_config = ShardedVectorConfig(
        [[("weight", (2, 3))]], dtype=torch.float64,
    )
    result, eigenvectors = solve_eigenpairs(
        lambda vector: matrix @ vector,
        vector_config,
        EigenSolveConfig(num_requested_eigenpairs=4, krylov_basis_vector_counts=(6,)),
    )
    final = result[-1]
    np.testing.assert_allclose(final["eigenvalues"], [9., 5., 2.], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        eigenvectors @ matrix.T,
        torch.tensor([9., 5., 2.], dtype=torch.float64)[:, None] * eigenvectors,
        atol=1e-12, rtol=1e-12,
    )
    assert final["stopping_reason"] == StoppingReason.SPAN_EXHAUSTED


@pytest.mark.parametrize("symmetric", [False, True])
def test_projected_eigensolve_uses_the_square_block(symmetric):
    square_projection = np.array([[4., 2.], [0., 2.]])
    projected_matrix = np.vstack([square_projection, [1e100, -1e100]])
    eigenvalues, coefficients, spectrum = solve_hessenberg_eigenpairs(
        projected_matrix, num_requested_eigenpairs=2, symmetric=symmetric,
    )
    if symmetric:
        square_projection = (square_projection + square_projection.T) / 2
        expected = [3. + np.sqrt(2.), 3. - np.sqrt(2.)]
    else:
        expected = [4., 2.]
    np.testing.assert_allclose(eigenvalues, expected, atol=1e-12)
    np.testing.assert_allclose(np.sort(spectrum), sorted(expected), atol=1e-12)
    np.testing.assert_allclose(
        square_projection @ coefficients, coefficients * eigenvalues, atol=1e-12,
    )


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("scalar", [0., 2.])
def test_exhausted_projection_retains_its_result_without_more_products(symmetric, scalar):
    vector_config = ShardedVectorConfig(
        [[("weight", (1, 6))]], dtype=torch.float64,
    )
    calls = []

    def matrix_vector_product(vector):
        calls.append(vector.clone())
        return scalar * vector

    basis_vectors, projected_matrix = initialize_krylov_basis(vector_config, start_seed=17)
    basis_extension_complete = False
    for _ in range(2):
        basis_vectors, projected_matrix, basis_extension_complete = extend_krylov_basis(
            matrix_vector_product,
            vector_config,
            basis_vectors,
            projected_matrix,
            6,
            breakdown_rtol=1e-12,
            krylov_basis_exhausted=basis_extension_complete,
        )
        assert basis_extension_complete
        assert len(calls) == len(basis_vectors) == 1
        np.testing.assert_allclose(projected_matrix, [[scalar], [0.]], atol=1e-14)
        values, coefficients, spectrum = solve_hessenberg_eigenpairs(
            projected_matrix, num_requested_eigenpairs=2, symmetric=symmetric,
        )
        np.testing.assert_allclose(values, [scalar] if scalar > 0 else [], atol=1e-14)
        assert coefficients.shape == (1, len(values))
        np.testing.assert_allclose(spectrum, [scalar], atol=1e-14)


@pytest.mark.parametrize("changed_values, rotated_span, exhausted, budget, expected", [
    (False, False, False, False, StoppingReason.AGREEMENT),
    (False, False, False, True, StoppingReason.AGREEMENT),
    (True, False, False, False, None),
    (False, True, False, False, None),
    (True, False, False, True, StoppingReason.BUDGET),
    (False, True, False, True, StoppingReason.BUDGET),
    (True, True, True, True, StoppingReason.SPAN_EXHAUSTED),
])
def test_stopping_decisions_use_eigenvalue_and_span_agreement(
    changed_values, rotated_span, exhausted, budget, expected,
):
    history = [np.array([10., 5.]), np.array([8. if changed_values else 10., 5.])]
    coefficients = np.eye(3)[:, [0, 2] if rotated_span else [0, 1]]
    readout = _evaluate_eigensolver_stopping_conditions(
        history, np.eye(2), coefficients,
        EigenSolveConfig(num_requested_eigenpairs=2),
        num_krylov_basis_vectors=3,
        krylov_basis_exhausted=exhausted, final_configured_basis_size_checked=budget,
    )
    assert readout["stopping_reason"] == expected


@pytest.mark.parametrize("value", [0., float("nan"), float("inf")])
def test_invalid_start_is_rejected_before_matrix_products(value):
    vector_config = ShardedVectorConfig(
        [[("weight", (1, 4))]], dtype=torch.float64,
    )
    calls = []

    def matrix_vector_product(vector):
        calls.append(vector)
        return vector

    with pytest.raises(ValueError, match="finite, positive norm"):
        solve_eigenpairs(
            matrix_vector_product, vector_config,
            EigenSolveConfig(num_requested_eigenpairs=1),
            initial_basis_vector=torch.full((4,), value, dtype=torch.float64),
        )
    assert calls == []
