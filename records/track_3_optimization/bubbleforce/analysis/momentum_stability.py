"""Frozen-mode stability for x[t+1]=(1-d)x[t]-kappa*sum(c[k]*x[t-k])."""

import numpy as np
from numpy.polynomial import chebyshev


def _compute_crossing_polynomial(kernel_coefficients: np.ndarray, decay_step: float) -> np.ndarray:
    """Build P(u) by combining U_n terms, then converting them to T_n terms.

    Collect the coefficient of each U_n as (1-d)*c[n+1] - c[n], using
    c[m]=0 beyond the kernel. Expand each U_n into T_n, T_{n-2}, ...,
    adding twice its coefficient at each degree. For even n, subtract
    its coefficient once at degree zero to obtain the T_0 coefficient:
    U_4 = 2*T_4 + 2*T_2 + T_0, for example.
    """
    u_coefficients = -kernel_coefficients
    # u_coefficients = [(1-d)c_k U_k + ]
    u_coefficients[:-1] += (1 - decay_step) * kernel_coefficients[1:]

    t_coefficients = np.zeros(len(kernel_coefficients))
    for u_degree, u_coefficient in enumerate(u_coefficients):
        # Add 2*b at T_n, T_{n-2}, ... for the term b*U_n.
        t_coefficients[u_degree::-2] += 2 * u_coefficient
        if u_degree % 2 == 0:
            # Reduce the constant contribution from 2*b to b because U_0=T_0.
            t_coefficients[0] -= u_coefficient
    return t_coefficients


def _find_crossing_angles(kernel_coefficients: np.ndarray, decay_step: float) -> np.ndarray:
    """Solve P(u)=0 for interior cosines, then include angles zero and pi."""
    t_coefficients = _compute_crossing_polynomial(kernel_coefficients, decay_step)
    cosine_roots = chebyshev.chebroots(np.trim_zeros(t_coefficients, "b"))
    # keep roots with negligible imaginary parts and roots strictly between -1 and 1
    interior_real_root_mask = (np.abs(cosine_roots.imag) < 1e-8) & (np.abs(cosine_roots.real) < 1)
    interior_cosines = cosine_roots.real[interior_real_root_mask]
    return np.concatenate(([0., np.pi], np.arccos(interior_cosines)))


def compute_bubbleforce(kernel: np.ndarray, *, decay_step: float) -> float:
    """Find the first positive curvature coupling with a unit-circle root.

    Substitute z=exp(i*theta) into the recurrence and solve for
    kappa=(1-d-z)/K(z), where K(z)=sum(c[k]*z**(-k)). To find angles
    giving real kappa, multiply the numerator and denominator by conj(K).
    Expand the imaginary part of (1-d-z)*conj(K) as sin(theta)*P(cos(theta)):

        P(u) = sum(c[k] * ((1-d)*U_{k-1}(u) - U_k(u))), with U_{-1}=0.

    For interior angles, divide by sin(theta) and use the second-kind
    Chebyshev identity U_n(cos(theta))=sin((n+1)*theta)/sin(theta).
    Convert P to first-kind Chebyshev coefficients before calling NumPy's
    chebroots, which solves sum(a[n]*T_n(u))=0. Apply
    U_n=2*T_n+2*T_{n-2}+... with coefficient 1 for T_0 when n is even.

    Keep real roots in (-1, 1), recover theta=arccos(u), and include
    theta=0 and pi separately because sin(theta)=0 at these endpoints.
    At each candidate with nonzero K, evaluate (1-d-z)/K to satisfy the
    remaining real equation. Return the smallest positive real coupling.

    Args:
        kernel: One-dimensional real kernel coefficients, current gradient first.
            Use the actual available-history coefficients at the checkpoint.
        decay_step: Learning rate times decoupled weight decay, d=eta*lambda_wd.

    Returns:
        First positive kappa at which a characteristic root touches the unit
        circle. The predicted positive curvature is kappa / learning_rate.

    Raises:
        ValueError: There is no small-positive stable branch under the supported
            assumptions, or no numerically resolved positive crossing.
    """
    kernel_coefficients = np.asarray(kernel, dtype=np.float64)
    if (kernel_coefficients.ndim != 1 or not len(kernel_coefficients)
            or not np.isfinite(kernel_coefficients).all() or kernel_coefficients.sum() <= 0
            or not 0 <= decay_step < 2):
        raise ValueError("require a finite positive-mass kernel and 0 <= eta*weight_decay < 2")

    crossing_angles = _find_crossing_angles(kernel_coefficients, decay_step)
    unit_circle_points = np.exp(1j * crossing_angles)
    kernel_response = np.polynomial.polynomial.polyval(1 / unit_circle_points, kernel_coefficients)
    nonzero_response_mask = (
        np.abs(kernel_response) > np.finfo(float).eps * np.abs(kernel_coefficients).sum()
    )
    candidate_couplings = (
        (1 - decay_step - unit_circle_points[nonzero_response_mask])
        / kernel_response[nonzero_response_mask]
    )
    positive_real_couplings = candidate_couplings.real[
        (candidate_couplings.real > 0)
        & (np.abs(candidate_couplings.imag) <= 1e-7 * np.maximum(1., np.abs(candidate_couplings.real)))
    ]
    if not len(positive_real_couplings):
        raise ValueError("no positive unit-circle crossing resolved for this kernel")
    return float(positive_real_couplings.min())
