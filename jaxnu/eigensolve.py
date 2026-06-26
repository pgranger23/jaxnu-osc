"""Analytic eigenvalues of a 3x3 Hermitian matrix.

Uses the stable trigonometric solution of the characteristic cubic (eigenvalues
of a Hermitian matrix are real).  Closed-form and differentiable, which makes it
preferable to a generic ``eigh`` for the small matrices appearing in 3-flavor
oscillations.  Only eigenvalues are produced; the :mod:`jaxnu.propagator`
``cayley`` backend builds ``exp(-iHL)`` from them via Cayley-Hamilton, so no
eigenvectors are required.

Reference: standard 3x3 symmetric eigenvalue algorithm (Smith 1961), extended to
Hermitian matrices by using squared off-diagonal magnitudes.
"""

from __future__ import annotations

import jax.numpy as jnp

_EPS = 1e-12


def _det3(m):
    """Determinant of a 3x3 matrix (complex-safe)."""
    return (
        m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
        - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
        + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
    )


def eigvalsh3(a):
    """Eigenvalues of a 3x3 Hermitian matrix ``a``, sorted ascending.

    Returns a real array of shape ``(3,)``.
    """
    a = jnp.asarray(a)

    # |off-diagonal|^2 sum.
    p1 = (
        jnp.abs(a[0, 1]) ** 2
        + jnp.abs(a[0, 2]) ** 2
        + jnp.abs(a[1, 2]) ** 2
    )

    diag = jnp.real(jnp.diagonal(a))
    q = jnp.sum(diag) / 3.0  # mean of diagonal == trace/3

    p2 = jnp.sum((diag - q) ** 2) + 2.0 * p1
    # Guard against the diagonal/degenerate limit (p2 -> 0).  Double-`where` so the
    # degenerate branch carries no gradient: ``sqrt`` has an infinite derivative at
    # 0, which (via 0*inf in the unselected branch) would otherwise NaN the
    # gradient through a zero / repeated-eigenvalue matrix (e.g. a zero-length
    # segment).
    big = p2 > _EPS
    p2_safe = jnp.where(big, p2, 1.0)
    p = jnp.where(big, jnp.sqrt(p2_safe / 6.0), 0.0)
    p_div = jnp.where(big, p, 1.0)  # safe denominator

    m = a - q * jnp.eye(3, dtype=a.dtype)
    detm = jnp.real(_det3(m))
    r = detm / (2.0 * p_div**3)
    # Clamp to the valid arccos domain (degenerate eigenvalues sit at r = +-1).
    r = jnp.clip(r, -1.0 + _EPS, 1.0 - _EPS)

    phi = jnp.arccos(r) / 3.0
    two_pi_3 = 2.0 * jnp.pi / 3.0

    e_hi = q + 2.0 * p * jnp.cos(phi)
    e_lo = q + 2.0 * p * jnp.cos(phi + two_pi_3)
    e_mid = 3.0 * q - e_hi - e_lo  # trace conservation

    eig = jnp.stack([e_lo, e_mid, e_hi])

    # If the matrix was essentially diagonal, fall back to the sorted diagonal.
    eig_diag = jnp.sort(diag)
    eig = jnp.where(big, eig, eig_diag)
    return eig
