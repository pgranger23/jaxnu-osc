"""Evolution operator ``S = exp(-i H L)`` for a constant-density segment.

Three interchangeable backends:

* ``cayley`` (default, 3x3): Cayley-Hamilton expansion ``exp(-iHL) = a0 I + a1 H
  + a2 H^2`` with coefficients from the analytic eigenvalues.  Fast, eigenvector-
  free, clean gradients.
* ``eigh``: reconstruct from ``jnp.linalg.eigh``.  Works for any N (steriles);
  used as the validation oracle.
* ``expm``: ``jax.scipy.linalg.expm`` of ``-iHL``.  Reference cross-check.

``H`` is in eV, ``L`` in eV^-1, so ``H * L`` is dimensionless.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

from .eigensolve import eigvalsh3


def propagator_eigh(h, length):
    """``exp(-i H L)`` via Hermitian eigendecomposition (any N)."""
    w, v = jnp.linalg.eigh(h)
    phase = jnp.exp(-1j * w * length)
    return (v * phase) @ jnp.conj(v).T


def propagator_expm(h, length):
    """``exp(-i H L)`` via matrix exponential (any N)."""
    return jsl.expm(-1j * h * length)


_DEGEN_EPS = 1e-7


def _f(x):
    return jnp.exp(-1j * x)


def _fp(x):
    return -1j * jnp.exp(-1j * x)


def _divdiff(a, b):
    """First divided difference ``(f(a)-f(b))/(a-b)`` with confluent limit.

    Uses a safe denominator so the *unused* branch never feeds NaN into the
    reverse-mode gradient (JAX ``where`` propagates NaNs from both branches).
    """
    d = a - b
    big = jnp.abs(d) > _DEGEN_EPS
    safe = jnp.where(big, d, 1.0)
    return jnp.where(big, (_f(a) - _f(b)) / safe, _fp(a))


def propagator_cayley(h, length):
    """``exp(-i H L)`` via Cayley-Hamilton / Newton divided differences (3x3).

    Works with the dimensionless matrix ``M = H L`` (eigenvalues O(1)) so the
    interpolation is well conditioned -- the raw Hamiltonian eigenvalues are
    ~1e-13 eV.  The Hermite (Newton) form

        S = f[m0] I + f[m0,m1] (M - m0 I) + f[m0,m1,m2] (M - m0 I)(M - m1 I)

    is exact for repeated eigenvalues, so zero-length segments (degenerate zero
    eigenvalues -> identity) and level crossings are handled without the
    singular Vandermonde solve.
    """
    m = h * length  # Hermitian, eigenvalues O(1)
    mu = eigvalsh3(m)  # real, ascending, shape (3,)
    m0, m1, m2 = mu[0], mu[1], mu[2]

    d01 = _divdiff(m0, m1)
    d12 = _divdiff(m1, m2)
    dd = m2 - m0
    big = jnp.abs(dd) > _DEGEN_EPS
    safe = jnp.where(big, dd, 1.0)
    d012 = jnp.where(big, (d12 - d01) / safe, -0.5 * _f(m0))  # f''(m0)/2

    eye = jnp.eye(3, dtype=jnp.complex128)
    a = m - m0 * eye
    b = m - m1 * eye
    return _f(m0) * eye + d01 * a + d012 * (a @ b)


_BACKENDS = {
    "cayley": propagator_cayley,
    "eigh": propagator_eigh,
    "expm": propagator_expm,
}


def propagator(h, length, backend="cayley"):
    """Dispatch to a propagator backend by name."""
    try:
        fn = _BACKENDS[backend]
    except KeyError:
        raise ValueError(
            f"unknown backend {backend!r}; choose from {sorted(_BACKENDS)}"
        )
    return fn(h, length)
