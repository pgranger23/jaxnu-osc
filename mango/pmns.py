"""PMNS leptonic mixing matrix construction.

Written generically in ``N`` flavors as a product of complex plane rotations so
that sterile-neutrino extensions only need to add rotations.  The 3-flavor PDG
parametrization ``U = R23 . U13(delta) . R12`` is provided as a helper.
"""

from __future__ import annotations

import jax.numpy as jnp


def plane_rotation(n: int, i: int, j: int, theta, delta=0.0):
    """Complex rotation in the ``(i, j)`` plane of an ``n``-dim space.

    Builds the unitary matrix that is the identity except for::

        U[i, i] = U[j, j] = cos(theta)
        U[i, j] =  sin(theta) * exp(-i delta)
        U[j, i] = -sin(theta) * exp(+i delta)

    With ``delta = 0`` this is a real rotation; a non-zero ``delta`` carries the
    CP-violating phase (used on the (1, 3) plane in the standard convention).
    """
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    phase = jnp.exp(-1j * jnp.asarray(delta, dtype=jnp.complex128))

    u = jnp.eye(n, dtype=jnp.complex128)
    u = u.at[i, i].set(c.astype(jnp.complex128))
    u = u.at[j, j].set(c.astype(jnp.complex128))
    u = u.at[i, j].set(s.astype(jnp.complex128) * phase)
    u = u.at[j, i].set(-s.astype(jnp.complex128) * jnp.conj(phase))
    return u


def pmns_3flavor(theta12, theta13, theta23, deltacp):
    """Standard PDG 3-flavor PMNS matrix ``U = R23 . U13(delta) . R12``."""
    r23 = plane_rotation(3, 1, 2, theta23, 0.0)
    u13 = plane_rotation(3, 0, 2, theta13, deltacp)
    r12 = plane_rotation(3, 0, 1, theta12, 0.0)
    return r23 @ u13 @ r12
