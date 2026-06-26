"""Sterile-neutrino front-end (3+N_sterile models).

The numerical core is already N-flavor; this module provides the user-facing
mixing-matrix builders and differentiable parameter containers.  Sterile flavors
occupy indices ``>= 3`` and feel only the relative neutral-current matter
potential (handled by the probability functions via ``n_active = 3``).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from . import pmns


def pmns_nflavor(n, rotations):
    """Build an ``n x n`` unitary mixing matrix from ordered plane rotations.

    ``rotations`` is a sequence of ``(i, j, theta, delta)`` applied left to right
    (the first entry is the left-most factor).  ``delta`` may be omitted (0).
    """
    u = jnp.eye(n, dtype=jnp.complex128)
    for rot in rotations:
        i, j, theta = rot[0], rot[1], rot[2]
        delta = rot[3] if len(rot) > 3 else 0.0
        u = u @ pmns.plane_rotation(n, i, j, theta, delta)
    return u


def pmns_3plus1(theta12, theta13, theta23, theta14, theta24, theta34,
                delta13=0.0, delta24=0.0):
    """3+1 PMNS matrix ``U = R34 R24 R14 R23 R13 R12`` (flavors e, mu, tau, s)."""
    return pmns_nflavor(4, [
        (2, 3, theta34),
        (1, 3, theta24, delta24),
        (0, 3, theta14),
        (1, 2, theta23),
        (0, 2, theta13, delta13),
        (0, 1, theta12),
    ])


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Sterile3plus1:
    """Differentiable 3+1 sterile parameters (angles in rad, Delta m^2 in eV^2)."""

    theta12: jax.Array
    theta13: jax.Array
    theta23: jax.Array
    theta14: jax.Array
    theta24: jax.Array
    theta34: jax.Array
    delta13: jax.Array
    delta24: jax.Array
    dm21: jax.Array
    dm31: jax.Array
    dm41: jax.Array

    n_active = 3  # class attribute (not a differentiable field)

    def pmns(self):
        return pmns_3plus1(self.theta12, self.theta13, self.theta23,
                           self.theta14, self.theta24, self.theta34,
                           self.delta13, self.delta24)

    def msquared(self):
        zero = jnp.zeros_like(self.dm21)
        return jnp.stack([zero, self.dm21, self.dm31, self.dm41])


@dataclasses.dataclass(frozen=True)
class NFlavorParams:
    """Generic N-flavor parameters holding a precomputed ``U`` and ``msq``.

    Use for arbitrary ``3 + N_sterile`` models (e.g. via :func:`pmns_nflavor`).
    Differentiable through ``U`` / ``msq`` entries; ``n_active`` is static.
    """

    U: jax.Array
    msq: jax.Array
    n_active: int = 3

    def pmns(self):
        return self.U

    def msquared(self):
        return self.msq


jax.tree_util.register_dataclass(
    NFlavorParams, data_fields=["U", "msq"], meta_fields=["n_active"])
