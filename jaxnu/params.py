"""Oscillation parameters as a differentiable JAX PyTree.

``OscParams`` holds the standard 3-flavor parameters.  It is registered as a
PyTree dataclass so it can be differentiated through directly
(``jax.grad(f)(params)`` returns an ``OscParams`` of gradients) and passed to
optimizers / samplers.

The framework is intentionally not hard-wired to 3 flavors: the numerical core
(:mod:`jaxnu.hamiltonian`) consumes a PMNS matrix ``U`` and a mass-squared vector
of *arbitrary* size.  ``OscParams.pmns()`` / ``OscParams.msqared()`` are the
3-flavor adapters; sterile / NSI extensions provide their own.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from . import pmns


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class OscParams:
    """Standard 3-flavor oscillation parameters (mixing angles in radians,
    mass-squared splittings in eV^2).

    ``dm21 = m2^2 - m1^2`` (solar), ``dm31 = m3^2 - m1^2`` (atmospheric).
    Normal ordering: ``dm31 > 0``; inverted: ``dm31 < 0``.
    """

    theta12: jax.Array
    theta13: jax.Array
    theta23: jax.Array
    deltacp: jax.Array
    dm21: jax.Array
    dm31: jax.Array

    def pmns(self) -> jax.Array:
        """3x3 complex PMNS matrix."""
        return pmns.pmns_3flavor(
            self.theta12, self.theta13, self.theta23, self.deltacp
        )

    def msquared(self) -> jax.Array:
        """Mass-squared values relative to m1: ``[0, dm21, dm31]`` (eV^2)."""
        return jnp.stack(
            [jnp.zeros_like(self.dm21), self.dm21, self.dm31]
        )


def nufit_no() -> OscParams:
    """A reasonable NuFIT-5.2-like normal-ordering benchmark point."""
    return OscParams(
        theta12=jnp.asarray(0.5836),   # ~33.4 deg
        theta13=jnp.asarray(0.1495),   # ~8.57 deg
        theta23=jnp.asarray(0.8587),   # ~49.2 deg
        deltacp=jnp.asarray(3.40),     # ~195 deg
        dm21=jnp.asarray(7.42e-5),
        dm31=jnp.asarray(2.515e-3),
    )
