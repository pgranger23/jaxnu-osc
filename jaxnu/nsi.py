"""Non-standard neutrino interactions (matter NSI).

Vector matter NSI modify the matter potential on the active block:

    H_matter = V_CC * (diag(1, 0, 0) + epsilon)

where ``epsilon`` is a Hermitian matrix of dimensionless parameters
``eps_{alpha beta}`` (relative to the charged-current potential ``V_CC``).
Diagonal entries are real; off-diagonal entries are complex.  Pass an :class:`NSI`
(or a raw matrix) to the ``nsi=`` argument of the probability functions.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NSI:
    """Matter-NSI parameters ``epsilon_{alpha beta}`` (relative to ``V_CC``).

    Off-diagonal parameters are complex; only the upper triangle is stored, the
    matrix is built Hermitian.  Defaults are 0 (standard interactions).
    """

    eps_ee: jax.Array = 0.0
    eps_mumu: jax.Array = 0.0
    eps_tautau: jax.Array = 0.0
    eps_emu: jax.Array = 0.0 + 0.0j
    eps_etau: jax.Array = 0.0 + 0.0j
    eps_mutau: jax.Array = 0.0 + 0.0j

    def matrix(self, n_active: int = 3) -> jax.Array:
        """Hermitian ``(n_active, n_active)`` NSI matrix (active block only)."""
        m = jnp.zeros((n_active, n_active), dtype=jnp.complex128)
        m = m.at[0, 0].set(self.eps_ee)
        m = m.at[1, 1].set(self.eps_mumu)
        m = m.at[2, 2].set(self.eps_tautau)
        m = m.at[0, 1].set(self.eps_emu)
        m = m.at[1, 0].set(jnp.conj(jnp.asarray(self.eps_emu)))
        m = m.at[0, 2].set(self.eps_etau)
        m = m.at[2, 0].set(jnp.conj(jnp.asarray(self.eps_etau)))
        m = m.at[1, 2].set(self.eps_mutau)
        m = m.at[2, 1].set(jnp.conj(jnp.asarray(self.eps_mutau)))
        return m
