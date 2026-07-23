"""Non-unitary leptonic mixing (alpha parametrization).

Heavy neutral leptons (or any high-scale mixing with extra states) make the
effective 3x3 leptonic mixing matrix non-unitary:

    N = (1 - alpha) U,     alpha = lower-triangular,
                           alpha_11/22/33 real >= 0, alpha_21/31/32 complex,

with ``U`` the standard (unitary) PMNS matrix (e.g. Escrihuela et al. 2015,
Blennow et al. 2017). Propagation happens in the vacuum-mass basis with

    H_mass = diag(0, dm21, dm31) / 2E  +  N^dag V_flavor N,
    V_flavor = diag(V_CC, 0, 0) - V_NC * I_3,     V_CC = sqrt2 G_F N_e,
                                                  V_NC = sqrt2 G_F N_n / 2,

where — unlike the unitary case — the flavor-universal neutral-current term does
*not* drop out, because ``N N^dag != 1``. Production and detection states are
normalized, giving the characteristic zero-distance effect

    P(a->b, L=0) = |(N N^dag)_ba|^2 / [(N N^dag)_aa (N N^dag)_bb].

The amplitude is ``A = N exp(-i H_mass L) N^dag`` (normalized per flavor), and
``P = |A_ba|^2``. In the unitary limit ``alpha = 0`` everything reduces exactly
to the standard probabilities. ``NonUnitarity`` is a differentiable PyTree, so
``jax.grad`` works through all six alpha parameters.

Scope: vacuum and constant-density matter (single layer).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from . import constants as C
from .oscillator import select


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NonUnitarity:
    """The alpha parameters (diagonal real, off-diagonal complex; all -> 0 = unitary)."""

    alpha11: jax.Array = 0.0
    alpha22: jax.Array = 0.0
    alpha33: jax.Array = 0.0
    alpha21: jax.Array = 0.0 + 0.0j
    alpha31: jax.Array = 0.0 + 0.0j
    alpha32: jax.Array = 0.0 + 0.0j

    def matrix(self):
        """The lower-triangular alpha matrix (3, 3)."""
        z = jnp.zeros((), dtype=jnp.complex128)
        return jnp.array(
            [[self.alpha11 + z, z, z],
             [self.alpha21 + z, self.alpha22 + z, z],
             [self.alpha31 + z, self.alpha32 + z, self.alpha33 + z]])

    def N(self, u):
        """Non-unitary mixing matrix ``N = (1 - alpha) U``."""
        return (jnp.eye(3, dtype=jnp.complex128) - self.matrix()) @ jnp.asarray(
            u, dtype=jnp.complex128)


def _prob_single(params, nu, energy_eV, length_invEV, v_cc, v_nc, anti):
    N = nu.N(params.pmns())
    if anti:
        N = jnp.conj(N)
        v_cc, v_nc = -v_cc, -v_nc
    msq = params.msquared()
    vf = (jnp.diag(jnp.array([v_cc, 0.0, 0.0], dtype=jnp.complex128))
          - v_nc * jnp.eye(3, dtype=jnp.complex128))
    h = jnp.diag((msq / (2.0 * energy_eV)).astype(jnp.complex128)) \
        + jnp.conj(N).T @ vf @ N
    w, V = jnp.linalg.eigh(h)
    S_mass = (V * jnp.exp(-1j * w * length_invEV)) @ jnp.conj(V).T
    A = N @ S_mass @ jnp.conj(N).T                      # A[b, a]
    NN = N @ jnp.conj(N).T
    norm = jnp.real(jnp.diag(NN))
    P = jnp.abs(A) ** 2 / (norm[:, None] * norm[None, :])
    return P


def probability(params, nu, energy_GeV, baseline_km, density=0.0, ye=0.5,
                anti=False, flavor_in=None, flavor_out=None):
    """Oscillation probabilities with non-unitary mixing.

    ``nu`` is a :class:`NonUnitarity`; ``energy_GeV`` may be scalar or 1-D.
    Returns ``P[..., out, in]`` (or a selected channel); includes the
    zero-distance effect and the non-cancelling NC matter term. Differentiable
    in ``params`` and all ``alpha`` parameters.
    """
    energy_eV = jnp.asarray(energy_GeV) * C.GEV_TO_EV
    length_invEV = jnp.asarray(baseline_km) * C.KM_TO_INV_EV
    v_cc, v_nc = C.matter_potentials(density, ye)

    def core(e):
        return _prob_single(params, nu, e, length_invEV, v_cc, v_nc, anti)

    p = core(energy_eV) if energy_eV.ndim == 0 else jax.vmap(core)(energy_eV)
    if flavor_in is not None and flavor_out is not None:
        return select(p, flavor_in, flavor_out)
    return p
