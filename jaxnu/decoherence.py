"""Quantum decoherence and wave-packet decoherence in neutrino oscillations.

Environmental (Lindblad-type) decoherence or wave-packet separation damp the
interference terms between (matter) Hamiltonian eigenstates:

    P(a->b) = sum_ij  V_bi V_ai* V_bj* V_aj  exp(-i (w_i - w_j) L)  D_ij(L)

where ``V``/``w`` are the eigenvectors/eigenvalues of the (matter) Hamiltonian
and ``D_ij`` is the damping factor (``D_ii = 1``):

* **Lindblad** (:class:`Decoherence`): ``D_ij = exp(-Gamma_ij L)`` with
  ``Gamma_ij = gamma_ij (E / E0)^n`` — the standard phenomenological model with
  a power-law energy dependence (``n`` in {-2, -1, 0, 1, 2} in the literature).
* **Wave packets** (:class:`WavePacket`): ``D_ij = exp(-(L/L_coh^ij)^2)`` with
  coherence length ``L_coh = 2 sqrt(2) E sigma_x / |dw_ij|`` (equivalently
  ``4 sqrt(2) E^2 sigma_x / |dm~^2_ij|``), driven by the production wave-packet
  size ``sigma_x``.

Both parameter sets are differentiable PyTrees, so ``jax.grad`` works through
``gamma`` / ``sigma_x`` — e.g. for decoherence sensitivity forecasts. In the
``gamma -> 0`` / ``sigma_x -> inf`` limit the standard oscillation probability is
recovered exactly; in the fully-decohered limit the interference-averaged
probability ``sum_i |V_bi|^2 |V_ai|^2`` is obtained.

Scope: vacuum and constant-density matter (a single layer). The damping indices
refer to the (matter) eigenstates sorted by eigenvalue — in vacuum with normal
ordering these are the usual mass states 1, 2, 3.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from . import constants as C
from .hamiltonian import matter_hamiltonian
from .oscillator import select


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Decoherence:
    """Lindblad decoherence: ``Gamma_ij = gamma_ij * (E / E0_GeV)^n`` (gammas in eV).

    ``n`` is a static (non-differentiable) power index.
    """

    gamma21: jax.Array = 0.0
    gamma31: jax.Array = 0.0
    gamma32: jax.Array = 0.0
    E0_GeV: jax.Array = 1.0
    n: int = dataclasses.field(default=0, metadata=dict(static=True))

    def damping(self, dw, energy_eV, length_invEV):
        """(3, 3) damping matrix for eigenvalue differences ``dw[i,j]``."""
        g = jnp.zeros((3, 3))
        g = g.at[1, 0].set(self.gamma21).at[0, 1].set(self.gamma21)
        g = g.at[2, 0].set(self.gamma31).at[0, 2].set(self.gamma31)
        g = g.at[2, 1].set(self.gamma32).at[1, 2].set(self.gamma32)
        scale = (energy_eV / (self.E0_GeV * C.GEV_TO_EV)) ** self.n
        return jnp.exp(-g * scale * length_invEV)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WavePacket:
    """Wave-packet decoherence with production wave-packet size ``sigma_x_m`` (meters)."""

    sigma_x_m: jax.Array = 1.0e-3

    def damping(self, dw, energy_eV, length_invEV):
        sigma_x = self.sigma_x_m * C.M_TO_INV_EV  # -> eV^-1
        arg = length_invEV * dw / (2.0 * jnp.sqrt(2.0) * energy_eV * sigma_x)
        return jnp.exp(-(arg ** 2))


def _prob_single(params, energy_eV, length_invEV, v_cc, model, anti):
    h = matter_hamiltonian(params.pmns(), params.msquared(), energy_eV, v_cc,
                           anti=anti)
    w, V = jnp.linalg.eigh(h)
    dw = w[:, None] - w[None, :]
    M = jnp.exp(-1j * dw * length_invEV) * model.damping(jnp.abs(dw), energy_eV,
                                                         length_invEV)
    Cmat = V[:, None, :] * jnp.conj(V)[None, :, :]      # C[b, a, i] = V_bi V_ai*
    P = jnp.einsum("bai,ij,baj->ba", Cmat, M, jnp.conj(Cmat))
    return jnp.real(P)


def probability(params, energy_GeV, baseline_km, model, density=0.0, ye=0.5,
                anti=False, flavor_in=None, flavor_out=None):
    """Oscillation probabilities with decoherence (vacuum or constant density).

    ``model`` is a :class:`Decoherence` or :class:`WavePacket`; ``energy_GeV``
    may be a scalar or 1-D array. Returns ``P[..., out, in]`` (or a selected
    channel). Differentiable in ``params`` *and* the decoherence parameters.
    """
    energy_eV = jnp.asarray(energy_GeV) * C.GEV_TO_EV
    length_invEV = jnp.asarray(baseline_km) * C.KM_TO_INV_EV
    v_cc = C.matter_potential_eV(density, ye)

    def core(e):
        return _prob_single(params, e, length_invEV, v_cc, model, anti)

    p = core(energy_eV) if energy_eV.ndim == 0 else jax.vmap(core)(energy_eV)
    if flavor_in is not None and flavor_out is not None:
        return select(p, flavor_in, flavor_out)
    return p
