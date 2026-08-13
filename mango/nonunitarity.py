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

**Layered / Earth matter** (:func:`probability_profile`, :func:`probability_earth`):
``N`` is applied *once*, at the source and detector -- it is not a per-layer
object. A stack of constant-density segments is handled by building the
*mass-basis* Hamiltonian ``H_mass,k = diag(0, dm21, dm31)/2E + N^dag V_flavor,k
N`` for each segment ``k`` (own ``V_CC,k``/``V_NC,k``), chaining the ordinary
unitary sub-propagators ``S_mass = S_mass,N ... S_mass,1`` exactly as
:func:`mango.layers.propagate_layers` does for the standard case, and only then
sandwiching the *total* chain with ``N ... N^dag``. Geometry/segments come from
:mod:`mango.earth` (:func:`mango.earth.chord_segments` /
:func:`mango.earth.layered_chord_segments`), unchanged.

Non-unitarity means probability need not be conserved column-by-column
(``sum_b P(a->b) != 1`` in general -- genuine unitarity-violation leakage into
the unobserved heavy sector, not a bug). The one probability that *is* exactly
conserved for any ``alpha``, energy or baseline is the diagonal at zero
distance: ``P(a->a; L=0) = 1`` identically, because ``A(L=0) = N N^dag`` and the
normalization divides by exactly that quantity's diagonal.

Scope: vacuum, constant-density matter (single layer), and now a fully layered
/ PREM-Earth profile.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from . import constants as C
from . import earth as _earth
from .oscillator import select, _grid_eval
from .propagator import propagator


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


# --- layered / Earth matter --------------------------------------------------
#
# N is applied once at the vertices, never per-layer: the per-segment object is
# the ordinary unitary mass-basis propagator S_mass,k, chained exactly as
# mango.layers.propagate_layers chains flavor-basis propagators, and only the
# *total* chain gets sandwiched by N ... N^dag. See the module docstring.


def _mass_matter_pieces(N_raw, anti):
    """Effective ``N`` (conjugated for antineutrinos) and its two energy- and
    baseline-independent flavor-projector pieces in the mass basis:

        A = N^dag P_e N,     B = N^dag N        (both Hermitian),

    so that ``N^dag V_flavor,k N = v_cc,k * A - v_nc,k * B`` for any segment.
    Computed once and reused across every segment and every energy (mirrors the
    vacuum-Hamiltonian hoisting in ``layers.propagate_layers``'s parallel path).
    """
    N = jnp.conj(N_raw) if anti else N_raw
    Pe = jnp.diag(jnp.array([1.0, 0.0, 0.0], dtype=jnp.complex128))
    Nd = jnp.conj(N).T
    A = Nd @ Pe @ N
    B = Nd @ N
    return N, A, B


def _propagate_mass_layers(msq, energy_eV, A, B, v_cc_eV, v_nc_eV, length_invEV,
                           backend="cayley", parallel=None):
    """Chain the mass-basis evolution operator through constant-density segments.

    Mirrors :func:`mango.layers.propagate_layers`'s two product strategies
    (sequential ``lax.scan`` on CPU, ``vmap`` + ``lax.associative_scan`` on
    GPU/TPU) applied to the mass-basis Hamiltonian ``H_mass,k = diag(msq/2E) +
    v_cc,k * A - v_nc,k * B`` instead of the flavor-basis
    :func:`mango.hamiltonian.matter_hamiltonian` -- ``A``/``B`` are the
    precomputed pieces from :func:`_mass_matter_pieces`. ``v_cc_eV``/``v_nc_eV``/
    ``length_invEV`` are per-segment arrays ordered source -> detector (segment 0
    applied first). Returns the ``(3, 3)`` mass-basis evolution operator
    ``S_mass``.
    """
    if parallel is None:
        parallel = jax.default_backend() != "cpu"
    h_vac = jnp.diag((msq / (2.0 * energy_eV)).astype(jnp.complex128))
    v_cc_eV = jnp.asarray(v_cc_eV)
    v_nc_eV = jnp.asarray(v_nc_eV)
    length_invEV = jnp.asarray(length_invEV)

    if not parallel:
        def step(carry, seg):
            v_k, vnc_k, l_k = seg
            h = h_vac + v_k * A - vnc_k * B
            return propagator(h, l_k, backend=backend) @ carry, None

        s0 = jnp.eye(3, dtype=jnp.complex128)
        s, _ = jax.lax.scan(step, s0, (v_cc_eV, v_nc_eV, length_invEV))
        return s

    def make_s(v_k, vnc_k, l_k):
        h = h_vac + v_k * A - vnc_k * B
        return propagator(h, l_k, backend=backend)

    s_all = jax.vmap(make_s)(v_cc_eV, v_nc_eV, length_invEV)
    prods = jax.lax.associative_scan(lambda a, b: jnp.matmul(b, a), s_all)
    return prods[-1]


def _amplitude_to_prob(N, s_mass):
    """``P[b, a]`` from the mass-basis chain, sandwiched/normalized once by ``N``
    (the zero-distance-effect formula of the module docstring, with
    ``s_mass = I`` at ``L = 0``)."""
    A = N @ s_mass @ jnp.conj(N).T
    NN = N @ jnp.conj(N).T
    norm = jnp.real(jnp.diag(NN))
    return jnp.abs(A) ** 2 / (norm[:, None] * norm[None, :])


def probability_profile(params, nu, energy_GeV, density_gcc, ye, length_km,
                        anti=False, backend="cayley", parallel=None,
                        flavor_in=None, flavor_out=None):
    """Non-unitary probabilities through a user-supplied piecewise-constant
    profile (cf. :func:`mango.oscillator.probability_profile`).

    ``density_gcc``, ``ye``, ``length_km`` are 1-D arrays (one entry per
    segment, ordered source -> detector); ``energy_GeV`` a scalar or 1-D array.
    A single segment reproduces :func:`probability` exactly (same physics, same
    formula -- see the module docstring). Differentiable in ``params``, all
    ``alpha`` parameters, and the segment densities/Y_e/lengths.
    """
    density_gcc = jnp.asarray(density_gcc)
    ye = jnp.broadcast_to(jnp.asarray(ye), density_gcc.shape)
    v_cc, v_nc = C.matter_potentials(density_gcc, ye)
    length_invEV = jnp.asarray(length_km) * C.KM_TO_INV_EV
    if anti:
        v_cc, v_nc = -v_cc, -v_nc

    N, Amat, Bmat = _mass_matter_pieces(nu.N(params.pmns()), anti)
    msq = params.msquared()

    def core(e):
        s_mass = _propagate_mass_layers(msq, e, Amat, Bmat, v_cc, v_nc,
                                        length_invEV, backend=backend,
                                        parallel=parallel)
        return _amplitude_to_prob(N, s_mass)

    energy_eV = jnp.asarray(energy_GeV) * C.GEV_TO_EV
    p = core(energy_eV) if energy_eV.ndim == 0 else jax.vmap(core)(energy_eV)
    if flavor_in is not None and flavor_out is not None:
        return select(p, flavor_in, flavor_out)
    return p


def probability_earth(params, nu, energy_GeV, cos_zenith, det_depth_km=0.0,
                      n_sub=4, h_atm_km=0.0, ye_core=_earth.YE_CORE_DEFAULT,
                      ye_mantle=_earth.YE_MANTLE_DEFAULT, earth_model=None,
                      anti=False, backend="cayley", parallel=None,
                      flavor_in=None, flavor_out=None):
    """Non-unitary oscillation probabilities through the PREM Earth (and
    atmosphere); cf. :func:`mango.oscillator.probability_earth`.

    Reuses :func:`mango.earth.chord_segments` (or, with ``earth_model``, a
    :class:`mango.earth.LayeredEarth` via
    :func:`mango.earth.layered_chord_segments`) for the geometry -- identical to
    the standard front-end -- and chains segments in the mass basis (see the
    module docstring). ``nu`` is a :class:`NonUnitarity`; other parameters match
    ``oscillator.probability_earth``. ``alpha -> 0`` reproduces the standard
    layered/PREM probability exactly (the non-cancelling ``N^dag N`` NC term
    collapses to an overall phase when ``N`` is unitary).
    """
    u, msq = params.pmns(), params.msquared()
    N, Amat, Bmat = _mass_matter_pieces(nu.N(u), anti)

    if earth_model is None:
        table = _earth.shell_table(n_sub)  # static boundaries; Y_e traced below
        segments = lambda cz: _earth.chord_segments(
            cz, table, h_atm_km=h_atm_km, det_depth_km=det_depth_km,
            ye_core=ye_core, ye_mantle=ye_mantle)
    else:
        segments = lambda cz: _earth.layered_chord_segments(
            earth_model, cz, h_atm_km=h_atm_km, det_depth_km=det_depth_km)

    def core(e_eV, cz):
        rho, ye, length_km = segments(cz)
        v_cc, v_nc = C.matter_potentials(rho, ye)
        if anti:
            v_cc, v_nc = -v_cc, -v_nc
        length_invEV = length_km * C.KM_TO_INV_EV
        s_mass = _propagate_mass_layers(msq, e_eV, Amat, Bmat, v_cc, v_nc,
                                        length_invEV, backend=backend,
                                        parallel=parallel)
        return _amplitude_to_prob(N, s_mass)

    energy_eV = jnp.asarray(energy_GeV) * C.GEV_TO_EV
    cz = jnp.asarray(cos_zenith, dtype=jnp.float64)
    p = _grid_eval(core, energy_eV, cz)

    if flavor_in is not None and flavor_out is not None:
        return select(p, flavor_in, flavor_out)
    return p
