"""High-level oscillation-probability API.

Single-point functional cores operate in natural units; the public functions
convert from GeV / km / (g/cm^3) and vectorize over energy (and zenith for the
Earth case) with ``vmap``.  Everything is ``jit``-able and differentiable.

Probability matrices are returned with shape ``(..., N, N)`` indexed
``P[..., beta, alpha] = P(nu_alpha -> nu_beta)``.
"""

from __future__ import annotations

import enum
import functools

import jax
import jax.numpy as jnp

from . import constants as C
from . import earth as _earth
from . import nufast as _nufast
from .hamiltonian import matter_hamiltonian
from .layers import propagate_layers
from .propagator import propagator


class Flavor(enum.IntEnum):
    E = 0
    MU = 1
    TAU = 2


def prob_from_amplitude(s):
    """``P[beta, alpha] = |S[beta, alpha]|^2`` for evolution operator ``s``."""
    return jnp.abs(s) ** 2


def select(p, flavor_in, flavor_out):
    """Pick ``P(nu_{flavor_in} -> nu_{flavor_out})`` from a probability matrix."""
    return p[..., int(flavor_out), int(flavor_in)]


# --- helpers ----------------------------------------------------------------

def n_flavors(params):
    """Number of neutrino flavors in ``params`` (3 for standard, 4 for 3+1, ...)."""
    return params.msquared().shape[0]


def _n_active(params):
    return getattr(params, "n_active", n_flavors(params))


def _resolve_backend(params, backend):
    # cayley is 3x3-only; fall back to eigh (works for any N) otherwise.
    if n_flavors(params) != 3 and backend == "cayley":
        return "eigh"
    return backend


def _nsi_matrix(nsi, n_active):
    """Coerce an NSI spec (NSI object or matrix) to an ``(n_active, n_active)``
    array, or ``None``."""
    if nsi is None:
        return None
    if hasattr(nsi, "matrix"):
        return nsi.matrix(n_active)
    return jnp.asarray(nsi, dtype=jnp.complex128)


# --- public API --------------------------------------------------------------

def probability_constant(params, energy_GeV, baseline_km, density=0.0, ye=0.5,
                         anti=False, backend="nufast", nsi=None,
                         flavor_in=None, flavor_out=None):
    """Oscillation probabilities through constant-density matter (or vacuum).

    ``energy_GeV`` may be a scalar or 1-D array (vectorized).  ``baseline_km``,
    ``density`` and ``ye`` are scalars.  Set ``density=0`` for vacuum.  ``nsi`` is
    an optional :class:`jaxnu.nsi.NSI` (or matrix) for non-standard interactions;
    ``params`` may carry sterile flavors (see :mod:`jaxnu.sterile`).

    ``backend="nufast"`` (default) uses the fast analytic NuFast formula
    (:mod:`jaxnu.nufast`) for standard 3-flavor; it transparently falls back to a
    matrix-exponential backend (``"cayley"`` for 3-flavor, ``"eigh"`` for N!=3)
    when NSI or steriles are present.  Returns a ``(..., N, N)`` matrix, or a
    scalar/1-D array if both ``flavor_in`` and ``flavor_out`` are given.
    """
    energy_eV = jnp.asarray(energy_GeV) * C.GEV_TO_EV
    length_invEV = jnp.asarray(baseline_km) * C.KM_TO_INV_EV
    v_cc, v_nc = C.matter_potentials(density, ye)
    na = _n_active(params)

    if backend == "nufast":
        if _nufast.eligible(params, nsi, na):
            def core(e_eV):
                return _nufast.prob_matrix(params, e_eV, length_invEV, v_cc, anti)
            p = (core(energy_eV) if energy_eV.ndim == 0
                 else jax.vmap(core)(energy_eV))
            if flavor_in is not None and flavor_out is not None:
                return select(p, flavor_in, flavor_out)
            return p
        backend = "cayley"  # not eligible -> matrix-exponential path

    u, msq = params.pmns(), params.msquared()
    nsi_mat = _nsi_matrix(nsi, na)
    backend = _resolve_backend(params, backend)

    def core(e_eV):
        h = matter_hamiltonian(u, msq, e_eV, v_cc, anti=anti, nsi=nsi_mat,
                               v_nc_eV=v_nc, n_active=na)
        return prob_from_amplitude(propagator(h, length_invEV, backend=backend))

    p = core(energy_eV) if energy_eV.ndim == 0 else jax.vmap(core)(energy_eV)
    if flavor_in is not None and flavor_out is not None:
        return select(p, flavor_in, flavor_out)
    return p


def probability_vacuum(params, energy_GeV, baseline_km, anti=False,
                       backend="nufast", flavor_in=None, flavor_out=None):
    """Vacuum oscillation probabilities (convenience: ``density=0``)."""
    return probability_constant(
        params, energy_GeV, baseline_km, density=0.0, anti=anti,
        backend=backend, flavor_in=flavor_in, flavor_out=flavor_out,
    )


def probability_profile(params, energy_GeV, density_gcc, ye, length_km,
                        anti=False, backend="cayley", nsi=None,
                        flavor_in=None, flavor_out=None):
    """Probabilities through a user-supplied piecewise-constant profile.

    ``density_gcc``, ``ye``, ``length_km`` are 1-D arrays (one entry per
    segment, ordered source -> detector).  ``energy_GeV`` is a scalar or array.
    """
    density_gcc = jnp.asarray(density_gcc)
    ye = jnp.broadcast_to(jnp.asarray(ye), density_gcc.shape)
    v_cc, v_nc = C.matter_potentials(density_gcc, ye)
    length_invEV = jnp.asarray(length_km) * C.KM_TO_INV_EV
    u, msq = params.pmns(), params.msquared()
    na = _n_active(params)
    nsi_mat = _nsi_matrix(nsi, na)
    backend = _resolve_backend(params, backend)

    def core(e_eV):
        s = propagate_layers(u, msq, e_eV, v_cc, length_invEV, anti=anti,
                             backend=backend, nsi=nsi_mat, v_nc=v_nc, n_active=na)
        return prob_from_amplitude(s)

    energy_eV = jnp.asarray(energy_GeV) * C.GEV_TO_EV
    p = core(energy_eV) if energy_eV.ndim == 0 else jax.vmap(core)(energy_eV)

    if flavor_in is not None and flavor_out is not None:
        return select(p, flavor_in, flavor_out)
    return p


def _grid_eval(core, energy_eV, cz):
    """Evaluate ``core(E, cz)`` over scalar/1-D combinations.

    Returns shape ``(n_cz, n_E, 3, 3)`` when both are arrays (an oscillogram
    grid), reducing leading axes for scalar inputs.
    """
    e_scalar = energy_eV.ndim == 0
    c_scalar = cz.ndim == 0
    if e_scalar and c_scalar:
        return core(energy_eV, cz)
    if c_scalar:
        return jax.vmap(lambda e: core(e, cz))(energy_eV)
    if e_scalar:
        return jax.vmap(lambda c: core(energy_eV, c))(cz)
    return jax.vmap(lambda c: jax.vmap(lambda e: core(e, c))(energy_eV))(cz)


def probability_earth(params, energy_GeV, cos_zenith, det_depth_km=0.0,
                      n_sub=4, h_atm_km=0.0, ye_core=_earth.YE_CORE_DEFAULT,
                      ye_mantle=_earth.YE_MANTLE_DEFAULT, anti=False,
                      backend="cayley", nsi=None,
                      flavor_in=None, flavor_out=None):
    """Oscillation probabilities through the PREM Earth (and atmosphere).

    ``energy_GeV`` and ``cos_zenith`` may each be scalar or 1-D.  With both 1-D
    the result is an oscillogram grid of shape ``(n_cz, n_E, N, N)``.
    ``cos_zenith < 0`` is up-going (through the Earth).  With ``h_atm_km > 0`` a
    production height is added so down-going / near-horizon directions
    (``cos_zenith >= 0``) get the correct vacuum baseline (atmospheric mode);
    with ``h_atm_km = 0`` (default) down-going is pure vacuum of zero length.

    ``n_sub`` subdivides each PREM region for accuracy; ``ye_core`` / ``ye_mantle``
    set the two-zone electron fraction (boundary at 3480 km); ``det_depth_km`` is
    the detector depth below the surface.  ``nsi`` and sterile ``params`` are
    supported (sterile flavors feel the relative NC potential).
    """
    table = _earth.shell_table(n_sub)  # static geometry (boundaries); Y_e set below
    u, msq = params.pmns(), params.msquared()
    na = _n_active(params)
    nsi_mat = _nsi_matrix(nsi, na)
    backend = _resolve_backend(params, backend)

    def core(e_eV, cz):
        rho, ye, length_km = _earth.chord_segments(
            cz, table, h_atm_km=h_atm_km, det_depth_km=det_depth_km,
            ye_core=ye_core, ye_mantle=ye_mantle)
        v_cc, v_nc = C.matter_potentials(rho, ye)
        length_invEV = length_km * C.KM_TO_INV_EV
        s = propagate_layers(u, msq, e_eV, v_cc, length_invEV, anti=anti,
                             backend=backend, nsi=nsi_mat, v_nc=v_nc, n_active=na)
        return prob_from_amplitude(s)

    energy_eV = jnp.asarray(energy_GeV) * C.GEV_TO_EV
    cz = jnp.asarray(cos_zenith, dtype=jnp.float64)
    p = _grid_eval(core, energy_eV, cz)

    if flavor_in is not None and flavor_out is not None:
        return select(p, flavor_in, flavor_out)
    return p
