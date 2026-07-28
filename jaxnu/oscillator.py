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

import numpy as _np

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
    """Pick ``P(nu_{flavor_in} -> nu_{flavor_out})`` from a probability matrix.

    Raises ``ValueError`` if either index is out of range for ``p``'s flavor
    dimension (``p.shape[-1]``) rather than silently clamping (JAX's default
    out-of-bounds indexing behaviour).
    """
    n = p.shape[-1]
    fi, fo = int(flavor_in), int(flavor_out)
    if not (0 <= fi < n):
        raise ValueError(
            f"flavor_in={fi} out of range for a {n}-flavor probability matrix "
            f"(valid indices: 0..{n - 1})"
        )
    if not (0 <= fo < n):
        raise ValueError(
            f"flavor_out={fo} out of range for a {n}-flavor probability matrix "
            f"(valid indices: 0..{n - 1})"
        )
    return p[..., fo, fi]


# --- helpers ----------------------------------------------------------------


def _is_concrete(x):
    """``True`` unless ``x`` is a JAX tracer (i.e. a value under ``jit``/``vmap``/
    ``grad``).  Gate every eager input-validation check on this: we can only
    inspect the *value* of a concrete input, never of a traced one -- checking a
    tracer's value would either raise a ``TracerBoolConversionError`` or (worse)
    silently bake a spurious concrete branch into the trace.  Under tracing we
    simply skip the check; validation still happens whenever the same code path
    runs on concrete inputs (e.g. once at the boundary of a jitted region, or in
    a pure-Python call).

    Always test concreteness of the *raw* argument, before any ``jax.numpy``
    conversion -- and, once confirmed concrete, do the actual numeric check with
    plain ``numpy`` (never ``jax.numpy``).  Under ``jit`` (omnistaging), *every*
    ``jax.numpy`` op executed while a trace is active is staged into the jaxpr
    and returns a fresh tracer, even one applied only to concrete/closed-over
    Python constants unrelated to the traced arguments; using ``jax.numpy`` for
    the check itself would therefore spuriously produce a tracer (and blow up on
    ``bool()``) purely from being lexically inside someone else's ``jit``, which
    is exactly the failure mode this helper exists to avoid.
    """
    return not isinstance(x, jax.core.Tracer)


def _check_positive(x, name, unit_hint):
    """Raise if concrete ``x`` has any non-positive entry."""
    if _is_concrete(x) and _np.any(_np.asarray(x) <= 0):
        raise ValueError(f"{name} must be positive (units: {unit_hint})")


def _check_nonneg(x, name, unit_hint):
    """Raise if concrete ``x`` has any negative entry."""
    if _is_concrete(x) and _np.any(_np.asarray(x) < 0):
        raise ValueError(f"{name} must be non-negative (units: {unit_hint})")


def _check_abs_le_one(x, name):
    """Raise if concrete ``x`` has any entry with magnitude > 1."""
    if _is_concrete(x) and _np.any(_np.abs(_np.asarray(x)) > 1.0):
        raise ValueError(
            f"{name} must satisfy |{name}| <= 1 (it is cos(zenith), dimensionless "
            "-- not an angle in degrees or radians)"
        )


def _check_flavor_index(idx, n, name):
    """Raise if the (always-static) flavor index ``idx`` is out of range for an
    ``n``-flavor ``params``."""
    if idx is None:
        return
    i = int(idx)
    if not (0 <= i < n):
        raise ValueError(
            f"{name}={i} out of range for the {n}-flavor params passed "
            f"(valid indices: 0..{n - 1})"
        )


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
    array, or ``None``.

    The :class:`jaxnu.nsi.NSI` dataclass path always builds a Hermitian matrix by
    construction. A raw matrix has no such guarantee, so it is checked here: a
    non-Hermitian matter-NSI matrix breaks the unitarity of the propagated
    amplitude (it silently injects non-physical gain/loss), so it is rejected
    rather than used as-is.
    """
    if nsi is None:
        return None
    if hasattr(nsi, "matrix"):
        return nsi.matrix(n_active)
    # Check concreteness of the *raw* nsi argument, before any jax.numpy
    # conversion, and do the actual check in plain numpy -- see the docstring
    # of `_is_concrete` for why: under `jit`, converting via jax.numpy first
    # and then testing `_is_concrete` on the *result* would spuriously report
    # "traced" (and a numpy/jnp check on that result would spuriously error)
    # any time this code merely happens to run lexically inside someone else's
    # jit trace, regardless of whether `nsi` itself is a traced value.
    if _is_concrete(nsi):
        m_np = _np.asarray(nsi, dtype=_np.complex128)
        if not _np.allclose(m_np, _np.conj(m_np).T, atol=1e-9):
            raise ValueError(
                "raw nsi= matrix must be Hermitian (eps_{alpha,beta} == "
                "conj(eps_{beta,alpha})); a non-Hermitian matter potential "
                "breaks unitarity of the propagated amplitude. Pass a "
                "jaxnu.nsi.NSI(...) instance instead if you want this "
                "enforced by construction."
            )
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
    _check_positive(energy_GeV, "energy_GeV", "GeV, not eV")
    _check_nonneg(baseline_km, "baseline_km", "km, not m")
    _check_nonneg(density, "density", "g/cm^3")
    _check_flavor_index(flavor_in, n_flavors(params), "flavor_in")
    _check_flavor_index(flavor_out, n_flavors(params), "flavor_out")

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
    _check_positive(energy_GeV, "energy_GeV", "GeV, not eV")
    _check_nonneg(density_gcc, "density_gcc", "g/cm^3")
    _check_nonneg(length_km, "length_km", "km, not m")
    _check_flavor_index(flavor_in, n_flavors(params), "flavor_in")
    _check_flavor_index(flavor_out, n_flavors(params), "flavor_out")

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
                      ye_mantle=_earth.YE_MANTLE_DEFAULT, earth_model=None,
                      anti=False, backend="cayley", nsi=None,
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

    Pass ``earth_model`` (a :class:`jaxnu.earth.LayeredEarth`) to use a fully
    parametric constant-density-shell Earth instead of the fixed PREM polynomials;
    probabilities are then differentiable w.r.t. the **shell boundary radii**,
    **densities** and **Y_e** (it overrides ``n_sub`` / ``ye_core`` / ``ye_mantle``).
    """
    _check_positive(energy_GeV, "energy_GeV", "GeV, not eV")
    _check_abs_le_one(cos_zenith, "cos_zenith")
    _check_nonneg(det_depth_km, "det_depth_km", "km, not m")
    _check_flavor_index(flavor_in, n_flavors(params), "flavor_in")
    _check_flavor_index(flavor_out, n_flavors(params), "flavor_out")
    u, msq = params.pmns(), params.msquared()
    na = _n_active(params)
    nsi_mat = _nsi_matrix(nsi, na)
    backend = _resolve_backend(params, backend)

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
