"""PREM Earth model and chord geometry.

Implements the Preliminary Reference Earth Model (Dziewonski & Anderson 1981):
density as piecewise polynomials in ``x = r / R_earth``.  A neutrino arriving with
``cos(zenith) = cz`` (up-going => ``cz < 0``) traverses a chord; we split it into
constant-density shells and hand the segments to the layer propagator.

Design choice for ``jit`` / ``vmap``: the shell radii and per-shell densities are
*static* (built from numpy, independent of ``cz``).  Only the per-segment path
**lengths** depend on ``cz``, computed with clamps so the array shape is fixed --
shells that the chord does not reach simply get zero length (identity
propagators).  Lengths are smooth in ``cz`` (=> differentiable in zenith angle).
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import jax.numpy as jnp

from .constants import R_EARTH_KM

# PREM regions: (outer_radius_km, density polynomial coeffs in x=r/R_E, Y_e).
# Y_e ~ 0.4656 in the (iron) core, ~0.4957 in the mantle/crust.
_PREM = [
    (1221.5, [13.0885, 0.0, -8.8381, 0.0], 0.4656),
    (3480.0, [12.5815, -1.2638, -3.6426, -5.5281], 0.4656),
    (5701.0, [7.9565, -6.4761, 5.5283, -3.0807], 0.4957),
    (5771.0, [5.3197, -1.4836, 0.0, 0.0], 0.4957),
    (5971.0, [11.2494, -8.0298, 0.0, 0.0], 0.4957),
    (6151.0, [7.1089, -3.8045, 0.0, 0.0], 0.4957),
    (6346.6, [2.6910, 0.6924, 0.0, 0.0], 0.4957),
    (6356.0, [2.9000, 0.0, 0.0, 0.0], 0.4957),
    (6368.0, [2.6000, 0.0, 0.0, 0.0], 0.4957),
    (R_EARTH_KM, [1.0200, 0.0, 0.0, 0.0], 0.4957),
]

ShellTable = namedtuple("ShellTable", ["outer", "inner", "rho", "ye"])

# Core / mantle boundary (outer core radius) — sets where Y_e switches.
CORE_RADIUS_KM = 3480.0
# Default electron fractions (PREM-standard: iron core vs silicate mantle).
YE_CORE_DEFAULT = 0.4656
YE_MANTLE_DEFAULT = 0.4957


def prem_density(r_km):
    """PREM density (g/cm^3) at radius ``r_km`` (scalar or array)."""
    r = np.atleast_1d(np.asarray(r_km, dtype=float))
    x = r / R_EARTH_KM
    rho = np.zeros_like(r)
    lower = 0.0
    filled = np.zeros_like(r, dtype=bool)
    for outer, coeffs, _ye in _PREM:
        mask = (~filled) & (r <= outer + 1e-9)
        val = sum(c * x**i for i, c in enumerate(coeffs))
        rho = np.where(mask, val, rho)
        filled |= mask
        lower = outer
    rho = np.where(filled, rho, 0.0)  # outside Earth
    return rho if np.ndim(r_km) else float(rho[0])


def shell_table(n_sub: int = 4, ye_core: float = YE_CORE_DEFAULT,
                ye_mantle: float = YE_MANTLE_DEFAULT) -> ShellTable:
    """Static table of constant-density shells (numpy).

    PREM region boundaries (density discontinuities) are always respected; within
    each region the number of equal-radius sub-shells is allocated **proportional
    to the region's radial thickness** (target step ``R_E / (10*n_sub)``, at least
    one per region).  This puts resolution in the thick core/mantle -- where the
    oscillation phase accumulates -- instead of wasting it on thin crust layers,
    converging much faster in ``n_sub`` than uniform subdivision.

    Density is a placeholder shell mid-radius value (``chord_segments`` resamples
    it at the path midpoint).  The electron fraction is two-zone: ``ye_core``
    inside ``CORE_RADIUS_KM`` (3480 km), ``ye_mantle`` outside (3480 km is a region
    boundary, so no shell straddles it).
    """
    dr_target = R_EARTH_KM / (10 * n_sub)
    outer, inner, rho, ye = [], [], [], []
    lo = 0.0
    for hi, coeffs, _ye_region in _PREM:
        n_k = max(1, int(round((hi - lo) / dr_target)))
        edges = np.linspace(lo, hi, n_k + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            mid = 0.5 * (a + b)
            x = mid / R_EARTH_KM
            inner.append(a)
            outer.append(b)
            rho.append(sum(c * x**i for i, c in enumerate(coeffs)))
            ye.append(ye_core if mid < CORE_RADIUS_KM else ye_mantle)
        lo = hi
    return ShellTable(
        outer=np.array(outer),
        inner=np.array(inner),
        rho=np.array(rho),
        ye=np.array(ye),
    )


_SQRT_FLOOR = 1e-9  # km^2


def _d(r2_minus_rmin2):
    """Distance from closest approach: ``sqrt(max(r^2 - r_min^2, 0))``.

    Gradient-safe: at the chord turning point ``r -> r_min`` the argument hits 0
    where ``d(sqrt)/dx = 1/(2 sqrt(x))`` diverges (the geometric vertical
    tangent), which would feed NaN into ``dP/dcz``.  Below the floor we return 0
    with zero gradient; the forward value is unchanged.
    """
    x = jnp.clip(r2_minus_rmin2, 0.0, None)
    big = x > _SQRT_FLOOR
    x_safe = jnp.where(big, x, 1.0)
    return jnp.where(big, jnp.sqrt(x_safe), 0.0)


def chord_segments(cz, table: ShellTable, h_atm_km=0.0, det_depth_km=0.0):
    """Path segments along the neutrino trajectory for ``cos(zenith) = cz``.

    Unified geometry covering the full range of ``cz``: up-going through the Earth,
    down-going, and the optional atmospheric production height ``h_atm_km``
    (neutrinos produced at ``R_E + h_atm``, giving a vacuum baseline for
    down-going / near-horizon directions). PREM shell boundaries are respected
    exactly (no density smearing). Returns ``(rho, ye, length_km)`` of fixed length
    ``2 * n_shells + 1`` ordered source -> detector; differentiable in ``cz``.

    Geometry uses the distance-from-closest-approach coordinate
    ``d(r) = sqrt(r^2 - r_min^2)`` with ``r_min = r_det*sqrt(1-cz^2)``. The
    descending leg (production side) spans ``d in [d_low, s_prod]`` and, for
    up-going only, an ascending leg spans ``d in [0, |s_det|]``; one vacuum segment
    carries the part above ``R_E``.
    """
    r_det = R_EARTH_KM - det_depth_km
    r_prod = R_EARTH_KM + h_atm_km
    outer = jnp.asarray(table.outer)
    inner = jnp.asarray(table.inner)
    ye = jnp.asarray(table.ye)  # two-zone, constant within a shell

    cz = jnp.asarray(cz, dtype=jnp.float64)
    upgoing = cz < 0.0
    rmin2 = r_det**2 * jnp.clip(1.0 - cz**2, 0.0, None)
    s_prod = _d(r_prod**2 - rmin2)            # production point (d-coordinate)
    s_det = r_det * cz                        # detector: + down-going, - up-going
    d_low = jnp.where(upgoing, 0.0, s_det)    # bottom of the descending leg

    d_out = _d(outer**2 - rmin2)
    d_in = _d(inner**2 - rmin2)
    sdet_abs = jnp.abs(s_det)

    # Per-leg clamped d-ranges -> segment lengths and path-midpoint radii.  Sampling
    # density at the *path* midpoint (not the shell's geometric midpoint) makes the
    # constant-per-segment approximation converge far faster in n_sub (NuFast-style).
    d_hi_desc = jnp.clip(d_out, d_low, s_prod)
    d_lo_desc = jnp.clip(d_in, d_low, s_prod)
    len_desc = d_hi_desc - d_lo_desc
    rho_desc = prem_density_jax(jnp.sqrt(rmin2 + (0.5 * (d_hi_desc + d_lo_desc)) ** 2))

    d_hi_asc = jnp.clip(d_out, 0.0, sdet_abs)
    d_lo_asc = jnp.clip(d_in, 0.0, sdet_abs)
    len_asc = jnp.where(upgoing, d_hi_asc - d_lo_asc, 0.0)
    rho_asc = prem_density_jax(jnp.sqrt(rmin2 + (0.5 * (d_hi_asc + d_lo_asc)) ** 2))

    # Atmospheric vacuum segment: descending part with radius in [R_E, r_prod].
    d_re = _d(R_EARTH_KM**2 - rmin2)
    len_atm = s_prod - jnp.clip(d_re, d_low, s_prod)

    # Source -> detector: atmosphere, descending (outer->inner), ascending.
    one = jnp.ones((1,))
    lengths = jnp.concatenate([len_atm[None], len_desc[::-1], len_asc])
    rho_seg = jnp.concatenate([0.0 * one, rho_desc[::-1], rho_asc])
    ye_seg = jnp.concatenate([YE_MANTLE_DEFAULT * one, ye[::-1], ye])
    return rho_seg, ye_seg, lengths


def earth_segments(cz, table: ShellTable, det_depth_km=0.0):
    """Earth-only path segments (``h_atm_km = 0``); see :func:`chord_segments`."""
    return chord_segments(cz, table, h_atm_km=0.0, det_depth_km=det_depth_km)


def baseline_km(cz, det_depth_km=0.0):
    """Total in-Earth chord length (km) for ``cos(zenith) = cz``."""
    r_det = R_EARTH_KM - det_depth_km
    cz = jnp.asarray(cz, dtype=jnp.float64)
    r_min2 = r_det**2 * jnp.clip(1.0 - cz**2, 0.0, None)
    desc = jnp.sqrt(jnp.clip(R_EARTH_KM**2 - r_min2, 0.0, None))
    asc = jnp.sqrt(jnp.clip(r_det**2 - r_min2, 0.0, None))
    return jnp.where(cz < 0.0, desc + asc, 0.0)


def prem_density_jax(r_km):
    """JAX-traceable PREM density (g/cm^3) at radius ``r_km`` (0 above R_E)."""
    r = jnp.asarray(r_km, dtype=jnp.float64)
    x = r / R_EARTH_KM
    rho = jnp.zeros_like(r)
    prev = -1.0
    for outer, coeffs, _ye in _PREM:
        val = sum(c * x**i for i, c in enumerate(coeffs))
        in_region = (r > prev) & (r <= outer)
        rho = jnp.where(in_region, val, rho)
        prev = outer
    return rho


