"""Solar matter profile and adiabatic MSW evolution.

Provides a standard-solar-model electron-density profile (loaded from a BS05-style
table, or an analytic exponential approximation) and the **adiabatic** mass-state
composition of a neutrino produced inside the Sun -- the mechanism behind the LMA
solar solution. Idea adapted from the nu-waves library.

In the adiabatic regime a neutrino produced as flavor ``alpha`` populates the
instantaneous matter eigenstates with fixed weights ``w_k = |<nu_k^m|nu_alpha>|^2``
(no level hopping); the observable vacuum mass-state fractions then evolve only
because the matter eigenstates rotate as the density drops:

    F_i(r) = sum_k w_k |<nu_i^vac | nu_k^m(r)>|^2 .
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp

from . import constants as C
from .hamiltonian import matter_hamiltonian

R_SUN_KM = 695700.0

SolarProfile = namedtuple("SolarProfile", ["r_over_rsun", "rho_ye", "R_sun_km"])


def load_bs05(path):
    """Load a BS05(-AGS,OP) standard solar model table.

    Returns a :class:`SolarProfile` with ``rho_ye = rho * Y_e`` (g/cm^3), using
    ``Y_e = (1 + X)/2`` from the hydrogen mass fraction X (column 7); radius is
    column 2 (r/R_sun), density column 4.
    """
    rows = []
    started = False
    with open(path) as f:
        for line in f:
            if not started:
                if line.strip().startswith("The Table begins"):
                    started = True
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                vals = [float(p) for p in parts[:7]]
            except ValueError:
                continue
            rows.append(vals)
    arr = np.array(rows)
    r = arr[:, 1]
    rho = arr[:, 3]
    X = arr[:, 6]
    ye = 0.5 * (1.0 + X)
    return SolarProfile(r_over_rsun=r, rho_ye=rho * ye, R_sun_km=R_SUN_KM)


def exponential_profile(ne0_over_NA=245.0, scale=10.54, n=400):
    """Bahcall exponential approximation ``n_e = ne0 * exp(-scale * r/R_sun)``.

    ``ne0_over_NA`` is the central electron density in units of N_A per cm^3, so
    ``rho_ye = n_e / N_A`` (g/cm^3-equivalent) ``= ne0_over_NA * exp(...)``.
    """
    r = np.linspace(0.0, 1.0, n)
    rho_ye = ne0_over_NA * np.exp(-scale * r)
    return SolarProfile(r_over_rsun=r, rho_ye=rho_ye, R_sun_km=R_SUN_KM)


def potential_eV(profile, r_km):
    """Matter potential V (eV) at radius ``r_km`` by interpolating the profile."""
    x = jnp.asarray(r_km) / profile.R_sun_km
    rho_ye = jnp.interp(x, jnp.asarray(profile.r_over_rsun),
                        jnp.asarray(profile.rho_ye))
    return C.matter_potential_eV(rho_ye, 1.0)


def adiabatic_mass_fractions(params, energy_GeV, profile, r_km,
                             r_emit_km, alpha=0, anti=False):
    """Vacuum mass-state fractions ``F_i(r)`` under adiabatic solar evolution.

    ``r_km`` is an array of radii (km) at which to evaluate; ``r_emit_km`` is the
    production radius. Returns array of shape ``(len(r_km), N)``.
    """
    u = params.pmns()
    if anti:
        u = jnp.conj(u)
    msq = params.msquared()
    energy_eV = energy_GeV * C.GEV_TO_EV

    def eigvecs(v_eV):
        h = matter_hamiltonian(u, msq, energy_eV, v_eV, anti=anti)
        _w, vecs = jnp.linalg.eigh(h)
        return vecs  # columns = matter eigenstates (flavor basis), asc. eigenvalue

    # Production weights w_k = |<nu_k^m(r_emit) | nu_alpha>|^2.
    v_emit = potential_eV(profile, r_emit_km)
    vecs_emit = eigvecs(v_emit)
    w = jnp.abs(vecs_emit[alpha, :]) ** 2  # (N,)

    udag = jnp.conj(u).T

    def frac_at(r):
        vecs = eigvecs(potential_eV(profile, r))
        a = udag @ vecs  # <nu_i^vac | nu_k^m> = (U^dag V)[i,k]
        return (jnp.abs(a) ** 2) @ w  # F_i = sum_k |a_ik|^2 w_k

    return jax.vmap(frac_at)(jnp.asarray(r_km))
