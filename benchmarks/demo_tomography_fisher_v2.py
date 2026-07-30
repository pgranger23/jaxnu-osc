"""Prototype: how much more realistic can the tomography demo get cheaply?

Two changes on top of ``demo_tomography_fisher.py``, both switchable so the
four combinations can be compared honestly:

  (a) ANTINEUTRINOS  -- a second (E, cos theta) sample with ``anti=True``,
      its own flux normalization, sharing theta23/dm31/density with the
      neutrino sample.  Toggle: INCLUDE_ANTINU.

  (b) RICHER DENSITY -- replace the two multiplicative scale factors
      (core, mantle) with N_ZONES independent radial density scale factors,
      built directly on jaxnu.earth.LayeredEarth (fully differentiable in
      per-shell density).  Toggle: N_ZONES in {2, 6}.  N_ZONES=2 uses the
      same core/mantle split as the original script, as a sanity check that
      this rewrite reproduces the original numbers before turning on the
      real change.

Everything else (grid, flux model, response, Fisher/Schur machinery, figure)
is left as close to the original as possible so the *only* things that move
are the two changes under test.

Run:  N_ZONES=6 INCLUDE_ANTINU=1 python demo_tomography_fisher_v2.py
      (defaults: N_ZONES=6, INCLUDE_ANTINU=1 -- the realistic configuration)
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import dataclasses

import jaxnu
import jaxnu.oscillator as _osc
import jaxnu.earth as _earth
from jaxnu import constants as C, nufit_no

# --- toggles under test -------------------------------------------------------
N_ZONES = int(os.environ.get("N_ZONES", "6"))
INCLUDE_ANTINU = bool(int(os.environ.get("INCLUDE_ANTINU", "1")))

# --- toy exposure (unchanged from v1) -----------------------------------------
N_EVENTS = 1.0e5          # nu_mu events in the sample (nubar is on top of this)
NUBAR_FRAC = 0.5          # toy nubar/nu event-count ratio (order-of-magnitude
                           # atmospheric nu_mu:nubar_mu correct; not a fit target)
E_MIN, E_MAX, N_E = 1.0, 30.0, 30       # GeV
CZ_MIN, CZ_MAX, N_CZ = -1.0, -0.05, 30  # up-going only
GAMMA, E_PIVOT = 2.7, 10.0              # atmospheric power law
H_ATM_KM = 15.0
CORE_CZ = -0.8376384419020163           # chord grazes the core-mantle boundary
N_BINS = N_E * N_CZ

_P0 = nufit_no()

# --- zoned, fully differentiable Earth (jaxnu.earth.LayeredEarth) ------------
# n_sub=4 matches the shell count used by v1's shell_table(4) (43 shells), so
# the per-bin propagation cost is unchanged; only the number of *free
# parameters* controlling the shell densities changes.
_BASE_EARTH = _earth.prem_layered(n_sub=4)


def _zone_edges(n_zones):
    if n_zones == 2:
        # same split as v1: core / mantle, boundary at the core-mantle radius
        return np.array([0.0, _earth.CORE_RADIUS_KM, C.R_EARTH_KM])
    if n_zones == 6:
        # inner core / outer core / lower mantle / transition zone /
        # upper mantle / crust -- PREM region boundaries grouped into six
        # radial zones (referee-facing: this is the "handful of radial
        # zones" the referee asked for)
        return np.array([0.0, 1221.5, 3480.0, 5701.0, 5971.0, 6346.6,
                         C.R_EARTH_KM])
    raise ValueError(f"no zone table defined for N_ZONES={n_zones}")


def _zone_index(base_earth, n_zones):
    """Static (numpy) per-shell zone index, built once from shell mid-radii."""
    outer = np.asarray(base_earth.outer)
    inner = np.concatenate([[0.0], outer[:-1]])
    mid = 0.5 * (inner + outer)
    edges = _zone_edges(n_zones)
    idx = np.searchsorted(edges, mid, side="right") - 1
    return jnp.asarray(np.clip(idx, 0, n_zones - 1))


_ZONE_IDX = _zone_index(_BASE_EARTH, N_ZONES)
ZONE_LABELS = {
    2: ["ln rho_core", "ln rho_mantle"],
    6: ["ln rho_inner_core", "ln rho_outer_core", "ln rho_lower_mantle",
        "ln rho_transition_zone", "ln rho_upper_mantle", "ln rho_crust"],
}[N_ZONES]

# parameter vector: [ln_sc_zone_0 .. ln_sc_zone_{N-1}, theta23, dm31,
#                     ln Phi0_nu, dgamma, ln Phi0_nubar]
N_DENS = N_ZONES
IDX_TH23, IDX_DM31, IDX_PHI, IDX_DGAMMA, IDX_PHIBAR = (
    N_DENS, N_DENS + 1, N_DENS + 2, N_DENS + 3, N_DENS + 4)
N_PAR = N_DENS + 5 if INCLUDE_ANTINU else N_DENS + 4

THETA0 = jnp.zeros(N_PAR).at[IDX_TH23].set(float(_P0.theta23)) \
                         .at[IDX_DM31].set(float(_P0.dm31))
LABELS = (ZONE_LABELS + ["theta23", "dm31", "ln Phi0_nu", "dgamma"]
          + (["ln Phi0_nubar"] if INCLUDE_ANTINU else []))


def _bin_centres():
    e_edges = np.logspace(np.log10(E_MIN), np.log10(E_MAX), N_E + 1)
    cz_edges = np.linspace(CZ_MIN, CZ_MAX, N_CZ + 1)
    e_c = np.sqrt(e_edges[:-1] * e_edges[1:])
    cz_c = 0.5 * (cz_edges[:-1] + cz_edges[1:])
    de = np.diff(e_edges)
    dcz = np.diff(cz_edges)
    E, CZ = np.meshgrid(e_c, cz_c, indexing="ij")
    dE, dCZ = np.meshgrid(de, dcz, indexing="ij")
    return (jnp.asarray(E.ravel()), jnp.asarray(CZ.ravel()),
            jnp.asarray((dE * dCZ).ravel()))


E_C, CZ_C, DOMEGA = _bin_centres()


def _prob_mumu(e_gev, cz, ln_sc, params, anti):
    """P(nu_mu -> nu_mu) [or nubar_mu -> nubar_mu] with per-zone density scale."""
    sc = jnp.exp(ln_sc)                          # (N_ZONES,)
    density = _BASE_EARTH.density * sc[_ZONE_IDX]
    model = dataclasses.replace(_BASE_EARTH, density=density)
    rho, ye, L = _earth.layered_chord_segments(model, cz, h_atm_km=H_ATM_KM,
                                                det_depth_km=0.0)
    v_cc, _ = C.matter_potentials(rho, ye)
    s = _osc.propagate_layers(params.pmns(), params.msquared(),
                              e_gev * C.GEV_TO_EV, v_cc,
                              L * C.KM_TO_INV_EV, anti=anti, backend="cayley")
    return _osc.prob_from_amplitude(s)[1, 1]


def _shape_nu(theta):
    ln_sc = theta[:N_DENS]
    th23, dm31, ln_phi, dgamma = (theta[IDX_TH23], theta[IDX_DM31],
                                  theta[IDX_PHI], theta[IDX_DGAMMA])
    p = dataclasses.replace(_P0, theta23=th23, dm31=dm31)
    prob = jax.vmap(_prob_mumu, in_axes=(0, 0, None, None, None))(
        E_C, CZ_C, ln_sc, p, False)
    flux = jnp.exp(ln_phi) * (E_C / E_PIVOT) ** (-(GAMMA + dgamma))
    return flux * prob * DOMEGA


def _shape_nubar(theta):
    ln_sc = theta[:N_DENS]
    th23, dm31, dgamma, ln_phibar = (theta[IDX_TH23], theta[IDX_DM31],
                                     theta[IDX_DGAMMA], theta[IDX_PHIBAR])
    p = dataclasses.replace(_P0, theta23=th23, dm31=dm31)
    prob = jax.vmap(_prob_mumu, in_axes=(0, 0, None, None, None))(
        E_C, CZ_C, ln_sc, p, True)
    # spectral index shared with nu (kept minimal); normalization independent
    flux = jnp.exp(ln_phibar) * (E_C / E_PIVOT) ** (-(GAMMA + dgamma))
    return flux * prob * DOMEGA


_NORM_NU = float(N_EVENTS / jnp.sum(_shape_nu(THETA0)))
_NORM_BAR = (float(NUBAR_FRAC * N_EVENTS / jnp.sum(_shape_nubar(THETA0)))
            if INCLUDE_ANTINU else 0.0)

RES_LNE = 0.20
RES_CZ = 0.10


def response(res_lne, res_cz):
    dlne = jnp.log(E_C)[:, None] - jnp.log(E_C)[None, :]
    dcz = CZ_C[:, None] - CZ_C[None, :]
    R = jnp.exp(-0.5 * (dlne / res_lne) ** 2 - 0.5 * (dcz / res_cz) ** 2)
    return R / jnp.sum(R, axis=0, keepdims=True)


def counts(theta, res_lne=RES_LNE, res_cz=RES_CZ):
    R = response(res_lne, res_cz)
    c_nu = R @ (_NORM_NU * _shape_nu(theta))
    if not INCLUDE_ANTINU:
        return c_nu
    c_bar = R @ (_NORM_BAR * _shape_nubar(theta))
    return jnp.concatenate([c_nu, c_bar])


def fisher(theta, res_lne=RES_LNE, res_cz=RES_CZ):
    mu = counts(theta, res_lne, res_cz)
    J = jax.jacfwd(counts)(theta, res_lne, res_cz)
    return J.T @ (J / mu[:, None]), mu, J


def sigma_core(res_cz, res_lne=RES_LNE):
    """Marginalized sigma on the FIRST density parameter vs angular resolution."""
    mu = counts(THETA0, res_lne, res_cz)
    J = jax.jacfwd(counts)(THETA0, res_lne, res_cz)
    F = J.T @ (J / mu[:, None])
    return jnp.sqrt(jnp.linalg.inv(F)[0, 0])


def main():
    print(f"N_ZONES={N_ZONES}  INCLUDE_ANTINU={INCLUDE_ANTINU}  "
          f"n_par={N_PAR}  n_bins={N_BINS * (2 if INCLUDE_ANTINU else 1)}")
    F, mu, J = fisher(THETA0)
    F = np.asarray(F)
    cov = np.linalg.inv(F)

    print(f"total events: {float(jnp.sum(mu)):.1f}")
    print("\n                          sigma (stat only)")
    print("parameter                fixed nuisances   marginalized    penalty")
    for i in range(N_DENS):
        fixed = 1.0 / np.sqrt(F[i, i])
        marg = np.sqrt(cov[i, i])
        print(f"{LABELS[i]:<24s} {fixed:>13.4f}   {marg:>12.4f}   "
              f"{marg/fixed:>8.2f}x")

    if N_DENS > 2:
        # bulk-core comparable to v1's single ln rho_core: sum of the
        # innermost two zones (inner + outer core), the same physical region
        # v1's density>9 g/cc mask selected.
        c = np.zeros(N_PAR)
        c[0] = c[1] = 1.0
        sigma_bulk_core = float(np.sqrt(c @ cov @ c))
        print(f"\nbulk core (zones 0+1 combined, marginalized): "
              f"{sigma_bulk_core:.4f}  <- compare directly to v1's 0.0281")

        print("\ncorrelation matrix among density-zone parameters:")
        d = np.sqrt(np.diag(cov[:N_DENS, :N_DENS]))
        corr = cov[:N_DENS, :N_DENS] / np.outer(d, d)
        hdr = "".join(f"{l.replace('ln rho_',''):>16s}" for l in LABELS[:N_DENS])
        print("                " + hdr)
        for i in range(N_DENS):
            row = "".join(f"{corr[i,j]:>16.2f}" for j in range(N_DENS))
            print(f"{LABELS[i]:<16s}{row}")

    print(f"\nstat scaling: sqrt(1e5/N_events) applies as in v1")

    s0 = float(sigma_core(RES_CZ))
    dsdr = float(jax.grad(sigma_core)(RES_CZ))
    h = 1e-3
    fd = (float(sigma_core(RES_CZ + h)) - float(sigma_core(RES_CZ - h))) / (2 * h)
    print(f"\nexperimental-design derivative (zone 0, through the matrix inverse):")
    print(f"  sigma at res_cz={RES_CZ}                    : {s0:.5f}")
    print(f"  d sigma / d res_cz  (autodiff)              : {dsdr:+.5f}")
    print(f"  d sigma / d res_cz  (central differences)   : {fd:+.5f}")


if __name__ == "__main__":
    main()
