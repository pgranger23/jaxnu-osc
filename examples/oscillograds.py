"""Oscillograds: maps of the gradient of the oscillation probability.

An *oscillograd* (arXiv:2512.16427) plots the partial derivative d P / d zeta of an
oscillation probability with respect to an oscillation parameter zeta, revealing
where in parameter space a probability is most sensitive to that parameter.

This is a natural showcase for a differentiable simulator: `jax.jacrev` gives the
oscillograd for *every* parameter in a single reverse-mode pass, and we can check
it rigorously against finite differences.

Produces three figures:
  1. oscillograds_lbl.jpg    -- P and d/d{deltaCP, theta23, dm31} over (E, deltaCP)
                                at the DUNE baseline (paper-style, constant density)
  2. oscillograds_atm.jpg    -- the same but over (E, cos theta_z) through the PREM
                                Earth: differentiable atmospheric oscillograds
  3. (stdout) autodiff vs central finite differences for all 6 parameters
"""

import sys
import dataclasses
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

import jaxnu
from jaxnu import OscParams, Flavor, probability_constant, probability_earth

OUT = Path(__file__).resolve().parent

# NuFIT/PDG-like normal-ordering point (nu-waves convention).
BASE = OscParams(
    theta12=jnp.asarray(np.deg2rad(33.4)),
    theta13=jnp.asarray(np.deg2rad(8.6)),
    theta23=jnp.asarray(np.deg2rad(49.0)),
    deltacp=jnp.asarray(np.deg2rad(195.0)),
    dm21=jnp.asarray(7.42e-5),
    dm31=jnp.asarray(2.517e-3),
)


# ---------------------------------------------------------------------------
# 1) Differentiability validation: autodiff vs central finite differences
# ---------------------------------------------------------------------------
def validate():
    print("=== Differentiability check: autodiff vs central finite differences ===")
    fields = ["theta12", "theta13", "theta23", "deltacp", "dm21", "dm31"]
    steps = {"theta12": 1e-5, "theta13": 1e-5, "theta23": 1e-5,
             "deltacp": 1e-5, "dm21": 1e-9, "dm31": 1e-8}

    cases = {
        "constant density (L=1300 km, rho=2.85)":
            lambda p: probability_constant(p, jnp.asarray(0.9), 1300.0, density=2.85,
                                           flavor_in=Flavor.MU, flavor_out=Flavor.E),
        "PREM Earth (E=4 GeV, cos z=-1)":
            lambda p: probability_earth(p, jnp.asarray(4.0), jnp.asarray(-1.0),
                                        flavor_in=Flavor.MU, flavor_out=Flavor.E),
    }
    for name, f in cases.items():
        grad = jax.grad(f)(BASE)
        print(f"\n  {name}")
        worst = 0.0
        for fld in fields:
            ad = float(getattr(grad, fld))
            h = steps[fld]
            pp = dataclasses.replace(BASE, **{fld: getattr(BASE, fld) + h})
            pm = dataclasses.replace(BASE, **{fld: getattr(BASE, fld) - h})
            fd = (float(f(pp)) - float(f(pm))) / (2 * h)
            rel = abs(ad - fd) / (abs(fd) + 1e-12)
            worst = max(worst, rel)
            print(f"    dP/d{fld:8s}  autodiff={ad:+.6e}  finite-diff={fd:+.6e}  rel={rel:.1e}")
        print(f"    -> worst relative error: {worst:.1e}")


# ---------------------------------------------------------------------------
# helpers: oscillograds via jacrev over a grid
# ---------------------------------------------------------------------------
def _params_with_dcp(dcp):
    return dataclasses.replace(BASE, deltacp=dcp)


def lbl_grids(E, dcp, L_km=1300.0, density=2.848):
    """P and dP/d{dcp, th23, dm31} over the (E, deltaCP) plane (constant density)."""
    def cell(e, d):
        p = _params_with_dcp(d)
        fn = lambda pp: probability_constant(pp, e, L_km, density=density,
                                             flavor_in=Flavor.MU, flavor_out=Flavor.E)
        val = fn(p)
        jac = jax.jacrev(fn)(p)
        return val, jac.deltacp, jac.theta23, jac.dm31
    f = jax.jit(lambda E, dcp: jax.vmap(lambda d: jax.vmap(lambda e: cell(e, d))(E))(dcp))
    return f(E, dcp)


def atm_grids(E, cz):
    """P and dP/d{dcp, th23, th13, dm31} over (E, cos z) through the PREM Earth."""
    def cell(e, c):
        fn = lambda pp: probability_earth(pp, e, c, n_sub=3,
                                          flavor_in=Flavor.MU, flavor_out=Flavor.E)
        val = fn(BASE)
        jac = jax.jacrev(fn)(BASE)
        return val, jac.deltacp, jac.theta23, jac.theta13, jac.dm31
    f = jax.jit(lambda E, cz: jax.vmap(lambda c: jax.vmap(lambda e: cell(e, c))(E))(cz))
    return f(E, cz)


def _panel(ax, X, Y, Z, title, diverging=True):
    if diverging:
        m = float(np.max(np.abs(Z))) or 1.0
        pc = ax.pcolormesh(X, Y, Z, cmap="RdBu_r", vmin=-m, vmax=m, shading="auto")
    else:
        pc = ax.pcolormesh(X, Y, Z, cmap="viridis", vmin=0, shading="auto")
    ax.set_title(title, fontsize=11)
    plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
    return pc


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
validate()

# --- LBL oscillograds (paper-style: (E, deltaCP) plane, DUNE baseline) ---
nE, nD = 240, 200
E = np.linspace(0.4, 5.0, nE)
dcp = np.linspace(0.0, 2 * np.pi, nD)
P, gD, g23, gM = (np.array(a) for a in lbl_grids(jnp.asarray(E), jnp.asarray(dcp)))
deg = np.rad2deg(dcp)
fig, axs = plt.subplots(1, 4, figsize=(18, 3.8), dpi=130, constrained_layout=True)
_panel(axs[0], E, deg, P, r"$P(\nu_\mu\to\nu_e)$", diverging=False)
_panel(axs[1], E, deg, gD, r"$\partial P/\partial\delta_{CP}$")
_panel(axs[2], E, deg, g23, r"$\partial P/\partial\theta_{23}$")
_panel(axs[3], E, deg, gM * 1e-3, r"$\partial P/\partial\Delta m^2_{31}\;[/10^{-3}\,\mathrm{eV}^2]$")
for ax in axs:
    ax.set_xlabel(r"$E_\nu$ [GeV]")
axs[0].set_ylabel(r"$\delta_{CP}$ [deg]")
fig.suptitle(r"Oscillograds, $\nu_\mu\to\nu_e$, DUNE-like $L=1300$ km (constant density)",
             fontsize=13)
fig.savefig(OUT / "oscillograds_lbl.jpg", dpi=130)
print("\nsaved", OUT / "oscillograds_lbl.jpg")

# --- Atmospheric oscillograds ((E, cos z) plane, PREM Earth) ---
nE2, nCZ = 200, 160
E2 = np.geomspace(0.5, 40.0, nE2)
cz = np.linspace(-1.0, -0.02, nCZ)
P2, gD2, g232, g132, gM2 = (np.array(a) for a in atm_grids(jnp.asarray(E2), jnp.asarray(cz)))
fig, axs = plt.subplots(1, 5, figsize=(22, 3.8), dpi=130, constrained_layout=True)
_panel(axs[0], E2, cz, P2, r"$P(\nu_\mu\to\nu_e)$", diverging=False)
_panel(axs[1], E2, cz, gD2, r"$\partial P/\partial\delta_{CP}$")
_panel(axs[2], E2, cz, g232, r"$\partial P/\partial\theta_{23}$")
_panel(axs[3], E2, cz, g132, r"$\partial P/\partial\theta_{13}$")
_panel(axs[4], E2, cz, gM2 * 1e-3, r"$\partial P/\partial\Delta m^2_{31}\;[/10^{-3}\,\mathrm{eV}^2]$")
for ax in axs:
    ax.set_xscale("log")
    ax.set_xlabel(r"$E_\nu$ [GeV]")
axs[0].set_ylabel(r"$\cos\theta_z$")
fig.suptitle(r"Atmospheric oscillograds, $\nu_\mu\to\nu_e$ through the PREM Earth "
             r"(differentiable matter propagation)", fontsize=13)
fig.savefig(OUT / "oscillograds_atm.jpg", dpi=130)
print("saved", OUT / "oscillograds_atm.jpg")
