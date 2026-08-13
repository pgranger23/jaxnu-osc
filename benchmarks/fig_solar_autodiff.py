"""fig_solar_autodiff.py: Solar MSW survival probability, 2D interior sensitivity,
and exact automatic differentiation landscape.

Produces Figure 5 of the jaxnu paper:
  (a) Solar P_ee(E) survival probability across 0.1 - 15 MeV, showing the transition
      from vacuum-averaged (0.55) to matter-dominated (0.30) regimes, along with
      exact AD derivatives w.r.t theta_12 (broad sensitivity) and dm^2_21 (concentrated
      resonance peak at E ~ 4.4 MeV).
  (b) 2D functional gradient map dP_ee / d(r_emit) across (E, r_emit), mapping the
      MSW resonance band inside the solar core (0.01 - 0.35 R_sun).

Run from repository root:
    python benchmarks/fig_solar_autodiff.py
    FIGONLY=1 python benchmarks/fig_solar_autodiff.py   # re-render from saved npz
"""

import dataclasses
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import jaxnu
from jaxnu import solar, nufit_no

# Output directories
BENCH_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PAPER_FIGS = "/Users/pgranger/jaxnu-paper/figures"
os.makedirs(BENCH_OUT, exist_ok=True)
os.makedirs(PAPER_FIGS, exist_ok=True)

NPZ_FILE = os.path.join(BENCH_OUT, "fig_solar_autodiff.npz")
FIG_PDF = os.path.join(PAPER_FIGS, "fig_solar_autodiff.pdf")
FIG_PNG = os.path.join(PAPER_FIGS, "fig_solar_autodiff.png")

FIGONLY = os.environ.get("FIGONLY", "") not in ("", "0")

if not FIGONLY or not os.path.exists(NPZ_FILE):
    print("Computing solar probability and gradient maps...", flush=True)
    t0 = time.time()

    p = nufit_no()
    bs05_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "data", "bs05_agsop.dat")
    if os.path.exists(bs05_path):
        prof = solar.load_bs05(bs05_path)
    else:
        prof = solar.exponential_profile()

    r_core_default = 0.05 * prof.R_sun_km

    # 1. 1D Energy Sweep (0.1 to 15 MeV)
    energies_MeV = jnp.logspace(-1, 1.2, 250)  # 250 points

    def calc_1d(e_mev):
        e_gev = e_mev * 1e-3
        # Forward Pee
        def pee_th12(th12):
            p_mod = dataclasses.replace(p, theta12=th12)
            F = solar.adiabatic_mass_fractions(p_mod, e_gev, prof, prof.R_sun_km, r_core_default, alpha=0)
            U = p_mod.pmns()
            return jnp.sum(F * jnp.abs(U[0]) ** 2)

        def pee_dm21(dm21):
            p_mod = dataclasses.replace(p, dm21=dm21)
            F = solar.adiabatic_mass_fractions(p_mod, e_gev, prof, prof.R_sun_km, r_core_default, alpha=0)
            U = p_mod.pmns()
            return jnp.sum(F * jnp.abs(U[0]) ** 2)

        P = pee_th12(p.theta12)
        g_th12 = jax.grad(pee_th12)(p.theta12)
        g_dm21 = jax.grad(pee_dm21)(p.dm21)
        return P, g_th12, g_dm21

    res_1d = jax.vmap(calc_1d)(energies_MeV)
    P_1d = np.array(res_1d[0])
    g_th12_1d = np.array(res_1d[1])
    g_dm21_1d = np.array(res_1d[2])

    # 2. 2D Energy x Radius Map
    energies_2d = jnp.logspace(-1, 1.2, 120)
    radii_ratio = jnp.linspace(0.01, 0.35, 100)  # r/R_sun

    def pee_at_er(e_mev, r_ratio):
        e_gev = e_mev * 1e-3
        r_km = r_ratio * prof.R_sun_km
        F = solar.adiabatic_mass_fractions(p, e_gev, prof, prof.R_sun_km, r_km, alpha=0)
        U = p.pmns()
        return jnp.sum(F * jnp.abs(U[0]) ** 2)

    # Gradient w.r.t r_emit (solar core radius)
    def grad_r(e_mev, r_ratio):
        return jax.grad(lambda r: pee_at_er(e_mev, r))(r_ratio)

    g_2d_r = jax.vmap(lambda e: jax.vmap(lambda r: grad_r(e, r))(radii_ratio))(energies_2d)
    g_2d_r = np.array(g_2d_r)

    np.savez(
        NPZ_FILE,
        energies_MeV=np.array(energies_MeV),
        P_1d=P_1d,
        g_th12_1d=g_th12_1d,
        g_dm21_1d=g_dm21_1d,
        energies_2d=np.array(energies_2d),
        radii_ratio=np.array(radii_ratio),
        g_2d_r=g_2d_r,
    )
    print(f"Data computed in {time.time() - t0:.2f}s and saved to {NPZ_FILE}", flush=True)

else:
    print(f"Loading precomputed data from {NPZ_FILE}...", flush=True)

data = np.load(NPZ_FILE)
energies_MeV = data["energies_MeV"]
P_1d = data["P_1d"]
g_th12_1d = data["g_th12_1d"]
g_dm21_1d = data["g_dm21_1d"]
energies_2d = data["energies_2d"]
radii_ratio = data["radii_ratio"]
g_2d_r = data["g_2d_r"]

# --- Plotting ---
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9.5,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.titlesize": 11,
    }
)

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(6.8, 6.8), sharex=False, gridspec_kw={"height_ratios": [1.1, 1.1, 1.3]}
)
fig.subplots_adjust(hspace=0.42, left=0.11, right=0.93, top=0.95, bottom=0.08)

# Panel 1: Survival probability P_ee(E)
ax1.plot(energies_MeV, P_1d, color="#1f77b4", lw=2.0, label=r"Solar $P_{ee}(E)$ (LMA adiabatic)")
ax1.axhline(0.55, color="gray", ls="--", lw=0.8, alpha=0.7, label=r"Vacuum average $\sum |U_{ei}|^4 \approx 0.55$")
ax1.axhline(0.30, color="gray", ls=":", lw=0.8, alpha=0.7, label=r"Matter limit $\cos^4\theta_{13}\sin^2\theta_{12} + \sin^4\theta_{13} \approx 0.30$")

ax1.set_xscale("log")
ax1.set_xlim(0.1, 15.0)
ax1.set_ylim(0.22, 0.62)
ax1.set_xlabel(r"Neutrino Energy $E$ [MeV]")
ax1.set_ylabel(r"Survival Prob. $P_{ee}$")
ax1.grid(True, which="both", ls="-", lw=0.3, alpha=0.5)
ax1.legend(loc="upper right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none")

# Annotations for solar flux components (above the curve for clarity)
ax1.text(0.2, 0.58, r"$pp$", fontsize=8.5, fontweight="bold", color="#2ca02c", ha="center")
ax1.text(0.86, 0.56, r"$^7\mathrm{Be}$", fontsize=8.5, fontweight="bold", color="#d62728", ha="center")
ax1.text(1.44, 0.50, r"$pep$", fontsize=8.5, fontweight="bold", color="#9467bd", ha="center")
ax1.text(7.5, 0.37, r"$^8\mathrm{B}$", fontsize=8.5, fontweight="bold", color="#8c564b", ha="center")
ax1.set_title(r"(a) Solar MSW survival probability $P_{ee}(E)$", loc="left", pad=4, fontweight="bold")

# Panel 2: Exact AD Derivatives w.r.t theta12 and dm21
ax2_twin = ax2.twinx()

line1 = ax2.plot(energies_MeV, g_th12_1d, color="#1f77b4", lw=1.8, label=r"$\partial P_{ee} / \partial \theta_{12}$ (rad$^{-1}$)")
line2 = ax2_twin.plot(energies_MeV, g_dm21_1d, color="#d62728", lw=1.8, ls="--", label=r"$\partial P_{ee} / \partial (\Delta m^2_{21})$ ($\mathrm{eV}^{-2}$)")

ax2.set_xscale("log")
ax2.set_xlim(0.1, 15.0)
ax2.set_xlabel(r"Neutrino Energy $E$ [MeV]")
ax2.set_ylabel(r"$\partial P_{ee} / \partial \theta_{12}$", color="#1f77b4")
ax2_twin.set_ylabel(r"$\partial P_{ee} / \partial (\Delta m^2_{21})$ [$\mathrm{eV}^{-2}$]", color="#d62728")

ax2.tick_params(axis="y", labelcolor="#1f77b4")
ax2_twin.tick_params(axis="y", labelcolor="#d62728")
ax2.grid(True, which="both", ls="-", lw=0.3, alpha=0.5)

# Peak annotation for dm21
idx_peak = np.argmax(g_dm21_1d)
e_peak = energies_MeV[idx_peak]
g_peak = g_dm21_1d[idx_peak]
ax2_twin.annotate(
    f"MSW Resonance Spike\nPeak: {g_peak:.1e} eV$^{{-2}}$\nat $E = {e_peak:.2f}$ MeV",
    xy=(e_peak, g_peak),
    xytext=(1.2, g_peak * 0.7),
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    fontsize=8,
    color="#d62728",
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", fc="#fbe8e8", ec="#d62728", lw=0.8),
)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc="upper right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none")
ax2.set_title(r"(b) Parameter sensitivities: broad $\theta_{12}$ vs localized $\Delta m^2_{21}$ resonance peak", loc="left", pad=4, fontweight="bold")

# Panel 3: 2D Solar Interior Core Sensitivity Map
cmap = plt.cm.YlOrRd

pc = ax3.pcolormesh(radii_ratio, energies_2d, np.abs(g_2d_r), cmap=cmap, shading="gouraud")
cbar = fig.colorbar(pc, ax=ax3, orientation="vertical", pad=0.02, aspect=15)
cbar.set_label(r"$|\partial P_{ee} / \partial (r_{\mathrm{emit}} / R_\odot)|$", fontsize=9)

ax3.set_yscale("log")
ax3.set_ylim(0.1, 15.0)
ax3.set_xlim(0.01, 0.35)
ax3.set_xlabel(r"Solar Production Radius $r_{\mathrm{emit}} / R_\odot$")
ax3.set_ylabel(r"Neutrino Energy $E$ [MeV]")
ax3.grid(True, which="both", ls=":", lw=0.3, alpha=0.5, color="gray")

ax3.set_title(r"(c) Functional core sensitivity map $|\partial P_{ee} / \partial (r_{\mathrm{emit}}/R_\odot)|$ across solar interior", loc="left", pad=4, fontweight="bold")

# Save outputs
fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
print(f"Saved figure to {FIG_PDF} and {FIG_PNG}")
