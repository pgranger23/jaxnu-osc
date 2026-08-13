"""fig_solar_functional_tomography.py: Functional Solar Core Tomography Kernel
K(E, r) = d P_bar_ee / d ln N_e(r) via automatic differentiation in jaxnu.

Demonstrates continuous functional shape differentiation across 60 spatial radius shells
in a single reverse-mode VJP backward pass.

Run from repository root:
    python benchmarks/fig_solar_functional_tomography.py
    FIGONLY=1 python benchmarks/fig_solar_functional_tomography.py   # re-render from saved npz
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

import jaxnu
from jaxnu import solar, nufit_no, constants as C

# Output directories
BENCH_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PAPER_FIGS = "/Users/pgranger/jaxnu-paper/figures"
os.makedirs(BENCH_OUT, exist_ok=True)
os.makedirs(PAPER_FIGS, exist_ok=True)

NPZ_FILE = os.path.join(BENCH_OUT, "fig_solar_functional_tomography.npz")
FIG_PDF = os.path.join(PAPER_FIGS, "fig_solar_functional_tomography.pdf")
FIG_PNG = os.path.join(PAPER_FIGS, "fig_solar_functional_tomography.png")

FIGONLY = os.environ.get("FIGONLY", "") not in ("", "0")

if not FIGONLY or not os.path.exists(NPZ_FILE):
    print("Computing 60-shell Functional Core Tomography Kernel matrix (150x60 grid)...", flush=True)
    t0 = time.time()

    p = nufit_no()
    bs05_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "data", "bs05_agsop.dat")
    if os.path.exists(bs05_path):
        prof = solar.load_bs05(bs05_path)
    else:
        prof = solar.exponential_profile()

    # 60 spatial radial perturbation shells across the solar core
    r_grid = jnp.linspace(0.01, 0.35, 60)
    r_table = np.array(prof.r_over_rsun)
    rho_table = np.array(prof.rho_ye)
    log_rho = np.log(rho_table)
    poly_coeffs = jnp.asarray(np.polyfit(r_table, log_rho, 8))

    def get_profile_with_delta(r_ratio, delta_vec):
        rho_base = jnp.exp(jnp.polyval(poly_coeffs, r_ratio))
        delta = jnp.interp(r_ratio, r_grid, delta_vec)
        return rho_base * (1.0 + delta)

    u = p.pmns()
    msq = p.msquared()

    def pee_single(e_mev, r_emit, delta_vec):
        e_gev = e_mev * 1e-3
        e_eV = e_gev * C.GEV_TO_EV

        rho_ye_emit = get_profile_with_delta(r_emit, delta_vec)
        v_emit = C.matter_potential_eV(rho_ye_emit, 1.0)

        h_emit = jaxnu.hamiltonian.matter_hamiltonian(u, msq, e_eV, v_emit)
        _, vecs_emit = jnp.linalg.eigh(h_emit)
        w = jnp.abs(vecs_emit[0, :]) ** 2

        rho_ye_surf = get_profile_with_delta(1.0, delta_vec)
        v_surf = C.matter_potential_eV(rho_ye_surf, 1.0)
        h_surf = jaxnu.hamiltonian.matter_hamiltonian(u, msq, e_eV, v_surf)
        _, vecs_surf = jnp.linalg.eigh(h_surf)

        a = jnp.conj(u).T @ vecs_surf
        F = (jnp.abs(a) ** 2) @ w
        return jnp.sum(F * jnp.abs(u[0]) ** 2)

    # 8B core production distribution: Gaussian centered at r = 0.04 R_sun
    r_nodes = jnp.linspace(0.01, 0.15, 20)
    weights_8b = jnp.exp(-0.5 * ((r_nodes - 0.04) / 0.025) ** 2)
    weights_8b = weights_8b / jnp.sum(weights_8b)

    def pee_obs_8b(e_mev, delta_vec):
        p_vals = jax.vmap(lambda r: pee_single(e_mev, r, delta_vec))(r_nodes)
        return jnp.sum(weights_8b * p_vals)

    delta_zero = jnp.zeros(len(r_grid))
    energies = jnp.logspace(-1, 1.48, 150)

    # Reverse-mode VJP evaluation over 60 functional density parameters
    K_8b = jax.vmap(lambda e: jax.grad(lambda d: pee_obs_8b(e, d))(delta_zero))(energies)
    K_8b = np.array(K_8b)

    np.savez(
        NPZ_FILE,
        energies=np.array(energies),
        r_grid=np.array(r_grid),
        K_8b=K_8b,
    )
    print(f"Data computed in {time.time() - t0:.2f}s and saved to {NPZ_FILE}", flush=True)

else:
    print(f"Loading precomputed data from {NPZ_FILE}...", flush=True)

data = np.load(NPZ_FILE)
energies = data["energies"]
r_grid = data["r_grid"]
K_8b = data["K_8b"]

# --- Plotting Standalone Figure ---
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.titlesize": 11,
    }
)

fig, ax = plt.subplots(figsize=(6.8, 4.0))
fig.subplots_adjust(left=0.11, right=0.92, top=0.90, bottom=0.14)

cmap = plt.cm.Blues

abs_K = np.abs(K_8b)

pc = ax.pcolormesh(r_grid, energies, abs_K, cmap=cmap, shading="gouraud")

cbar = fig.colorbar(pc, ax=ax, orientation="vertical", pad=0.02, aspect=18)
cbar.set_label(r"Functional Tomography Kernel $\left|\frac{\delta \bar{P}_{ee}^{(^8\mathrm{B})}}{\delta \ln N_e(r)}\right|$", fontsize=10.5)

cs = ax.contour(r_grid, energies, abs_K, levels=[0.002, 0.005, 0.010, 0.015, 0.018], colors="navy", linewidths=0.5, alpha=0.4)

ax.set_yscale("log")
ax.set_ylim(0.1, 30.0)
ax.set_xlim(0.01, 0.35)
ax.set_xlabel(r"Solar Radial Position $r / R_\odot$", fontsize=11)
ax.set_ylabel(r"Neutrino Energy $E$ [MeV]", fontsize=11)
ax.grid(True, which="both", ls=":", lw=0.3, alpha=0.5, color="gray")

# Annotations
ax.axhline(0.42, color="#2ca02c", ls="--", lw=0.9, alpha=0.8)
ax.text(0.34, 0.45, r"$pp$ end", color="#2ca02c", fontsize=8.5, ha="right", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

ax.axhline(0.86, color="#d62728", ls="--", lw=0.9, alpha=0.8)
ax.text(0.34, 0.92, r"$^7\mathrm{Be}$", color="#d62728", fontsize=8.5, ha="right", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

ax.axhline(18.8, color="#8c564b", ls="--", lw=0.9, alpha=0.8)
ax.text(0.34, 19.8, r"$hep$ end", color="#8c564b", fontsize=8.5, ha="right", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

ax.annotate(
    "Functional Tomographic Window\n" + r"(60 radial shells computed in 1 VJP pass)",
    xy=(0.04, 4.4),
    xytext=(0.11, 2.2),
    arrowprops=dict(arrowstyle="->", color="#08519c", lw=1.3),
    fontsize=9,
    color="#08519c",
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", fc="#eff3ff", ec="#08519c", lw=0.85),
)

ax.set_title(r"Functional Solar Core Tomography Kernel $\delta \bar{P}_{ee}^{(^8\mathrm{B})} / \delta \ln N_e(r)$ via AD", loc="left", pad=7, fontweight="bold")

# Save standalone outputs
fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
print(f"Saved standalone functional core tomography figure to {FIG_PDF} and {FIG_PNG}")
