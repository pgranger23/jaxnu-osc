"""fig_solar_autodiff.py: Standalone 2D functional sensitivity map of the solar interior core
|\partial P_{ee} / \partial (r_{\rm emit} / R_\odot)|.

Generates a focused, publication-quality single-panel figure showing how jaxnu's
automatic differentiation maps the MSW resonance band inside the solar core
(0.02 to 0.50 R_sun) across neutrino energies 0.1 - 30 MeV, annotated with
key solar neutrino flux regimes (pp, 7Be, pep, 8B, hep).

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
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter

import jaxnu
from jaxnu import solar, nufit_no, constants as C

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
    print("Computing 2D solar core gradient map with smooth C^2 profile (400x350 grid)...", flush=True)
    t0 = time.time()

    p = nufit_no()
    bs05_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "data", "bs05_agsop.dat")
    if os.path.exists(bs05_path):
        prof = solar.load_bs05(bs05_path)
    else:
        prof = solar.exponential_profile()

    # Use UnivariateSpline with mild smoothing to smooth out 4-digit rounding steps in bs05 table
    spl = UnivariateSpline(prof.r_over_rsun, prof.rho_ye, s=1e-3, k=3)
    r_knots_fine = np.linspace(0.0, 1.0, 500)
    rho_ye_smooth = spl(r_knots_fine)

    # Re-spline for exact C^2 evaluation
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(r_knots_fine, rho_ye_smooth)
    x_knots = jnp.asarray(cs.x)
    c_coeffs = jnp.asarray(cs.c)

    def jax_cubic_spline_eval(x):
        idx = jnp.searchsorted(x_knots, x, side="right") - 1
        idx = jnp.clip(idx, 0, len(x_knots) - 2)
        dx = x - x_knots[idx]
        return c_coeffs[0, idx] * dx**3 + c_coeffs[1, idx] * dx**2 + c_coeffs[2, idx] * dx + c_coeffs[3, idx]

    # High-resolution energy range: 400 log-spaced points from 0.1 to 30 MeV
    energies_2d = jnp.logspace(-1, 1.48, 400)
    # Physical core radius range: 0.02 to 0.50 R_sun (350 linear points)
    radii_ratio = jnp.linspace(0.02, 0.50, 350)

    u = p.pmns()
    msq = p.msquared()

    def pee_smooth_er(e_mev, r_ratio):
        e_gev = e_mev * 1e-3
        e_eV = e_gev * C.GEV_TO_EV

        # Smooth C^2 density potential at production radius
        rho_ye_emit = jax_cubic_spline_eval(r_ratio)
        v_emit = C.matter_potential_eV(rho_ye_emit, 1.0)

        h_emit = jaxnu.hamiltonian.matter_hamiltonian(u, msq, e_eV, v_emit)
        _, vecs_emit = jnp.linalg.eigh(h_emit)
        w = jnp.abs(vecs_emit[0, :]) ** 2

        # Surface potential
        rho_ye_surf = jax_cubic_spline_eval(1.0)
        v_surf = C.matter_potential_eV(rho_ye_surf, 1.0)
        h_surf = jaxnu.hamiltonian.matter_hamiltonian(u, msq, e_eV, v_surf)
        _, vecs_surf = jnp.linalg.eigh(h_surf)

        a = jnp.conj(u).T @ vecs_surf
        F = (jnp.abs(a) ** 2) @ w
        return jnp.sum(F * jnp.abs(u[0]) ** 2)

    def grad_r(e_mev, r_ratio):
        return jax.grad(lambda r: pee_smooth_er(e_mev, r))(r_ratio)

    # Vectorized evaluation over 140,000 points
    g_2d_r = jax.vmap(lambda e: jax.vmap(lambda r: grad_r(e, r))(radii_ratio))(energies_2d)
    g_2d_r = np.array(g_2d_r)

    np.savez(
        NPZ_FILE,
        energies_2d=np.array(energies_2d),
        radii_ratio=np.array(radii_ratio),
        g_2d_r=g_2d_r,
    )
    print(f"Data computed in {time.time() - t0:.2f}s and saved to {NPZ_FILE}", flush=True)

else:
    print(f"Loading precomputed data from {NPZ_FILE}...", flush=True)

data = np.load(NPZ_FILE)
energies_2d = data["energies_2d"]
radii_ratio = data["radii_ratio"]
g_2d_r = data["g_2d_r"]

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

cmap = plt.cm.YlOrRd

abs_g = np.abs(g_2d_r)

# High-resolution rendering
pc = ax.pcolormesh(radii_ratio, energies_2d, abs_g, cmap=cmap, vmin=0.0, vmax=1.35, shading="gouraud")

cbar = fig.colorbar(pc, ax=ax, orientation="vertical", pad=0.02, aspect=18)
cbar.set_label(r"Core sensitivity $|\partial P_{ee} / \partial (r_{\mathrm{emit}} / R_\odot)|$", fontsize=10.5)

# Smooth contours
cs = ax.contour(radii_ratio, energies_2d, abs_g, levels=[0.2, 0.4, 0.7, 0.9, 1.1, 1.25], colors="black", linewidths=0.5, alpha=0.35)

ax.set_yscale("log")
ax.set_ylim(0.1, 30.0)
ax.set_xlim(0.02, 0.50)
ax.set_xlabel(r"Solar Production Radius $r_{\mathrm{emit}} / R_\odot$", fontsize=11)
ax.set_ylabel(r"Neutrino Energy $E$ [MeV]", fontsize=11)
ax.grid(True, which="both", ls=":", lw=0.3, alpha=0.5, color="gray")

# Annotations for key solar flux boundaries and components
ax.axhline(0.42, color="#2ca02c", ls="--", lw=0.9, alpha=0.8)
ax.text(0.48, 0.45, r"$pp$ end (0.42 MeV)", color="#2ca02c", fontsize=8.5, ha="right", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

ax.axhline(0.86, color="#d62728", ls="--", lw=0.9, alpha=0.8)
ax.text(0.48, 0.92, r"$^7\mathrm{Be}$ (0.86 MeV)", color="#d62728", fontsize=8.5, ha="right", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

ax.axhline(1.44, color="#9467bd", ls="--", lw=0.9, alpha=0.8)
ax.text(0.48, 1.52, r"$pep$ (1.44 MeV)", color="#9467bd", fontsize=8.5, ha="right", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

ax.axhline(18.8, color="#8c564b", ls="--", lw=0.9, alpha=0.8)
ax.text(0.48, 19.8, r"$hep$ end (18.8 MeV)", color="#8c564b", fontsize=8.5, ha="right", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

# Annotate MSW resonance band
ax.annotate(
    "MSW Resonance Band\n(Peak core sensitivity)",
    xy=(0.17, 7.5),
    xytext=(0.04, 13.5),
    arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1.3),
    fontsize=9,
    color="#b2182b",
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", fc="#fbe8e8", ec="#b2182b", lw=0.85),
)

ax.set_title(r"Solar interior core sensitivity map $|\partial P_{ee} / \partial (r_{\mathrm{emit}}/R_\odot)|$", loc="left", pad=7, fontweight="bold")

# Save standalone outputs
fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
print(f"Saved standalone figure to {FIG_PDF} and {FIG_PNG}")
