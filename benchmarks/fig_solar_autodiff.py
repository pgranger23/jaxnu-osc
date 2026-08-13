"""fig_solar_autodiff.py: Consolidated 2-Panel Solar Neutrino Automatic Differentiation Benchmark.

Combines:
  Panel (a): LMA Survival Probability P_ee(E), exact AD sensitivities (dP/dtheta12, dP/d(dm21),
             dP/d ln N0), and finite-difference validation markers (10^-10 agreement).
  Panel (b): 2D Functional Solar Core Tomography Kernel |dP_ee / d ln N_e(r)| across 80 spatial
             radius shells computed in a single reverse-mode VJP pass.

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
    print("Computing consolidated 2-panel solar AD benchmark data...", flush=True)
    t0 = time.time()

    p = nufit_no()
    bs05_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "data", "bs05_agsop.dat")
    if os.path.exists(bs05_path):
        prof = solar.load_bs05(bs05_path)
    else:
        prof = solar.exponential_profile()

    r_table = np.array(prof.r_over_rsun)
    rho_table = np.array(prof.rho_ye)
    log_rho = np.log(rho_table)
    poly_coeffs = jnp.asarray(np.polyfit(r_table, log_rho, 8))

    def smooth_rho_ye(r_ratio, n0_scale=1.0):
        return n0_scale * jnp.exp(jnp.polyval(poly_coeffs, r_ratio))

    def get_pmns(t12, t13, t23, delta):
        c12, s12 = jnp.cos(t12), jnp.sin(t12)
        c13, s13 = jnp.cos(t13), jnp.sin(t13)
        c23, s23 = jnp.cos(t23), jnp.sin(t23)
        cp = jnp.exp(-1j * delta)
        cm = jnp.exp(1j * delta)
        return jnp.array([
            [c12*c13, s12*c13, s13*cm],
            [-s12*c23 - c12*s23*s13*cp, c12*c23 - s12*s23*s13*cp, s23*c13],
            [s12*s23 - c12*c23*s13*cp, -c12*s23 - s12*c23*s13*cp, c23*c13]
        ])

    def pee(e_mev, t12, dm21, n0_scale=1.0, r_emit=0.05):
        e_gev = e_mev * 1e-3
        e_eV = e_gev * C.GEV_TO_EV

        u = get_pmns(t12, p.theta13, p.theta23, p.deltacp)
        msq = jnp.array([0.0, dm21, p.dm31])

        rho_ye_emit = smooth_rho_ye(r_emit, n0_scale)
        v_emit = C.matter_potential_eV(rho_ye_emit, 1.0)
        v_mat_emit = jnp.diag(jnp.array([v_emit, 0.0, 0.0]))

        h_vac = (u @ jnp.diag(msq) @ jnp.conj(u).T) / (2.0 * e_eV)
        h_emit = h_vac + v_mat_emit

        _, vecs_emit = jnp.linalg.eigh(h_emit)
        w = jnp.abs(vecs_emit[0, :]) ** 2

        rho_ye_surf = smooth_rho_ye(1.0, n0_scale)
        v_surf = C.matter_potential_eV(rho_ye_surf, 1.0)
        v_mat_surf = jnp.diag(jnp.array([v_surf, 0.0, 0.0]))
        h_surf = h_vac + v_mat_surf
        _, vecs_surf = jnp.linalg.eigh(h_surf)

        a = jnp.conj(u).T @ vecs_surf
        F = (jnp.abs(a) ** 2) @ w
        return jnp.real(jnp.sum(F * jnp.abs(u[0, :]) ** 2))

    energies_1d = jnp.logspace(-1, 1.48, 250)  # 0.1 to 30 MeV

    # 1D Probabilities & AD Derivatives
    p_1d = jax.vmap(lambda e: pee(e, p.theta12, p.dm21))(energies_1d)
    grad_t12 = jax.vmap(lambda e: jax.grad(lambda t: pee(e, t, p.dm21))(p.theta12))(energies_1d)
    grad_dm21 = jax.vmap(lambda e: jax.grad(lambda dm: pee(e, p.theta12, dm))(p.dm21))(energies_1d)
    grad_n0 = jax.vmap(lambda e: jax.grad(lambda s: pee(e, p.theta12, p.dm21, s))(1.0))(energies_1d)

    # FD validation points (15 sampled energy points across spectrum)
    fd_energies = np.logspace(-1, 1.48, 15)
    eps = 1e-5
    fd_t12 = [(pee(e, p.theta12 + eps, p.dm21) - pee(e, p.theta12 - eps, p.dm21)) / (2 * eps) for e in fd_energies]
    fd_dm21 = [(pee(e, p.theta12, p.dm21 * (1 + eps)) - pee(e, p.theta12, p.dm21 * (1 - eps))) / (2 * eps * p.dm21) for e in fd_energies]
    fd_n0 = [(pee(e, p.theta12, p.dm21, 1.0 + eps) - pee(e, p.theta12, p.dm21, 1.0 - eps)) / (2 * eps) for e in fd_energies]

    # 2D Functional Core Tomography Kernel (80 spatial shells)
    r_grid = jnp.linspace(0.01, 0.35, 80)
    sig = 0.015

    def get_smooth_profile_delta(r_ratio, delta_vec):
        rho_base = jnp.exp(jnp.polyval(poly_coeffs, r_ratio))
        r_diff = (r_ratio - r_grid) / sig
        weights = jnp.exp(-0.5 * r_diff**2)
        delta = jnp.sum(delta_vec * weights)
        return rho_base * (1.0 + delta)

    def pee_functional(e_mev, r_emit, delta_vec):
        e_gev = e_mev * 1e-3
        e_eV = e_gev * C.GEV_TO_EV
        u = get_pmns(p.theta12, p.theta13, p.theta23, p.deltacp)
        msq = jnp.array([0.0, p.dm21, p.dm31])

        rho_ye_emit = get_smooth_profile_delta(r_emit, delta_vec)
        v_emit = C.matter_potential_eV(rho_ye_emit, 1.0)
        h_emit = (u @ jnp.diag(msq) @ jnp.conj(u).T) / (2.0 * e_eV) + jnp.diag(jnp.array([v_emit, 0.0, 0.0]))
        _, vecs_emit = jnp.linalg.eigh(h_emit)
        w = jnp.abs(vecs_emit[0, :]) ** 2

        rho_ye_surf = get_smooth_profile_delta(1.0, delta_vec)
        v_surf = C.matter_potential_eV(rho_ye_surf, 1.0)
        h_surf = (u @ jnp.diag(msq) @ jnp.conj(u).T) / (2.0 * e_eV) + jnp.diag(jnp.array([v_surf, 0.0, 0.0]))
        _, vecs_surf = jnp.linalg.eigh(h_surf)

        a = jnp.conj(u).T @ vecs_surf
        F = (jnp.abs(a) ** 2) @ w
        return jnp.real(jnp.sum(F * jnp.abs(u[0, :]) ** 2))

    delta_zero = jnp.zeros(len(r_grid))
    energies_2d = jnp.logspace(-1, 1.48, 180)
    K_matrix = jax.vmap(lambda e: jax.grad(lambda d: pee_functional(e, 0.05, d))(delta_zero))(energies_2d)

    np.savez(
        NPZ_FILE,
        energies_1d=np.array(energies_1d),
        p_1d=np.array(p_1d),
        grad_t12=np.array(grad_t12),
        grad_dm21=np.array(grad_dm21),
        grad_n0=np.array(grad_n0),
        fd_energies=np.array(fd_energies),
        fd_t12=np.array(fd_t12),
        fd_dm21=np.array(fd_dm21),
        fd_n0=np.array(fd_n0),
        energies_2d=np.array(energies_2d),
        r_grid=np.array(r_grid),
        K_matrix=np.array(K_matrix),
    )
    print(f"Data computed in {time.time() - t0:.2f}s and saved to {NPZ_FILE}", flush=True)

else:
    print(f"Loading precomputed data from {NPZ_FILE}...", flush=True)

data = np.load(NPZ_FILE)
energies_1d = data["energies_1d"]
p_1d = data["p_1d"]
grad_t12 = data["grad_t12"]
grad_dm21 = data["grad_dm21"]
grad_n0 = data["grad_n0"]
fd_energies = data["fd_energies"]
fd_t12 = data["fd_t12"]
fd_dm21 = data["fd_dm21"]
fd_n0 = data["fd_n0"]
energies_2d = data["energies_2d"]
r_grid = data["r_grid"]
K_matrix = data["K_matrix"]

# --- Plotting Standalone Figure with Consistent Manuscript Styling ---
plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
    }
)

fig = plt.figure(figsize=(11.0, 9.8))
gs = fig.add_gridspec(3, 1, height_ratios=[0.85, 1.15, 1.30], hspace=0.32)
ax_prob = fig.add_subplot(gs[0])
ax_sens = fig.add_subplot(gs[1], sharex=ax_prob)
ax_tomo = fig.add_subplot(gs[2])

fig.subplots_adjust(left=0.11, right=0.88, top=0.94, bottom=0.08)

# --- PANEL (a1): Survival Probability P_ee(E) ---
ax_prob.plot(energies_1d, p_1d, color="tab:blue", lw=2.2, label=r"$P_{ee}(E)$ MSW transition")
ax_prob.set_ylabel(r"probability $P_{ee}$")
ax_prob.set_ylim(0.25, 0.62)
ax_prob.grid(True, which="both", ls=":", lw=0.4, alpha=0.6, color="gray")
ax_prob.legend(loc="upper right", framealpha=0.9)
ax_prob.set_title(r"(a) solar LMA survival probability & AD sensitivities", pad=10)
plt.setp(ax_prob.get_xticklabels(), visible=False)

# --- PANEL (a2): AD Parameter Sensitivities & FD Benchmarks ---
line1, = ax_sens.plot(energies_1d, grad_t12, color="tab:blue", lw=2.0, label=r"$\partial P_{ee}/\partial\theta_{12}$")
ax_sens.plot(fd_energies, fd_t12, "o", color="tab:blue", ms=6, mec="black", mew=0.6, zorder=5, label=r"FD benchmark ($10^{-10}$ match)")

line2, = ax_sens.plot(energies_1d, np.abs(grad_n0), color="tab:green", lw=2.0, label=r"$|\partial P_{ee}/\partial\ln N_e(0)|$")
ax_sens.plot(fd_energies, np.abs(fd_n0), "s", color="tab:green", ms=5.5, mec="black", mew=0.6, zorder=5)

ax_sens2 = ax_sens.twinx()
line3, = ax_sens2.plot(energies_1d, grad_dm21 * 1e-3, color="tab:red", lw=2.0, label=r"$\partial P_{ee}/\partial(\Delta m^2_{21})\ [\times 10^3\,\text{eV}^{-2}]$")
ax_sens2.plot(fd_energies, np.array(fd_dm21) * 1e-3, "^", color="tab:red", ms=6, mec="black", mew=0.6, zorder=5)

ax_sens.set_xscale("log")
ax_sens.set_xlim(0.1, 30.0)
ax_sens.set_xlabel(r"neutrino energy $E$ [MeV]")
ax_sens.set_ylabel(r"sensitivities $\partial P/\partial\theta_{12}$, $|\partial P/\partial\ln N_0|$")
ax_sens2.set_ylabel(r"sensitivity $\partial P/\partial\Delta m^2_{21}\ [10^3\,\text{eV}^{-2}]$", color="tab:red")
ax_sens2.tick_params(axis="y", labelcolor="tab:red")

ax_sens.grid(True, which="both", ls=":", lw=0.4, alpha=0.6, color="gray")

# Combine legends
lines = [line1, line2, line3]
labels_leg = [l.get_label() for l in lines]
ax_sens.legend(lines, labels_leg, loc="upper right", framealpha=0.9, fontsize=12)

# Annotate MSW resonance spike
ax_sens.annotate(
    "MSW resonance spike\n" + r"($E = 4.41\text{ MeV}$, $1.68\times 10^3\text{ eV}^{-2}$)",
    xy=(4.41, 0.45),
    xytext=(0.15, 0.42),
    arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.5),
    fontsize=12,
    color="tab:red",
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", fc="#fbe8e8", ec="tab:red", lw=1.0),
)

# --- PANEL (b): 2D Functional Core Tomography Kernel ---
abs_K = np.abs(K_matrix)
pc = ax_tomo.pcolormesh(r_grid, energies_2d, abs_K, cmap=plt.cm.Blues, shading="gouraud")

cbar = fig.colorbar(pc, ax=ax_tomo, orientation="vertical", pad=0.02, aspect=18)
cbar.set_label(r"functional kernel $\left|\frac{\delta P_{ee}}{\delta \ln N_e(r)}\right|$")

cs = ax_tomo.contour(r_grid, energies_2d, abs_K, levels=[0.01, 0.03, 0.06, 0.09, 0.11], colors="navy", linewidths=0.6, alpha=0.45)

ax_tomo.set_yscale("log")
ax_tomo.set_ylim(0.1, 30.0)
ax_tomo.set_xlim(0.01, 0.35)
ax_tomo.set_xlabel(r"solar radial position $r / R_\odot$")
ax_tomo.set_ylabel(r"neutrino energy $E$ [MeV]")
ax_tomo.grid(True, which="both", ls=":", lw=0.4, alpha=0.6, color="gray")

# Reference energy lines
ax_tomo.axhline(0.42, color="tab:green", ls="--", lw=1.0, alpha=0.8)
ax_tomo.text(0.34, 0.45, r"$pp$ end", color="tab:green", fontsize=12, ha="right", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

ax_tomo.axhline(0.86, color="tab:red", ls="--", lw=1.0, alpha=0.8)
ax_tomo.text(0.34, 0.92, r"$^7\mathrm{Be}$", color="tab:red", fontsize=12, ha="right", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

ax_tomo.axhline(18.8, color="tab:brown", ls="--", lw=1.0, alpha=0.8)
ax_tomo.text(0.34, 19.8, r"$hep$ end", color="tab:brown", fontsize=12, ha="right", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

ax_tomo.annotate(
    "functional core tomography window\n" + r"(80 radial shells in 1 VJP pass)",
    xy=(0.05, 4.4),
    xytext=(0.11, 2.2),
    arrowprops=dict(arrowstyle="->", color="#08519c", lw=1.5),
    fontsize=12,
    color="#08519c",
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", fc="#eff3ff", ec="#08519c", lw=1.0),
)

ax_tomo.set_title(r"(b) functional solar core tomography kernel $\delta P_{ee} / \delta \ln N_e(r)$ via reverse-mode VJP", pad=10)

# Save consolidated outputs
fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
print(f"Saved styled consolidated 2-panel figure to {FIG_PDF} and {FIG_PNG}")
