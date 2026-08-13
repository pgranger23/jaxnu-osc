"""fig_solar_density_sensitivity.py: Core electron density sensitivity d P_ee / d ln N_e(0)
demonstrating solar structure and metallicity profiling via automatic differentiation in mango.

Generates a publication-quality standalone figure showing how solar neutrino survival
probabilities at different production radii (r_emit = 0.04, 0.06, 0.10, 0.18 R_sun) respond
to central solar core electron density perturbations N_e(0).

Run from repository root:
    python benchmarks/fig_solar_density_sensitivity.py
    FIGONLY=1 python benchmarks/fig_solar_density_sensitivity.py   # re-render from saved npz
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

import mango
from mango import solar, nufit_no, constants as C

# Output directories
BENCH_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PAPER_FIGS = "/Users/pgranger/jaxnu-paper/figures"
os.makedirs(BENCH_OUT, exist_ok=True)
os.makedirs(PAPER_FIGS, exist_ok=True)

NPZ_FILE = os.path.join(BENCH_OUT, "fig_solar_density_sensitivity.npz")
FIG_PDF = os.path.join(PAPER_FIGS, "fig_solar_density_sensitivity.pdf")
FIG_PNG = os.path.join(PAPER_FIGS, "fig_solar_density_sensitivity.png")

FIGONLY = os.environ.get("FIGONLY", "") not in ("", "0")

if not FIGONLY or not os.path.exists(NPZ_FILE):
    print("Computing solar core electron density sensitivity curves (dP_ee / d ln N_0)...", flush=True)
    t0 = time.time()

    p = nufit_no()
    bs05_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "data", "bs05_agsop.dat")
    if os.path.exists(bs05_path):
        prof = solar.load_bs05(bs05_path)
    else:
        prof = solar.exponential_profile()

    r = np.array(prof.r_over_rsun)
    rho = np.array(prof.rho_ye)
    log_rho = np.log(rho)
    poly_coeffs = jnp.asarray(np.polyfit(r, log_rho, 8))

    def smooth_rho_ye(r_ratio, n0_scale=1.0):
        return n0_scale * jnp.exp(jnp.polyval(poly_coeffs, r_ratio))

    energies = jnp.logspace(-1, 1.48, 300)  # 0.1 to 30 MeV
    radii_list = [0.04, 0.06, 0.10, 0.18]   # Representative solar production radii

    u = p.pmns()
    msq = p.msquared()

    def pee_n0(e_mev, r_ratio, n0_scale=1.0):
        e_gev = e_mev * 1e-3
        e_eV = e_gev * C.GEV_TO_EV

        rho_ye_emit = smooth_rho_ye(r_ratio, n0_scale)
        v_emit = C.matter_potential_eV(rho_ye_emit, 1.0)

        h_emit = mango.hamiltonian.matter_hamiltonian(u, msq, e_eV, v_emit)
        _, vecs_emit = jnp.linalg.eigh(h_emit)
        w = jnp.abs(vecs_emit[0, :]) ** 2

        rho_ye_surf = smooth_rho_ye(1.0, n0_scale)
        v_surf = C.matter_potential_eV(rho_ye_surf, 1.0)
        h_surf = mango.hamiltonian.matter_hamiltonian(u, msq, e_eV, v_surf)
        _, vecs_surf = jnp.linalg.eigh(h_surf)

        a = jnp.conj(u).T @ vecs_surf
        F = (jnp.abs(a) ** 2) @ w
        return jnp.sum(F * jnp.abs(u[0]) ** 2)

    sensitivities = {}
    probabilities = {}
    for r_val in radii_list:
        # Exact AD derivative with respect to log core density scale
        d_n0 = jax.vmap(lambda e: jax.grad(lambda s: pee_n0(e, r_val, s))(1.0))(energies)
        p_val = jax.vmap(lambda e: pee_n0(e, r_val, 1.0))(energies)
        sensitivities[r_val] = np.array(d_n0)
        probabilities[r_val] = np.array(p_val)

    np.savez(
        NPZ_FILE,
        energies=np.array(energies),
        radii_list=np.array(radii_list),
        **{f"sens_{r_val:.2f}": sensitivities[r_val] for r_val in radii_list},
        **{f"prob_{r_val:.2f}": probabilities[r_val] for r_val in radii_list},
    )
    print(f"Data computed in {time.time() - t0:.2f}s and saved to {NPZ_FILE}", flush=True)

else:
    print(f"Loading precomputed data from {NPZ_FILE}...", flush=True)

data = np.load(NPZ_FILE)
energies = data["energies"]
radii_list = data["radii_list"]
sensitivities = {r_val: data[f"sens_{r_val:.2f}"] for r_val in radii_list}
probabilities = {r_val: data[f"prob_{r_val:.2f}"] for r_val in radii_list}

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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.8, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.0], "hspace": 0.08})
fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.10)

colors = {
    0.04: "#d62728",  # 8B emission region (deep core)
    0.06: "#9467bd",  # 7Be / pep emission region
    0.10: "#2ca02c",  # pp emission region
    0.18: "#1f77b4",  # Outer core boundary
}

labels = {
    0.04: r"$r_{\mathrm{emit}} = 0.04\,R_\odot$ ($^8\mathrm{B}$ core peak)",
    0.06: r"$r_{\mathrm{emit}} = 0.06\,R_\odot$ ($^7\mathrm{Be}$ / $pep$ peak)",
    0.10: r"$r_{\mathrm{emit}} = 0.10\,R_\odot$ ($pp$ peak)",
    0.18: r"$r_{\mathrm{emit}} = 0.18\,R_\odot$ (Outer core)",
}

# --- Upper Panel: Survival Probabilities P_ee(E) ---
for r_val in radii_list:
    ax1.plot(energies, probabilities[r_val], color=colors[r_val], lw=1.8, label=labels[r_val])

ax1.set_ylabel(r"Survival Probability $P_{ee}(E)$", fontsize=11)
ax1.set_ylim(0.25, 0.62)
ax1.grid(True, which="both", ls=":", lw=0.3, alpha=0.5, color="gray")
ax1.legend(loc="lower left", framealpha=0.9, fontsize=8.5)
ax1.set_title(r"Solar core density sensitivity $\partial P_{ee}/\partial \ln N_e(0)$ via automatic differentiation", loc="left", pad=7, fontweight="bold")

# --- Lower Panel: Core Density AD Sensitivity dP_ee / d ln N_e(0) ---
for r_val in radii_list:
    ax2.plot(energies, np.abs(sensitivities[r_val]), color=colors[r_val], lw=1.8, label=labels[r_val])

ax2.set_xscale("log")
ax2.set_xlim(0.1, 30.0)
ax2.set_xlabel(r"Neutrino Energy $E$ [MeV]", fontsize=11)
ax2.set_ylabel(r"Sensitivity $|\partial P_{ee} / \partial \ln N_e(0)|$", fontsize=11)
ax2.set_ylim(-0.005, 0.145)
ax2.grid(True, which="both", ls=":", lw=0.3, alpha=0.5, color="gray")

# Annotations for solar flux components
ax2.axvline(0.42, color="#2ca02c", ls="--", lw=0.8, alpha=0.7)
ax2.text(0.43, 0.13, r"$pp$ end", color="#2ca02c", fontsize=8.5, rotation=90, va="top", fontweight="bold")

ax2.axvline(0.86, color="#d62728", ls="--", lw=0.8, alpha=0.7)
ax2.text(0.89, 0.13, r"$^7\mathrm{Be}$", color="#d62728", fontsize=8.5, rotation=90, va="top", fontweight="bold")

ax2.axvline(1.44, color="#9467bd", ls="--", lw=0.8, alpha=0.7)
ax2.text(1.49, 0.13, r"$pep$", color="#9467bd", fontsize=8.5, rotation=90, va="top", fontweight="bold")

ax2.axvline(18.8, color="#8c564b", ls="--", lw=0.8, alpha=0.7)
ax2.text(19.5, 0.13, r"$hep$ end", color="#8c564b", fontsize=8.5, rotation=90, va="top", fontweight="bold")

# Annotate core density constraint peak
ax2.annotate(
    "Peak Core Density Sensitivity\n" + r"($^8\mathrm{B}$ probes $N_e(0)$ to $\sim 1\%$)",
    xy=(4.4, 0.125),
    xytext=(0.15, 0.085),
    arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1.2),
    fontsize=8.5,
    color="#b2182b",
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", fc="#fbe8e8", ec="#b2182b", lw=0.8),
)

# Save standalone outputs
fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
print(f"Saved standalone core density sensitivity figure to {FIG_PDF} and {FIG_PNG}")
