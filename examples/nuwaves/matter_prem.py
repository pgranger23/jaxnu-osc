"""Reproduction of nu-waves `matter_prem_test.jpg`: 4-panel atmospheric oscillogram.

P(nu_mu->nu_e), P(nu_mu->nu_mu) and the antineutrino versions over
(E in [0.1, 100] GeV, cos theta_z in [-1, 1]) through the PREM Earth, with a
15 km atmospheric production height (so down-going directions oscillate too).
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax

from _common import nuwaves_params, FIGDIR
from jaxnu import probability_earth, Flavor

p = nuwaves_params("NO")
nE, nCZ = 300, 300
E = np.geomspace(0.1, 100.0, nE)
cz = np.linspace(-1.0, 1.0, nCZ)

# Single unified entrypoint: h_atm_km>0 enables the atmospheric (full-cosz) mode.
osc = jax.jit(lambda anti: probability_earth(
    p, E, cz, n_sub=3, h_atm_km=15.0, anti=anti), static_argnums=0)

t0 = time.perf_counter()
Pnu = np.array(osc(False))   # (nCZ, nE, 3, 3)
Pbar = np.array(osc(True))
print(f"oscillograms in {time.perf_counter() - t0:.1f} s")

panels = [
    (Pnu[..., Flavor.E, Flavor.MU],   r"$P_{\nu_\mu \to \nu_e}$", "black"),
    (Pnu[..., Flavor.MU, Flavor.MU],  r"$P_{\nu_\mu \to \nu_\mu}$", "white"),
    (Pbar[..., Flavor.E, Flavor.MU],  r"$P_{\bar\nu_\mu \to \bar\nu_e}$", "black"),
    (Pbar[..., Flavor.MU, Flavor.MU], r"$P_{\bar\nu_\mu \to \bar\nu_\mu}$", "white"),
]

fig, axs = plt.subplots(2, 2, figsize=(9.8, 8.0), dpi=140, constrained_layout=True)
for ax, (Z, lab, col) in zip(axs.flat, panels):
    pc = ax.pcolormesh(E, cz, Z, vmin=0, vmax=1, shading="auto", cmap="inferno_r")
    ax.set_xscale("log")
    ax.grid(True, which="both", color="w", alpha=0.3, lw=0.4, ls="--")
    ax.text(0.96, 0.96, lab, transform=ax.transAxes, ha="right", va="top",
            color=col, fontsize=18, weight="bold")
for ax in axs[1, :]:
    ax.set_xlabel(r"$E_\nu$ [GeV]")
for ax in axs[:, 0]:
    ax.set_ylabel(r"$\cos\theta_z$")
cbar = fig.colorbar(axs.flat[0].collections[0], ax=axs, location="right",
                    fraction=0.05, pad=0.03)
cbar.set_label("Oscillation probability", fontsize=12)
plt.savefig(FIGDIR / "matter_prem_test.jpg", dpi=140)
print("saved", FIGDIR / "matter_prem_test.jpg")
