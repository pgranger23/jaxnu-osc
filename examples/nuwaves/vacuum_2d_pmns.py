"""Reproduction of nu-waves `vacuum_2d_pmns.jpg`: P over (E, L) grid."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import jax

from _common import nuwaves_params, FIGDIR
from jaxnu import probability_vacuum, Flavor

p = nuwaves_params("NO")
E = np.linspace(0.2, 5.0, 400)   # GeV
L = np.linspace(1000.0, 2000.0, 400)  # km

# vmap baseline over rows; probability_vacuum already vectorizes energy.
row = lambda Lk: probability_vacuum(p, E, baseline_km=Lk)
P = jax.vmap(row)(L)  # (nL, nE, 3, 3)
Pme = np.array(P[..., Flavor.E, Flavor.MU])
Pmm = np.array(P[..., Flavor.MU, Flavor.MU])

fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), dpi=140)
for ax, Z, title in [(axs[0], Pme, r"$P(\nu_\mu \to \nu_e)$"),
                     (axs[1], Pmm, r"$P(\nu_\mu \to \nu_\mu)$")]:
    pc = ax.pcolormesh(E, L, np.clip(Z, 1e-3, 1.0), norm=LogNorm(1e-3, 1.0),
                       shading="auto", cmap="inferno")
    ax.axhline(1300, ls="--", color="w", alpha=0.7)
    ax.set_xlabel(r"$E_\nu$ [GeV]")
    ax.set_ylabel(r"$L$ [km]")
    ax.set_title(title)
    fig.colorbar(pc, ax=ax, label="Probability")
plt.tight_layout()
plt.savefig(FIGDIR / "vacuum_2d_pmns.jpg", dpi=140)
print("saved", FIGDIR / "vacuum_2d_pmns.jpg")
