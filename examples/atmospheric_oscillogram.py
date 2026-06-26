"""Atmospheric nu_mu -> nu_e oscillogram through the PREM Earth.

Computes P(nu_mu -> nu_e) on an (energy, cos_zenith) grid and saves a plot if
matplotlib is available.  Demonstrates the jit-compiled, vmapped Earth path.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp

import jaxnu
from jaxnu import nufit_no, probability_earth, Flavor

p = nufit_no()

energy = jnp.linspace(1.0, 25.0, 200)        # GeV
coszen = jnp.linspace(-1.0, -0.05, 150)      # up-going

osc = jax.jit(lambda E, cz: probability_earth(
    p, E, cz, flavor_in=Flavor.MU, flavor_out=Flavor.E))

osc(energy[:2], coszen[:2]).block_until_ready()  # warm up / compile
t0 = time.perf_counter()
P = osc(energy, coszen).block_until_ready()      # shape (n_cz, n_E)
dt = time.perf_counter() - t0
print(f"grid {P.shape} computed in {dt*1e3:.1f} ms; "
      f"P in [{float(P.min()):.3f}, {float(P.max()):.3f}]")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(7, 5))
    plt.pcolormesh(np.array(energy), np.array(coszen), np.array(P),
                   shading="auto", cmap="viridis")
    plt.colorbar(label=r"$P(\nu_\mu \to \nu_e)$")
    plt.xlabel("Neutrino energy [GeV]")
    plt.ylabel(r"$\cos\theta_z$")
    plt.title("Atmospheric oscillogram (PREM Earth)")
    out = Path(__file__).resolve().parent / "atmospheric_oscillogram.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"saved {out}")
except ImportError:
    print("matplotlib not installed; skipping plot")
