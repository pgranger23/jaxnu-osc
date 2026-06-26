"""Reproduction of nu-waves `vacuum_pmns.jpg`: T2K-like vacuum oscillation."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import nuwaves_params, FIGDIR
import jaxnu
from jaxnu import probability_vacuum, Flavor

p = nuwaves_params("NO")
E = np.linspace(0.2, 3.0, 400)
P = probability_vacuum(p, E, baseline_km=295.0)  # (nE, 3, 3) [out, in]

Pme = np.array(jaxnu.select(P, Flavor.MU, Flavor.E))
Pmm = np.array(jaxnu.select(P, Flavor.MU, Flavor.MU))
Pmt = np.array(jaxnu.select(P, Flavor.MU, Flavor.TAU))

plt.figure(figsize=(6.5, 4.0), dpi=150)
plt.plot(E, Pme, label=r"$P_{\mu e}$ appearance", lw=2)
plt.plot(E, Pmm, label=r"$P_{\mu\mu}$ disappearance", lw=2)
plt.plot(E, Pmt, label=r"$P_{\mu\tau}$ appearance", lw=2)
plt.plot(E, Pme + Pmm + Pmt, "--", color="red", label="Total probability", lw=1.5)
plt.xlabel(r"$E_\nu$ [GeV]")
plt.ylabel("Probability")
plt.title(r"T2K-like vacuum oscillation ($L = 295\,$km)")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig(FIGDIR / "vacuum_pmns.jpg", dpi=150)
print("saved", FIGDIR / "vacuum_pmns.jpg")
