"""Reproduction of nu-waves `matter_constant_test.jpg`: DUNE-like NO vs IO."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import nuwaves_params, FIGDIR
from jaxnu import probability_constant, Flavor

E = np.linspace(0.2, 5.0, 600)
kw = dict(baseline_km=1300.0, density=2.8, ye=0.5,
          flavor_in=Flavor.MU, flavor_out=Flavor.E)

P_no = np.array(probability_constant(nuwaves_params("NO"), E, **kw))
P_io = np.array(probability_constant(nuwaves_params("IO"), E, **kw))

plt.figure(figsize=(7, 4.3), dpi=150)
plt.plot(E, P_no, label=r"$\nu_\mu\!\to\!\nu_e$ (matter) NO", lw=2)
plt.plot(E, P_io, label=r"$\nu_\mu\!\to\!\nu_e$ (matter) IO", lw=2)
plt.xlabel(r"$E_\nu$ [GeV]")
plt.ylabel("Probability")
plt.title("DUNE-like oscillation, L=1300 km (vacuum vs matter)")
plt.xlim(E.min(), E.max())
plt.ylim(0, 0.30)
plt.legend(ncol=2, frameon=False)
plt.tight_layout()
plt.savefig(FIGDIR / "matter_constant_test.jpg", dpi=150)
print("saved", FIGDIR / "matter_constant_test.jpg")
