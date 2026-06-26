"""Reproduction of nu-waves `adiabatic_sun_ssm.jpg`: solar adiabatic MSW.

Vacuum mass-state fractions of a nu_e produced at r = 0.05 R_sun, evolving
adiabatically out through the BS05(AGS,OP) standard-solar-model density profile.
This is the textbook averaged-adiabatic result: the nu_e emerges predominantly as
nu_2 (the LMA-MSW solution), with nu_3 ~ sin^2(theta_13) throughout.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import nuwaves_params, FIGDIR, DATADIR
from jaxnu import solar

p = nuwaves_params("NO")
prof = solar.load_bs05(str(DATADIR / "bs05_agsop.dat"))

E_GeV = 0.008                       # 8 MeV
r_emit = 0.05 * prof.R_sun_km
r = np.concatenate(([r_emit], np.geomspace(r_emit + 1.0, prof.R_sun_km, 400)))
F = np.array(solar.adiabatic_mass_fractions(p, E_GeV, prof, r, r_emit, alpha=0))

x = r / prof.R_sun_km
plt.figure(figsize=(7.5, 4.2), dpi=150)
for i, (lab, col) in enumerate([(r"$\nu_1$", "C0"), (r"$\nu_2$", "C1"),
                                (r"$\nu_3$", "C2")]):
    plt.plot(x, F[:, i], lw=2, label=lab, color=col)
    plt.scatter([x[0], x[-1]], [F[0, i], F[-1, i]], s=25, color=col, zorder=5)
plt.xscale("log")
plt.xlabel(r"$r/R_\odot$")
plt.ylabel("Mass-state fraction (adiabatic)")
plt.ylim(0, 1.0)
plt.grid(True, which="both", alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(FIGDIR / "adiabatic_sun_ssm_test.jpg", dpi=150)
print("saved", FIGDIR / "adiabatic_sun_ssm_test.jpg")
