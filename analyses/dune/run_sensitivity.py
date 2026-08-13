"""DUNE CP-violation and mass-ordering sensitivity, reproduced with mango.

The chi-square is profiled over theta23 (both octants), theta13 (prior), Delta m^2_31,
delta_CP, the matter density (2% prior) and the 9 GLoBES normalization systematics,
using the *exact gradient from mango* (jax.value_and_grad -> L-BFGS). 624 kt-MW-yr.
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dune

x = np.linspace(-1, 1, 41)
t0 = time.time()
cpv = dune.cpv_sensitivity(x)
print("CPV scan done (%.0fs)" % (time.time() - t0))
mo = dune.mass_ordering_sensitivity(x)
print("MO scan done (%.0fs)" % (time.time() - t0))
np.savez("sensitivity_scan.npz", x=x, cpv=cpv, mo=mo)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))

a1.plot(x, cpv, "b-", lw=2)
for lvl, lab in [(3, r"$3\sigma$"), (5, r"$5\sigma$")]:
    a1.axhline(lvl, color="gray", ls=":", lw=1)
    a1.text(-0.97, lvl + 0.12, lab, color="gray", fontsize=9)
a1.set_xlabel(r"$\delta_{CP}/\pi$"); a1.set_ylabel(r"$\sigma=\sqrt{\Delta\chi^2}$")
a1.set_title("CP Violation Sensitivity"); a1.set_xlim(-1, 1); a1.set_ylim(0, 11)
a1.text(-0.95, 9.8, "jaxnu + DUNE TDR GLoBES\n624 kt-MW-yr, Normal Ordering", fontsize=9)

a2.plot(x, mo, "r-", lw=2)
a2.set_xlabel(r"$\delta_{CP}/\pi$"); a2.set_ylabel(r"$\sqrt{\Delta\chi^2}$")
a2.set_title("Mass Ordering Sensitivity"); a2.set_xlim(-1, 1); a2.set_ylim(0, 50)
a2.text(-0.95, 45, "jaxnu + DUNE TDR GLoBES\n624 kt-MW-yr, Normal Ordering", fontsize=9)

fig.tight_layout()
fig.savefig("dune_sensitivity.png", dpi=120)
print("saved dune_sensitivity.png")
print("CPV peaks: %.2f (neg), %.2f (pos)   [DUNE ref ~6.8, ~7.1]"
      % (cpv[x < 0].max(), cpv[x > 0].max()))
print("MO range: %.1f - %.1f   [DUNE ref ~13.5 - 24.5]" % (mo.min(), mo.max()))
