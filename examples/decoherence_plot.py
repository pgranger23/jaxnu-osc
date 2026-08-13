"""Standard decoherence plots.

Left:  Lindblad decoherence at a DUNE-like beam — nu_mu survival vs energy with
       increasing gamma damps the oscillations toward the interference-averaged
       probability (gamma L ~ 0 ... >> 1).
Right: wave-packet decoherence at a JUNO-like reactor baseline — the fast
       Delta m^2_31 wiggles wash out as the production wave packet sigma_x
       shrinks, while the slow solar oscillation survives (the classic reactor
       wave-packet signature).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

import mango
from mango import nufit_no, Flavor, Decoherence, WavePacket, decoherence

p = nufit_no()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=130)

# --- Lindblad @ DUNE-like beam ----------------------------------------------
E = jnp.linspace(0.4, 5.0, 500)
L, RHO = 1300.0, 2.8
for gam, color in [(0.0, "k"), (5e-14, "C0"), (2e-13, "C1"), (1e-12, "C3")]:
    d = Decoherence(gamma21=gam, gamma31=gam, gamma32=gam, n=0)
    P = decoherence.probability(p, E, L, d, density=RHO,
                                flavor_in=Flavor.MU, flavor_out=Flavor.MU)
    gl = gam * L * mango.constants.KM_TO_INV_EV
    lab = "standard" if gam == 0 else (
        rf"$\gamma={gam:.0e}$ eV  ($\gamma L\approx{gl:.1f}$)")
    a1.plot(E, np.asarray(P), color=color, lw=1.6, label=lab)
a1.set_xlabel(r"$E_\nu$ [GeV]"); a1.set_ylabel(r"$P(\nu_\mu\to\nu_\mu)$")
a1.set_title(rf"Lindblad decoherence, $L={L:.0f}$ km, $\rho={RHO}$ (n=0)")
a1.set_ylim(0, 1.05); a1.legend(fontsize=8, loc="lower right")

# --- wave packets @ JUNO-like reactor ---------------------------------------
E2 = jnp.linspace(0.0018, 0.008, 1200)     # 1.8-8 MeV
L2 = 52.5
for sx, color in [(1.0, "k"), (3e-12, "C0"), (1e-12, "C1"), (3e-13, "C3")]:
    P = decoherence.probability(p, E2, L2, WavePacket(sigma_x_m=sx), anti=True,
                                flavor_in=Flavor.E, flavor_out=Flavor.E)
    lab = "plane wave" if sx == 1.0 else rf"$\sigma_x={sx*1e12:g}$ pm"
    a2.plot(np.asarray(E2) * 1e3, np.asarray(P), color=color, lw=1.3, label=lab)
a2.set_xlabel(r"$E_\nu$ [MeV]"); a2.set_ylabel(r"$P(\bar\nu_e\to\bar\nu_e)$")
a2.set_title(rf"Wave-packet decoherence, reactor $L={L2}$ km")
a2.set_ylim(0, 1.05); a2.legend(fontsize=8, loc="lower right")

fig.suptitle("Decoherence with jaxnu (differentiable in $\\gamma$ / $\\sigma_x$)",
             fontsize=12)
fig.tight_layout()
out = Path(__file__).resolve().parent / "decoherence.jpg"
fig.savefig(out)
print("saved", out)
