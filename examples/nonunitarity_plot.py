"""Standard non-unitarity plots.

Left:  DUNE-like nu_mu -> nu_e appearance with |alpha_21| = 0.02 and different
       phases — non-unitary mixing shifts the appearance probability in a way
       that can mimic / bias delta_CP (the classic NU-vs-CP degeneracy).
Right: the zero-distance effect — P(nu_mu -> nu_e) vs baseline at fixed energy:
       with alpha_21 != 0 the probability starts at |(NN^dag)_e mu|^2 != 0
       already at L = 0 (flavor violation without oscillation).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from mango import nufit_no, Flavor, NonUnitarity, nonunitarity

p = nufit_no()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=130)

# --- appearance vs energy with alpha_21 phases -------------------------------
E = jnp.linspace(0.4, 5.0, 400)
L, RHO = 1300.0, 2.8
a_mag = 0.02
cases = [(None, "k", "unitary ($\\alpha=0$)")] + [
    (a_mag * np.exp(1j * ph), c, rf"$|\alpha_{{21}}|={a_mag}$, $\phi_{{21}}={lab}$")
    for ph, c, lab in [(0.0, "C0", "0"), (-np.pi / 2, "C1", r"-\pi/2"),
                       (np.pi, "C3", r"\pi")]]
for a21, color, lab in cases:
    nu = NonUnitarity() if a21 is None else NonUnitarity(alpha21=jnp.asarray(a21))
    P = nonunitarity.probability(p, nu, E, L, density=RHO,
                                 flavor_in=Flavor.MU, flavor_out=Flavor.E)
    a1.plot(E, np.asarray(P), color=color, lw=1.6, label=lab)
a1.set_xlabel(r"$E_\nu$ [GeV]"); a1.set_ylabel(r"$P(\nu_\mu\to\nu_e)$")
a1.set_title(rf"Non-unitary appearance, $L={L:.0f}$ km (matter)")
a1.legend(fontsize=8)

# --- zero-distance effect vs baseline ----------------------------------------
Ls = jnp.geomspace(0.5, 3000.0, 300)
E0 = 2.5
for a21, color, lab in [(0.0, "k", "unitary"),
                        (0.02, "C0", r"$|\alpha_{21}|=0.02$"),
                        (0.05, "C3", r"$|\alpha_{21}|=0.05$")]:
    nu = NonUnitarity(alpha21=jnp.asarray(a21 + 0.0j))
    f = lambda Lk: nonunitarity.probability(p, nu, jnp.asarray(E0), Lk,
                                            density=RHO, flavor_in=Flavor.MU,
                                            flavor_out=Flavor.E)
    P = jax.vmap(f)(Ls)
    a2.loglog(np.asarray(Ls), np.clip(np.asarray(P), 1e-8, None),
              color=color, lw=1.6, label=lab)
    if a21 > 0:  # analytic zero-distance value
        NN = np.asarray(nu.N(p.pmns()) @ jnp.conj(nu.N(p.pmns())).T)
        dg = np.real(np.diag(NN))
        a2.axhline(abs(NN[0, 1]) ** 2 / (dg[0] * dg[1]), color=color, ls=":", lw=1)
a2.set_xlabel(r"$L$ [km]"); a2.set_ylabel(r"$P(\nu_\mu\to\nu_e)$")
a2.set_title(rf"Zero-distance effect, $E_\nu={E0}$ GeV "
             "(dotted: analytic $L\\to 0$ limit)")
a2.legend(fontsize=8, loc="upper left")

fig.suptitle("Non-unitary mixing with jaxnu (differentiable in $\\alpha$)",
             fontsize=12)
fig.tight_layout()
out = Path(__file__).resolve().parent / "nonunitarity.jpg"
fig.savefig(out)
print("saved", out)
