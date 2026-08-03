"""jaxnu code paper: derivatives with respect to the BSM parameters themselves.

The point of the beyond-standard-model section is not that jaxnu can *evaluate*
decoherence or non-unitary mixing -- other codes do that -- but that the
derivatives with respect to the new parameters come out of the same reverse
pass as any other gradient. Those are the quantities a sensitivity forecast for
such a model needs, and they exist nowhere in closed form.

Both derivatives are taken at the standard-model point (gamma = 0, alpha = 0),
which is where a limit-setting analysis linearizes.

Left:  Lindblad decoherence through the Earth's diameter at constant density,
       dP(numu->numu)/d gamma_ij, in units of 1e-23 GeV = 1e-14 eV (the scale
       of current atmospheric bounds), so the numbers are O(1).
Right: non-unitary mixing through the PREM Earth at cos(theta_z) = -1,
       dP(numu->numu)/d alpha_ij; the alphas are dimensionless.

Run from the repo root:
    JAX_PLATFORMS=cpu python benchmarks/bench_bsm_derivatives.py
Artefacts go to benchmarks/output/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from jaxnu import nufit_no, decoherence, nonunitarity
from jaxnu.decoherence import Decoherence
from jaxnu.nonunitarity import NonUnitarity

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTDIR, exist_ok=True)

P = nufit_no()
# 2-30 GeV: below ~2 GeV the oscillation phase through a diameter turns
# over faster than the eye (or the page) resolves, and every curve
# degenerates into a band. The atmospheric sensitivity to both sectors
# sits in the resolved region anyway.
NE = 400
E = jnp.asarray(np.logspace(np.log10(2.0), np.log10(30.0), NE))

# --- (a) decoherence -------------------------------------------------------
L_DIAM_KM = 12742.0        # Earth diameter: the longest atmospheric baseline
RHO_MEAN = 5.5             # mean Earth density; the decoherence front-end is
YE = 0.5                   # constant-density, so use a representative value
GAMMA_UNIT = 1.0e-14       # eV  ( = 1e-23 GeV, the scale of current bounds)


def p_decoh(g21, g31, g32):
    """P(numu->numu) with Lindblad damping, gammas in units of GAMMA_UNIT."""
    model = Decoherence(gamma21=g21 * GAMMA_UNIT, gamma31=g31 * GAMMA_UNIT,
                        gamma32=g32 * GAMMA_UNIT)
    return decoherence.probability(P, E, L_DIAM_KM, model,
                                   density=RHO_MEAN, ye=YE)[..., 1, 1]


z = jnp.asarray(0.0)
P_dec = np.asarray(p_decoh(z, z, z))
J_dec = jax.jacfwd(p_decoh, argnums=(0, 1, 2))(z, z, z)
J_dec = [np.asarray(j) for j in J_dec]

# --- (b) non-unitarity -----------------------------------------------------
CZ = -1.0                  # straight through the core


def p_nonu(a11, a22, a33, a21):
    nu = NonUnitarity(alpha11=a11, alpha22=a22, alpha33=a33,
                      alpha21=a21 + 0.0j)
    return nonunitarity.probability_earth(P, nu, E, CZ, n_sub=4)[..., 1, 1]


P_nu = np.asarray(p_nonu(z, z, z, z))
J_nu = jax.jacfwd(p_nonu, argnums=(0, 1, 2, 3))(z, z, z, z)
J_nu = [np.asarray(j) for j in J_nu]

Eg = np.asarray(E)
print(f"{'quantity':34s} {'min':>12s} {'max':>12s}")
for lab, j in zip(("dP/dgamma21", "dP/dgamma31", "dP/dgamma32"), J_dec):
    print(f"{lab:34s} {j.min():12.4g} {j.max():12.4g}")
for lab, j in zip(("dP/dalpha11", "dP/dalpha22", "dP/dalpha33",
                   "dP/dRe(alpha21)"), J_nu):
    print(f"{lab:34s} {j.min():12.4g} {j.max():12.4g}")

np.savez(os.path.join(OUTDIR, "jaxnu_bsm_derivatives.npz"), E=Eg,
         P_dec=P_dec, J_dec=np.array(J_dec), P_nu=P_nu, J_nu=np.array(J_nu))

# --- figure ----------------------------------------------------------------
# Placed at \textwidth (~6.3 in); a 12 in canvas scales by ~0.52, so set fonts
# so labels land near 10 pt printed.
plt.rcParams.update({"font.size": 17, "axes.titlesize": 17,
                     "axes.labelsize": 17, "xtick.labelsize": 14,
                     "ytick.labelsize": 14, "legend.fontsize": 13})
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)

dec_style = [(r"$\partial P/\partial\gamma_{21}$", "tab:blue", "-"),
             (r"$\partial P/\partial\gamma_{31}$", "tab:red", "--"),
             (r"$\partial P/\partial\gamma_{32}$", "tab:green", "-.")]
for (lab, c, ls), j in zip(dec_style, J_dec):
    axes[0].plot(Eg, j, color=c, ls=ls, lw=2.0, label=lab)
axes[0].set_title("decoherence (Lindblad)")
axes[0].set_ylabel(r"$\partial P/\partial\gamma$  "
                   r"[per $10^{-23}\,$GeV]")

nu_style = [(r"$\partial P/\partial\alpha_{11}$", "tab:blue", "-"),
            (r"$\partial P/\partial\alpha_{22}$", "tab:red", "--"),
            (r"$\partial P/\partial\alpha_{33}$", "tab:green", "-."),
            (r"$\partial P/\partial\,\mathrm{Re}\,\alpha_{21}$",
             "tab:purple", ":")]
for (lab, c, ls), j in zip(nu_style, J_nu):
    axes[1].plot(Eg, j, color=c, ls=ls, lw=2.0, label=lab)
axes[1].set_title("non-unitary mixing")
axes[1].set_ylabel(r"$\partial P/\partial\alpha$")

xt = [t for t in (2, 3, 5, 10, 20, 30) if Eg.min() <= t <= Eg.max()]
for ax, prob in zip(axes, (P_dec, P_nu)):
    ax.set_xscale("log")
    ax.set_xlim(Eg.min(), Eg.max())
    # plain numerals: the default log formatter writes these as "2 x 10^0"
    # and the minor labels collide into an unreadable band
    ax.set_xticks(xt)
    ax.set_xticklabels([str(t) for t in xt])
    ax.minorticks_off()
    ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    ax.set_xlabel(r"$E_\nu$ [GeV]")
    # headroom: these curves fill the panel, and with three or four entries
    # "best" has nowhere to put the legend that is not on top of a line
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.42 * (hi - lo))
    ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=1.2)
    ax.grid(alpha=0.25)
    # the probability itself, for context, on a twin axis
    tw = ax.twinx()
    tw.plot(Eg, prob, color="0.6", lw=1.0, alpha=0.75, zorder=0)
    tw.set_ylim(0, 1.05)
    tw.set_ylabel(r"$P(\nu_\mu\to\nu_\mu)$ (grey)", fontsize=14, color="0.4")
    tw.tick_params(axis="y", labelsize=12, colors="0.4")

fp = os.path.join(OUTDIR, "jaxnu_bsm_derivatives")
fig.savefig(fp + ".png", dpi=140, bbox_inches="tight")
fig.savefig(fp + ".pdf", bbox_inches="tight")
print("saved", fp + ".pdf")
