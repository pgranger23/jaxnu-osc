"""TDR-style DUNE spectra straight from the generic GLoBES loader.

Loads the official DUNE GLoBES configuration with ``jaxnu.globes.load`` and plots
the nu_e / nu_e-bar appearance spectra for delta_CP = -pi/2, 0, +pi/2 (the classic
DUNE figure showing the CP-phase dependence), with the summed backgrounds shaded.
Everything downstream of the file parse is differentiable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from jaxnu import OscParams, globes

GLB = Path(__file__).resolve().parents[1] / "analyses/dune/globes/DUNE_GLoBES.glb"
exp = globes.load(GLB, scale=1.21)


def params(dcp):
    return OscParams(theta12=jnp.asarray(0.5903), theta13=jnp.asarray(0.150),
                     theta23=jnp.asarray(0.866), deltacp=jnp.asarray(dcp),
                     dm21=jnp.asarray(7.39e-5), dm31=jnp.asarray(2.451e-3 + 7.39e-5))


def split(rule, p):
    comps = exp.components(p)[rule]
    sig = sum(a for (_s, is_sig, a) in comps if is_sig)
    bkg = sum(a for (_s, is_sig, a) in comps if not is_sig)
    return np.asarray(sig), np.asarray(bkg)


def rebin(y):  # 0.125 -> 0.25 GeV bins below 8 GeV
    m = exp.reco_c < 8.0
    yy = y[m]
    n = (len(yy) // 2) * 2
    return exp.reco_c[m][:n:2] + 0.0625, yy[:n].reshape(-1, 2).sum(1)


fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=130)
titles = {"nue_app": r"$\nu_e$ appearance (FHC)",
          "nuebar_app": r"$\bar\nu_e$ appearance (RHC)"}
for ax, rule in zip(axs, ["nue_app", "nuebar_app"]):
    _, bkg = split(rule, params(0.0))
    x, b = rebin(bkg)
    ax.fill_between(x, b, step="mid", color="0.8", label="total background")
    for dcp, color, lab in [(-np.pi / 2, "C0", r"$\delta_{CP}=-\pi/2$"),
                            (0.0, "k", r"$\delta_{CP}=0$"),
                            (np.pi / 2, "C3", r"$\delta_{CP}=+\pi/2$")]:
        sig, bkg = split(rule, params(dcp))
        x, t = rebin(sig + bkg)
        ax.step(x, t, where="mid", color=color, lw=1.6, label=lab)
    ax.set_xlim(0.5, 8); ax.set_ylim(bottom=0)
    ax.set_xlabel("Reconstructed energy [GeV]")
    ax.set_ylabel("Events / 0.25 GeV")
    ax.set_title(titles[rule])
    ax.legend(fontsize=9)
fig.suptitle("DUNE TDR spectra from jaxnu.globes.load (official GLoBES config, "
             "6.5 yr/mode, NO)", fontsize=12)
fig.tight_layout()
out = Path(__file__).resolve().parent / "dune_globes.jpg"
fig.savefig(out)
print("saved", out)
for rule in titles:
    sig, bkg = split(rule, params(-np.pi / 2))
    w = exp.windows[rule]
    print(f"  {rule}: signal={sig[w].sum():.0f}  bkg={bkg[w].sum():.0f} "
          f"(deltaCP=-pi/2, 0.5-18 GeV)")
