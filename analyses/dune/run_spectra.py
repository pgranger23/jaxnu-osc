"""Reproduce the DUNE TDR far-detector event spectra (5-year staged, NO, dCP=0)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp

import dune

d = dune.data()
p = dune.params_from(jnp.array([dune.NUFIT_NO["theta23"], dune.NUFIT_NO["theta13"],
                                 dune.NUFIT_NO["dm31"], 0.0]))
comps = dune.rule_components(p)


def split(rule):
    sig = np.zeros(d.nreco)
    bkg = {k: np.zeros(d.nreco) for k in ("beam_nue", "numu", "nutau", "NC")}
    for (sys, is_sig, arr) in comps[rule]:
        a = np.asarray(arr)
        if is_sig:
            sig = sig + a
        elif "nue" in sys:
            bkg["beam_nue"] = bkg["beam_nue"] + a
        elif "nutau" in sys:
            bkg["nutau"] = bkg["nutau"] + a
        elif "nc" in sys:
            bkg["NC"] = bkg["NC"] + a
        else:
            bkg["numu"] = bkg["numu"] + a
    return sig, bkg


def rebin025(y):  # merge the 0.125 GeV bins to 0.25 in the 0-8 GeV region
    m = d.reco_c < 8.0
    yy = y[m]; n = (len(yy) // 2) * 2
    return d.reco_c[m][:n:2] + 0.125, yy[:n].reshape(-1, 2).sum(1)


fig, axs = plt.subplots(2, 2, figsize=(11, 8))
titles = {"nue_app": r"$\nu_e$ Appearance (FHC)", "nuebar_app": r"$\bar\nu_e$ Appearance (RHC)",
          "numu_dis": r"$\nu_\mu$ Disappearance (FHC)", "numubar_dis": r"$\bar\nu_\mu$ Disappearance (RHC)"}
for ax, rule in zip(axs.flat, d.rules):
    sig, bkg = split(rule)
    x, s = rebin025(sig)
    ax.step(x, s, where="mid", color="k", lw=1.5, label="Signal CC")
    for name, col in [("beam_nue", "b"), ("numu", "g"), ("nutau", "m"), ("NC", "r")]:
        _, b = rebin025(bkg[name])
        if b.sum() > 1:
            ax.step(x, b, where="mid", color=col, lw=1.1, label=name)
    ax.set_xlim(0, 8); ax.set_ylim(bottom=0)
    ax.set_xlabel("Reconstructed Energy (GeV)"); ax.set_ylabel("Events / 0.25 GeV")
    ax.set_title(titles[rule] + f"  (tot={np.asarray(dune.total_spectra(p)[rule])[d.win].sum():.0f})")
    ax.legend(fontsize=8)
fig.suptitle("DUNE far-detector spectra reproduced with jaxnu (5 yr staged, NO, "
             r"$\delta_{CP}=0$)", fontsize=13)
fig.tight_layout()
fig.savefig("dune_spectra.png", dpi=120)
print("saved dune_spectra.png")
for rule in d.rules:
    sig, bkg = split(rule)
    print("%-12s signal=%.0f  bkg=%.0f" % (rule, sig[d.win].sum(),
          sum(v[d.win].sum() if hasattr(v, "__len__") else 0 for v in bkg.values())))
