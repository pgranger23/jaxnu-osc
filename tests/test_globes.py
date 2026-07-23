"""Tests for the generic GLoBES (AEDL) loader.

Two layers of validation:
1. the real DUNE TDR configuration (if the analyses/dune data is present) —
   totals, gradients, chi2 self-consistency;
2. a synthetic mini-config exercising the non-DUNE code paths: uniform ``$bins``,
   analytic Gaussian smearing (``@type=1``), ``@signalerror`` fallback
   systematics, and ``$profiletype = 2``.
"""

import textwrap
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import jaxnu
from jaxnu import OscParams, globes

DUNE_GLB = Path(__file__).resolve().parents[1] / "analyses/dune/globes/DUNE_GLoBES.glb"

P0 = OscParams(theta12=jnp.asarray(0.5903), theta13=jnp.asarray(0.150),
               theta23=jnp.asarray(0.866), deltacp=jnp.asarray(0.0),
               dm21=jnp.asarray(7.39e-5), dm31=jnp.asarray(2.451e-3 + 7.39e-5))


def test_dune_config_totals_and_grad():
    if not DUNE_GLB.exists():  # data not shipped with pip installs
        return
    exp = globes.load(DUNE_GLB, scale=1.21)
    assert len(exp.channels) == 32 and len(exp.rules) == 4
    assert abs(exp.baseline - 1284.9) < 1e-6 and exp.target_mass == 40.0
    sp = exp.spectra(P0, backend="eigh")
    # totals in the file's 0.5-18 GeV window (validated vs the bespoke parser)
    expected = {"nue_app": 3554.1, "nuebar_app": 1291.8,
                "numu_dis": 23017.4, "numubar_dis": 13112.2}
    for r, v in expected.items():
        tot = float(np.asarray(sp[r])[exp.windows[r]].sum())
        assert abs(tot / v - 1) < 5e-3, (r, tot, v)
    # differentiable through the loader
    g = jax.grad(lambda p: exp.spectra(p)["nue_app"].sum())(P0)
    assert np.isfinite(float(g.deltacp)) and abs(float(g.deltacp)) > 1.0
    # chi2(self) == 0 and differentiable in systematics
    dat = {r: jnp.asarray(np.asarray(v)) for r, v in sp.items()}
    xi = jnp.zeros(len(exp.sys_labels))
    assert abs(float(exp.chi2(P0, xi, dat, backend="eigh"))) < 1e-8
    gx = jax.grad(lambda x: exp.chi2(P0, x, dat, backend="eigh"))(xi)
    assert np.all(np.isfinite(np.asarray(gx)))


def _write_mini_config(tmp):
    """A minimal non-DUNE-style config: uniform bins, Gaussian smearing,
    @signalerror systematics, profiletype 2."""
    E = np.linspace(0.25, 9.75, 20)
    flux = np.zeros((20, 7))
    flux[:, 0] = E
    flux[:, 2] = 1e-11 * np.exp(-0.5 * ((E - 2.5) / 1.0) ** 2)   # numu
    flux[:, 1] = 0.01 * flux[:, 2]                               # beam nue
    np.savetxt(tmp / "flux.txt", flux)
    logE = np.linspace(-1, 1, 21)
    xs = np.zeros((21, 7)); xs[:, 1:] = 0.7                      # sigma/E const
    np.savetxt(tmp / "xsec.txt", np.column_stack([logE, xs[:, 1:]]))
    (tmp / "MINI.glb").write_text(textwrap.dedent("""\
        %!GLoBES
        $version="3.2.13"
        $profiletype = 2
        $densitytab = {2.8}
        $lengthtab = {1300.0}
        $target_mass = 10.0
        $emin = 0.0
        $emax = 10.0
        $bins = 40
        nuflux(#f)<
          @flux_file="./flux.txt"
          @time = 5.0
          @power = 10.0
          @norm = 1.0
        >
        cross(#CC)<
          @cross_file = "./xsec.txt"
        >
        energy(#sm)<
          @type = 1
          @sigma_e = {0.15, 0.0, 0.0}
        >
        channel(#sig)<
          @channel = #f: +: m: e: #CC: #sm
        >
        channel(#bkg)<
          @channel = #f: +: e: e: #CC: #sm
        >
        rule(#app)<
          @signal = 1.0@#sig
          @background = 1.0@#bkg
          @signalerror = 0.02 : 0.0001
          @backgrounderror = 0.05 : 0.0001
          @energy_window = 0.5 : 8.0
        >
    """))
    return tmp / "MINI.glb"


def test_mini_config_gaussian_smearing_and_fallback_sys(tmp_path):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tilt-ignored warning
        exp = globes.load(_write_mini_config(tmp_path))
    assert exp.nreco == 40 and abs(exp.baseline - 1300.0) < 1e-9
    # Gaussian smearing rows are (near-)normalized inside the range
    R = exp.smear_matrix("sm")
    mid = np.argmin(np.abs(exp.samp_c - 5.0))
    assert abs(R[:, mid].sum() - 1.0) < 1e-3
    # fallback @signalerror -> synthesized labels with the right sigmas
    assert exp.sys_sigma["app_sig"] == 0.02 and exp.sys_sigma["app_bkg"] == 0.05
    sp = exp.spectra(P0)
    tot = float(np.asarray(sp["app"])[exp.windows["app"]].sum())
    assert tot > 1.0  # non-trivial rate
    g = jax.grad(lambda p: exp.spectra(p)["app"].sum())(P0)
    assert np.isfinite(float(g.deltacp))


def test_profiletype1_prem_chord(tmp_path):
    glb = _write_mini_config(tmp_path)
    text = glb.read_text().replace("$profiletype = 2", "$profiletype = 1")
    text = text.replace("$densitytab = {2.8}\n", "")
    glb.write_text(text)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp = globes.load(glb)
    # PREM chord for 1300 km: several crust/mantle layers, correct total length
    assert len(exp.lengths) >= 2
    assert abs(exp.baseline - 1300.0) < 1.0
    assert np.all(exp.density > 0) and np.max(exp.density) < 4.0  # crust/upper mantle
