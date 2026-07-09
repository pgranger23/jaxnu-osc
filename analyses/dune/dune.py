"""DUNE long-baseline oscillation analysis with jaxnu.

Reproduces the DUNE TDR GLoBES configuration (arXiv:2103.04797): the far-detector
event spectra and the CP-violation / mass-ordering sensitivities, using jaxnu for
the (differentiable) oscillation probabilities and the official DUNE GLoBES inputs
(flux, cross-sections, per-channel migration matrices, efficiencies, and the
normalization systematics) shipped under ``globes/``.

The forward model per reconstructed bin i and channel c:

    N_c[i] = norm * post_eff_c[i] * sum_j  R_c[i,j] * flux_c[j] * xsec_c[j] * P_c[j]

summed over channels within each rule (nu_e app, nu_e-bar app, nu_mu dis,
nu_mu-bar dis). Only the oscillation probabilities P_c depend on the physics
parameters, and they come from jaxnu -> the whole spectrum is differentiable.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import jaxnu
from jaxnu import OscParams

HERE = Path(__file__).resolve().parent
G = HERE / "globes"

L_KM = 1284.9
RHO = 2.848            # g/cm^3, average crust density for this baseline
# NuFIT 4.0 (the paper's Table); normal ordering central values.
NUFIT_NO = dict(theta12=0.5903, theta13=0.150, theta23=0.866,
                dm21=7.39e-5, dm31=2.451e-3 + 7.39e-5)
# relative (1-sigma-equivalent) priors used by DUNE
PRIORS = dict(theta12=0.023, theta13=0.015, dm21=0.028, rho=0.02)

# exposure: single mode = @time 6.5 yr, @power 11 (1e20 POT/yr), 40 kt.
POT_PER_MODE = 11.0 * 1e20 * 6.5
N_NUCLEON = 40.0 * 1e9 * 6.022e23
# overall normalization: POT * N_nucleon * (m^2->cm^2) * (xsec unit 1e-38);
# CAL_NORM calibrates to the DUNE reference spectrum (convention factor).
CAL_NORM = 1.21
NORM = CAL_NORM * POT_PER_MODE * N_NUCLEON * 1e-4 * 1e-38

E_, M_, T_ = 0, 1, 2  # flavor indices


def _read(path):
    return Path(path).read_text()


class DuneData:
    """Parses the GLoBES config and builds the (jaxnu-driven) forward model."""

    def __init__(self):
        glb = _read(G / "DUNE_GLoBES.glb")

        def arr(name):
            m = re.search(r"\$%s\s*=\s*\{([^}]*)\}" % name, glb)
            return np.array([float(x) for x in m.group(1).split(",")])

        emin = float(re.search(r"\$emin\s*=\s*([0-9.]+)", glb).group(1))
        smin = float(re.search(r"\$sampling_min\s*=\s*([0-9.]+)", glb).group(1))
        self.reco_edges = np.concatenate([[emin], emin + np.cumsum(arr("binsize"))])
        self.reco_c = 0.5 * (self.reco_edges[:-1] + self.reco_edges[1:])
        samp_edges = np.concatenate([[smin], smin + np.cumsum(arr("sampling_stepsize"))])
        self.samp_c = 0.5 * (samp_edges[:-1] + samp_edges[1:])
        self.samp_w = np.diff(samp_edges)
        self.nreco, self.nsamp = len(self.reco_c), len(self.samp_c)

        fl = "flux/histos_g4lbne_v3r5p4_QGSP_BERT_OptimizedEngineeredNov2017_%s_LBNEFD_globes_flux.txt"
        self.flux = {"FHC": np.loadtxt(G / (fl % "neutrino"))[:, :7],
                     "RHC": np.loadtxt(G / (fl % "antineutrino"))[:, :7]}
        self.xc = np.loadtxt(G / "xsec/xsec_cc.dat")
        self.xn = np.loadtxt(G / "xsec/xsec_nc.dat")
        self._smr, self._eff = {}, {}

        # systematic (normalization) uncertainties
        d = _read(G / "definitions.inc")
        def defval(k):
            return float(re.search(r"%s\s*=\s*([0-9.]+)" % k, d).group(1))
        self.sys_sigma = {
            "nue_sig": defval("ERR_NUE_SIG"), "nue_sigbar": defval("ERR_NUE_SIGBAR"),
            "numu_sig": defval("ERR_NUMU_SIG"), "numu_sigbar": defval("ERR_NUMU_SIGBAR"),
            "numu_bg": defval("ERR_NUMU_BG"), "nue_bg": defval("ERR_NUE_BG"),
            "nutau_bg": defval("ERR_NUTAU_BG"), "nue_bgbar": defval("ERR_NUE_BGBAR"),
            "nc_bgdis": defval("ERR_NC_BGDIS")}
        self.sys_index = {k: i for i, k in enumerate(self.sys_sigma)}
        self.sigma_vec = jnp.asarray([self.sys_sigma[k] for k in self.sys_sigma])

        self._build_channels()
        self._precompute()

    # --- inputs -----------------------------------------------------------
    def smear(self, name):
        if name not in self._smr:
            rows = re.findall(r"\{(\d+)\s*,\s*(\d+)\s*,([^}]*)\}",
                              _read(G / f"smr/{name}.txt"))
            R = np.zeros((self.nreco, self.nsamp))
            for i, (a, b, vals) in enumerate(rows):
                a, b = int(a), int(b)
                v = np.array([float(x) for x in vals.split(",") if x.strip()])
                R[i, a:b + 1] = v[:b - a + 1]
            self._smr[name] = R
        return self._smr[name]

    def posteff(self, name):
        if name not in self._eff:
            v = re.search(r"\{([^}]*)\}", _read(G / f"eff/post_{name}.txt")).group(1)
            self._eff[name] = np.array([float(x) for x in v.split(",")])
        return self._eff[name]

    def _flux_interp(self, mode, col):
        f = self.flux[mode]
        return np.interp(self.samp_c, f[:, 0], f[:, col])

    def _xsec_interp(self, table, col):  # sigma = (sigma/E) * E
        return np.interp(np.log10(np.maximum(self.samp_c, 1e-6)),
                         table[:, 0], table[:, 1 + col]) * self.samp_c

    # --- channel table ----------------------------------------------------
    def _build_channels(self):
        # (mode, flux_col, xsec_table, xsec_col, (out,in) or None for NC,
        #  smear, posteff, sys_label, is_signal)
        cc, nc = self.xc, self.xn
        def app(mode, tag):  # tag = FHC/RHC posteff prefix
            s = "sig" if tag == "FHC" else "sigbar"  # only affects signal sys below
            return [
                # signal
                (mode, 2, cc, 0, (E_, M_), False, "app_nue_sig", f"app_{tag}_nue_sig",
                 "nue_sig" if tag == "FHC" else "nue_sigbar", True),
                (mode, 5, cc, 3, (E_, M_), True, "app_nuebar_sig", f"app_{tag}_nuebar_sig",
                 "nue_sig" if tag == "FHC" else "nue_sigbar", True),
                # backgrounds
                (mode, 1, cc, 0, (E_, E_), False, "app_nue_bkg", f"app_{tag}_nue_bkg",
                 "nue_bg" if tag == "FHC" else "nue_bgbar", False),
                (mode, 4, cc, 3, (E_, E_), True, "app_nuebar_bkg", f"app_{tag}_nuebar_bkg",
                 "nue_bg" if tag == "FHC" else "nue_bgbar", False),
                (mode, 2, cc, 1, (M_, M_), False, "app_numu_bkg", f"app_{tag}_numu_bkg", "numu_bg", False),
                (mode, 5, cc, 4, (M_, M_), True, "app_numubar_bkg", f"app_{tag}_numubar_bkg", "numu_bg", False),
                (mode, 2, cc, 2, (T_, M_), False, "app_nutau_bkg", f"app_{tag}_nutau_bkg", "nutau_bg", False),
                (mode, 5, cc, 5, (T_, M_), True, "app_nutaubar_bkg", f"app_{tag}_nutaubar_bkg", "nutau_bg", False),
                (mode, 2, nc, 1, None, False, "app_NC_bkg", f"app_{tag}_NC_bkg", "numu_bg", False),
                (mode, 5, nc, 4, None, True, "app_aNC_bkg" if tag == "RHC" else "app_NC_bkg",
                 f"app_{tag}_aNC_bkg", "numu_bg", False),
            ]
        def dis(mode, tag):
            return [
                (mode, 2, cc, 1, (M_, M_), False, "dis_numu_sig", f"dis_{tag}_numu_sig",
                 "numu_sig" if tag == "FHC" else "numu_sigbar", True),
                (mode, 5, cc, 4, (M_, M_), True, "dis_numubar_sig", f"dis_{tag}_numubar_sig",
                 "numu_sig" if tag == "FHC" else "numu_sigbar", True),
                (mode, 2, cc, 2, (T_, M_), False, "dis_nutau_bkg", f"dis_{tag}_nutau_bkg", "nutau_bg", False),
                (mode, 5, cc, 5, (T_, M_), True, "dis_nutaubar_bkg", f"dis_{tag}_nutaubar_bkg", "nutau_bg", False),
                (mode, 2, nc, 1, None, False, "dis_NC_bkg", f"dis_{tag}_NC_bkg", "nc_bgdis", False),
                (mode, 5, nc, 4, None, True, "dis_aNC_bkg", f"dis_{tag}_NC_bkg", "nc_bgdis", False),
            ]
        self.rules = {"nue_app": app("FHC", "FHC"), "nuebar_app": app("RHC", "RHC"),
                      "numu_dis": dis("FHC", "FHC"), "numubar_dis": dis("RHC", "RHC")}

    def _precompute(self):
        # For every channel precompute the energy-independent kernel
        #   K_c[i,j] = post_eff_c[i] * R_c[i,j] * flux_c[j] * xsec_c[j] * NORM * samp_w[j]
        # so that N_c[i] = sum_j K_c[i,j] * P_c[j]  (P from jaxnu).
        self.kernels, self.meta = {}, {}
        for rule, chans in self.rules.items():
            Ks, ms = [], []
            for (mode, fcol, xt, xcol, oscch, bar, smr, eff, sys, sig) in chans:
                base = (self._flux_interp(mode, fcol) * self._xsec_interp(xt, xcol)
                        * NORM * self.samp_w)
                K = (self.posteff(eff)[:, None] * self.smear(smr)) * base[None, :]
                Ks.append(jnp.asarray(K))
                ms.append((oscch, bar, sys, sig))
            self.kernels[rule] = Ks
            self.meta[rule] = ms
        self.win = (self.reco_c > 0.5) & (self.reco_c < 8.0)  # energy window (GeV)


# --- oscillation + spectra (jaxnu) ---------------------------------------

def _osc_probs(params, rho=RHO):
    """P[out,in] for nu and nubar at the sampling energies, from jaxnu."""
    data = _DATA
    E = jnp.asarray(data.samp_c)
    Pn = jaxnu.probability_constant(params, E, L_KM, density=rho, ye=0.5, backend="eigh")
    Pb = jaxnu.probability_constant(params, E, L_KM, density=rho, ye=0.5, anti=True, backend="eigh")
    return Pn, Pb


def rule_components(params, rho=RHO):
    """Per rule, list of (sys_label, is_signal, reco_spectrum) — differentiable."""
    data = _DATA
    Pn, Pb = _osc_probs(params, rho)
    one = jnp.ones(data.nsamp)
    out = {}
    for rule in data.rules:
        comps = []
        for K, (oscch, bar, sys, sig) in zip(data.kernels[rule], data.meta[rule]):
            if oscch is None:
                prob = one
            else:
                o, i = oscch
                prob = (Pb if bar else Pn)[:, o, i]
            comps.append((sys, sig, K @ prob))
        out[rule] = comps
    return out


def params_from(vec):
    # vec = [theta23, theta13, dm31, deltacp]
    return OscParams(theta12=jnp.asarray(NUFIT_NO["theta12"]),
                     theta13=vec[1], theta23=vec[0], deltacp=vec[3],
                     dm21=jnp.asarray(NUFIT_NO["dm21"]), dm31=vec[2])


# --- data singleton -------------------------------------------------------
_DATA = None


def data():
    global _DATA
    if _DATA is None:
        _DATA = DuneData()
    return _DATA


def total_spectra(params):
    """Per-rule total (signal+bkg) reconstructed spectrum (for plotting/Asimov)."""
    comps = rule_components(params)
    return {r: sum(a for (_s, _sig, a) in comps) for r, comps in comps.items()}


def asimov(deltacp, **over):
    """Asimov 'data': total spectra at NuFIT-NO true params with given delta_cp."""
    kw = dict(NUFIT_NO); kw.update(over)
    p = OscParams(theta12=jnp.asarray(kw["theta12"]), theta13=jnp.asarray(kw["theta13"]),
                  theta23=jnp.asarray(kw["theta23"]), deltacp=jnp.asarray(deltacp),
                  dm21=jnp.asarray(kw["dm21"]), dm31=jnp.asarray(kw["dm31"]))
    return {r: np.asarray(v) for r, v in total_spectra(p).items()}


# --- chi-square with normalization systematics ---------------------------

def fine_W():
    """Rebinning matrix that just selects the fine bins in the energy window."""
    d = data()
    idx = np.where(d.win)[0]
    W = np.zeros((len(idx), d.nreco))
    W[np.arange(len(idx)), idx] = 1.0
    return jnp.asarray(W)


def rebin_matrix(edges):
    """Rebinning matrix ``W`` (K, nreco) for K analysis bins with the given edges.

    ``W[k, i]`` is the fraction of native fine-bin ``i`` (width-weighted overlap)
    that falls in analysis bin ``k = [edges[k], edges[k+1]]``, so
    ``coarse_spectrum = W @ fine_spectrum``. **Differentiable in ``edges``** (via
    clip/min/max), which is the intended hook for a binning-optimization agent:
    parametrize ``edges``, pass ``W = rebin_matrix(edges)`` to
    :func:`cpv_sensitivity` (evaluation) or build a Fisher-information objective from
    ``W @ jacobian`` (optimization), and gradient-ascend the sensitivity.
    """
    d = data()
    lo = jnp.asarray(d.reco_edges[:-1]); hi = jnp.asarray(d.reco_edges[1:])
    edges = jnp.asarray(edges)
    overlap = jnp.clip(jnp.minimum(hi[None, :], edges[1:, None])
                       - jnp.maximum(lo[None, :], edges[:-1, None]), 0.0, None)
    return overlap / (hi - lo)[None, :]


def chi2(free, data_tuple, W):
    """free = [theta23, theta13, dm31, deltacp, drho, xi_0..xi_8] (15).

    ``W`` (K, nreco) rebins the fine model into the K analysis bins used in the
    Poisson chi2. ``drho`` is the fractional matter-density pull."""
    d = data()
    th23, th13, dm31, dcp, drho = free[0], free[1], free[2], free[3], free[4]
    xi = free[5:14]
    params = OscParams(theta12=jnp.asarray(NUFIT_NO["theta12"]), theta13=th13,
                       theta23=th23, deltacp=dcp,
                       dm21=jnp.asarray(NUFIT_NO["dm21"]), dm31=dm31)
    comps = rule_components(params, RHO * (1.0 + drho))
    chi = 0.0
    for rule, dat in zip(d.rules, data_tuple):
        model = jnp.zeros(d.nreco)
        for (sys, _sig, arr) in comps[rule]:
            model = model + (1.0 + xi[d.sys_index[sys]]) * arr
        m = jnp.clip(W @ model, 1e-9, None); obs = W @ dat
        chi = chi + 2.0 * jnp.sum(m - obs + obs * jnp.log(obs / m))
    chi = chi + jnp.sum((xi / d.sigma_vec) ** 2)                 # syst penalties
    chi = chi + ((th13 - NUFIT_NO["theta13"]) /
                 (NUFIT_NO["theta13"] * PRIORS["theta13"])) ** 2  # theta13 prior
    chi = chi + (drho / PRIORS["rho"]) ** 2                       # matter density prior
    return chi


chi2_vg = jax.jit(jax.value_and_grad(chi2))


def _minimize(data_tuple, starts, bounds, W):
    """Gradient-based profile of chi2 (jaxnu gives the exact gradient)."""
    from scipy.optimize import minimize
    best = np.inf
    for x0 in starts:
        def fun(x):
            v, g = chi2_vg(jnp.asarray(x), data_tuple, W)
            return float(v), np.asarray(g)
        r = minimize(fun, np.asarray(x0), jac=True, method="L-BFGS-B",
                     bounds=bounds, options=dict(maxiter=300, ftol=1e-10))
        best = min(best, r.fun)
    return best


def _octants():
    th = NUFIT_NO["theta23"]
    return [th, np.arcsin(np.sqrt(1 - np.sin(th) ** 2))]


def cpv_sensitivity(deltacp_true_over_pi, W=None):
    """sigma = sqrt(dChi2) for CP violation vs true delta_cp. ``W`` = binning."""
    d = data()
    if W is None:
        W = fine_W()
    XI0 = [0.0] * 9
    out = []
    for x in deltacp_true_over_pi:
        dat = tuple(jnp.asarray(asimov(x * np.pi)[r]) for r in d.rules)
        chi = np.inf
        for dtest in (0.0, np.pi):  # CP-conserving hypotheses
            bounds = [(0.60, 0.95), (0.10, 0.20), (2.30e-3, 2.75e-3),
                      (dtest, dtest), (-0.1, 0.1)] + [(-0.6, 0.6)] * 9
            starts = [[o, NUFIT_NO["theta13"], NUFIT_NO["dm31"], dtest, 0.0] + XI0
                      for o in _octants()]
            chi = min(chi, _minimize(dat, starts, bounds, W))
        out.append(np.sqrt(max(chi, 0.0)))
    return np.array(out)


def mass_ordering_sensitivity(deltacp_true_over_pi, W=None):
    """sigma = sqrt(dChi2) to reject the wrong (inverted) ordering, true NO."""
    d = data()
    if W is None:
        W = fine_W()
    XI0 = [0.0] * 9
    dm31_io = -2.512e-3 + NUFIT_NO["dm21"]        # NuFIT 4.0 IO
    bounds = [(0.60, 0.95), (0.10, 0.20), (-2.75e-3, -2.30e-3),
              (-np.pi, np.pi), (-0.1, 0.1)] + [(-0.6, 0.6)] * 9  # fit inverted ordering
    out = []
    for x in deltacp_true_over_pi:
        dat = tuple(jnp.asarray(asimov(x * np.pi)[r]) for r in d.rules)
        starts = [[o, NUFIT_NO["theta13"], dm31_io, dt, 0.0] + XI0
                  for o in _octants() for dt in (-np.pi / 2, 0.0, np.pi / 2, np.pi)]
        out.append(np.sqrt(max(_minimize(dat, starts, bounds, W), 0.0)))
    return np.array(out)
