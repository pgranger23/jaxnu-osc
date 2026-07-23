"""Generic GLoBES (AEDL) experiment loader — a "differentiable GLoBES".

Parses a GLoBES experiment definition (``.glb`` + its includes) and builds a
differentiable jaxnu forward model: per-channel event rates

    N_c[i] = norm * post_eff_c[i] * sum_j R_c[i,j] * (flux_c[j] * xsec_c[j]
             * P_c[j] * dE_j * pre_eff_c[j] + pre_bkg_c[j]) + post_bkg_c[i]

with the oscillation probabilities ``P_c`` supplied by jaxnu (so every rate is
differentiable in the oscillation parameters), plus rule assembly (signal +
backgrounds), energy windows, and the normalization systematics.

Supported AEDL subset (pragmatic; covers the published DUNE TDR configuration and
similar modern configs):

* ``include "file"`` (recursive, paths relative to the main ``.glb``), C comments,
  scalar definitions (``NAME = 3.5``) and list variables (``%name = {...}`` with
  ``copy(%name)``);
* ``$emin/$emax/$binsize|$bins``, ``$sampling_*`` (defaults to the analysis
  binning), ``$target_mass``;
* ``$profiletype`` 1 (PREM chord via :mod:`jaxnu.earth`), 2/3 (``$densitytab`` /
  ``$lengthtab`` layers);
* ``nuflux`` (``@flux_file/@time/@power/@norm``), ``cross`` (``@cross_file``),
  ``energy`` smearing as explicit migration rows ``{jmin,jmax, v...}`` **or**
  analytic Gaussian ``@type=1, @sigma_e={a,b,c}`` (sigma = a*E + b*sqrt(E) + c);
* ``channel`` with ``NOSC_`` flavors and pre/post smearing efficiencies and
  backgrounds; ``rule`` with ``coef@#channel`` lists, ``@energy_window``, and
  normalization systematics via ``@sys_on_multiex_errors_*`` sys lists or
  ``@signalerror/@backgrounderror`` (first component; tilts are ignored with a
  warning).

**Normalization.** Rates use the physical convention
``POT(@time * @power * pot_unit) x nucleons($target_mass) x flux x sigma``, times
an overall ``scale`` you can calibrate against a published rate. The file's
``@norm`` is *stored but not applied* — GLoBES files use it as an internal,
convention-dependent fudge factor that is not portable (for the DUNE TDR files,
``scale = 1.21`` reproduces the published spectra).
"""

from __future__ import annotations

import dataclasses
import re
import warnings
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import erf

from . import constants as C
from . import earth as _earth
from .oscillator import probability_constant, probability_profile

_FLAV = {"e": 0, "m": 1, "t": 2}


# ---------------------------------------------------------------------------
# text-level AEDL handling
# ---------------------------------------------------------------------------

def _strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def _resolve_includes(text, root):
    def repl(m):
        sub = (root / m.group(1)).read_text()
        return _resolve_includes(_strip_comments(sub), root)
    return re.sub(r'include\s+"([^"]+)"', repl, text)


def _blocks(text, kind):
    """All ``kind(#name)< body >`` blocks as {name: [bodies...]}."""
    out = {}
    for m in re.finditer(kind + r"\s*\(\s*#(\w+)\s*\)\s*<(.*?)>", text, flags=re.S):
        out.setdefault(m.group(1), []).append(m.group(2))
    return out


class _Defs:
    """Scalar definitions (NAME = value) with numeric lookup."""

    def __init__(self, text):
        self.d = {m.group(1): float(m.group(2)) for m in re.finditer(
            r"^\s*([A-Za-z_]\w*)\s*=\s*([0-9.eE+\-]+)\s*$", text, flags=re.M)}

    def num(self, token, default=None):
        token = token.strip()
        if not token:
            return default
        try:
            return float(token)
        except ValueError:
            if token in self.d:
                return self.d[token]
            if default is not None:
                return default
            raise ValueError(f"cannot resolve numeric token {token!r}")


def _attr(body, name, default=None):
    m = re.search(r"@" + name + r"\s*=\s*([^\n@]+)", body)
    return m.group(1).strip() if m else default


def _floatlist(s):
    return np.array([float(x) for x in s.replace("{", "").replace("}", "").split(",")
                     if x.strip()])


# ---------------------------------------------------------------------------
# parsed structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Channel:
    name: str
    flux: str
    anti: bool
    src: int            # initial flavor index (0,1,2)
    dst: int            # final flavor index
    noosc: bool         # NOSC_ -> no oscillation applied
    xsec: str
    smear: str
    pre_eff: np.ndarray | None = None
    post_eff: np.ndarray | None = None
    pre_bkg: np.ndarray | None = None
    post_bkg: np.ndarray | None = None


@dataclasses.dataclass
class Rule:
    name: str
    signal: list        # [(coef, channel_name)]
    background: list
    sys_sig: list       # per-channel systematic label
    sys_bkg: list
    window: tuple


class GlobesExperiment:
    """A GLoBES experiment as a differentiable jaxnu forward model."""

    def __init__(self, path, scale=1.0, ye=0.5, pot_unit=1e20):
        path = Path(path)
        self.root = path.parent
        self.scale, self.ye, self.pot_unit = scale, ye, pot_unit
        text = _resolve_includes(_strip_comments(path.read_text()), self.root)
        self._parse(text)
        self._precompute()

    # --- parsing -----------------------------------------------------------
    def _parse(self, text):
        defs = _Defs(text)
        self.defs = defs

        def svar(name, default=None):
            m = re.search(r"\$" + name + r"\s*=\s*(\{[^}]*\}|[^\n]+)", text)
            return m.group(1).strip() if m else default

        # binning
        emin = defs.num(svar("emin", "0"))
        emax = defs.num(svar("emax", "0"))
        binsize = svar("binsize")
        if binsize:
            self.reco_edges = np.concatenate([[emin], emin + np.cumsum(_floatlist(binsize))])
        else:
            nb = int(defs.num(svar("bins")))
            self.reco_edges = np.linspace(emin, emax, nb + 1)
        self.reco_c = 0.5 * (self.reco_edges[:-1] + self.reco_edges[1:])
        smin = defs.num(svar("sampling_min", str(emin)))
        sstep = svar("sampling_stepsize")
        if sstep:
            se = np.concatenate([[smin], smin + np.cumsum(_floatlist(sstep))])
        elif svar("sampling_points"):
            smax = defs.num(svar("sampling_max", str(emax)))
            se = np.linspace(smin, smax, int(defs.num(svar("sampling_points"))) + 1)
        else:
            se = self.reco_edges
        self.samp_c = 0.5 * (se[:-1] + se[1:])
        self.samp_w = np.diff(se)
        self.nreco, self.nsamp = len(self.reco_c), len(self.samp_c)

        # matter profile
        ptype = int(defs.num(svar("profiletype", "3")))
        if ptype in (2, 3):
            self.density = _floatlist(svar("densitytab"))
            self.lengths = _floatlist(svar("lengthtab"))
        else:  # profiletype 1: PREM chord for the given baseline
            L = defs.num(svar("baselinelength", svar("lengthtab", "0").strip("{}")))
            cz = -L / (2.0 * _earth.R_EARTH_KM)
            rho, ye_seg, seg = _earth.chord_segments(cz, _earth.shell_table(2))
            keep = np.asarray(seg) > 1e-6
            self.density = np.asarray(rho)[keep]
            self.lengths = np.asarray(seg)[keep]
        self.baseline = float(np.sum(self.lengths))
        self.target_mass = defs.num(svar("target_mass", "1"))

        # list variables
        self.lists = {m.group(1): _floatlist(m.group(2)) for m in
                      re.finditer(r"%(\w+)\s*=\s*\{([^}]*)\}", text)}

        # fluxes
        self.flux = {}
        for name, bodies in _blocks(text, "nuflux").items():
            b = bodies[-1]
            f = re.search(r'@flux_file\s*=\s*"([^"]+)"', b).group(1)
            self.flux[name] = dict(
                table=np.loadtxt(self.root / f)[:, :7],
                time=defs.num(_attr(b, "time", "1")),
                power=defs.num(_attr(b, "power", "1")),
                norm=defs.num(_attr(b, "norm", "1")))

        # cross sections: log10(E), sigma/E per flavor (units 1e-38 cm^2/GeV)
        self.xsec = {}
        for name, bodies in _blocks(text, "cross").items():
            f = re.search(r'@cross_file\s*=\s*"([^"]+)"', bodies[-1]).group(1)
            self.xsec[name] = np.loadtxt(self.root / f)

        # smearing (merge multiple blocks with the same name; keep non-empty)
        self._smear_bodies = {}
        for name, bodies in _blocks(text, "energy").items():
            body = max(bodies, key=len)
            self._smear_bodies[name] = body
        self._smear_cache = {}

        # systematics
        self.sys_sigma = {}
        for name, bodies in _blocks(text, "sys").items():
            self.sys_sigma[name] = defs.num(_attr(bodies[-1], "error"))

        # channels
        self.channels = {}
        for name, bodies in _blocks(text, "channel").items():
            b = bodies[-1]
            toks = [t.strip() for t in _attr(b, "channel").split(":")]
            fluxn, sign, src, dst, xsec, smear = toks[:6]
            ch = Channel(
                name=name, flux=fluxn.lstrip("#"), anti=(sign == "-"),
                src=_FLAV[src.replace("NOSC_", "")],
                dst=_FLAV[dst.replace("NOSC_", "")],
                noosc=("NOSC_" in src or "NOSC_" in dst),
                xsec=xsec.lstrip("#"), smear=smear.lstrip("#"))
            for attr, field in [("pre_smearing_efficiencies", "pre_eff"),
                                ("post_smearing_efficiencies", "post_eff"),
                                ("pre_smearing_background", "pre_bkg"),
                                ("post_smearing_background", "post_bkg")]:
                v = _attr(b, attr)
                if v:
                    m = re.match(r"copy\(\s*%(\w+)\s*\)", v)
                    arr = self.lists[m.group(1)] if m else _floatlist(v)
                    setattr(ch, field, arr)
            self.channels[name] = ch

        # rules
        self.rules = {}
        for name, bodies in _blocks(text, "rule").items():
            b = bodies[-1]

            def chan_list(key):
                # NB: values like "1.0@#chan : ..." contain '@', so capture the
                # full line rather than using the generic _attr.
                m = re.search(r"@" + key + r"\s*=\s*([^\n]+)", b)
                out = []
                if m:
                    for tok in m.group(1).split(":"):
                        t = re.match(r"\s*([0-9.eE+\-]+)\s*@\s*#(\w+)", tok)
                        if t:
                            out.append((float(t.group(1)), t.group(2)))
                return out

            def sys_list(key, chans, fallback_err, label):
                m = re.search(r"@" + key + r"\s*=\s*((?:\{[^}]*\}\s*:?\s*)+)", b)
                if m:
                    names = re.findall(r"\{\s*#(\w+)", m.group(1))
                    return [n for n in names]
                # fall back to @signalerror / @backgrounderror = (norm, tilt)
                v = _attr(b, fallback_err)
                if v:
                    nums = re.findall(r"[0-9.eE+\-]+", v)
                    lab = f"{name}_{label}"
                    self.sys_sigma[lab] = float(nums[0])
                    if len(nums) > 1 and float(nums[1]) > 0:
                        warnings.warn(f"rule {name}: spectral tilt error ignored "
                                      "(normalization systematics only)")
                    return [lab] * len(chans)
                return [None] * len(chans)

            sig = chan_list("signal")
            bkg = chan_list("background")
            win = _attr(b, "energy_window")
            window = tuple(float(x) for x in win.split(":")) if win else (0.0, np.inf)
            self.rules[name] = Rule(
                name=name, signal=sig, background=bkg,
                sys_sig=sys_list("sys_on_multiex_errors_sig", sig, "signalerror", "sig"),
                sys_bkg=sys_list("sys_on_multiex_errors_bg", bkg, "backgrounderror", "bkg"),
                window=window)

        self.sys_labels = sorted({s for r in self.rules.values()
                                  for s in r.sys_sig + r.sys_bkg if s})
        self.sys_index = {k: i for i, k in enumerate(self.sys_labels)}
        self.sigma_vec = jnp.asarray([self.sys_sigma[k] for k in self.sys_labels])

    # --- smearing ----------------------------------------------------------
    def smear_matrix(self, name):
        if name in self._smear_cache:
            return self._smear_cache[name]
        body = self._smear_bodies[name]
        rows = re.findall(r"\{(\d+)\s*,\s*(\d+)\s*,([^}]*)\}", body)
        if rows and "@energy" in body:
            R = np.zeros((self.nreco, self.nsamp))
            for i, (a, bb, vals) in enumerate(rows):
                a, bb = int(a), int(bb)
                v = np.array([float(x) for x in vals.split(",") if x.strip()])
                R[i, a:bb + 1] = v[:bb - a + 1]
        else:  # analytic Gaussian: sigma(E) = a*E + b*sqrt(E) + c
            se = _attr(body, "sigma_e")
            a, bsq, c = (_floatlist(se).tolist() + [0.0, 0.0, 0.0])[:3]
            E = self.samp_c
            sig = np.maximum(a * E + bsq * np.sqrt(np.maximum(E, 0)) + c, 1e-6)
            lo = (self.reco_edges[:-1][:, None] - E[None, :]) / (np.sqrt(2) * sig)
            hi = (self.reco_edges[1:][:, None] - E[None, :]) / (np.sqrt(2) * sig)
            R = 0.5 * np.asarray(erf(hi) - erf(lo))
        self._smear_cache[name] = R
        return R

    # --- forward model -----------------------------------------------------
    def _norm(self, fluxname):
        f = self.flux[fluxname]
        pot = f["time"] * f["power"] * self.pot_unit
        nucleons = self.target_mass * 1e9 * 6.02214076e23
        return self.scale * pot * nucleons * 1e-4 * 1e-38

    def _precompute(self):
        """Per-channel static kernels K[i,j] with N_c = K @ P + const."""
        self.kernels, self.const = {}, {}
        for name, ch in self.channels.items():
            f = self.flux[ch.flux]["table"]
            fcol = 1 + (3 if ch.anti else 0) + ch.src
            flux = np.interp(self.samp_c, f[:, 0], f[:, fcol])
            xt = self.xsec[ch.xsec]
            xcol = 1 + (3 if ch.anti else 0) + ch.dst
            sigma = np.interp(np.log10(np.maximum(self.samp_c, 1e-6)),
                              xt[:, 0], xt[:, xcol]) * self.samp_c
            base = flux * sigma * self.samp_w * self._norm(ch.flux)
            if ch.pre_eff is not None:
                base = base * ch.pre_eff
            R = self.smear_matrix(ch.smear)
            post = ch.post_eff if ch.post_eff is not None else 1.0
            K = (R * base[None, :])
            if np.ndim(post):
                K = post[:, None] * K
            const = np.zeros(self.nreco)
            if ch.pre_bkg is not None:
                v = R @ ch.pre_bkg
                const += (post * v) if np.ndim(post) else post * v
            if ch.post_bkg is not None:
                const += ch.post_bkg
            self.kernels[name] = jnp.asarray(K)
            self.const[name] = jnp.asarray(const)
        self.windows = {r.name: (self.reco_c > r.window[0]) & (self.reco_c < r.window[1])
                        for r in self.rules.values()}

    def _osc(self, params, anti, **osc_kw):
        E = jnp.asarray(self.samp_c)
        if len(self.lengths) == 1:
            return probability_constant(params, E, float(self.lengths[0]),
                                        density=float(self.density[0]), ye=self.ye,
                                        anti=anti, **osc_kw)
        return probability_profile(params, E, self.density,
                                   self.ye * np.ones_like(self.density),
                                   self.lengths, anti=anti, **osc_kw)

    def channel_rates(self, params, **osc_kw):
        """dict channel -> reconstructed-energy spectrum (differentiable)."""
        Pn = self._osc(params, False, **osc_kw)
        Pb = self._osc(params, True, **osc_kw)
        out = {}
        for name, ch in self.channels.items():
            if ch.noosc:
                prob = jnp.ones(self.nsamp)
            else:
                prob = (Pb if ch.anti else Pn)[:, ch.dst, ch.src]
            out[name] = self.kernels[name] @ prob + self.const[name]
        return out

    def components(self, params, **osc_kw):
        """dict rule -> [(sys_label, is_signal, spectrum)] (differentiable)."""
        rates = self.channel_rates(params, **osc_kw)
        out = {}
        for r in self.rules.values():
            comp = []
            for (coef, cn), sys in zip(r.signal, r.sys_sig):
                comp.append((sys, True, coef * rates[cn]))
            for (coef, cn), sys in zip(r.background, r.sys_bkg):
                comp.append((sys, False, coef * rates[cn]))
            out[r.name] = comp
        return out

    def spectra(self, params, **osc_kw):
        """dict rule -> total (signal + background) spectrum."""
        return {r: sum(a for (_s, _i, a) in comp)
                for r, comp in self.components(params, **osc_kw).items()}

    def chi2(self, params, xi, data, **osc_kw):
        """Poisson chi2 with normalization systematics ``xi`` (len sys_labels).

        ``data``: dict rule -> observed spectrum on the fine bins. Add your own
        oscillation-parameter priors externally. Differentiable in ``params``
        and ``xi``.
        """
        comps = self.components(params, **osc_kw)
        chi = jnp.sum((xi / self.sigma_vec) ** 2)
        for r, comp in comps.items():
            model = jnp.zeros(self.nreco)
            for (sys, _is_sig, arr) in comp:
                f = 1.0 + (xi[self.sys_index[sys]] if sys else 0.0)
                model = model + f * arr
            w = self.windows[r]
            m = jnp.clip(model[w], 1e-9, None)
            o = jnp.clip(data[r][w], 0.0, None)
            chi = chi + 2.0 * jnp.sum(m - o + jnp.where(o > 0, o * jnp.log(o / m), 0.0))
        return chi


def load(path, scale=1.0, ye=0.5, pot_unit=1e20) -> GlobesExperiment:
    """Load a GLoBES ``.glb`` experiment file (see module docstring)."""
    return GlobesExperiment(path, scale=scale, ye=ye, pot_unit=pot_unit)
