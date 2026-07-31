"""jaxnu code paper, benchmark set 1: (a) backend timing comparison
(Table "Backend performance per energy point" in the paper), (b) matter-
density derivative oscillograms dP/dln(rho) for core and mantle -- the
tomography showcase that constant-density analytic codes cannot provide
(Figure "exact autodiff derivatives w.r.t. PREM shell densities").

Standalone: only imports jaxnu / numpy / jax / matplotlib. Writes plots and
raw numbers to ./benchmarks/output/ (created if missing) rather than to any
path from the private analysis repo the paper's own numbers were produced in.

Run from the repo root:
    JAX_PLATFORMS=cpu python benchmarks/bench_timing_and_gradients.py
(drop JAX_PLATFORMS=cpu to use a GPU if one is visible to JAX).

See benchmarks/README.md for expected runtimes and which paper table/figure
this corresponds to.
"""
import dataclasses
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

import jaxnu
import jaxnu.oscillator as _osc
from jaxnu import nufit_no, probability_constant, probability_earth

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTDIR, exist_ok=True)

P = nufit_no()
dev = jax.devices()[0].platform
print(f"backend: {dev}", flush=True)


def timeit(fn, *a, n=20):
    r = fn(*a)
    jax.block_until_ready(r)
    t0 = time.perf_counter()
    for _ in range(n):
        jax.block_until_ready(fn(*a))
    return (time.perf_counter() - t0) / n


out = {}
# ---------- (a) timing ----------
# FIGONLY=1 re-renders the oscillogram alone and carries the previously
# measured timings through untouched. The timing rows are device-labelled
# ("dev"), so regenerating the figure on whatever machine happens to be free
# must not overwrite them with numbers from a different one.
FIGONLY = os.environ.get("FIGONLY", "") not in ("", "0")
if FIGONLY:
    _prev = np.load(os.path.join(OUTDIR, "jaxnu_bench_timing_and_gradients.npz"),
                    allow_pickle=True)
    for _k in ("timing_names", "timing_sec", "ratio", "dev"):
        out[_k] = _prev[_k]
    print(f"FIGONLY: carrying forward timings measured on {out['dev']}",
          flush=True)
else:
    NB = 2000
    E = jnp.linspace(0.5, 20.0, NB)
    cz = jnp.full(NB, -0.8)
    rows = []
    for bk in ("cayley", "eigh", "expm"):
        f = jax.jit(lambda e, b=bk: probability_constant(P, e, 1300.0, density=2.85, ye=0.5, backend=b))
        rows.append((f"const-density {bk}", timeit(f, E)))
        print(rows[-1], flush=True)
    try:
        from jaxnu.nufast import prob_matrix as _nufast_prob_matrix
        from jaxnu import constants as C
        v_cc, _ = C.matter_potentials(jnp.asarray(2.85), jnp.asarray(0.5))
        L = jnp.asarray(1300.0) * C.KM_TO_INV_EV
        f = jax.jit(lambda e: _nufast_prob_matrix(P, e * C.GEV_TO_EV, L, v_cc))
        rows.append(("const-density nufast-port", timeit(f, E)))
        print(rows[-1], flush=True)
    except Exception as ex:
        print("nufast port timing skipped:", ex, flush=True)

    # NE=40, n=3 keeps this runnable in a few minutes on a busy/shared CPU node
    # -- the paper used NE=200, n=10 on 8 dedicated EPYC 7542 cores; raise these
    # back for closer agreement with the paper's per-point ns figures (see
    # benchmarks/README.md).
    NE = 40
    Ee = jnp.linspace(0.5, 20.0, NE)
    cze = jnp.full(NE, -0.8)
    fe = jax.jit(lambda e, c: probability_earth(P, e, c))
    rows.append(("PREM earth (default)", timeit(fe, Ee, cze, n=3)))
    print(rows[-1], flush=True)

    names = ("theta12", "theta13", "theta23", "deltacp", "dm21", "dm31")
    vals = [jnp.asarray(getattr(P, n), dtype=jnp.float64) for n in names]


    def fsum(*vs):
        p = dataclasses.replace(P, **dict(zip(names, vs)))
        return probability_earth(p, Ee, cze)[..., 1, 1].sum()


    gf = jax.jit(jax.grad(fsum, argnums=tuple(range(6))))
    t_fwd = timeit(jax.jit(fsum), *vals, n=3)
    t_grd = timeit(gf, *vals, n=3)
    rows.append(("PREM earth fwd (scalar sum)", t_fwd))
    rows.append(("PREM earth + all-6 gradient", t_grd))
    print(f"\n{'configuration':32s} {'time [ms]':>12s} {'ns/point':>10s}")
    for k, t in rows:
        npts = NB if "const" in k else NE
        print(f"{k:32s} {t*1e3:12.3f} {t/npts*1e9:10.1f}")
    print(f"\ngradient/forward ratio (6 params, PREM): {t_grd/t_fwd:.2f}")
    out["timing_names"] = np.array([r[0] for r in rows])
    out["timing_sec"] = np.array([r[1] for r in rows])
    out["ratio"] = np.array([t_grd / t_fwd])
    out["dev"] = np.array([dev])

# ---------- (b) density-derivative oscillograms ----------
table = _osc._earth.shell_table(4)

# Zenith at which the chord starts to graze the outer core: the boundary
# beyond which the core-density derivative is exactly zero. Taken from the
# same PREM constant the shell table uses, not hard-coded.
CZ_CORE = -float(np.sqrt(1.0 - (_osc._earth.CORE_RADIUS_KM
                                / _osc._earth.R_EARTH_KM) ** 2))


def prob_all(sc_core, sc_mantle, czi, h_atm, th23, e_eV, anti=False):
    """P(nu_mu -> nu_mu), differentiable in one representative input from each
    of the three derivative classes the paper distinguishes:

      matter     -- sc_core, sc_mantle  (multiplicative shell-density scalings)
      geometry   -- czi, h_atm          (zenith angle, production height)
      parameters -- th23                (a standard oscillation parameter)

    Written as a single function so one jacfwd produces all five derivative
    maps on the same grid, which is what the figure shows.
    """
    rho, ye, L = _osc._earth.chord_segments(czi, table, h_atm_km=h_atm,
                                            det_depth_km=0.0)
    core = rho > 9.0                       # PREM: outer core >= 9.9, mantle <= 5.6 g/cm^3
    rho_s = rho * jnp.where(core, sc_core, sc_mantle)
    v_cc, _ = _osc.C.matter_potentials(rho_s, ye)
    p_th = dataclasses.replace(P, theta23=th23)
    s = _osc.propagate_layers(p_th.pmns(), p_th.msquared(), e_eV, v_cc,
                              L * _osc.C.KM_TO_INV_EV, anti=anti, backend="cayley")
    return _osc.prob_from_amplitude(s)


def prob_scaled(sc_core, sc_mantle, e_eV, czi, anti=False):
    """Back-compatible wrapper: density scalings only, nominal geometry."""
    return prob_all(sc_core, sc_mantle, czi, 15.0, P.theta23, e_eV, anti=anti)


# NG=220 (48400 jax.jacfwd PREM-Earth evaluations) is what the paper figure
# uses; NG=35 (~1200 evaluations) keeps the same qualitative structure
# (core/mantle sign pattern, MSW resonance) but runs in a fraction of the
# time on a busy/shared CPU node. Raise NG back to 220 for a publication-
# quality, smoothly-sampled figure.
#
# The resolution is set by the fringes at the bottom of the energy range, not
# by how the figure looks overall. The oscillation phase through a diameter is
# ~40 rad at 1 GeV and scales as 1/E, so d(phase)/d(ln E) ~ 40; NG=90 over
# [1,30] GeV samples it every 1.5 rad -- about four points per cycle, which
# aliases into a moire that a log colour scale then amplifies into what looks
# like noise. NG=220 samples every 0.63 rad (~10 per cycle) and resolves it.
NG = int(os.environ.get("NG", 220))
Eg = np.logspace(np.log10(1.0), np.log10(30.0), NG)
czg = np.linspace(-1.0, -0.05, NG)
Em, Cm = np.meshgrid(Eg, czg, indexing="ij")
one = jnp.array(1.0)


_H0 = jnp.asarray(15.0)          # nominal production height [km]
_T23 = jnp.asarray(float(P.theta23))


def cell(e, c):
    """All five derivatives at one grid point, from a single jacfwd."""
    d = jax.jacfwd(
        lambda a, b, cz, h, t: prob_all(a, b, cz, h, t,
                                        e * _osc.C.GEV_TO_EV)[1, 1],
        argnums=(0, 1, 2, 3, 4))(one, one, c, _H0, _T23)
    return jnp.stack(d)


def _chunked(fn, e, c, chunk=8192):
    """vmap in slices: one shot over NG^2 = 48400 points builds a jacfwd
    tangent batch large enough to exhaust GPU memory."""
    f = jax.jit(jax.vmap(fn))
    return np.concatenate([np.asarray(f(e[i:i + chunk], c[i:i + chunk]))
                           for i in range(0, e.size, chunk)])


_e, _c = jnp.asarray(Em.ravel()), jnp.asarray(Cm.ravel())
G = _chunked(cell, _e, _c).reshape(NG, NG, 5)
Pmm = _chunked(lambda e, c: prob_scaled(one, one, e * _osc.C.GEV_TO_EV, c)[1, 1],
               _e, _c).reshape(NG, NG)
out["Eg"] = Eg
out["czg"] = czg
out["G"] = G
out["Pmm"] = Pmm
np.savez(os.path.join(OUTDIR, "jaxnu_bench_timing_and_gradients.npz"), **out)

# Placed at \textwidth (~6.3 in), so a 15 in canvas is scaled by 0.42 and every
# font size set here prints at 0.42x. Sizes are chosen so that lands near 9 pt
# for titles and labels and 7.5 pt for ticks; scale them with the canvas if the
# aspect ratio is changed again.
plt.rcParams.update({"font.size": 22, "axes.titlesize": 22,
                     "axes.labelsize": 22, "xtick.labelsize": 18,
                     "ytick.labelsize": 18})

# Layout notes. Two things were wasting the figure:
#  * tight_layout with six independent colourbars left large gutters;
#    constrained_layout with small pads packs the same axes far tighter, and
#    a wide-and-short canvas suits data that spans a narrow cos(theta_z) band.
#  * the derivative panels were on a LINEAR symmetric scale, so each one was
#    dominated by its single largest excursion (near the core-crossing edge)
#    and everything else washed out to white. The interesting structure --
#    the near-horizon fringes, the mantle response over the open sky -- spans
#    several decades in magnitude. A symmetric log scale keeps the sign, keeps
#    zero white, and makes that structure visible.
fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.4), sharey=True, sharex=True,
                         constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02,
                                wspace=0.02, hspace=0.03)

# rasterized: a 220x220 pcolormesh is one vector path per cell, so six panels
# come to a 7 MB PDF that older pdfTeX will not embed. Rasterizing the mesh
# alone (at savefig dpi) keeps all text, axes and ticks vector.
im = axes[0, 0].pcolormesh(czg, Eg, Pmm, cmap="viridis", shading="auto",
                           vmin=0, vmax=1, rasterized=True)
axes[0, 0].set_title(r"$P(\nu_\mu\to\nu_\mu)$")
fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.015)

# one representative derivative from each class; index into G matches the
# argnums order of `cell` above.
panels = [
    (0, axes[0, 1], r"matter:  $\partial P/\partial\ln\rho_{\rm core}$"),
    (1, axes[0, 2], r"matter:  $\partial P/\partial\ln\rho_{\rm mantle}$"),
    (2, axes[1, 0], r"geometry:  $\partial P/\partial\cos\theta_z$"),
    (3, axes[1, 1], r"geometry:  $\partial P/\partial h_{\rm atm}$  [km$^{-1}$]"),
    (4, axes[1, 2], r"parameter:  $\partial P/\partial\theta_{23}$"),
]
def symticks(v, lt):
    """Decade ticks for a SymLogNorm colourbar, plus zero.

    The default symlog locator crowds a "0" label against +-linthresh (they
    sit one linear window apart, i.e. adjacent on the bar). Emitting only the
    decades at or above linthresh, and zero, keeps every label legible.
    """
    dec = [10.0 ** k for k in range(int(np.ceil(np.log10(lt))),
                                    int(np.floor(np.log10(v))) + 1)]
    return [-d for d in dec[::-1]] + [0.0] + dec


for k, ax, ttl in panels:
    d = G[:, :, k]
    v = np.abs(d).max()
    lt = v / 100.0
    # linear within 1/100 of the peak, logarithmic outside: two decades of
    # structure without a discontinuity at zero.
    norm = SymLogNorm(linthresh=lt, vmin=-v, vmax=v, base=10)
    im = ax.pcolormesh(czg, Eg, d, cmap="RdBu_r", shading="auto", norm=norm,
                       rasterized=True)
    ax.set_title(ttl)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.015, ticks=symticks(v, lt))
    cb.ax.tick_params(labelsize=15)

# The core panel is identically zero over most of its area -- that is the
# point of it, so say so rather than leaving the reader to wonder.
axes[0, 1].axvline(CZ_CORE, color="0.35", lw=1.2, ls="--")
axes[0, 1].text(CZ_CORE + 0.02, Eg.max() * 0.72, "no core crossing\n"
                r"$\Rightarrow \partial P/\partial\ln\rho_{\rm core}\equiv 0$",
                fontsize=15, color="0.30", ha="left", va="top")

yt = [t for t in (1, 2, 3, 5, 10, 20, 30) if Eg.min() <= t <= Eg.max()]
for ax in axes.ravel():
    ax.set_yscale("log")
    ax.set_xlim(czg.min(), czg.max())
    ax.set_ylim(Eg.min(), Eg.max())
    # plain numerals: the default log formatter writes these as "3 x 10^0"
    ax.set_yticks(yt)
    ax.set_yticklabels([str(t) for t in yt])
    ax.minorticks_off()
for ax in axes[1, :]:
    ax.set_xlabel(r"$\cos\theta_z$")
for ax in axes[:, 0]:
    # short form: spelled out, this label is taller than the panel at these
    # font sizes and constrained_layout lets it collide with the ticks.
    ax.set_ylabel(r"$E_\nu$ [GeV]")
figpath = os.path.join(OUTDIR, "jaxnu_rho_grad.png")
fig.savefig(figpath, dpi=140, bbox_inches="tight")
# dpi sets the resolution of the rasterized mesh layer; the figure prints at
# ~0.42x, so 300 here lands near 700 dpi on paper.
fig.savefig(figpath.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
print(f"saved {figpath}")
