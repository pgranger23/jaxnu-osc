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
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

import dataclasses
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


def prob_scaled(sc_core, sc_mantle, e_eV, czi, anti=False):
    rho, ye, L = _osc._earth.chord_segments(czi, table, h_atm_km=15.0, det_depth_km=0.0)
    core = rho > 9.0                       # PREM: outer core >= 9.9, mantle <= 5.6 g/cm^3
    rho_s = rho * jnp.where(core, sc_core, sc_mantle)
    v_cc, _ = _osc.C.matter_potentials(rho_s, ye)
    u, msq = P.pmns(), P.msquared()
    s = _osc.propagate_layers(u, msq, e_eV, v_cc, L * _osc.C.KM_TO_INV_EV,
                              anti=anti, backend="cayley")
    return _osc.prob_from_amplitude(s)


# NG=110 (12100 jax.jacfwd PREM-Earth evaluations) is what the paper figure
# uses; NG=35 (~1200 evaluations) keeps the same qualitative structure
# (core/mantle sign pattern, MSW resonance) but runs in a fraction of the
# time on a busy/shared CPU node. Raise NG back to 110 for a publication-
# quality, smoothly-sampled figure.
NG = 35
Eg = np.logspace(np.log10(1.0), np.log10(30.0), NG)
czg = np.linspace(-1.0, -0.05, NG)
Em, Cm = np.meshgrid(Eg, czg, indexing="ij")
one = jnp.array(1.0)


def cell(e, c):
    d = jax.jacfwd(lambda a, b: prob_scaled(a, b, e * _osc.C.GEV_TO_EV, c)[1, 1],
                   argnums=(0, 1))(one, one)
    return jnp.stack(d)


G = jax.vmap(cell)(jnp.asarray(Em.ravel()), jnp.asarray(Cm.ravel()))
G = np.asarray(G).reshape(NG, NG, 2)
Pmm = np.asarray(jax.vmap(lambda e, c: prob_scaled(one, one, e * _osc.C.GEV_TO_EV, c)[1, 1])(
    jnp.asarray(Em.ravel()), jnp.asarray(Cm.ravel()))).reshape(NG, NG)
out["Eg"] = Eg
out["czg"] = czg
out["G"] = G
out["Pmm"] = Pmm
np.savez(os.path.join(OUTDIR, "jaxnu_bench_timing_and_gradients.npz"), **out)

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), sharey=True)
im = axes[0].pcolormesh(czg, Eg, Pmm, cmap="viridis", shading="auto", vmin=0, vmax=1)
axes[0].set_title(r"$P(\nu_\mu\to\nu_\mu)$")
plt.colorbar(im, ax=axes[0])
for k, (lab, ttl) in enumerate([("core", r"$\partial P/\partial\ln\rho_{\rm core}$"),
                                ("mantle", r"$\partial P/\partial\ln\rho_{\rm mantle}$")]):
    v = np.abs(G[:, :, k]).max()
    im = axes[k + 1].pcolormesh(czg, Eg, G[:, :, k], cmap="RdBu_r", shading="auto",
                                vmin=-v, vmax=v)
    axes[k + 1].set_title(ttl)
    plt.colorbar(im, ax=axes[k + 1])
for ax in axes:
    ax.set_yscale("log")
    ax.set_xlabel(r"$\cos\theta_z$")
axes[0].set_ylabel("neutrino energy [GeV]")
fig.suptitle("jaxnu: exact autodiff derivatives with respect to PREM shell densities "
             "(core/mantle), through the layered Earth", fontsize=12)
fig.tight_layout()
figpath = os.path.join(OUTDIR, "jaxnu_rho_grad.png")
fig.savefig(figpath, dpi=140, bbox_inches="tight")
print(f"saved {figpath}")
