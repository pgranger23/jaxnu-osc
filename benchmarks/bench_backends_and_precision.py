"""jaxnu code paper, benchmark set 2 (addendum to bench_timing_and_gradients.py):
  (a) NuFast-port timing standalone (uses jaxnu.nufast.prob_matrix directly)
  (b) same timing rows, meant to be run once on CPU and once on GPU so the
      paper's timing table is not GPU-only
  (c) AD-vs-central-difference validation of all three derivative classes:
      oscillation parameters, geometry (cos(zenith), production height), and
      matter density (core/mantle scale factors) -- this is the evidence
      behind the paper's "agrees with finite differences at the 1e-8 level or
      better" claim (section "Validation")
  (d) float32 vs float64 demonstration for the "double precision is
      mandatory" claim

Standalone: only imports jaxnu / numpy / jax. Writes JSON summaries to
./benchmarks/output/ (created if missing).

Run as (from the repo root):
    JAX_PLATFORMS=cpu python benchmarks/bench_backends_and_precision.py timing
    JAX_PLATFORMS=cpu python benchmarks/bench_backends_and_precision.py grad
    JAX_PLATFORMS=cpu python benchmarks/bench_backends_and_precision.py prec

See benchmarks/README.md for expected runtimes and which paper table this
corresponds to.
"""
import os
import sys
import time
import json
import numpy as np
import jax
import jax.numpy as jnp

MODE = sys.argv[1] if len(sys.argv) > 1 else "timing"
OUT = {}

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTDIR, exist_ok=True)


def timeit(fn, *a, n=20):
    r = fn(*a)
    jax.block_until_ready(r)
    t0 = time.perf_counter()
    for _ in range(n):
        jax.block_until_ready(fn(*a))
    return (time.perf_counter() - t0) / n


if MODE == "prec":
    # must run BEFORE heavy work; jaxnu turns x64 on at import, we turn it back off
    import jaxnu
    from jaxnu import nufit_no, probability_earth
    P64 = nufit_no()
    NE = 400
    E = jnp.linspace(1.0, 20.0, NE)
    cz = jnp.full(NE, -1.0)
    p64 = np.asarray(probability_earth(P64, E, cz)[..., 1, 1])
    jax.config.update("jax_enable_x64", False)
    import dataclasses
    P32 = dataclasses.replace(P64, **{f.name: jnp.asarray(np.float32(getattr(P64, f.name)))
                                      for f in dataclasses.fields(P64)})
    E32 = jnp.asarray(np.asarray(E, dtype=np.float32))
    cz32 = jnp.asarray(np.asarray(cz, dtype=np.float32))
    p32 = np.asarray(probability_earth(P32, E32, cz32)[..., 1, 1])
    d = np.abs(p64 - p32)
    print(f"float32 vs float64, PREM cz=-1, 1-20 GeV: max |dP| = {d.max():.3e}, "
          f"rms = {d.std():.3e}, dtype32 = {p32.dtype}")
    OUT["prec_max"] = float(d.max())
    OUT["prec_rms"] = float(d.std())
    json.dump(OUT, open(os.path.join(OUTDIR, "jaxnu_bench2_prec.json"), "w"), indent=1)
    print("ALL STEPS DONE")
    sys.exit()

import jaxnu
import jaxnu.oscillator as _osc
from jaxnu import nufit_no, probability_constant, probability_earth
from jaxnu import constants as C
from jaxnu import nufast as NF
import dataclasses

P = nufit_no()
dev = jax.devices()[0].platform
print(f"backend: {dev}", flush=True)
OUT["dev"] = dev

if MODE == "timing":
    NB = 2000
    E = jnp.linspace(0.5, 20.0, NB)
    rows = []
    for bk in ("cayley", "eigh", "expm"):
        f = jax.jit(lambda e, b=bk: probability_constant(
            P, e, 1300.0, density=2.85, ye=0.5, backend=b))
        rows.append((f"const-density {bk}", timeit(f, E), NB))
        print(rows[-1], flush=True)
    # NuFast port: takes natural units directly
    v_cc, _ = C.matter_potentials(jnp.asarray(2.85), jnp.asarray(0.5))
    L = jnp.asarray(1300.0) * C.KM_TO_INV_EV
    fnf = jax.jit(lambda e: NF.prob_matrix(P, e * C.GEV_TO_EV, L, v_cc))
    rows.append(("const-density nufast-port", timeit(fnf, E), NB))
    print(rows[-1], flush=True)
    # sanity: nufast agrees with the general core
    a = np.asarray(fnf(E))
    b = np.asarray(probability_constant(P, E, 1300.0, density=2.85, ye=0.5, backend="cayley"))
    print(f"  nufast-port vs cayley max|dP| = {np.abs(a-b).max():.2e}", flush=True)
    OUT["nufast_vs_cayley"] = float(np.abs(a - b).max())

    # NE=40, n=3 keeps this runnable in a couple of minutes on a busy/shared
    # CPU node -- the paper used NE=200, n=10 on 8 dedicated EPYC 7542 cores;
    # raise these back if you have dedicated hardware and want closer
    # agreement with the paper's per-point ns figures (see benchmarks/README.md).
    NE = 40
    Ee = jnp.linspace(0.5, 20.0, NE)
    cze = jnp.full(NE, -0.8)
    fe = jax.jit(lambda e, c: probability_earth(P, e, c))
    rows.append(("PREM earth", timeit(fe, Ee, cze, n=3), NE))
    print(rows[-1], flush=True)
    names = ("theta12", "theta13", "theta23", "deltacp", "dm21", "dm31")
    vals = [jnp.asarray(getattr(P, n), dtype=jnp.float64) for n in names]

    def fsum(*vs):
        p = dataclasses.replace(P, **dict(zip(names, vs)))
        return probability_earth(p, Ee, cze)[..., 1, 1].sum()
    t_fwd = timeit(jax.jit(fsum), *vals, n=3)
    t_grd = timeit(jax.jit(jax.grad(fsum, argnums=tuple(range(6)))), *vals, n=3)
    rows.append(("PREM earth fwd (scalar)", t_fwd, NE))
    rows.append(("PREM earth + all-6 grad", t_grd, NE))
    print(f"\n{'configuration':30s} {'time [ms]':>11s} {'per point':>12s}")
    for k, t, n in rows:
        print(f"{k:30s} {t*1e3:11.3f} {t/n*1e9:10.1f} ns")
    print(f"\ngrad/fwd ratio (6 params, PREM): {t_grd/t_fwd:.2f}")
    OUT["rows"] = [[k, t, n] for k, t, n in rows]
    OUT["ratio"] = t_grd / t_fwd
    json.dump(OUT, open(os.path.join(OUTDIR, f"jaxnu_bench2_timing_{dev}.json"), "w"), indent=1)

elif MODE == "grad":
    # ---- AD vs central finite differences, all three derivative classes ----
    # NE=24 is deliberately small: 8 separate derivative kinds below are each
    # jit-compiled from scratch through the PREM Earth path, and compilation
    # -- not NE -- dominates the wall time here, especially on a busy/shared
    # CPU node.
    NE = 24
    Ee = jnp.linspace(1.0, 20.0, NE)
    # Include the nadir endpoint cz = -1 exactly.  That point is where the
    # geometry gradient used to be silently halved by a clip tie, so excluding
    # it from the AD-vs-FD table (as an earlier version of this script did, to
    # keep cz +- h inside [-1, 1]) is exactly the wrong choice: it is the point
    # most worth checking.  Central differences cannot step past -1, so the cz
    # row uses a one-sided difference taken from the physical side there.
    cze = jnp.concatenate([jnp.array([-1.0]), jnp.linspace(-0.999, -0.2, NE - 1)])
    table = _osc._earth.shell_table(4)

    def prob_full(p, cz_, h_km, sc_core, sc_mantle, e_gev):
        rho, ye, L = _osc._earth.chord_segments(cz_, table, h_atm_km=h_km,
                                                det_depth_km=0.0)
        core = rho > 9.0
        rho_s = rho * jnp.where(core, sc_core, sc_mantle)
        v_cc, _ = C.matter_potentials(rho_s, ye)
        s = _osc.propagate_layers(p.pmns(), p.msquared(), e_gev * C.GEV_TO_EV,
                                  v_cc, L * C.KM_TO_INV_EV, anti=False,
                                  backend="cayley")
        return _osc.prob_from_amplitude(s)[1, 1]

    one = jnp.asarray(1.0)
    base = dict(h_km=jnp.asarray(15.0), sc_core=one, sc_mantle=one)

    def make(kind):
        if kind in ("theta23", "deltacp", "dm31", "theta13"):
            def f(x, e, c):
                p = dataclasses.replace(P, **{kind: x})
                return prob_full(p, c, base["h_km"], one, one, e)
            return f, jnp.asarray(getattr(P, kind))
        if kind == "cz":
            return (lambda x, e, c: prob_full(P, x, base["h_km"], one, one, e)), None
        if kind == "h_atm":
            return (lambda x, e, c: prob_full(P, c, x, one, one, e)), jnp.asarray(15.0)
        if kind == "rho_core":
            return (lambda x, e, c: prob_full(P, c, base["h_km"], x, one, e)), one
        if kind == "rho_mantle":
            return (lambda x, e, c: prob_full(P, c, base["h_km"], one, x, e)), one

    KINDS = [("theta13", 1e-6), ("theta23", 1e-6), ("deltacp", 1e-5),
             ("dm31", 1e-9), ("cz", 1e-6), ("h_atm", 1e-3),
             ("rho_core", 1e-6), ("rho_mantle", 1e-6)]
    print(f"\n{'derivative':14s} {'max |AD-FD|':>13s} {'max rel.':>11s} {'scale':>11s}")
    res = {}
    for kind, h in KINDS:
        f, x0 = make(kind)
        gfun = jax.jit(jax.vmap(lambda e, c, x0=x0, f=f: jax.grad(f)(
            c if x0 is None else x0, e, c)))
        ad = np.asarray(gfun(Ee, cze))

        def fd_one(e, c, f=f, x0=x0, h=h, kind=kind):
            x = c if x0 is None else x0
            if kind == "cz":
                # one-sided from the physical side wherever cz - h would leave
                # [-1, 1]; second-order accurate three-point forward formula.
                inside = x - h >= -1.0
                central = (f(x + h, e, c) - f(x - h, e, c)) / (2 * h)
                forward = (-3 * f(x, e, c) + 4 * f(x + h, e, c)
                           - f(x + 2 * h, e, c)) / (2 * h)
                return jnp.where(inside, central, forward)
            return (f(x + h, e, c) - f(x - h, e, c)) / (2 * h)
        fd = np.asarray(jax.jit(jax.vmap(fd_one))(Ee, cze))
        num = np.abs(ad - fd).max()
        sc = np.abs(ad).max()
        print(f"{kind:14s} {num:13.2e} {num/max(sc,1e-30):11.2e} {sc:11.2e}")
        res[kind] = dict(abs=float(num), rel=float(num / max(sc, 1e-30)),
                         scale=float(sc))
    OUT["grad"] = res
    json.dump(OUT, open(os.path.join(OUTDIR, "jaxnu_bench2_grad.json"), "w"), indent=1)

print("ALL STEPS DONE")
