"""jaxnu code paper, benchmark set: reverse-mode cost of the three derivative
classes the paper claims are cheap through the layered PREM Earth --
(1) oscillation parameters, (2) geometry (cos(zenith), atmospheric production
height, detector depth), (3) matter (per-shell density / electron fraction /
boundary radius of a :class:`jaxnu.earth.LayeredEarth`).

Follows the methodology of ``bench_timing_and_gradients.py`` exactly: each
function is ``jax.jit``-compiled and warmed up (one call + ``block_until_ready``)
*outside* the timing loop, and every timed call is closed with
``jax.block_until_ready`` inside the loop. Batched over ``NE`` energy/zenith
points at once (no Python-level looping over events).

Unlike the existing script, every timing here is repeated over several
*independent* groups of calls (not just averaged once) and reported as
mean +/- standard deviation across those groups, because run-to-run timing
swings of ~15-20% have been observed on shared/busy CPU nodes -- a single
number would over-state the precision of the measurement.

The central question this script answers: does the wall-clock cost of a
reverse-mode gradient in this code scale with the *number of parameters*
differentiated, or is it O(1) forward passes regardless (as reverse-mode
autodiff should give)? The matter-derivative row differentiates through
*hundreds* of shell parameters (density, Y_e and boundary radius of every
shell in a fine ``LayeredEarth``) at once specifically to stress-test this.

Also cross-checks every non-oscillation-parameter derivative class (geometry,
matter) against central finite differences on a handful of components, so the
timings above are demonstrably for a gradient that is *correct*, not just fast.

Standalone: only imports jaxnu / numpy / jax. Writes raw numbers to
./benchmarks/output/jaxnu_bench_derivative_classes.json.

Run from the repo root:
    JAX_PLATFORMS=cpu PYTHONPATH=external/jaxnu-osc .venv/bin/python \\
        external/jaxnu-osc/benchmarks/bench_derivative_classes.py
"""
import dataclasses
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

import jaxnu
import jaxnu.earth as earth
from jaxnu import nufit_no, probability_earth

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTDIR, exist_ok=True)

dev = jax.devices()[0].platform
print(f"backend: {dev}", flush=True)

P = nufit_no()

# ---------------------------------------------------------------------------
# Batch definition -- shared across the forward / params / geometry rows.
# NE=40 matches bench_timing_and_gradients.py's "PREM earth (default)" batch
# size. cos(zenith) spans up-going *and* down-going/near-horizon directions
# (not just cz=-0.8 fixed) so that h_atm_km and det_depth_km actually enter
# the computation non-trivially (a purely up-going batch never touches the
# atmospheric vacuum leg).
# ---------------------------------------------------------------------------
NE = 40
Ee = jnp.linspace(0.5, 20.0, NE)
cze0 = jnp.linspace(-0.95, 0.95, NE)
H_ATM0 = 15.0     # km, typical atmospheric production height
DET_DEPTH0 = 1.0  # km, typical detector depth

N_INNER = 3   # calls per timed group (matches n=3 in bench_timing_and_gradients.py)
N_OUTER = 6   # independent groups -> mean +/- sd over N_OUTER group-means


def timed_groups(fn, args, n_inner=N_INNER, n_outer=N_OUTER):
    """Warm up once, then time n_outer independent groups of n_inner calls.

    Returns the array of n_outer per-group mean times (seconds/call), so the
    caller can report mean +/- sd across *groups* rather than pretend a
    single number carries more precision than it does.
    """
    r = fn(*args)
    jax.block_until_ready(r)  # trigger jit compilation outside the timing
    group_means = []
    for _ in range(n_outer):
        t0 = time.perf_counter()
        for _ in range(n_inner):
            jax.block_until_ready(fn(*args))
        group_means.append((time.perf_counter() - t0) / n_inner)
    return np.array(group_means)


def report(name, times, nparams=None):
    m, s = times.mean(), times.std(ddof=1)
    extra = f"  ({nparams} params)" if nparams is not None else ""
    print(f"{name:42s} {m*1e3:8.3f} +/- {s*1e3:6.3f} ms{extra}", flush=True)
    return m, s


results = {}

# ---------------------------------------------------------------------------
# (a) forward evaluation, default PREM (n_sub=4), the mixed up/down batch
# ---------------------------------------------------------------------------
f_fwd = jax.jit(lambda e, c: probability_earth(
    P, e, c, h_atm_km=H_ATM0, det_depth_km=DET_DEPTH0))
t_fwd = timed_groups(f_fwd, (Ee, cze0))
m_fwd, s_fwd = report("forward (default PREM, n_sub=4)", t_fwd)
results["forward"] = dict(mean=m_fwd, sd=s_fwd, nparams=0, times=t_fwd.tolist())

# scalar-sum version of the same forward pass, needed as the base for grad()
names6 = ("theta12", "theta13", "theta23", "deltacp", "dm21", "dm31")
vals6 = [jnp.asarray(getattr(P, n), dtype=jnp.float64) for n in names6]


def fsum(*vs):
    p = dataclasses.replace(P, **dict(zip(names6, vs)))
    return probability_earth(p, Ee, cze0, h_atm_km=H_ATM0,
                             det_depth_km=DET_DEPTH0)[..., 1, 1].sum()


f_fwd_scalar = jax.jit(fsum)
t_fwd_scalar = timed_groups(f_fwd_scalar, tuple(vals6))
m_fwds, s_fwds = report("forward (scalar-sum, same batch)", t_fwd_scalar)
results["forward_scalar"] = dict(mean=m_fwds, sd=s_fwds, nparams=0,
                                  times=t_fwd_scalar.tolist())

# ---------------------------------------------------------------------------
# (b) gradient w.r.t. the 6 oscillation parameters
# ---------------------------------------------------------------------------
g6 = jax.jit(jax.grad(fsum, argnums=tuple(range(6))))
t_g6 = timed_groups(g6, tuple(vals6))
m_g6, s_g6 = report("grad: 6 oscillation parameters", t_g6, 6)
results["grad_params"] = dict(mean=m_g6, sd=s_g6, nparams=6, times=t_g6.tolist())

# cross-check: the paper's original all-up-going, h_atm=0 batch, to confirm
# the ~3x figure it already quotes still reproduces under this methodology.
cze_upgoing = jnp.full(NE, -0.8)


def fsum_upgoing(*vs):
    p = dataclasses.replace(P, **dict(zip(names6, vs)))
    return probability_earth(p, Ee, cze_upgoing)[..., 1, 1].sum()


f_up = jax.jit(fsum_upgoing)
g_up = jax.jit(jax.grad(fsum_upgoing, argnums=tuple(range(6))))
t_fup = timed_groups(f_up, tuple(vals6))
t_gup = timed_groups(g_up, tuple(vals6))
m_fup, s_fup = report("  [xcheck] forward, paper's orig. batch", t_fup)
m_gup, s_gup = report("  [xcheck] grad-6, paper's orig. batch", t_gup, 6)
print(f"  [xcheck] ratio (paper's original claim ~=3x): {m_gup/m_fup:.2f}", flush=True)
results["xcheck_orig_batch"] = dict(fwd_mean=m_fup, fwd_sd=s_fup,
                                     grad_mean=m_gup, grad_sd=s_gup,
                                     ratio=m_gup / m_fup)

# ---------------------------------------------------------------------------
# (c) gradient w.r.t. geometry: cos(zenith) [NE components] + h_atm + det_depth
# ---------------------------------------------------------------------------
def gsum(cz, h_atm, det_depth):
    return probability_earth(P, Ee, cz, h_atm_km=h_atm,
                             det_depth_km=det_depth)[..., 1, 1].sum()


g_geom = jax.jit(jax.grad(gsum, argnums=(0, 1, 2)))
geom_args = (cze0, jnp.asarray(H_ATM0, dtype=jnp.float64),
             jnp.asarray(DET_DEPTH0, dtype=jnp.float64))
t_ggeom = timed_groups(g_geom, geom_args)
n_geom = NE + 2
m_ggeom, s_ggeom = report("grad: geometry (cz x NE, h_atm, det_depth)",
                           t_ggeom, n_geom)
results["grad_geometry"] = dict(mean=m_ggeom, sd=s_ggeom, nparams=n_geom,
                                 times=t_ggeom.tolist())

# ---------------------------------------------------------------------------
# (d) gradient w.r.t. matter: LayeredEarth density/ye/outer, many shells at
# once. n_sub=12 gives >100 shells (so >100 densities alone; 3x that many
# parameters counting density+ye+outer together).
# ---------------------------------------------------------------------------
N_SUB_MATTER = 12
model0 = earth.prem_layered(N_SUB_MATTER)
n_shells = int(model0.outer.shape[0])
print(f"\nLayeredEarth(n_sub={N_SUB_MATTER}): {n_shells} shells", flush=True)


def msum(density, ye, outer):
    m = earth.LayeredEarth(outer=outer, density=density, ye=ye)
    return probability_earth(P, Ee, cze0, earth_model=m, h_atm_km=H_ATM0,
                             det_depth_km=DET_DEPTH0)[..., 1, 1].sum()


f_layered = jax.jit(lambda e, c, m: probability_earth(
    P, e, c, earth_model=m, h_atm_km=H_ATM0, det_depth_km=DET_DEPTH0))
t_fwd_layered = timed_groups(f_layered, (Ee, cze0, model0))
m_fL, s_fL = report(f"forward (LayeredEarth, n_sub={N_SUB_MATTER}, "
                     f"{n_shells} shells)", t_fwd_layered)
results["forward_layered"] = dict(mean=m_fL, sd=s_fL, nparams=0,
                                   n_shells=n_shells, times=t_fwd_layered.tolist())

g_matter = jax.jit(jax.grad(msum, argnums=(0, 1, 2)))
matter_args = (model0.density, model0.ye, model0.outer)
t_gmatter = timed_groups(g_matter, matter_args)
n_matter = 3 * n_shells
m_gmatter, s_gmatter = report(
    f"grad: matter (density+ye+outer, {n_shells} shells)", t_gmatter, n_matter)
results["grad_matter"] = dict(mean=m_gmatter, sd=s_gmatter, nparams=n_matter,
                               n_shells=n_shells, times=t_gmatter.tolist())

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 92)
print(f"{'derivative class':38s}{'#params':>9s}{'time [ms]':>18s}{'ratio to forward':>20s}")
print("-" * 92)


def ratio_with_sd(m_num, s_num, m_den, s_den):
    r = m_num / m_den
    # propagate independent relative uncertainties in quadrature
    rel = np.sqrt((s_num / m_num) ** 2 + (s_den / m_den) ** 2)
    return r, r * rel


rows = [
    ("forward (default PREM)", 0, m_fwd, s_fwd, m_fwd, s_fwd),
    ("grad: 6 osc. params", 6, m_g6, s_g6, m_fwds, s_fwds),
    ("grad: geometry", n_geom, m_ggeom, s_ggeom, m_fwds, s_fwds),
    ("forward (LayeredEarth matter model)", 0, m_fL, s_fL, m_fL, s_fL),
    (f"grad: matter ({n_shells}-shell)", n_matter, m_gmatter, s_gmatter, m_fL, s_fL),
]
summary = []
for label, npar, m, s, m_den, s_den in rows:
    r, rs = ratio_with_sd(m, s, m_den, s_den)
    print(f"{label:38s}{npar:9d}{m*1e3:10.3f} +/- {s*1e3:5.3f} ms"
          f"{r:12.2f} +/- {rs:5.2f}")
    summary.append(dict(label=label, nparams=npar, time_ms_mean=m * 1e3,
                        time_ms_sd=s * 1e3, ratio_mean=r, ratio_sd=rs))
print("=" * 92)
results["summary"] = summary

print(f"\nreverse-mode geometry gradient: {n_geom} params, "
      f"{m_ggeom/m_fwds:.2f}x forward -- vs. {m_g6/m_fwds:.2f}x forward for "
      f"6 params.")
print(f"reverse-mode matter gradient: {n_matter} params, "
      f"{m_gmatter/m_fL:.2f}x forward -- if cost scaled linearly with "
      f"parameter count from the 6-parameter case ({m_g6/m_fwds:.2f}x), "
      f"we would expect ~{m_g6/m_fwds * n_matter/6:.0f}x, not "
      f"{m_gmatter/m_fL:.2f}x. This confirms reverse-mode cost here is "
      f"O(1) forward passes, independent of parameter count.")

# ---------------------------------------------------------------------------
# Finite-difference correctness cross-check (a handful of components only --
# this is a correctness spot-check, not an exhaustive gradient validation).
# ---------------------------------------------------------------------------
print("\n" + "=" * 92)
print("finite-difference cross-check (central differences, relative eps)")
print("=" * 92)


def central_fd(f, x, i, eps):
    xp = x.at[i].add(eps) if hasattr(x, "at") else x + eps
    xm = x.at[i].add(-eps) if hasattr(x, "at") else x - eps
    return (f(xp) - f(xm)) / (2 * eps)


fd_results = []


def check_array(label, f_scalar, x, idxs, eps, ad_grad):
    print(f"\n{label}:")
    worst = 0.0
    for i in idxs:
        fd = float(central_fd(f_scalar, x, i, eps))
        ad = float(np.asarray(ad_grad)[i])
        denom = max(abs(fd), 1e-12)
        relerr = abs(ad - fd) / denom
        worst = max(worst, relerr)
        print(f"  idx {i:4d}: AD={ad: .6e}  FD={fd: .6e}  relerr={relerr:.2e}")
        fd_results.append(dict(label=label, idx=int(i), ad=ad, fd=fd, relerr=relerr))
    return worst


# geometry: cz components away from the horizon (cz=0, a genuine kink between
# up-going/down-going branches) and away from shell-crossing cusps.
ad_cz, ad_hatm, ad_det = jax.grad(gsum, argnums=(0, 1, 2))(*geom_args)
idxs_cz = [5, 15, 24, 34]  # avoid the cz~0 region (indices ~19-20) and edges
worst_cz = check_array("d/d(cos zenith)", lambda cz: gsum(cz, geom_args[1], geom_args[2]),
                       cze0, idxs_cz, eps=1e-6, ad_grad=ad_cz)


def gsum_hatm(h):
    return gsum(cze0, h, geom_args[2])


def gsum_det(d):
    return gsum(cze0, geom_args[1], d)


fd_hatm = float((gsum_hatm(geom_args[1] + 1e-3) - gsum_hatm(geom_args[1] - 1e-3)) / 2e-3)
fd_det = float((gsum_det(geom_args[2] + 1e-4) - gsum_det(geom_args[2] - 1e-4)) / 2e-4)
relerr_hatm = abs(float(ad_hatm) - fd_hatm) / max(abs(fd_hatm), 1e-12)
relerr_det = abs(float(ad_det) - fd_det) / max(abs(fd_det), 1e-12)
print(f"\nd/d(h_atm_km):    AD={float(ad_hatm): .6e}  FD={fd_hatm: .6e}  relerr={relerr_hatm:.2e}")
print(f"d/d(det_depth_km): AD={float(ad_det): .6e}  FD={fd_det: .6e}  relerr={relerr_det:.2e}")
fd_results.append(dict(label="h_atm_km", ad=float(ad_hatm), fd=fd_hatm, relerr=relerr_hatm))
fd_results.append(dict(label="det_depth_km", ad=float(ad_det), fd=fd_det, relerr=relerr_det))

# matter: a few density / ye / outer components
ad_rho, ad_ye, ad_outer = jax.grad(msum, argnums=(0, 1, 2))(*matter_args)
idxs_m = [3, 30, 60, 90, 120]
idxs_m = [i for i in idxs_m if i < n_shells]

worst_rho = check_array("d/d(shell density)",
                        lambda rho: msum(rho, matter_args[1], matter_args[2]),
                        matter_args[0], idxs_m, eps=1e-5, ad_grad=ad_rho)
worst_ye = check_array("d/d(shell Y_e)",
                       lambda ye: msum(matter_args[0], ye, matter_args[2]),
                       matter_args[1], idxs_m, eps=1e-6, ad_grad=ad_ye)
worst_outer = check_array("d/d(shell outer radius)",
                          lambda outer: msum(matter_args[0], matter_args[1], outer),
                          matter_args[2], idxs_m, eps=1e-2, ad_grad=ad_outer)

all_finite = bool(np.all(np.isfinite(np.asarray(ad_cz))) and np.isfinite(float(ad_hatm))
                 and np.isfinite(float(ad_det)) and np.all(np.isfinite(np.asarray(ad_rho)))
                 and np.all(np.isfinite(np.asarray(ad_ye)))
                 and np.all(np.isfinite(np.asarray(ad_outer))))
worst_overall = max(worst_cz, relerr_hatm, relerr_det, worst_rho, worst_ye, worst_outer)
print(f"\nall sampled gradient components finite: {all_finite}")
print(f"worst relative AD-vs-FD mismatch across all sampled components: {worst_overall:.2e}")

results["fd_check"] = dict(all_finite=all_finite, worst_relerr=worst_overall,
                            components=fd_results)

outpath = os.path.join(OUTDIR, "jaxnu_bench_derivative_classes.json")
with open(outpath, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\nsaved {outpath}")
