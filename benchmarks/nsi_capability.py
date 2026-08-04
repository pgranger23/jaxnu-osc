"""jaxnu code paper: what the NSI derivatives are actually for.

The eps_ee/density degeneracy in ``demo_tomography_fisher.py`` is a
correctness check -- it is visible by inspection of
``H_matter = V_CC [diag(1,0,0) + eps]`` without running any code, and the
Fisher matrix rediscovering it on its own is a nice consistency check but not
a capability demonstration.

This script asks a different question: is there a genuine use case for
*having* dP/d(NSI) as a first-class, cheaply-obtained derivative, rather than
a convenience for a computation someone could do by hand? Two candidates,
both measured rather than asserted:

(a) COST. A realistic matter-NSI fit frees the *whole* epsilon matrix (up to
    8 real degrees of freedom here; eps_mumu is fixed, since subtracting a
    multiple of the identity from eps is unobservable). Profiling all of them
    means assembling a Jacobian with 8 more columns than the standard-only
    fit. jaxnu gets the full Jacobian from one batched ``jax.jacfwd`` call;
    the alternative -- differencing the same forward function once per
    parameter, unbatched, blocked between calls -- is what a fitting pipeline
    built around a black-box (non-vmappable) simulator is stuck with. Measured
    on the SAME jaxnu forward function both ways (no external code is
    benchmarked; see the "Validation scope" note in the paper for why not),
    so this isolates the batched-Jacobian-vs-serialized-calls effect from any
    cross-language comparison.

(b) WHERE THE NSI INFORMATION IS. Given the full per-bin Jacobian, the
    profiled (nuisance-eliminated) information on a single NSI direction has
    a bin-by-bin decomposition: project each bin's d(mu)/d(eps_dir) onto the
    subspace orthogonal to every other free parameter's gradient (in the
    1/mu-weighted inner product), and square. This sums to the exact Schur-
    complement profiled Fisher information, so it is not an approximation --
    but producing the *per-bin* map, rather than just the summed number,
    needs the full per-bin gradient vectors that only a differentiable
    end-to-end model supplies at this cost. This is the direct NSI analogue
    of panel (b), "where the information is", in ``demo_tomography_fisher.py``.
    It surfaces a real asymmetry: Re(eps_mutau) and Im(eps_mutau) are the
    real and imaginary parts of the *same* Hermitian entry, yet the profiled
    P(nu_mu->nu_mu)-only fit constrains them to ~0.003 and ~0.8 respectively
    -- a ~300x difference, confirmed against a direct dP/d(eps) check away
    from the Fisher machinery (Re dominates Im by 2-3 orders of magnitude at
    every sampled point), consistent with matter NSI entering an atmospheric
    disappearance channel at leading order through the real part only.

Reuses ``demo_tomography_fisher.py``'s Earth model, flux, response and event
count (import, not copy) so the numbers are on the same footing as the rest
of the worked example. Standalone otherwise: no path or module from any
private analysis repository.

Run from the repository root:
    JAX_PLATFORMS=cpu python benchmarks/nsi_capability.py
    FIGONLY=1 JAX_PLATFORMS=cpu python benchmarks/nsi_capability.py   # re-plot only

Takes several minutes on CPU: assembling the Fisher-sized Jacobian nine
times (k=0..8 free NSI parameters) is the same 900-bin, layered-PREM,
Cayley-backend computation `demo_tomography_fisher.py` needs one of, done
repeatedly to trace out the cost curve.
"""
import dataclasses
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

import demo_tomography_fisher as D
from jaxnu.nsi import NSI

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTDIR, exist_ok=True)


def _out(name):
    return os.path.join(OUTDIR, name)


dev = jax.devices()[0].platform
print(f"backend: {dev}", flush=True)

# FIGONLY=1 re-renders the figure from the saved .npz without repeating the
# ~10-minute Jacobian sweep (same pattern as bench_timing_and_gradients.py).
FIGONLY = os.environ.get("FIGONLY", "") not in ("", "0")

# --- the 8 real, independent NSI degrees of freedom (eps_mumu fixed: an -----
# overall multiple of the identity added to eps is unobservable, since it
# just rescales V_CC uniformly) -------------------------------------------
N_EPS = 8
EPS_LABELS = ["eps_ee", "eps_tautau", "Re eps_emu", "Im eps_emu",
              "Re eps_etau", "Im eps_etau", "Re eps_mutau", "Im eps_mutau"]


def eps_matrix(e):
    return NSI(eps_ee=e[0], eps_mumu=0.0, eps_tautau=e[1],
               eps_emu=e[2] + 1j * e[3], eps_etau=e[4] + 1j * e[5],
               eps_mutau=e[6] + 1j * e[7]).matrix(3)


def _shape_eps(theta_std, e8, anti):
    ln_sc = theta_std[:D.N_DENS]
    eps = eps_matrix(e8)
    p = dataclasses.replace(D._P0, theta23=theta_std[D.IDX_TH23],
                            dm31=theta_std[D.IDX_DM31])
    pr = jax.vmap(D._prob_mumu, in_axes=(0, 0, None, None, None, None))(
        D.E_C, D.CZ_C, ln_sc, p, anti, eps)
    ln_phi = (theta_std[D.IDX_PHIBAR] if (anti and D.INCLUDE_ANTINU)
             else theta_std[D.IDX_PHI])
    flux = jnp.exp(ln_phi) * (D.E_C / D.E_PIVOT) ** (-(D.GAMMA + theta_std[D.IDX_DGAMMA]))
    return flux * pr * D.DOMEGA


def counts_eps(theta_std, e8):
    """Same response, flux and event count as demo_tomography_fisher.counts,
    with the NSI matrix threaded through the propagator."""
    R = D.response(D.RES_LNE, D.RES_CZ)
    c = R @ (D._NORM_NU * _shape_eps(theta_std, e8, False))
    if not D.INCLUDE_ANTINU:
        return c
    return jnp.concatenate([c, R @ (D._NORM_BAR * _shape_eps(theta_std, e8, True))])


def full_fn(k):
    """counts as a function of one (N_PAR+k)-vector: the standard parameters
    plus the first k of the 8 NSI degrees of freedom (fixed order); the
    remaining 8-k are held at exactly zero, not differentiated."""
    def f(theta_full):
        e8 = (jnp.zeros(N_EPS).at[:k].set(theta_full[D.N_PAR:D.N_PAR + k])
              if k else jnp.zeros(N_EPS))
        return counts_eps(theta_full[:D.N_PAR], e8)
    return f


def timed_groups(fn, args, n_inner=3, n_outer=6):
    r = fn(*args)
    jax.block_until_ready(r)
    means = []
    for _ in range(n_outer):
        t0 = time.perf_counter()
        for _ in range(n_inner):
            jax.block_until_ready(fn(*args))
        means.append((time.perf_counter() - t0) / n_inner)
    return np.array(means)


if FIGONLY:
    _prev = np.load(_out("nsi_capability.npz"))
    ks = _prev["ks"]
    ad_ms, ad_sd = list(_prev["ad_ms"]), list(_prev["ad_sd"])
    fd_ms, fd_sd = list(_prev["fd_ms"]), list(_prev["fd_sd"])
    slope_ad, icpt_ad = float(_prev["slope_ad"]), float(_prev["icpt_ad"])
    slope_fd, icpt_fd = float(_prev["slope_fd"]), float(_prev["icpt_fd"])
    info_re, info_im = _prev["info_re"], _prev["info_im"]
    mu_full = _prev["mu_full"]
    sigma_re, sigma_im = float(_prev["sigma_re"]), float(_prev["sigma_im"])
    frac_core_re = float(_prev["frac_core_re"])
    frac_core_im = float(_prev["frac_core_im"])
    frac_core_density = float(_prev["frac_core_density"])
    n_e, n_cz, n_bins = int(_prev["n_e"]), int(_prev["n_cz"]), int(_prev["n_bins"])
    core_cz = float(_prev["core_cz"])
    E_C_np, CZ_C_np = _prev["E_C"], _prev["CZ_C"]
    print(f"FIGONLY: loaded {_out('nsi_capability.npz')}", flush=True)
else:
    # =========================================================================
    # (a) cost vs number of free NSI parameters, k = 0..8
    # =========================================================================
    print("=" * 78)
    print("(a) cost: batched AD Jacobian vs. serialized finite differences")
    print("=" * 78)
    print(f"\n{'k':>3s} {'AD jacfwd [ms]':>18s} {'FD, k+N_PAR+1 calls [ms]':>26s} {'AD/FD':>8s}")

    ad_ms, ad_sd, fd_ms, fd_sd = [], [], [], []
    J_full = mu_full = None   # captured at k=N_EPS for (b)/(c) below
    for k in range(0, N_EPS + 1):
        n = D.N_PAR + k
        theta0 = jnp.concatenate([D.THETA0, jnp.zeros(k)])

        f = jax.jit(full_fn(k))
        g = jax.jit(jax.jacfwd(full_fn(k)))
        t_ad = timed_groups(g, (theta0,))

        def fd_group(theta, f=f, n=n):
            base = jax.block_until_ready(f(theta))
            for i in range(n):
                step = 1e-4 * max(abs(float(theta[i])), 1.0)
                jax.block_until_ready(f(theta.at[i].add(step)))
            return base

        t_fd = timed_groups(fd_group, (theta0,), n_inner=1, n_outer=6)

        m_ad, s_ad = float(t_ad.mean()), float(t_ad.std(ddof=1))
        m_fd, s_fd = float(t_fd.mean()), float(t_fd.std(ddof=1))
        ad_ms.append(m_ad * 1e3); ad_sd.append(s_ad * 1e3)
        fd_ms.append(m_fd * 1e3); fd_sd.append(s_fd * 1e3)
        print(f"{k:3d} {m_ad*1e3:12.1f}+-{s_ad*1e3:5.1f} "
              f"{m_fd*1e3:19.1f}+-{s_fd*1e3:5.1f} {m_ad/m_fd:8.3f}")

        if k == N_EPS:
            J_full = np.asarray(g(theta0))          # (n_bins, N_PAR+N_EPS)
            mu_full = np.asarray(f(theta0))          # (n_bins,)

    ks = np.arange(N_EPS + 1)
    slope_ad, icpt_ad = np.polyfit(ks, ad_ms, 1)
    slope_fd, icpt_fd = np.polyfit(ks, fd_ms, 1)
    print(f"\nlinear fit, ms per additional free NSI parameter:")
    print(f"  AD:  {slope_ad:6.1f} ms/param  (intercept {icpt_ad:7.1f} ms)")
    print(f"  FD:  {slope_fd:6.1f} ms/param  (intercept {icpt_fd:7.1f} ms)")
    print(f"  marginal-cost ratio (FD/AD slope): {slope_fd/slope_ad:.2f}x")

    # =========================================================================
    # (b) sigma(bulk core) vs k -- a consistency check, NOT a figure/paper
    # result. eps_ee is freed first in EPS_LABELS order, so every k>=1 point
    # trivially inherits the exact eps_ee/density degeneracy already
    # established in nsi_degeneracy_study; the curve says nothing about the
    # other 7 components individually. Kept as confirmation the degeneracy is
    # stable under this larger parameter set, not promoted below.
    # =========================================================================
    print("\n" + "=" * 78)
    print("(b) sigma(bulk core) vs number of free NSI parameters [consistency check]")
    print("=" * 78)

    v_bulk = np.zeros(D.N_PAR)
    for i in D.CORE_ZONE_IDXS:
        v_bulk[i] = 1.0
    n_core = float(v_bulk @ v_bulk)

    def sigma_bulk(k):
        n = D.N_PAR + k
        Jk = J_full[:, :n]
        Fk = Jk.T @ (Jk / mu_full[:, None])
        vv = np.zeros(n); vv[:D.N_PAR] = v_bulk
        lam, U = np.linalg.eigh(Fk)
        c = U.T @ vv
        flat = lam <= lam.max() * 1e-13
        if np.any(flat) and np.sqrt((c[flat] ** 2).sum()) > 1e-6 * np.linalg.norm(c):
            return None
        return float(np.sqrt((c[~flat] ** 2 / lam[~flat]).sum())) / n_core

    for k in range(N_EPS + 1):
        s = sigma_bulk(k)
        label = EPS_LABELS[k - 1] if k else "(standard only)"
        print(f"  k={k}  [+{label:14s}]  sigma(bulk core) = "
              f"{'unbounded' if s is None else f'{s:.4f}'}")

    # =========================================================================
    # (c) where the NSI information is: per-bin profiled information on a
    # chosen eps direction, orthogonalized (in the 1/mu-weighted inner
    # product) against every OTHER free parameter -- standard AND the other 7
    # NSI dof, including eps_ee even though it is degenerate with the density
    # scale: the projector onto a rank-deficient nuisance space is still
    # well-defined (lstsq handles it); only the projection, not any
    # coefficient vector, is meaningful.
    # =========================================================================
    print("\n" + "=" * 78)
    print("(c) per-bin profiled information: Re eps_mutau and Im eps_mutau")
    print("=" * 78)

    IDX_RE_MUTAU, IDX_IM_MUTAU = D.N_PAR + 6, D.N_PAR + 7
    n_tot = D.N_PAR + N_EPS
    w = 1.0 / mu_full                              # weight (1/mu) per bin

    def profiled_info_map(idx_target):
        """Per-bin contribution to the profiled information on parameter
        idx_target, after projecting out every other column of J_full
        (weighted least squares in the Fisher metric). Sums to the exact
        Schur-complement scalar."""
        others = [i for i in range(n_tot) if i != idx_target]
        Jo = J_full[:, others]                      # (n_bins, n_tot-1)
        g_t = J_full[:, idx_target]                  # (n_bins,)
        sw = np.sqrt(w)
        A = Jo * sw[:, None]
        b = g_t * sw
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        resid = g_t - Jo @ coef                      # orthogonal component
        info_bin = w * resid ** 2
        return info_bin, float(info_bin.sum())

    info_re, F_re = profiled_info_map(IDX_RE_MUTAU)
    info_im, F_im = profiled_info_map(IDX_IM_MUTAU)
    sigma_re = 1.0 / np.sqrt(F_re)
    sigma_im = 1.0 / np.sqrt(F_im)
    print(f"  profiled sigma(Re eps_mutau) = {sigma_re:.4f}")
    print(f"  profiled sigma(Im eps_mutau) = {sigma_im:.4f}")

    n_bins = D.N_BINS
    n_e, n_cz = D.N_E, D.N_CZ
    core_cz = D.CORE_CZ
    E_C_np, CZ_C_np = np.asarray(D.E_C), np.asarray(D.CZ_C)
    core_mask = CZ_C_np < core_cz
    frac_core_re = info_re[:n_bins][core_mask].sum() / info_re[:n_bins].sum()
    frac_core_im = info_im[:n_bins][core_mask].sum() / info_im[:n_bins].sum()
    print(f"  fraction of nu-channel Re eps_mutau info in core-crossing bins: "
          f"{frac_core_re*100:.1f}%")
    print(f"  fraction of nu-channel Im eps_mutau info in core-crossing bins: "
          f"{frac_core_im*100:.1f}%")
    # for comparison, the bulk-core DENSITY direction on the same (k=8)
    # Jacobian, same footing
    Jv = J_full[:, :D.N_PAR] @ v_bulk
    info_density = (Jv ** 2) * w
    frac_core_density = (info_density[:n_bins][core_mask].sum()
                         / info_density[:n_bins].sum())
    print(f"  (for comparison, bulk-core DENSITY direction: "
          f"{frac_core_density*100:.1f}% in core-crossing bins)")

    np.savez(_out("nsi_capability.npz"),
             ks=ks, ad_ms=ad_ms, ad_sd=ad_sd, fd_ms=fd_ms, fd_sd=fd_sd,
             slope_ad=slope_ad, icpt_ad=icpt_ad,
             slope_fd=slope_fd, icpt_fd=icpt_fd,
             info_re=info_re, info_im=info_im, mu_full=mu_full,
             sigma_re=sigma_re, sigma_im=sigma_im,
             frac_core_re=frac_core_re, frac_core_im=frac_core_im,
             frac_core_density=frac_core_density,
             E_C=E_C_np, CZ_C=CZ_C_np,
             n_e=n_e, n_cz=n_cz, n_bins=n_bins, core_cz=core_cz)
    print(f"\nsaved {_out('nsi_capability.npz')}")


# =============================================================================
# figure: (a) cost-vs-k curve, (b)/(c) profiled Re/Im eps_mutau oscillograms
# =============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.rcParams.update({
    "font.size": 17, "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
    "axes.linewidth": 1.2, "lines.linewidth": 2.0,
})

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

# (a) cost-vs-k
ax = axes[0]
ks_f = np.asarray(ks, dtype=float)
ax.errorbar(ks_f, ad_ms, yerr=ad_sd, fmt="o-", color="tab:blue", ms=5,
           label="AD (1 batched jacfwd call)")
ax.errorbar(ks_f, fd_ms, yerr=fd_sd, fmt="s-", color="tab:red", ms=5,
           label=r"FD ($k$+12 serialized calls)")
qs = np.linspace(0, 8, 2)
ax.plot(qs, icpt_ad + slope_ad * qs, "--", color="tab:blue", lw=1.3, alpha=0.6)
ax.plot(qs, icpt_fd + slope_fd * qs, "--", color="tab:red", lw=1.3, alpha=0.6)
ax.set_xlabel(r"free NSI parameters $k$")
ax.set_ylabel("wall time [ms]")
ax.set_title("(a) cost of extending the fit", pad=12)
ax.annotate(f"marginal cost:\n{slope_ad:.0f} ms/param (AD)\n"
           f"{slope_fd:.0f} ms/param (FD)",
           xy=(0.97, 0.30), xycoords="axes fraction", va="top", ha="right",
           fontsize=11.5)
ax.legend(loc="upper left", frameon=False, fontsize=11)
ax.grid(alpha=0.3)

# (b)/(c) profiled information oscillograms, nu channel, shared colour scale
# so the Re/Im contrast is visually literal, not just a number in the caption
E2 = E_C_np[:n_bins].reshape(n_e, n_cz)
CZ2 = CZ_C_np[:n_bins].reshape(n_e, n_cz)
info_re_nu = info_re[:n_bins].reshape(n_e, n_cz)
info_im_nu = info_im[:n_bins].reshape(n_e, n_cz)

vmax = max(info_re_nu.max(), info_im_nu.max())
vmin = vmax * 1e-6
norm = LogNorm(vmin=vmin, vmax=vmax)

for ax, info, ttl, sig in ((axes[1], info_re_nu,
                            r"(b) profiled info: $\mathrm{Re}\,\varepsilon_{\mu\tau}$",
                            sigma_re),
                           (axes[2], info_im_nu,
                            r"(c) profiled info: $\mathrm{Im}\,\varepsilon_{\mu\tau}$",
                            sigma_im)):
    m = ax.pcolormesh(CZ2, E2, np.maximum(info, vmin), cmap="viridis",
                      norm=norm, shading="auto", rasterized=True)
    ax.axvline(core_cz, color="w", ls="--", lw=1.1, alpha=0.8)
    ax.set_yscale("log")
    ax.set_ylim(E2.min(), 23.0)
    ax.set_xlabel(r"$\cos\theta_z$ (reconstructed)")
    ax.set_title(ttl, pad=12)
    ax.annotate(fr"profiled $\sigma={sig:.3f}$", xy=(0.04, 0.95),
               xycoords="axes fraction", color="w", fontsize=12,
               fontweight="bold", va="top")

axes[1].set_ylabel("energy [GeV]")
axes[2].sharey(axes[1])
axes[2].tick_params(labelleft=False)
cb = fig.colorbar(m, ax=axes[1:3].tolist(), fraction=0.05, pad=0.02,
                  label="per-bin profiled information")

fig.savefig(_out("nsi_capability.png"), dpi=150, bbox_inches="tight",
           pad_inches=0.08)
fig.savefig(_out("nsi_capability.pdf"), dpi=220, bbox_inches="tight",
           pad_inches=0.08)
print(f"saved {_out('nsi_capability.png/.pdf')}")
