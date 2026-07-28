"""End-to-end differentiability demo: Earth-tomography Fisher information.

This is the composability demonstration for the jaxnu paper.  The quantity it
produces -- a marginalized uncertainty on the density of the Earth's core -- is
*not* an oscillation probability.  Getting it requires differentiating through

    PREM shell densities -> chord geometry -> oscillation probability
        -> atmospheric flux weighting -> (E, cos theta) histogram
        -> detector response matrix -> Poisson Fisher information
        -> matrix inverse -> sigma

in one pass.  A single ``jax.jacfwd`` covers the whole chain, which is the
property that distinguishes an autodiff engine from an analytic probability
library: the analytic library gives you dP/dtheta and stops.

It also differentiates the *final uncertainty* with respect to the angular
resolution of the detector -- i.e. through the matrix inverse as well -- which
is an experimental-design derivative with no analytic counterpart at all.

This is a DEMONSTRATION OF THE MACHINERY, not a sensitivity forecast.  The
"detector" is a placeholder: unit efficiency, Gaussian energy/angle smearing
with no tails, muon neutrinos only, no antineutrinos, no atmospheric-flux
shape uncertainty beyond the two nuisances below.  Absolute numbers should be
read only as scaling with the stated event count.

Run:  python demo_tomography_fisher.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import dataclasses

import jaxnu
import jaxnu.oscillator as _osc
from jaxnu import constants as C, nufit_no

# --- toy exposure ------------------------------------------------------------
N_EVENTS = 1.0e5          # total muon-neutrino events in the sample
E_MIN, E_MAX, N_E = 1.0, 30.0, 30       # GeV
CZ_MIN, CZ_MAX, N_CZ = -1.0, -0.05, 30  # up-going only
GAMMA, E_PIVOT = 2.7, 10.0              # atmospheric power law
H_ATM_KM = 15.0
CORE_CZ = -0.8376384419020163           # chord grazes the core-mantle boundary

_TABLE = _osc._earth.shell_table(4)
_P0 = nufit_no()

# parameter vector: [ln rho_core, ln rho_mantle, theta23, dm31, ln Phi0, dgamma]
THETA0 = jnp.array([0.0, 0.0, float(_P0.theta23), float(_P0.dm31), 0.0, 0.0])
LABELS = ["ln rho_core", "ln rho_mantle", "theta23", "dm31", "ln Phi0", "dgamma"]


def _bin_centres():
    e_edges = np.logspace(np.log10(E_MIN), np.log10(E_MAX), N_E + 1)
    cz_edges = np.linspace(CZ_MIN, CZ_MAX, N_CZ + 1)
    e_c = np.sqrt(e_edges[:-1] * e_edges[1:])          # log centres
    cz_c = 0.5 * (cz_edges[:-1] + cz_edges[1:])
    de = np.diff(e_edges)
    dcz = np.diff(cz_edges)
    E, CZ = np.meshgrid(e_c, cz_c, indexing="ij")
    dE, dCZ = np.meshgrid(de, dcz, indexing="ij")
    return (jnp.asarray(E.ravel()), jnp.asarray(CZ.ravel()),
            jnp.asarray((dE * dCZ).ravel()))


E_C, CZ_C, DOMEGA = _bin_centres()


def _prob_mumu(e_gev, cz, sc_core, sc_mantle, params):
    """P(nu_mu -> nu_mu) with the core and mantle densities scaled."""
    rho, ye, L = _osc._earth.chord_segments(cz, _TABLE, h_atm_km=H_ATM_KM,
                                            det_depth_km=0.0)
    core = rho > 9.0                       # PREM: outer core >= 9.9 g/cm^3
    rho_s = rho * jnp.where(core, sc_core, sc_mantle)
    v_cc, _ = C.matter_potentials(rho_s, ye)
    s = _osc.propagate_layers(params.pmns(), params.msquared(),
                              e_gev * C.GEV_TO_EV, v_cc,
                              L * C.KM_TO_INV_EV, anti=False, backend="cayley")
    return _osc.prob_from_amplitude(s)[1, 1]


def _shape(theta):
    """Un-normalized expected counts per bin (the full forward model)."""
    ln_rc, ln_rm, th23, dm31, ln_phi, dgamma = theta
    p = dataclasses.replace(_P0, theta23=th23, dm31=dm31)
    prob = jax.vmap(_prob_mumu, in_axes=(0, 0, None, None, None))(
        E_C, CZ_C, jnp.exp(ln_rc), jnp.exp(ln_rm), p)
    flux = jnp.exp(ln_phi) * (E_C / E_PIVOT) ** (-(GAMMA + dgamma))
    return flux * prob * DOMEGA


# absolute normalization is fixed once, at the nominal point, so that the
# flux-norm nuisance is a genuine free parameter rather than a definition
_NORM = float(N_EVENTS / jnp.sum(_shape(THETA0)))


RES_LNE = 0.20      # energy resolution, fractional (Gaussian in ln E)
RES_CZ = 0.10       # angular resolution in cos(zenith)


def response(res_lne, res_cz):
    """Differentiable (n_bins x n_bins) migration matrix, reco <- true.

    A Gaussian smearing kernel in (ln E, cos theta_z), column-normalized so it
    conserves events.  Nothing here is special -- the point is only that it is
    a JAX function of the resolution widths, so those widths are differentiable
    inputs like any other.
    """
    dlne = jnp.log(E_C)[:, None] - jnp.log(E_C)[None, :]
    dcz = CZ_C[:, None] - CZ_C[None, :]
    R = jnp.exp(-0.5 * (dlne / res_lne) ** 2 - 0.5 * (dcz / res_cz) ** 2)
    return R / jnp.sum(R, axis=0, keepdims=True)


def counts(theta, res_lne=RES_LNE, res_cz=RES_CZ):
    return response(res_lne, res_cz) @ (_NORM * _shape(theta))


def fisher(theta, res_lne=RES_LNE, res_cz=RES_CZ):
    """Poisson Fisher matrix F_ij = sum_b (1/mu_b) dmu_b/dth_i dmu_b/dth_j."""
    mu = counts(theta, res_lne, res_cz)
    J = jax.jacfwd(counts)(theta, res_lne, res_cz)   # (n_bins, n_par)
    return J.T @ (J / mu[:, None]), mu, J


def _true_space():
    """mu and dmu/dtheta BEFORE the response matrix, computed once.

    The response matrix depends on the resolutions but not on theta, and it
    enters linearly, so mu(res) = R(res) @ mu_true and J(res) = R(res) @ J_true.
    Factoring it out this way turns a resolution scan -- which would otherwise
    need a full jacfwd through the layered Earth per point -- into a sequence of
    small matrix products.
    """
    mu_true = _NORM * _shape(THETA0)
    J_true = _NORM * jax.jacfwd(_shape)(THETA0)
    return mu_true, J_true


def sigma_from_response(res_cz, mu_true, J_true, res_lne=RES_LNE):
    """sigma(ln rho_core) for a given angular resolution, reusing the true-space
    Jacobian.  Must agree with sigma_core() below, which recomputes everything."""
    R = response(res_lne, res_cz)
    mu = R @ mu_true
    J = R @ J_true
    F = J.T @ (J / mu[:, None])
    return float(jnp.sqrt(jnp.linalg.inv(F)[0, 0]))


def sigma_core(res_cz, res_lne=RES_LNE):
    """Marginalized sigma(ln rho_core) as a function of angular resolution.

    Differentiating THIS is the composability claim in one line: the gradient
    runs through the oscillation probability, the flux, the histogram, the
    response matrix, the Fisher matrix and its inverse.
    """
    mu = counts(THETA0, res_lne, res_cz)
    J = jax.jacfwd(counts)(THETA0, res_lne, res_cz)
    F = J.T @ (J / mu[:, None])
    return jnp.sqrt(jnp.linalg.inv(F)[0, 0])


def main():
    print(f"grid: {N_E} x {N_CZ} = {N_E*N_CZ} bins, {N_EVENTS:.0e} events, "
          f"up-going nu_mu only")
    F, mu, J = fisher(THETA0)
    F = np.asarray(F)
    cov = np.linalg.inv(F)

    print(f"\ntotal events check: {float(jnp.sum(mu)):.1f}")
    print("\n                      sigma (stat only)")
    print("parameter          fixed nuisances   marginalized    penalty")
    for i in (0, 1):
        fixed = 1.0 / np.sqrt(F[i, i])
        marg = np.sqrt(cov[i, i])
        print(f"{LABELS[i]:<18s} {fixed:>13.4f}   {marg:>12.4f}   "
              f"{marg/fixed:>8.2f}x")

    # which nuisances actually cost us? (Schur complement over subsets)
    def marg(keep):
        """sigma(par 0) marginalizing only over the indices in `keep`."""
        idx = [0] + sorted(keep)
        sub = F[np.ix_(idx, idx)]
        return np.sqrt(np.linalg.inv(sub)[0, 0])

    print("\nwhere the marginalization cost comes from:")
    print(f"  all other parameters fixed          : {1/np.sqrt(F[0,0]):.4f}")
    print(f"  marginalize flux only (5,6)         : {marg([4, 5]):.4f}")
    print(f"  marginalize theta23, dm31 only (3,4): {marg([2, 3]):.4f}")
    print(f"  marginalize ln rho_mantle only      : {marg([1]):.4f}")
    print(f"  marginalize everything              : {np.sqrt(cov[0,0]):.4f}")

    # where does the core information actually come from?
    Jn = np.asarray(J)
    mun = np.asarray(mu)
    per_bin = (Jn[:, 0] ** 2) / mun            # diagonal Fisher contribution
    core_mask = np.asarray(CZ_C) < CORE_CZ
    frac_info = per_bin[core_mask].sum() / per_bin.sum()
    frac_bins = core_mask.mean()
    frac_evts = mun[core_mask].sum() / mun.sum()
    print(f"\ncore-crossing bins (cos theta_z < {CORE_CZ:.4f}):")
    print(f"  {frac_bins*100:.1f}% of bins, {frac_evts*100:.1f}% of events, "
          f"but {frac_info*100:.1f}% of F_00")
    print("  (F_00 is the NUISANCE-FIXED information on ln rho_core -- that is "
          "the\n   quantity that decomposes additively over bins; the "
          "marginalized\n   information does not split this way.)")

    print(f"\nstatistical scaling: sigma(ln rho_core) = "
          f"{np.sqrt(cov[0,0])*np.sqrt(N_EVENTS/1e5):.4f} "
          f"x sqrt(1e5 / N_events)")

    # --- the composability one-liner: d sigma / d (angular resolution) -------
    s0 = float(sigma_core(RES_CZ))
    dsdr = float(jax.grad(sigma_core)(RES_CZ))
    h = 1e-3
    fd = (float(sigma_core(RES_CZ + h)) - float(sigma_core(RES_CZ - h))) / (2 * h)
    print(f"\nexperimental-design derivative (through the matrix inverse):")
    print(f"  sigma(ln rho_core) at res_cz={RES_CZ}      : {s0:.5f}")
    print(f"  d sigma / d res_cz  (autodiff)             : {dsdr:+.5f}")
    print(f"  d sigma / d res_cz  (central differences)  : {fd:+.5f}")
    print(f"  -> sharpening cos(theta) resolution by 0.01 reduces "
          f"sigma(ln rho_core) by {dsdr*0.01/s0*100:.2f}%")

    np.savez("tomography_fisher.npz", F=F, cov=cov, mu=mun,
             theta0=np.asarray(THETA0), labels=np.array(LABELS),
             frac_info=frac_info)
    print("\nsaved tomography_fisher.npz")

    _figure(Jn, mun, per_bin, frac_info, s0, dsdr, frac_bins * 100)


def _figure(Jn, mun, per_bin, frac_info, s0, dsdr, frac_bins_pct):
    """Three panels: the derivative after the full chain, where the information
    lands, and what the design derivative buys."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    E = np.asarray(E_C).reshape(N_E, N_CZ)
    CZ = np.asarray(CZ_C).reshape(N_E, N_CZ)
    dmu = Jn[:, 0].reshape(N_E, N_CZ)
    info = per_bin.reshape(N_E, N_CZ)

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # (a) dmu/dln rho_core in RECONSTRUCTED space -- i.e. after flux weighting
    #     and after the response matrix, unlike Fig. 2 which is true space.
    v = np.abs(dmu).max()
    m = ax[0].pcolormesh(CZ, E, dmu, cmap="RdBu_r", vmin=-v, vmax=v,
                         shading="auto")
    plt.colorbar(m, ax=ax[0], label=r"$\partial\mu_b/\partial\ln\rho_{\rm core}$")
    ax[0].set_title(r"(a) count derivative, reconstructed space")

    # (b) per-bin Fisher information on ln rho_core.  Log colour scale: the
    #     dynamic range is several decades and a linear scale shows only the
    #     single brightest cell, hiding the structure that makes the point.
    from matplotlib.colors import LogNorm
    pos = info[info > 0]
    m = ax[1].pcolormesh(CZ, E, np.maximum(info, pos.min()), cmap="viridis",
                         norm=LogNorm(vmin=max(pos.min(), pos.max() * 1e-5),
                                      vmax=pos.max()), shading="auto")
    plt.colorbar(m, ax=ax[1], label=r"per-bin contribution to $F_{00}$")
    ax[1].set_title(r"(b) where the information is")
    ax[1].annotate(f"{frac_info*100:.0f}% of $F_{{00}}$\n"
                   f"left of the line\n({frac_bins_pct:.0f}% of the bins)",
                   xy=(0.42, 0.88), xycoords="axes fraction", color="w",
                   fontsize=10, fontweight="bold", va="top")

    for a in ax[:2]:
        a.axvline(CORE_CZ, color="k", ls="--", lw=1.4)
        a.set_yscale("log")
        a.set_ylim(E.min(), 20.0)
        a.set_xlabel(r"$\cos\theta_z$ (reconstructed)")
        a.annotate("core-crossing", xy=(CORE_CZ, 19.0), xytext=(-4, 0),
                   textcoords="offset points", rotation=90, ha="right",
                   va="top", fontsize=8.5)
    ax[0].set_ylabel("reconstructed energy [GeV]")

    # (c) the design derivative: AD tangent against an explicit scan
    mu_true, J_true = _true_space()
    grid = np.linspace(0.04, 0.20, 17)
    scan = [sigma_from_response(float(r), mu_true, J_true) for r in grid]
    ax[2].plot(grid, scan, "o-", ms=3.5, color="0.25",
               label=r"explicit scan (17 Fisher evaluations)")
    tan = np.linspace(RES_CZ - 0.045, RES_CZ + 0.045, 2)
    ax[2].plot(tan, s0 + dsdr * (tan - RES_CZ), "r-", lw=2.2,
               label=fr"AD tangent, $\partial\sigma/\partial\sigma_{{c}}={dsdr:.4f}$"
                     "\n(one gradient call)")
    ax[2].plot([RES_CZ], [s0], "r*", ms=13, zorder=5)
    ax[2].set_xlabel(r"angular resolution $\sigma_{\cos\theta_z}$")
    ax[2].set_ylabel(r"$\sigma(\ln\rho_{\rm core})$, marginalized")
    ax[2].set_title("(c) the experimental-design derivative")
    ax[2].legend(fontsize=8.5, loc="upper left")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"tomography_fisher.{ext}", dpi=150, bbox_inches="tight")
    print("saved tomography_fisher.png/.pdf")

    # the scan and the AD tangent are independent routes to the same slope
    i = int(np.argmin(np.abs(grid - RES_CZ)))
    num = (scan[i + 1] - scan[i - 1]) / (grid[i + 1] - grid[i - 1])
    print(f"  cross-check: scan slope at the operating point = {num:+.5f} "
          f"vs AD {dsdr:+.5f}")


if __name__ == "__main__":
    main()
