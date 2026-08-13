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
with no tails, perfect nu/nubar separation (the response never mixes the two
species), no atmospheric-flux shape uncertainty beyond the nuisances below,
no backgrounds.  Absolute numbers should be read only as scaling with the
stated event count.

Earth model (default): the density is parametrized by N_ZONES=6 independent
radial scale factors -- inner core, outer core, lower mantle, transition
zone, upper mantle, crust -- built on ``jaxnu.earth.LayeredEarth`` (every
per-shell density is a differentiable input).  This replaces an earlier
2-parameter (core/mantle) version, which a referee correctly flagged as
removing the inter-shell degeneracies that dominate a real inversion: with
only two density parameters sigma(ln rho_core) came out an order of
magnitude tighter than published forecasts for the same MSW mechanism.  The
6-zone model reproduces that "tens of percent" regime (see the module
docstring numbers logged by ``main()``) and exposes the degeneracy directly
in the zone-zone correlation matrix (figure panel (d)).

The sample includes both neutrinos and antineutrinos by default (matter
effects differ in sign for the two; jaxnu's propagator takes ``anti=True``
natively), sharing the density and oscillation parameters but each with its
own flux normalization.

Both realism changes are controlled by environment variables so the original
2-zone, neutrino-only configuration remains reproducible for comparison:

    N_ZONES=2 INCLUDE_ANTINU=0 python demo_tomography_fisher.py

reproduces the original 2-parameter result to the last stated digit.

Run:  python demo_tomography_fisher.py   (defaults: N_ZONES=6, antinu on)

Artefacts go to ``benchmarks/output/`` regardless of the working directory,
alongside the other benchmark outputs.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import dataclasses

import jaxnu
import jaxnu.oscillator as _osc
import jaxnu.earth as _earth
from jaxnu import constants as C, nufit_no

# --- realism toggles (env vars so the old configuration stays reproducible) --
N_ZONES = int(os.environ.get("N_ZONES", "6"))
INCLUDE_ANTINU = bool(int(os.environ.get("INCLUDE_ANTINU", "1")))

# --- toy exposure ------------------------------------------------------------
_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(_OUTDIR, exist_ok=True)


def _out(name):
    """Artefact path, alongside the other benchmark outputs."""
    return os.path.join(_OUTDIR, name)


N_EVENTS = 1.0e5          # nu_mu events in the sample (nubar sits on top of this)
NUBAR_FRAC = 0.5          # toy nubar/nu event-count ratio (order-of-magnitude
                           # atmospheric nu_mu:nubar_mu correct; not a fit target)
E_MIN, E_MAX, N_E = 1.0, 30.0, 30       # GeV
CZ_MIN, CZ_MAX, N_CZ = -1.0, -0.05, 30  # up-going only
GAMMA, E_PIVOT = 2.7, 10.0              # atmospheric power law
H_ATM_KM = 15.0
CORE_CZ = -0.8376384419020163           # chord grazes the core-mantle boundary
N_BINS = N_E * N_CZ

_P0 = nufit_no()

# --- zoned, fully differentiable Earth (jaxnu.earth.LayeredEarth) ------------
# n_sub=4 matches the shell count used by the original shell_table(4) (43
# shells), so the per-bin propagation cost is unchanged; only the number of
# *free parameters* controlling the shell densities changes.
_BASE_EARTH = _earth.prem_layered(n_sub=4)


def _zone_edges(n_zones):
    if n_zones == 2:
        # original split: core / mantle, boundary at the core-mantle radius
        return np.array([0.0, _earth.CORE_RADIUS_KM, C.R_EARTH_KM])
    if n_zones == 6:
        # inner core / outer core / lower mantle / transition zone /
        # upper mantle / crust -- PREM region boundaries grouped into six
        # radial zones
        return np.array([0.0, 1221.5, 3480.0, 5701.0, 5971.0, 6346.6,
                         C.R_EARTH_KM])
    raise ValueError(f"no zone table defined for N_ZONES={n_zones}")


def _zone_index(base_earth, n_zones):
    """Static (numpy) per-shell zone index, built once from shell mid-radii."""
    outer = np.asarray(base_earth.outer)
    inner = np.concatenate([[0.0], outer[:-1]])
    mid = 0.5 * (inner + outer)
    edges = _zone_edges(n_zones)
    idx = np.searchsorted(edges, mid, side="right") - 1
    return jnp.asarray(np.clip(idx, 0, n_zones - 1))


_ZONE_IDX = _zone_index(_BASE_EARTH, N_ZONES)
_ZONE_EDGES_KM = _zone_edges(N_ZONES)
ZONE_LABELS = {
    2: ["ln rho_core", "ln rho_mantle"],
    6: ["ln rho_inner_core", "ln rho_outer_core", "ln rho_lower_mantle",
        "ln rho_transition_zone", "ln rho_upper_mantle", "ln rho_crust"],
}[N_ZONES]

# parameter vector: [ln_sc_zone_0 .. ln_sc_zone_{N-1}, theta23, dm31,
#                     ln Phi0_nu, dgamma, ln Phi0_nubar]
N_DENS = N_ZONES
IDX_TH23, IDX_DM31, IDX_PHI, IDX_DGAMMA, IDX_PHIBAR = (
    N_DENS, N_DENS + 1, N_DENS + 2, N_DENS + 3, N_DENS + 4)
N_PAR = N_DENS + 5 if INCLUDE_ANTINU else N_DENS + 4

THETA0 = jnp.zeros(N_PAR).at[IDX_TH23].set(float(_P0.theta23)) \
                         .at[IDX_DM31].set(float(_P0.dm31))
LABELS = (ZONE_LABELS + ["theta23", "dm31", "ln Phi0_nu", "dgamma"]
          + (["ln Phi0_nubar"] if INCLUDE_ANTINU else []))

# "bulk core" direction: the uniform scaling of whichever zones make up the
# physical core (density > 9 g/cc / r < 3480 km) -- the direct analogue of
# the original single ln_rho_core parameter.  For N_ZONES=2 that is zone 0
# alone (the whole core is one parameter); for N_ZONES=6 it is zones 0+1
# (inner + outer core).  NOT "the first two zones" in general -- zone 1 is
# "mantle" in the 2-zone config, so summing zones 0+1 there would silently
# mean "the whole Earth", not "the core".  Used for the headline sigma, the
# figure panels, and the design derivative, so the new numbers stay
# comparable in kind to the old ones.
CORE_ZONE_IDXS = {2: [0], 6: [0, 1]}.get(N_ZONES, [0])
BULK_CORE_VEC = jnp.zeros(N_PAR)
for _i in CORE_ZONE_IDXS:
    BULK_CORE_VEC = BULK_CORE_VEC.at[_i].set(1.0)
IS_MULTI_ZONE_CORE = len(CORE_ZONE_IDXS) > 1


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


def _prob_mumu(e_gev, cz, ln_sc, params, anti, eps=None):
    """P(nu_mu -> nu_mu) [or nubar_mu -> nubar_mu] with per-zone density scale.

    ``eps`` is an optional (3, 3) matter-NSI matrix. It is threaded here rather
    than bolted on elsewhere because the whole point of the NSI study below is
    that epsilon and the shell densities enter the same Hamiltonian term.
    """
    sc = jnp.exp(ln_sc)                          # (N_ZONES,)
    density = _BASE_EARTH.density * sc[_ZONE_IDX]
    model = dataclasses.replace(_BASE_EARTH, density=density)
    rho, ye, L = _earth.layered_chord_segments(model, cz, h_atm_km=H_ATM_KM,
                                                det_depth_km=0.0)
    v_cc, _ = C.matter_potentials(rho, ye)
    s = _osc.propagate_layers(params.pmns(), params.msquared(),
                              e_gev * C.GEV_TO_EV, v_cc,
                              L * C.KM_TO_INV_EV, anti=anti, backend="cayley",
                              nsi=eps)
    return _osc.prob_from_amplitude(s)[1, 1]


def _shape_nu(theta):
    """Un-normalized expected nu_mu counts per bin."""
    ln_sc = theta[:N_DENS]
    th23, dm31, ln_phi, dgamma = (theta[IDX_TH23], theta[IDX_DM31],
                                  theta[IDX_PHI], theta[IDX_DGAMMA])
    p = dataclasses.replace(_P0, theta23=th23, dm31=dm31)
    prob = jax.vmap(_prob_mumu, in_axes=(0, 0, None, None, None))(
        E_C, CZ_C, ln_sc, p, False)
    flux = jnp.exp(ln_phi) * (E_C / E_PIVOT) ** (-(GAMMA + dgamma))
    return flux * prob * DOMEGA


def _shape_nubar(theta):
    """Un-normalized expected nubar_mu counts per bin (anti=True)."""
    ln_sc = theta[:N_DENS]
    th23, dm31, dgamma, ln_phibar = (theta[IDX_TH23], theta[IDX_DM31],
                                     theta[IDX_DGAMMA], theta[IDX_PHIBAR])
    p = dataclasses.replace(_P0, theta23=th23, dm31=dm31)
    prob = jax.vmap(_prob_mumu, in_axes=(0, 0, None, None, None))(
        E_C, CZ_C, ln_sc, p, True)
    # spectral index shared with nu (kept minimal); normalization independent
    flux = jnp.exp(ln_phibar) * (E_C / E_PIVOT) ** (-(GAMMA + dgamma))
    return flux * prob * DOMEGA


def _shape(theta):
    """Un-normalized counts, nu block then nubar block (or nu only)."""
    if not INCLUDE_ANTINU:
        return _shape_nu(theta)
    return jnp.concatenate([_shape_nu(theta), _shape_nubar(theta)])


# absolute normalization is fixed once, at the nominal point, per species, so
# that the flux-norm nuisances are genuine free parameters rather than
# definitions
_NORM_NU = float(N_EVENTS / jnp.sum(_shape_nu(THETA0)))
_NORM_BAR = (float(NUBAR_FRAC * N_EVENTS / jnp.sum(_shape_nubar(THETA0)))
            if INCLUDE_ANTINU else 0.0)


RES_LNE = 0.20      # energy resolution, fractional (Gaussian in ln E)
RES_CZ = 0.10       # angular resolution in cos(zenith)


def response(res_lne, res_cz):
    """Differentiable (n_bins x n_bins) migration matrix, reco <- true.

    A Gaussian smearing kernel in (ln E, cos theta_z), column-normalized so it
    conserves events.  Nothing here is special -- the point is only that it is
    a JAX function of the resolution widths, so those widths are differentiable
    inputs like any other.  The same detector response applies to the nu and
    nubar samples (same reconstruction).
    """
    dlne = jnp.log(E_C)[:, None] - jnp.log(E_C)[None, :]
    dcz = CZ_C[:, None] - CZ_C[None, :]
    R = jnp.exp(-0.5 * (dlne / res_lne) ** 2 - 0.5 * (dcz / res_cz) ** 2)
    return R / jnp.sum(R, axis=0, keepdims=True)


def _apply_response_block(R, x):
    """Apply the (n_bins x n_bins) response to a (possibly nu+nubar) stack.

    ``x`` has leading dimension ``N_BINS`` (nu only) or ``2*N_BINS`` (nu then
    nubar); the response never mixes species, so it applies block-wise.
    """
    if not INCLUDE_ANTINU:
        return R @ x
    return jnp.concatenate([R @ x[:N_BINS], R @ x[N_BINS:]], axis=0)


def counts(theta, res_lne=RES_LNE, res_cz=RES_CZ):
    R = response(res_lne, res_cz)
    c_nu = R @ (_NORM_NU * _shape_nu(theta))
    if not INCLUDE_ANTINU:
        return c_nu
    c_bar = R @ (_NORM_BAR * _shape_nubar(theta))
    return jnp.concatenate([c_nu, c_bar])


def fisher(theta, res_lne=RES_LNE, res_cz=RES_CZ):
    """Poisson Fisher matrix F_ij = sum_b (1/mu_b) dmu_b/dth_i dmu_b/dth_j."""
    mu = counts(theta, res_lne, res_cz)
    J = jax.jacfwd(counts)(theta, res_lne, res_cz)   # (n_bins, n_par)
    return J.T @ (J / mu[:, None]), mu, J


def _true_space():
    """mu and dmu/dtheta BEFORE the response matrix, computed once.

    The response matrix depends on the resolutions but not on theta, and it
    enters linearly (block-wise per species), so mu(res) = R(res) @ mu_true
    and J(res) = R(res) @ J_true within each species block.  Factoring it out
    this way turns a resolution scan -- which would otherwise need a full
    jacfwd through the layered Earth per point -- into a sequence of small
    matrix products.
    """
    mu_true_nu = _NORM_NU * _shape_nu(THETA0)
    J_true_nu = _NORM_NU * jax.jacfwd(_shape_nu)(THETA0)
    if not INCLUDE_ANTINU:
        return mu_true_nu, J_true_nu
    mu_true_bar = _NORM_BAR * _shape_nubar(THETA0)
    J_true_bar = _NORM_BAR * jax.jacfwd(_shape_nubar)(THETA0)
    return (jnp.concatenate([mu_true_nu, mu_true_bar]),
            jnp.concatenate([J_true_nu, J_true_bar]))


def _bulk_norm(v):
    """Number of zones in the bulk direction; see the note in main() -- the
    common-mode estimator is the average, so sigma carries a 1/n."""
    return jnp.sum(v * v)


def sigma_from_response(res_cz, mu_true, J_true, v=BULK_CORE_VEC, res_lne=RES_LNE):
    """sigma(v . theta) for a given angular resolution, reusing the true-space
    Jacobian.  Must agree with sigma_core() below, which recomputes everything."""
    R = response(res_lne, res_cz)
    mu = _apply_response_block(R, mu_true)
    J = _apply_response_block(R, J_true)
    F = J.T @ (J / mu[:, None])
    return float(jnp.sqrt(v @ jnp.linalg.inv(F) @ v) / _bulk_norm(v))


def sigma_core(res_cz, res_lne=RES_LNE, v=BULK_CORE_VEC):
    """Marginalized sigma(v . theta) as a function of angular resolution.

    Differentiating THIS is the composability claim in one line: the gradient
    runs through the oscillation probability, the flux, the histogram, the
    response matrix, the Fisher matrix and its inverse.  Defaults to the
    "bulk core" direction (inner + outer core), the direct analogue of the
    original single ln_rho_core parameter.
    """
    mu = counts(THETA0, res_lne, res_cz)
    J = jax.jacfwd(counts)(THETA0, res_lne, res_cz)
    F = J.T @ (J / mu[:, None])
    return jnp.sqrt(v @ jnp.linalg.inv(F) @ v) / _bulk_norm(v)


def nsi_degeneracy_study():
    """Matter NSI and the shell densities enter the same Hamiltonian term.

    H_matter = V_CC(x) [diag(1,0,0) + eps]. Scaling every shell density by
    (1+d) and setting eps_ee = d are therefore the *same* operation, exactly,
    not approximately. Adding eps_ee to a per-zone density fit must then leave
    the Fisher matrix with an exactly flat direction: raise all the densities
    and lower eps_ee to compensate.

    Nothing below derives that. The Fisher matrix is assembled by automatic
    differentiation exactly as in main(), and the degeneracy shows up on its
    own in the eigen-decomposition. What survives is the part of the profile
    orthogonal to it, which is the physically interesting statement: with NSI
    free, oscillation tomography constrains the *shape* of the density profile
    but not its overall scale.
    """
    from jaxnu.nsi import NSI

    def shape_eps(theta, anti):
        ln_sc = theta[:N_DENS]
        eps = NSI(eps_ee=theta[N_PAR], eps_etau=theta[N_PAR + 1] + 0.0j).matrix(3)
        p = dataclasses.replace(_P0, theta23=theta[IDX_TH23], dm31=theta[IDX_DM31])
        pr = jax.vmap(_prob_mumu, in_axes=(0, 0, None, None, None, None))(
            E_C, CZ_C, ln_sc, p, anti, eps)
        ln_phi = theta[IDX_PHIBAR] if (anti and INCLUDE_ANTINU) else theta[IDX_PHI]
        flux = jnp.exp(ln_phi) * (E_C / E_PIVOT) ** (-(GAMMA + theta[IDX_DGAMMA]))
        return flux * pr * DOMEGA

    def counts_eps(theta):
        R = response(RES_LNE, RES_CZ)
        c = R @ (_NORM_NU * shape_eps(theta, False))
        if not INCLUDE_ANTINU:
            return c
        return jnp.concatenate([c, R @ (_NORM_BAR * shape_eps(theta, True))])

    th0 = jnp.concatenate([THETA0, jnp.zeros(2)])
    mu = np.asarray(counts_eps(th0))
    J = np.asarray(jax.jacfwd(counts_eps)(th0))
    F = J.T @ (J / mu[:, None])
    labels = LABELS + ["eps_ee", "Re eps_etau"]

    print("\n" + "=" * 70)
    print("NSI and the density profile: an exact degeneracy, found by the "
          "Fisher matrix")
    print("=" * 70)

    # (a) the flat direction, exhibited rather than asserted
    w, V = np.linalg.eigh(F)
    v0 = V[:, 0] / np.abs(V[:, 0]).max()
    print(f"  cond(F) with eps free      : {np.linalg.cond(F):.3e}")
    print(f"  lambda_min / lambda_max    : {w[0] / w[-1]:.3e}"
          f"   (float64 noise floor: the direction is flat)")
    print("  flattest direction:")
    for lab, c in zip(labels, v0):
        if abs(c) > 0.05:
            print(f"      {lab:24s} {c:+.4f}")
    print("  i.e. raise every shell density and lower eps_ee to match.")

    # (b) which combinations survive. The uniform mode is the degenerate one;
    #     the core-minus-mantle contrast is orthogonal to it by construction.
    n_par2 = N_PAR + 2
    uniform = np.zeros(n_par2); uniform[:N_DENS] = 1.0 / N_DENS
    contrast = np.zeros(n_par2)
    for i in CORE_ZONE_IDXS:
        contrast[i] = 1.0 / len(CORE_ZONE_IDXS)
    mantle = [i for i in range(N_DENS) if i not in CORE_ZONE_IDXS]
    for i in mantle:
        contrast[i] = -1.0 / len(mantle)
    bulk = np.zeros(n_par2)
    bulk[:N_PAR] = np.asarray(BULK_CORE_VEC)

    def sigma_of(F_, v, norm):
        """sigma along v, or None if v overlaps a flat direction.

        np.linalg.pinv must NOT be used here. On a singular Fisher matrix it
        truncates the null space, which imposes an infinitely *strong*
        constraint on the flat direction rather than none at all, and returns a
        sigma smaller than the non-degenerate case. The variance along v is
        sum_i (v.e_i)^2 / lambda_i over the eigenbasis; a direction with
        support on a null eigenvector simply has no bound.
        """
        lam, U = np.linalg.eigh(F_)
        c = U.T @ v
        flat = lam <= lam.max() * 1e-14
        if np.any(flat) and np.sqrt((c[flat] ** 2).sum()) > 1e-6 * np.linalg.norm(c):
            return None
        return float(np.sqrt((c[~flat] ** 2 / lam[~flat]).sum())) / norm

    F_std = F[:N_PAR, :N_PAR]                      # eps fixed at zero
    # a Gaussian prior on eps, without which the flat direction is unbounded
    SIG_EPS = 0.1
    F_pri = F.copy()
    F_pri[N_PAR, N_PAR] += 1.0 / SIG_EPS ** 2
    F_pri[N_PAR + 1, N_PAR + 1] += 1.0 / SIG_EPS ** 2

    print(f"\n  {'direction':28s} {'eps fixed':>12s} {'eps free':>12s}"
          f" {'eps, prior ' + str(SIG_EPS):>16s}")
    for name, v, nrm in (("uniform density scale", uniform, 1.0),
                         ("core/mantle contrast", contrast, 1.0),
                         ("bulk core (headline)", bulk,
                          float(np.asarray(BULK_CORE_VEC) @ np.asarray(BULK_CORE_VEC)))):
        fmt = lambda x: "  unbounded" if x is None else f"{x:10.4f}"
        print(f"  {name:28s} {fmt(sigma_of(F_std, v[:N_PAR], nrm)):>12s}"
              f" {fmt(sigma_of(F, v, nrm)):>12s}"
              f" {fmt(sigma_of(F_pri, v, nrm)):>16s}")
    print("\n  The uniform scale is lost entirely; the contrast is untouched.")
    print("  With eps free, oscillation tomography measures the shape of the")
    print("  density profile, not its normalization.")

    # (c) eps_etau has a different flavor structure and is not degenerate
    F_etau = np.delete(np.delete(F, N_PAR, 0), N_PAR, 1)     # drop eps_ee only
    w_etau = np.linalg.eigvalsh(F_etau)
    print(f"\n  with eps_etau free but eps_ee fixed: "
          f"lambda_min/lambda_max = {w_etau[0] / w_etau[-1]:.3e}")
    print("  The degeneracy is specific to eps_ee, which shares the density's")
    print("  flavor structure. Off-diagonal NSI does not reproduce it.")


def main():
    print(f"grid: {N_E} x {N_CZ} = {N_BINS} bins per species, "
          f"N_ZONES={N_ZONES}, INCLUDE_ANTINU={INCLUDE_ANTINU}, "
          f"n_par={N_PAR}")
    if N_DENS == 6:
        print("zone radial ranges (km): " + ", ".join(
            f"{lbl.replace('ln rho_', '')}=[{lo:.1f},{hi:.1f}]"
            for lbl, lo, hi in zip(ZONE_LABELS, _ZONE_EDGES_KM[:-1],
                                   _ZONE_EDGES_KM[1:])))

    F, mu, J = fisher(THETA0)
    F = np.asarray(F)
    cov = np.linalg.inv(F)
    Jn = np.asarray(J)
    mun = np.asarray(mu)

    print(f"\ntotal events check: {float(jnp.sum(mu)):.1f}")

    print("\n                             sigma (stat only)")
    print("parameter                fixed nuisances   marginalized    penalty")
    for i in range(N_DENS):
        fixed = 1.0 / np.sqrt(F[i, i])
        marg = np.sqrt(cov[i, i])
        print(f"{LABELS[i]:<24s} {fixed:>13.4f}   {marg:>12.4f}   "
              f"{marg/fixed:>8.2f}x")

    # --- the headline number: bulk core, same physical region the original
    # 2-parameter model's single ln_rho_core covered (zone 0 alone for
    # N_ZONES=2; zones 0+1 = inner+outer core for N_ZONES=6) ----------------
    v = np.asarray(BULK_CORE_VEC)
    core_zone_names = " + ".join(LABELS[i] for i in CORE_ZONE_IDXS)
    # Both numbers must describe the SAME quantity. Take it to be q, the
    # common-mode scaling: theta = theta0 + q*v, i.e. every core zone shifts by
    # the same q. Then the reduced-model (all-else-fixed) uncertainty is
    # 1/sqrt(v^T F v), and the marginalized one is sqrt(v^T C v)/n, where
    # n = v.v is the number of zones summed -- the estimator of a common shift
    # is the AVERAGE of the per-zone shifts, not their sum. Omitting that 1/n
    # compares sigma(a1+a2) against sigma(q) and overstates the degradation by
    # exactly n.
    n_core = float(v @ v)
    bulk_fixed = 1.0 / np.sqrt(v @ F @ v)
    bulk_marg = float(np.sqrt(v @ cov @ v)) / n_core
    print(f"\nbulk core ({core_zone_names}, same physical region as the old "
          f"single ln rho_core):")
    print(f"  fixed nuisances : {bulk_fixed:.4f}")
    print(f"  marginalized    : {bulk_marg:.4f}")

    if N_DENS > 1:
        d = np.sqrt(np.diag(cov[:N_DENS, :N_DENS]))
        corr = cov[:N_DENS, :N_DENS] / np.outer(d, d)
        print("\ncorrelation matrix among density-zone parameters:")
        short = [l.replace("ln rho_", "") for l in LABELS[:N_DENS]]
        hdr = "".join(f"{s:>16s}" for s in short)
        print("                " + hdr)
        for i in range(N_DENS):
            row = "".join(f"{corr[i,j]:>16.2f}" for j in range(N_DENS))
            print(f"{short[i]:<16s}{row}")
        if IS_MULTI_ZONE_CORE:
            i0, i1 = CORE_ZONE_IDXS[0], CORE_ZONE_IDXS[1]
            print(f"\n{short[i0]} / {short[i1]} correlation: {corr[i0,i1]:+.3f}")

    # which nuisances actually cost the (coordinate) parameter LABELS[0]?
    # (Schur complement over subsets -- only valid for coordinate directions,
    # so this uses zone 0 directly rather than the bulk-core combination.)
    def marg(keep):
        idx = [0] + sorted(keep)
        sub = F[np.ix_(idx, idx)]
        return np.sqrt(np.linalg.inv(sub)[0, 0])

    print(f"\nwhere the marginalization cost on {LABELS[0]} comes from:")
    print(f"  all other parameters fixed              : {1/np.sqrt(F[0,0]):.4f}")
    if N_DENS > 1:
        print(f"  marginalize {LABELS[1].replace('ln rho_',''):<16s} only     : "
              f"{marg([1]):.4f}")
    if N_DENS > 2:
        print(f"  marginalize all other density zones     : "
              f"{marg(list(range(1, N_DENS))):.4f}")
    print(f"  marginalize theta23, dm31 only           : "
          f"{marg([IDX_TH23, IDX_DM31]):.4f}")
    print(f"  marginalize everything                   : {np.sqrt(cov[0,0]):.4f}")

    # --- condition number sanity check --------------------------------------
    cond_F = np.linalg.cond(F)
    resid = np.abs(F @ cov - np.eye(len(F))).max()
    print(f"\ncondition number of F: {cond_F:.3e}")
    print(f"  max |F @ F^-1 - I| = {resid:.2e} "
          f"(float64 eps ~1e-16)")
    # A small residual is a backward-error statement; at this condition number
    # it does not on its own bound the forward error. Compare against an
    # independent inversion route (SVD pseudo-inverse) instead of asserting it.
    rel_svd = float((np.abs(np.linalg.pinv(F) - cov)
                     / np.abs(cov)).max())
    print(f"  max relative deviation, SVD pseudo-inverse vs solve: "
          f"{rel_svd:.2e} (this is what says the inverse is trustworthy)")

    # --- where does the (bulk) core information actually come from? --------
    Jv = Jn @ v                                # (n_bins_total,) dmu/d(bulk core)
    per_bin_all = (Jv ** 2) / mun
    cz_full = (np.asarray(CZ_C) if not INCLUDE_ANTINU
              else np.concatenate([np.asarray(CZ_C), np.asarray(CZ_C)]))
    core_mask = cz_full < CORE_CZ
    frac_info = per_bin_all[core_mask].sum() / per_bin_all.sum()
    frac_bins = core_mask.mean()
    frac_evts = mun[core_mask].sum() / mun.sum()
    print(f"\ncore-crossing bins (cos theta_z < {CORE_CZ:.4f}), bulk-core "
          f"({core_zone_names}) direction:")
    print(f"  {frac_bins*100:.1f}% of bins, {frac_evts*100:.1f}% of events, "
          f"but {frac_info*100:.1f}% of the fixed-nuisance information")
    print(f"  (this additive-over-bins decomposition holds for ANY linear "
          f"combination of\n   coordinate parameters -- including the bulk-"
          f"core direction -- so it still makes\n   sense with N_ZONES="
          f"{N_ZONES}; it is the marginalized information that does not "
          f"split this way.)")

    print(f"\nstatistical scaling: sigma(bulk core) = "
          f"{bulk_marg*np.sqrt(N_EVENTS/1e5):.4f} x sqrt(1e5 / N_events)")

    # --- the composability one-liner: d sigma / d (angular resolution) -------
    # sigma_core already runs jacfwd over the parameters, so grad(sigma_core)
    # is grad-of-jacfwd: a mixed second derivative d2 mu / d theta d res.
    # Differentiating once more gives the curvature, which says how far the
    # tangent can be extrapolated.  It is nearly free: the inner Jacobian is
    # already N_PAR-way forward mode, so one more tangent direction adds
    # about 1/N_PAR of the work.
    s0 = float(sigma_core(RES_CZ))
    dsdr = float(jax.grad(sigma_core)(RES_CZ))
    d2sdr2 = float(jax.grad(jax.grad(sigma_core))(RES_CZ))
    h = 1e-3
    fd = (float(sigma_core(RES_CZ + h)) - float(sigma_core(RES_CZ - h))) / (2 * h)
    g = jax.jit(jax.grad(sigma_core))
    h2 = 2e-3
    fd2 = (float(g(RES_CZ + h2)) - float(g(RES_CZ - h2))) / (2 * h2)
    print(f"\nexperimental-design derivative (bulk core, through the matrix "
          f"inverse):")
    print(f"  sigma(bulk core) at res_cz={RES_CZ}        : {s0:.5f}")
    print(f"  d sigma / d res_cz  (autodiff)             : {dsdr:+.5f}")
    print(f"  d sigma / d res_cz  (central differences)  : {fd:+.5f}")
    print(f"  d2 sigma / d res_cz^2  (autodiff)          : {d2sdr2:+.5f}")
    print(f"  d2 sigma / d res_cz^2  (FD of exact grad)  : {fd2:+.5f}")
    print(f"  -> sharpening cos(theta) resolution by 0.01 reduces "
          f"sigma(bulk core) by {dsdr*0.01/s0*100:.2f}%")
    # how far the linear extrapolation can be trusted
    mu_t, J_t = _true_space()
    print("  range of validity of the tangent:")
    for dr in (0.01, 0.02, 0.05):
        lin = s0 + dsdr * (-dr)
        quad = lin + 0.5 * d2sdr2 * dr ** 2
        exact = sigma_from_response(RES_CZ - dr, mu_t, J_t)
        print(f"    sharpen by {dr:.2f}: linear {lin:.5f} ({100*abs(lin-exact)/exact:4.2f}% off)"
              f"  quadratic {quad:.5f} ({100*abs(quad-exact)/exact:4.2f}% off)"
              f"  exact {exact:.5f}")

    np.savez(_out("tomography_fisher.npz"), F=F, cov=cov, mu=mun,
             theta0=np.asarray(THETA0), labels=np.array(LABELS),
             frac_info=frac_info, n_zones=N_ZONES,
             include_antinu=INCLUDE_ANTINU)
    print(f"\nsaved {_out('tomography_fisher.npz')}")

    _figure(Jn, mun, per_bin_all, frac_info, s0, dsdr, frac_bins * 100, cov,
            d2sdr2)

    nsi_degeneracy_study()


def _figure(Jn, mun, per_bin_all, frac_info, s0, dsdr, frac_bins_pct, cov,
            d2sdr2=None):
    """Four panels: the derivative after the full chain, where the
    information lands, what the design derivative buys, and (new) the
    density-parameter correlation matrix -- the exhibit a 2-parameter density
    model cannot produce."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The figure is placed at \textwidth (~6.3 in) in a single-column journal
    # layout, so a 16.5 in canvas is scaled down by ~0.38.  Default 10 pt labels
    # would print at under 4 pt.  Scale the fonts up so they land at ~9-10 pt on
    # paper, and keep the canvas modest so the scale factor is small.
    plt.rcParams.update({
        "font.size": 17, "axes.titlesize": 19, "axes.labelsize": 17,
        "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14,
        "axes.linewidth": 1.2, "lines.linewidth": 2.0,
    })

    E = np.asarray(E_C).reshape(N_E, N_CZ)
    CZ = np.asarray(CZ_C).reshape(N_E, N_CZ)
    # nu-channel only for the spatial maps (a)/(b) -- the antinu channel
    # enters the Fisher matrix identically and is not re-plotted here.
    dmu = sum(Jn[:N_BINS, i] for i in CORE_ZONE_IDXS).reshape(N_E, N_CZ)
    info = per_bin_all[:N_BINS].reshape(N_E, N_CZ)

    # 2x2: (a)(b)(c) as before, (d) is now the correlation-matrix exhibit.
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.6))
    ax = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    # (a) dmu/d(bulk core) in RECONSTRUCTED space (nu channel) -- i.e. after
    #     flux weighting and after the response matrix.
    v = np.abs(dmu).max()
    m = ax[0].pcolormesh(CZ, E, dmu, cmap="RdBu_r", vmin=-v, vmax=v,
                         shading="auto")
    plt.colorbar(m, ax=ax[0], label=r"$\partial\mu_b/\partial\ln\rho_{\rm core}$")
    ax[0].set_title(r"(a) count derivative, reconstructed space", pad=14)

    # (b) per-bin Fisher information on the bulk-core direction (nu channel).
    #     Log colour scale: the dynamic range is several decades and a linear
    #     scale shows only the single brightest cell, hiding the structure
    #     that makes the point.
    from matplotlib.colors import LogNorm
    pos = info[info > 0]
    m = ax[1].pcolormesh(CZ, E, np.maximum(info, pos.min()), cmap="viridis",
                         norm=LogNorm(vmin=max(pos.min(), pos.max() * 1e-5),
                                      vmax=pos.max()), shading="auto")
    plt.colorbar(m, ax=ax[1], label=r"per-bin contribution to $F_{00}$")
    ax[1].set_title(r"(b) where the information is", pad=14)
    ax[1].annotate(f"{frac_info*100:.0f}% of info\nleft of the line\n"
                   f"({frac_bins_pct:.0f}% of the bins)",
                   xy=(0.42, 0.88), xycoords="axes fraction", color="w",
                   fontsize=10, fontweight="bold", va="top")

    for i, a in enumerate(ax[:2]):
        col = "w" if i == 1 else "k"
        a.axvline(CORE_CZ, color=col, ls="--", lw=1.4)
        a.set_yscale("log")
        # ylim upper deliberately > the 2e1 major tick (not exactly on it): a
        # tick that sits exactly on the axis boundary has its label straddle
        # the top spine and collide with the title text above.
        a.set_ylim(E.min(), 23.0)
        a.set_xlabel(r"$\cos\theta_z$ (reconstructed)")
        a.annotate("core-crossing", xy=(CORE_CZ, 19.0), xytext=(-4, 0),
                   textcoords="offset points", rotation=90, ha="right",
                   va="top", fontsize=8.5, color=col)
    # short label: "reconstructed" is already established by the panel title
    # and the x-axis; the full string was tall enough (rotated) to collide
    # with the title above it.
    ax[0].set_ylabel("energy [GeV]")

    # (c) the design derivative: AD tangent against an explicit scan
    mu_true, J_true = _true_space()
    grid = np.linspace(0.04, 0.20, 17)
    scan = [sigma_from_response(float(r), mu_true, J_true) for r in grid]
    ax[2].plot(grid, scan, "o-", ms=3.5, color="0.25",
               label="explicit scan (17 evaluations)")
    # drawn over the full scan, not just near the operating point: the whole
    # point of the quadratic term is where the tangent stops working, and
    # that is only visible if the tangent is extrapolated far enough to fail.
    tan = np.linspace(grid[0], grid[-1], 2)
    ax[2].plot(tan, s0 + dsdr * (tan - RES_CZ), "r-", lw=2.2,
               label=fr"tangent: $\partial\sigma/\partial\sigma_c={dsdr:.4f}$")
    if d2sdr2 is not None:
        # one more nested derivative: where the tangent stops being a good
        # extrapolation.  Drawn over the full scan so the departure is visible.
        q = np.linspace(grid[0], grid[-1], 60)
        ax[2].plot(q, s0 + dsdr * (q - RES_CZ) + 0.5 * d2sdr2 * (q - RES_CZ) ** 2,
                   color="tab:blue", ls="--", lw=1.8,
                   label=fr"$+$ curvature: $\partial^2\sigma/"
                         fr"\partial\sigma_c^2={d2sdr2:.3f}$")
    ax[2].plot([RES_CZ], [s0], "r*", ms=13, zorder=5)
    ax[2].set_xlabel(r"angular resolution $\sigma_{\cos\theta_z}$")
    ax[2].set_ylabel(r"$\sigma(\ln\rho_{\rm core,\,bulk})$")
    ax[2].set_title("(c) experimental-design derivative", pad=14)
    # short labels: with three entries the long "(one gradient call)"
    # annotations no longer fit against the diagonal curve, and the cost
    # contrast they carried is stated in the paper caption instead
    # headroom so the three-entry legend clears the extrapolated tangent,
    # which now runs to the top-right corner of the panel
    ax[2].set_ylim(top=max(scan) * 1.28)
    ax[2].legend(loc="upper left", frameon=False, fontsize=11)
    ax[2].grid(alpha=0.3)

    # (d) NEW: the density-parameter correlation matrix -- the exhibit a
    #     2-parameter density model structurally cannot produce.  Diverging
    #     colour scale centred at zero; annotated with the numeric values.
    if N_DENS > 1:
        d = np.sqrt(np.diag(cov[:N_DENS, :N_DENS]))
        corr = cov[:N_DENS, :N_DENS] / np.outer(d, d)
        short = [l.replace("ln rho_", "").replace("_", "\n")
                for l in LABELS[:N_DENS]]
        m = ax[3].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax[3].set_xticks(range(N_DENS))
        ax[3].set_yticks(range(N_DENS))
        ax[3].set_xticklabels(short, fontsize=10, rotation=45, ha="right")
        ax[3].set_yticklabels(short, fontsize=10)
        for i in range(N_DENS):
            for j in range(N_DENS):
                txt_color = "white" if abs(corr[i, j]) > 0.6 else "black"
                ax[3].text(j, i, f"{corr[i,j]:+.2f}", ha="center", va="center",
                          fontsize=9.5, color=txt_color)
        plt.colorbar(m, ax=ax[3], label="correlation coefficient", shrink=0.85)
        ax[3].set_title("(d) density-zone correlations", pad=14)
    else:
        ax[3].axis("off")
        ax[3].annotate("correlation matrix needs N_ZONES>1\n"
                       "(run with the default N_ZONES=6)",
                       xy=(0.5, 0.5), xycoords="axes fraction", ha="center")

    fig.tight_layout(w_pad=2.6, h_pad=2.2)
    for ext in ("png", "pdf"):
        fig.savefig(_out(f"tomography_fisher.{ext}"), dpi=150, bbox_inches="tight",
                   pad_inches=0.08)
    print(f"saved {_out('tomography_fisher.png/.pdf')}")

    # the scan and the AD tangent are independent routes to the same slope
    i = int(np.argmin(np.abs(grid - RES_CZ)))
    num = (scan[i + 1] - scan[i - 1]) / (grid[i + 1] - grid[i - 1])
    print(f"  cross-check: scan slope at the operating point = {num:+.5f} "
          f"vs AD {dsdr:+.5f}")


if __name__ == "__main__":
    main()
