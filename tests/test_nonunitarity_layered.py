"""Tests for layered/Earth non-unitary mixing (jaxnu.nonunitarity.{probability_profile,
probability_earth}).

Covers the deliverables of the layered non-unitarity extension:
  1. a single-layer profile reproduces the existing constant-density
     ``nonunitarity.probability`` to machine precision (the key regression test);
  2. ``alpha -> 0`` reproduces the standard layered/PREM probability
     (``oscillator.probability_earth``) to machine precision;
  3. the correct invariant -- ``P(a->a; L=0) = 1`` exactly, for any alpha/energy,
     even though columns need NOT sum to 1 at finite baseline (real unitarity
     violation, not a bug);
  4. the zero-distance (flavor-violating) effect survives through a full layered
     chain, matching the closed-form ``N N^dag`` formula;
  5. gradients flow through both the alpha parameters and the geometry
     (baseline/segment length, segment density, cos(zenith)), matching finite
     differences.
"""

import numpy as np
import jax
import jax.numpy as jnp

from jaxnu import nufit_no, Flavor, nonunitarity, oscillator as osc, earth as _earth, prem_layered
from jaxnu.nonunitarity import NonUnitarity

P = nufit_no()
E = jnp.linspace(0.5, 8.0, 15)

# A generic (non-tiny, non-degenerate) alpha, real+complex, used throughout.
NU = NonUnitarity(alpha11=0.02, alpha22=0.015, alpha33=0.01,
                  alpha21=0.01 + 0.005j, alpha31=0.008 - 0.003j,
                  alpha32=0.006 + 0.002j)


# --- 1. single-layer regression -----------------------------------------------

def test_profile_single_segment_matches_constant_density():
    dens, ye, L = 2.8, 0.5, 1300.0
    for anti in (False, True):
        Pref = nonunitarity.probability(P, NU, E, L, density=dens, ye=ye, anti=anti)
        Pprof = nonunitarity.probability_profile(
            P, NU, E, jnp.array([dens]), jnp.array([ye]), jnp.array([L]),
            anti=anti, backend="eigh")
        assert float(jnp.max(jnp.abs(Pref - Pprof))) < 1e-12, anti


def test_profile_vacuum_single_segment_matches_constant_density():
    L = 500.0
    Pref = nonunitarity.probability(P, NU, E, L, density=0.0)
    Pprof = nonunitarity.probability_profile(
        P, NU, E, jnp.array([0.0]), jnp.array([0.5]), jnp.array([L]), backend="eigh")
    assert float(jnp.max(jnp.abs(Pref - Pprof))) < 1e-12


def test_profile_splitting_a_segment_is_a_noop():
    """Splitting one constant-density segment into two equal halves must not
    change the total propagator (same physical path, same physics)."""
    dens, ye, L = 3.3, 0.49, 900.0
    P1 = nonunitarity.probability_profile(
        P, NU, jnp.asarray(2.0), jnp.array([dens]), jnp.array([ye]), jnp.array([L]),
        backend="eigh")
    P2 = nonunitarity.probability_profile(
        P, NU, jnp.asarray(2.0), jnp.array([dens, dens]), jnp.array([ye, ye]),
        jnp.array([L / 2, L / 2]), backend="eigh")
    assert float(jnp.max(jnp.abs(P1 - P2))) < 1e-12


# --- 2. alpha -> 0 reproduces the standard layered/PREM probability ----------

def test_earth_alpha_zero_matches_standard_prem():
    nu0 = NonUnitarity()
    cz = jnp.array([-0.95, -0.6, -0.3, -0.05, 0.2, 0.8])
    Pn = nonunitarity.probability_earth(P, nu0, E, cz, backend="eigh")
    Pstd = osc.probability_earth(P, E, cz, backend="eigh")
    assert float(jnp.max(jnp.abs(Pn - Pstd))) < 1e-12


def test_earth_alpha_zero_matches_standard_with_layered_earth_model():
    nu0 = NonUnitarity()
    cz = jnp.array([-0.8, -0.4, 0.1])
    model = prem_layered(n_sub=2)
    Pn = nonunitarity.probability_earth(P, nu0, E, cz, earth_model=model, backend="eigh")
    Pstd = osc.probability_earth(P, E, cz, earth_model=model, backend="eigh")
    assert float(jnp.max(jnp.abs(Pn - Pstd))) < 1e-12


def test_earth_alpha_zero_matches_standard_atmospheric_and_anti():
    nu0 = NonUnitarity()
    cz = jnp.array([-0.5, 0.3, 0.9])
    for anti in (False, True):
        Pn = nonunitarity.probability_earth(
            P, nu0, jnp.asarray(1.5), cz, h_atm_km=15.0, anti=anti, backend="eigh")
        Pstd = osc.probability_earth(
            P, jnp.asarray(1.5), cz, h_atm_km=15.0, anti=anti, backend="eigh")
        assert float(jnp.max(jnp.abs(Pn - Pstd))) < 1e-12, anti


# --- 3. the correct invariant: P(a->a; L=0) = 1 exactly -----------------------
#
# Non-unitarity means sum_b P(a->b) need NOT equal 1 at finite baseline: that
# would-be probability leaks into the unobserved heavy sector this alpha
# parametrization integrates out. The one quantity that *is* exactly conserved
# for any alpha/energy/baseline is diag(P) at zero distance: A(L=0) = N N^dag,
# so P(a->a;0) = |NN^dag_aa|^2 / (NN^dag_aa)^2 = 1 identically by construction.


def test_zero_distance_diagonal_invariant_holds_through_a_layered_chain():
    table = _earth.shell_table(4)
    rho_seg, ye_seg, len_seg = _earth.chord_segments(-0.8, table)
    P0 = nonunitarity.probability_profile(
        P, NU, jnp.asarray(2.0), rho_seg, ye_seg, jnp.zeros_like(len_seg), backend="eigh")
    assert np.max(np.abs(np.diag(np.asarray(P0)) - 1.0)) < 1e-12


def test_finite_baseline_columns_need_not_sum_to_one_but_unitary_limit_does():
    table = _earth.shell_table(4)
    rho_seg, ye_seg, len_seg = _earth.chord_segments(-0.8, table)
    Pnu = nonunitarity.probability_profile(
        P, NU, jnp.asarray(2.0), rho_seg, ye_seg, len_seg, backend="eigh")
    colsum = np.asarray(jnp.sum(Pnu, axis=0))
    # genuine leakage: measurably away from 1 (not just float noise)
    assert np.max(np.abs(colsum - 1.0)) > 1e-4
    # but every probability is still a sane number
    assert np.all(np.asarray(Pnu) >= -1e-12)

    # the alpha=0 limit through the identical chain is exactly conservative
    Pstd = nonunitarity.probability_profile(
        P, NonUnitarity(), jnp.asarray(2.0), rho_seg, ye_seg, len_seg, backend="eigh")
    colsum_std = np.asarray(jnp.sum(Pstd, axis=0))
    assert np.max(np.abs(colsum_std - 1.0)) < 1e-10


# --- 4. zero-distance (flavor-violating) effect through a full layered chain -

def test_earth_zero_distance_effect_matches_closed_form():
    Pzd = nonunitarity.probability_earth(P, NU, jnp.asarray(1.0), jnp.asarray(0.999999999),
                                         backend="eigh")
    Nmat = NU.N(P.pmns())
    NN = Nmat @ jnp.conj(Nmat).T
    norm = jnp.real(jnp.diag(NN))
    expect = jnp.abs(NN) ** 2 / jnp.outer(norm, norm)
    # cos_zenith ~ 1 (straight down, ~0 baseline through Earth+atmosphere at
    # h_atm_km=0) reduces the chain to (numerically) zero length everywhere
    assert float(jnp.max(jnp.abs(Pzd - expect))) < 1e-6
    # flavor-violating off-diagonal present (the hallmark of non-unitarity)
    assert float(Pzd[0, 1]) > 1e-6


# --- 5. gradients: alpha parameters and geometry, vs finite differences ------

def _fd_grad(f, x0, h):
    return (float(f(x0 + h)) - float(f(x0 - h))) / (2 * h)


def test_gradient_wrt_alpha_through_earth():
    def f(a):
        nu = NonUnitarity(alpha11=a, alpha21=0.01 + 0.005j)
        return nonunitarity.probability_earth(
            P, nu, jnp.asarray(3.0), jnp.asarray(-0.6), backend="eigh",
            flavor_in=Flavor.MU, flavor_out=Flavor.E)

    a0 = jnp.asarray(0.02)
    ad = float(jax.grad(f)(a0))
    fd = _fd_grad(f, a0, 1e-6)
    assert np.isfinite(ad) and abs(ad - fd) < 1e-5 * (1 + abs(fd))


def test_gradient_wrt_cos_zenith_geometry():
    def f(cz):
        return nonunitarity.probability_earth(
            P, NU, jnp.asarray(3.0), cz, backend="eigh",
            flavor_in=Flavor.MU, flavor_out=Flavor.MU)

    cz0 = jnp.asarray(-0.6)
    ad = float(jax.grad(f)(cz0))
    fd = _fd_grad(f, cz0, 1e-6)
    assert np.isfinite(ad) and abs(ad - fd) < 1e-5 * (1 + abs(fd))


def test_gradient_wrt_segment_length_geometry():
    def f(l2):
        lens = jnp.array([500.0, l2])
        return nonunitarity.probability_profile(
            P, NU, jnp.asarray(3.0), jnp.array([2.8, 4.5]), jnp.array([0.5, 0.49]),
            lens, backend="eigh", flavor_in=Flavor.MU, flavor_out=Flavor.TAU)

    l0 = jnp.asarray(700.0)
    ad = float(jax.grad(f)(l0))
    fd = _fd_grad(f, l0, 1e-4)
    assert np.isfinite(ad) and abs(ad - fd) < 1e-5 * (1 + abs(fd))


def test_gradient_wrt_segment_density_geometry():
    def f(rho2):
        dens = jnp.array([2.8, rho2])
        return nonunitarity.probability_profile(
            P, NU, jnp.asarray(3.0), dens, jnp.array([0.5, 0.49]),
            jnp.array([500.0, 700.0]), backend="eigh",
            flavor_in=Flavor.E, flavor_out=Flavor.E)

    r0 = jnp.asarray(4.5)
    ad = float(jax.grad(f)(r0))
    fd = _fd_grad(f, r0, 1e-4)
    assert np.isfinite(ad) and abs(ad - fd) < 1e-5 * (1 + abs(fd))


def test_gradient_wrt_layered_earth_model_shell_boundary():
    """Differentiable in a LayeredEarth's shell boundary radii too."""
    model = prem_layered(n_sub=2)

    def f(r_last):
        m = model.__class__(
            outer=model.outer.at[-1].set(r_last), density=model.density, ye=model.ye)
        return nonunitarity.probability_earth(
            P, NU, jnp.asarray(3.0), jnp.asarray(-0.5), earth_model=m, backend="eigh",
            flavor_in=Flavor.E, flavor_out=Flavor.MU)

    r0 = model.outer[-1]
    ad = float(jax.grad(f)(r0))
    fd = _fd_grad(f, r0, 1e-2)
    assert np.isfinite(ad) and abs(ad - fd) < 1e-4 * (1 + abs(fd))


# --- consistency: product strategies, jit, vmap, oscillogram shape -----------

def test_parallel_and_sequential_scan_agree():
    table = _earth.shell_table(4)
    rho_seg, ye_seg, len_seg = _earth.chord_segments(-0.7, table)
    Pseq = nonunitarity.probability_profile(
        P, NU, jnp.asarray(2.0), rho_seg, ye_seg, len_seg, backend="eigh", parallel=False)
    Ppar = nonunitarity.probability_profile(
        P, NU, jnp.asarray(2.0), rho_seg, ye_seg, len_seg, backend="eigh", parallel=True)
    assert float(jnp.max(jnp.abs(Pseq - Ppar))) < 1e-10


def test_earth_oscillogram_grid_shape():
    cz = jnp.array([-0.9, -0.5, -0.1])
    Egrid = jnp.array([1.0, 2.0, 3.0])
    Pgrid = nonunitarity.probability_earth(P, NU, Egrid, cz, backend="eigh")
    assert Pgrid.shape == (3, 3, 3, 3)


def test_probability_earth_is_jittable():
    f = jax.jit(lambda cz: nonunitarity.probability_earth(
        P, NU, jnp.asarray(2.0), cz, backend="eigh"))
    out = f(jnp.asarray(-0.4))
    ref = nonunitarity.probability_earth(P, NU, jnp.asarray(2.0), jnp.asarray(-0.4),
                                         backend="eigh")
    assert float(jnp.max(jnp.abs(out - ref))) < 1e-12
