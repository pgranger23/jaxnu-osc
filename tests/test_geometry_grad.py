"""Regression tests for the Earth-geometry gradient bugs (see jaxnu/earth.py):

  BUG 1: ``rmin2 = r_det**2 * jnp.clip(1 - cz**2, 0, None)`` landed exactly on
  the ``jnp.clip`` (== ``jnp.maximum``) tie at ``cz = +-1``, halving
  ``d(rmin2)/dcz`` and everything downstream (baselines, ``dP/dcz``) there.

  BUG 2: ``shell_table`` built its ``ye`` column with ``np.array(...)`` on a
  list that can hold JAX tracers (``ye_core`` / ``ye_mantle``), so
  ``jax.grad(..., ye_core=...)`` raised ``TracerArrayConversionError``.

Also covers the new ``critical_cos_zenith`` diagnostic helper.  Style follows
``tests/test_grad.py``: autodiff is checked against finite differences.
"""

import jax
import jax.numpy as jnp

from mango import earth as E
from mango import nufit_no, probability_earth, probability_profile, Flavor


# --- BUG 1: cos(zenith) pole gradient ---------------------------------------

def test_grad_baseline_km_at_poles():
    """d(baseline_km)/d(cz) at cz=+-1 must match a one-sided FD taken from the
    physical side, not be halved by the old clip-tie bug."""
    f = lambda cz: E.baseline_km(cz, det_depth_km=1.9)
    h = 1e-6

    # cz = -1: only cz > -1 is physical (up-going); FD from above.
    ad = float(jax.grad(f)(jnp.asarray(-1.0)))
    fd = (float(f(jnp.asarray(-1.0 + h))) - float(f(jnp.asarray(-1.0)))) / h
    assert abs(ad - fd) < 1e-3 * (1.0 + abs(fd)), (ad, fd)
    # Would be roughly half of this (~ -6368) if the clip-tie bug regressed.
    assert ad < -12000.0, ad
    assert float(jax.jacfwd(f)(jnp.asarray(-1.0))) < -12000.0

    # cz = +1: only cz < 1 is physical (down-going, zero baseline); FD from below.
    ad = float(jax.grad(f)(jnp.asarray(1.0)))
    fd = (float(f(jnp.asarray(1.0))) - float(f(jnp.asarray(1.0 - h)))) / h
    assert ad == 0.0 and fd == 0.0
    assert abs(ad - fd) < 1e-3 * (1.0 + abs(fd))


def test_grad_dPdcz_at_poles_atmospheric():
    """Same regression for dP/dcz through the full oscillation chain.  Using
    h_atm_km > 0 makes cz=+1 non-trivial too (pure-vacuum atmospheric leg),
    so both poles actually exercise the buggy ``rmin2`` line."""
    p = nufit_no()
    f = lambda cz: probability_earth(
        p, jnp.asarray(4.0), cz, h_atm_km=15.0,
        flavor_in=Flavor.MU, flavor_out=Flavor.E)
    h = 1e-6
    # (cz0, sign of step towards the physical side)
    for cz0, sign in ((-1.0, +1.0), (1.0, -1.0)):
        ad = float(jax.grad(f)(jnp.asarray(cz0)))
        ad_jf = float(jax.jacfwd(f)(jnp.asarray(cz0)))
        fd = (float(f(jnp.asarray(cz0 + sign * h))) - float(f(jnp.asarray(cz0)))) / (sign * h)
        assert abs(ad - fd) < 1e-3 * (1.0 + abs(fd)), (cz0, ad, fd)
        assert abs(ad_jf - fd) < 1e-3 * (1.0 + abs(fd)), (cz0, ad_jf, fd)


def test_grad_cz_control_point_unaffected():
    """A control point away from the poles was never bugged; guard against a
    fix that overcorrects there."""
    p = nufit_no()
    f = lambda cz: probability_earth(
        p, jnp.asarray(4.0), cz, flavor_in=Flavor.MU, flavor_out=Flavor.E)
    cz0 = -0.6
    h = 1e-5
    ad = float(jax.grad(f)(jnp.asarray(cz0)))
    fd = (float(f(jnp.asarray(cz0 + h))) - float(f(jnp.asarray(cz0 - h)))) / (2 * h)
    assert abs(ad - fd) < 1e-3 * (1.0 + abs(fd)), (ad, fd)


# --- dP/d(h_atm_km) ----------------------------------------------------------

def test_grad_h_atm_km():
    p = nufit_no()
    f = lambda h: probability_earth(
        p, jnp.asarray(4.0), jnp.asarray(-0.5), h_atm_km=h,
        flavor_in=Flavor.MU, flavor_out=Flavor.E)
    h0, step = 15.0, 1e-2
    ad = float(jax.grad(f)(jnp.asarray(h0)))
    fd = (float(f(jnp.asarray(h0 + step))) - float(f(jnp.asarray(h0 - step)))) / (2 * step)
    rel = abs(ad - fd) / abs(ad)
    assert rel < 1e-8, (ad, fd, rel)


# --- dP/d(ln rho) for core and mantle ---------------------------------------

def test_grad_ln_rho_core_mantle():
    """Scale the densities of shells above/below the 3480 km core boundary by
    a differentiable factor exp(ln_scale) (the paper's Table convention) and
    check dP/d(ln_scale) at scale=1 -- i.e. dP/d(ln rho) -- against a central
    FD, for a core-crossing chord."""
    p = nufit_no()
    table = E.shell_table(4)
    cz = jnp.asarray(-0.95)  # steeply up-going: chord crosses the core.
    rho, ye, length_km = E.chord_segments(cz, table, h_atm_km=15.0, det_depth_km=0.0)
    core = rho > 9.0  # PREM: outer core >~9.9 g/cm^3, mantle <~5.6 g/cm^3.
    assert bool(jnp.any(core)) and bool(jnp.any((rho > 0) & ~core))

    def f(ln_core, ln_mantle):
        scale = jnp.where(core, jnp.exp(ln_core), jnp.exp(ln_mantle))
        return probability_profile(
            p, jnp.asarray(4.0), rho * scale, ye, length_km,
            flavor_in=Flavor.MU, flavor_out=Flavor.E)

    g_core, g_mantle = jax.grad(f, argnums=(0, 1))(jnp.asarray(0.0), jnp.asarray(0.0))
    g_core, g_mantle = float(g_core), float(g_mantle)

    step = 1e-5
    fd_core = (float(f(jnp.asarray(step), jnp.asarray(0.0)))
               - float(f(jnp.asarray(-step), jnp.asarray(0.0)))) / (2 * step)
    fd_mantle = (float(f(jnp.asarray(0.0), jnp.asarray(step)))
                 - float(f(jnp.asarray(0.0), jnp.asarray(-step)))) / (2 * step)

    assert abs(g_core - fd_core) / abs(g_core) < 1e-8, (g_core, fd_core)
    assert abs(g_mantle - fd_mantle) / abs(g_mantle) < 1e-8, (g_mantle, fd_mantle)


# --- BUG 2: dP/d(ye_core), dP/d(ye_mantle) ----------------------------------

def test_grad_ye_core_mantle():
    """Regression: shell_table used to build ``ye`` with ``np.array()`` on a
    list that can hold JAX tracers, raising TracerArrayConversionError under
    jax.grad(..., ye_core=...) / (..., ye_mantle=...)."""
    p = nufit_no()
    cz = jnp.asarray(-0.95)  # core-crossing, so both ye zones matter.

    f_core = lambda ye_core: probability_earth(
        p, jnp.asarray(4.0), cz, ye_core=ye_core,
        flavor_in=Flavor.MU, flavor_out=Flavor.E)
    f_mantle = lambda ye_mantle: probability_earth(
        p, jnp.asarray(4.0), cz, ye_mantle=ye_mantle,
        flavor_in=Flavor.MU, flavor_out=Flavor.E)

    ye_c0, ye_m0 = E.YE_CORE_DEFAULT, E.YE_MANTLE_DEFAULT
    g_core = float(jax.grad(f_core)(jnp.asarray(ye_c0)))
    g_mantle = float(jax.grad(f_mantle)(jnp.asarray(ye_m0)))

    step = 1e-6
    fd_core = (float(f_core(jnp.asarray(ye_c0 + step)))
               - float(f_core(jnp.asarray(ye_c0 - step)))) / (2 * step)
    fd_mantle = (float(f_mantle(jnp.asarray(ye_m0 + step)))
                 - float(f_mantle(jnp.asarray(ye_m0 - step)))) / (2 * step)

    assert abs(g_core - fd_core) / abs(g_core) < 1e-8, (g_core, fd_core)
    assert abs(g_mantle - fd_mantle) / abs(g_mantle) < 1e-8, (g_mantle, fd_mantle)

    # Cross-check against the low-level probability_profile route (hand-built
    # ye array), which never goes through shell_table's ye column at all.
    table = E.shell_table(4)
    rho_seg, _, length_km = E.chord_segments(cz, table, det_depth_km=0.0)
    core_mask = rho_seg > 9.0

    def g(ye_core_val):
        ye_seg = jnp.where(core_mask, ye_core_val, E.YE_MANTLE_DEFAULT)
        return probability_profile(
            p, jnp.asarray(4.0), rho_seg, ye_seg, length_km,
            flavor_in=Flavor.MU, flavor_out=Flavor.E)

    g_lowlevel = float(jax.grad(g)(jnp.asarray(ye_c0)))
    fd_lowlevel = (float(g(jnp.asarray(ye_c0 + step)))
                   - float(g(jnp.asarray(ye_c0 - step)))) / (2 * step)
    assert abs(g_lowlevel - fd_lowlevel) / abs(g_lowlevel) < 1e-8, (g_lowlevel, fd_lowlevel)


# --- ADDITION: critical_cos_zenith ------------------------------------------

def test_critical_cos_zenith_core_mantle_boundary():
    """The core-mantle boundary (3480 km) must appear in critical_cos_zenith,
    matching the analytic grazing angle exactly, and it must coincide with a
    genuine (unbounded, not smoothed-away) gradient anomaly in dP/dcz."""
    cc = E.critical_cos_zenith()
    assert cc.ndim == 1
    assert (cc[:-1] <= cc[1:]).all()  # sorted
    assert (cc < 0.0).all()  # up-going only

    expected = -((1.0 - (E.CORE_RADIUS_KM / E.R_EARTH_KM) ** 2) ** 0.5)
    closest = cc[abs(cc - expected).argmin()]
    assert abs(closest - expected) < 1e-9, (closest, expected)

    p = nufit_no()
    f = lambda cz: probability_earth(
        p, jnp.asarray(4.0), cz, flavor_in=Flavor.MU, flavor_out=Flavor.E)
    g_far = abs(float(jax.grad(f)(jnp.asarray(expected - 0.01))))
    g_near = abs(float(jax.grad(f)(jnp.asarray(expected - 1e-6))))
    assert g_near > 50.0 * g_far, (g_near, g_far)
