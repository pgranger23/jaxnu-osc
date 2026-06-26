"""Tests for the atmospheric (production height) and solar adiabatic features."""

import os

import numpy as np
import jax.numpy as jnp

import jaxnu
from jaxnu import (nufit_no, probability_earth, probability_vacuum,
                   Flavor, earth, solar)


def test_atmospheric_downgoing_is_vacuum():
    # cos_zenith = +1 -> pure vacuum over the production-height baseline.
    p = nufit_no()
    Pv = float(probability_vacuum(p, jnp.asarray(0.5), 15.0,
                                  flavor_in=Flavor.MU, flavor_out=Flavor.MU))
    Pa = float(probability_earth(p, jnp.asarray(0.5), jnp.asarray(1.0),
                                 h_atm_km=15.0,
                                 flavor_in=Flavor.MU, flavor_out=Flavor.MU))
    assert abs(Pv - Pa) < 1e-6


def test_chord_segments_lengths():
    # Diameter + atmosphere for cz=-1; pure atmosphere for cz=+1.
    table = earth.shell_table(4)
    _, _, L1 = earth.chord_segments(-1.0, table, h_atm_km=15.0)
    assert abs(float(L1.sum()) - (2 * earth.R_EARTH_KM + 15.0)) < 1e-3
    _, _, L2 = earth.chord_segments(1.0, table, h_atm_km=15.0)
    assert abs(float(L2.sum()) - 15.0) < 1e-6
    # Up-going through the core, no atmosphere -> Earth diameter.
    _, _, L3 = earth.chord_segments(-1.0, table, h_atm_km=0.0)
    assert abs(float(L3.sum()) - 2 * earth.R_EARTH_KM) < 1e-3


def test_ye_configurable_changes_result():
    p = nufit_no()
    a = probability_earth(p, jnp.asarray(6.0), jnp.asarray(-1.0),
                          ye_core=0.4656, ye_mantle=0.4957)
    b = probability_earth(p, jnp.asarray(6.0), jnp.asarray(-1.0),
                          ye_core=0.40, ye_mantle=0.40)
    assert float(jnp.max(jnp.abs(a - b))) > 1e-3


def test_ye_core_differentiable_and_geometrically_localized():
    import jax
    p = nufit_no()
    yc0 = jnp.asarray(0.466)
    f = lambda yc, c: probability_earth(p, jnp.asarray(6.0), jnp.asarray(c),
                                        ye_core=yc, flavor_in=Flavor.MU,
                                        flavor_out=Flavor.E)
    # core-crossing path: gradient is non-trivial and matches finite differences
    ad = float(jax.grad(lambda yc: f(yc, -1.0))(yc0))
    h = 1e-6
    fd = (float(f(yc0 + h, -1.0)) - float(f(yc0 - h, -1.0))) / (2 * h)
    assert abs(ad) > 1e-2 and abs(ad - fd) < 1e-4 * (1.0 + abs(fd))
    # mantle-only path (cos z = -0.3, closest approach above the 3480 km core):
    # the gradient w.r.t. the *core* electron fraction is exactly zero.
    assert abs(float(jax.grad(lambda yc: f(yc, -0.3))(yc0))) < 1e-12


def test_solar_adiabatic_unitarity_and_lma():
    p = nufit_no()
    prof = solar.exponential_profile()  # no external data needed
    R = prof.R_sun_km
    r = np.geomspace(0.05 * R, R, 50)
    F = np.array(solar.adiabatic_mass_fractions(p, 0.008, prof, r,
                                                0.05 * R, alpha=0))
    assert np.allclose(F.sum(axis=1), 1.0, atol=1e-9)
    # LMA: nu_e emerges predominantly as nu_2 at the surface.
    assert F[-1, 1] > F[-1, 0] and F[-1, 1] > 0.5
