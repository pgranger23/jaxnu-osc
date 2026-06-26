"""Tests for the NSI and sterile-neutrino front-ends."""

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp

import jaxnu
from jaxnu import (OscParams, NSI, Sterile3plus1, NFlavorParams, pmns_nflavor,
                   nufit_no, probability_constant, probability_vacuum,
                   probability_earth, Flavor)


def _sterile(p3, **over):
    kw = dict(theta12=p3.theta12, theta13=p3.theta13, theta23=p3.theta23,
              theta14=jnp.asarray(0.0), theta24=jnp.asarray(0.0),
              theta34=jnp.asarray(0.0), delta13=p3.deltacp,
              delta24=jnp.asarray(0.0), dm21=p3.dm21, dm31=p3.dm31,
              dm41=jnp.asarray(1.0))
    kw.update({k: jnp.asarray(v) for k, v in over.items()})
    return Sterile3plus1(**kw)


# --- NSI ---------------------------------------------------------------------

def test_nsi_zero_is_standard():
    p = nufit_no()
    E = jnp.linspace(0.5, 5.0, 16)
    P0 = probability_constant(p, E, 1300.0, density=2.8)
    Pn = probability_constant(p, E, 1300.0, density=2.8, nsi=NSI())
    assert float(jnp.max(jnp.abs(P0 - Pn))) < 1e-12


def test_nsi_changes_result_and_unitary():
    p = nufit_no()
    E = jnp.linspace(0.5, 5.0, 16)
    P0 = probability_constant(p, E, 1300.0, density=2.8)
    Pn = probability_constant(p, E, 1300.0, density=2.8,
                              nsi=NSI(eps_emu=0.1 + 0.05j, eps_ee=0.05))
    assert float(jnp.max(jnp.abs(P0 - Pn))) > 1e-3
    assert float(jnp.max(jnp.abs(Pn.sum(axis=-2) - 1.0))) < 1e-10


def test_nsi_gradient():
    p = nufit_no()
    f = lambda e: probability_constant(p, jnp.asarray(2.0), 1300.0, density=2.8,
                                       nsi=NSI(eps_ee=e), flavor_in=Flavor.MU,
                                       flavor_out=Flavor.E)
    ad = float(jax.grad(f)(jnp.asarray(0.05)))
    h = 1e-5
    fd = (float(f(jnp.asarray(0.05 + h))) - float(f(jnp.asarray(0.05 - h)))) / (2 * h)
    assert abs(ad - fd) < 1e-4


# --- sterile -----------------------------------------------------------------

def test_sterile_reduces_to_three_flavor():
    p = nufit_no()
    st = _sterile(p)  # all theta_i4 = 0
    E = jnp.linspace(0.5, 5.0, 16)
    P3 = np.array(probability_constant(p, E, 1300.0, density=2.8))
    P4 = np.array(probability_constant(st, E, 1300.0, density=2.8))
    assert P4.shape[-1] == 4
    assert np.max(np.abs(P4[:, :3, :3] - P3)) < 1e-12


def test_sterile_raa_vacuum_depth():
    # Reactor antineutrino sterile: vacuum P_ee minimum = 1 - sin^2(2 theta14).
    p = nufit_no()
    s2 = 0.1
    th14 = 0.5 * np.arcsin(np.sqrt(s2))
    st = _sterile(p, theta14=th14, dm41=1.0)
    E = 0.004
    L = 1.2369 * E  # dm41 oscillation maximum (1.267*dm41*L/E = pi/2)
    Pee = float(probability_vacuum(st, jnp.asarray(E), L, anti=True,
                                   flavor_in=Flavor.E, flavor_out=Flavor.E))
    assert abs(Pee - (1.0 - s2)) < 2e-3


def test_sterile_in_matter_unitary_and_differentiable():
    p = nufit_no()
    st = _sterile(p, theta14=0.15, theta24=0.1, dm41=0.5)
    P = probability_earth(st, jnp.asarray(3.0), jnp.asarray(-1.0))
    assert P.shape == (4, 4)
    assert float(jnp.max(jnp.abs(P.sum(axis=-2) - 1.0))) < 1e-10

    def f(th14):
        s = _sterile(p, theta14=th14, theta24=0.1, dm41=0.5)
        return probability_earth(s, jnp.asarray(3.0), jnp.asarray(-1.0),
                                 flavor_in=Flavor.MU, flavor_out=Flavor.MU)
    ad = float(jax.grad(f)(jnp.asarray(0.15)))
    h = 1e-5
    fd = (float(f(jnp.asarray(0.15 + h))) - float(f(jnp.asarray(0.15 - h)))) / (2 * h)
    assert not np.isnan(ad) and abs(ad - fd) < 1e-3


def test_nflavor_3plus2_unitary():
    # Build a 3+2 model via the generic rotation builder.
    p = nufit_no()
    U = pmns_nflavor(5, [
        (3, 4, 0.1), (2, 4, 0.05), (1, 4, 0.05), (0, 4, 0.05),
        (2, 3, 0.1), (1, 3, 0.05), (0, 3, 0.05),
        (1, 2, float(p.theta23)), (0, 2, float(p.theta13)),
        (0, 1, float(p.theta12)),
    ])
    msq = jnp.asarray([0.0, float(p.dm21), float(p.dm31), 0.5, 1.0])
    params = NFlavorParams(U=U, msq=msq, n_active=3)
    P = probability_earth(params, jnp.asarray(3.0), jnp.asarray(-0.8))
    assert P.shape == (5, 5)
    assert float(jnp.max(jnp.abs(P.sum(axis=-2) - 1.0))) < 1e-9
