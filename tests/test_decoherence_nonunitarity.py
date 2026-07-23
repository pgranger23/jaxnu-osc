"""Tests for the decoherence and non-unitarity front-ends."""

import numpy as np
import jax
import jax.numpy as jnp

import jaxnu
from jaxnu import (nufit_no, probability_constant, decoherence, nonunitarity,
                   Decoherence, WavePacket, NonUnitarity, Flavor)

P = nufit_no()
E = jnp.linspace(0.5, 10.0, 20)


# --- decoherence -------------------------------------------------------------

def test_decoherence_zero_gamma_is_standard():
    P0 = decoherence.probability(P, E, 1300.0, Decoherence(), density=2.8)
    Pref = probability_constant(P, E, 1300.0, density=2.8, backend="eigh")
    assert float(jnp.max(jnp.abs(P0 - Pref))) < 1e-12


def test_decoherence_full_damping_is_averaged():
    from jaxnu.hamiltonian import matter_hamiltonian
    from jaxnu import constants as C
    d = Decoherence(gamma21=1e-10, gamma31=1e-10, gamma32=1e-10)
    Pd = decoherence.probability(P, jnp.asarray(2.0), 1300.0, d, density=2.8)
    h = matter_hamiltonian(P.pmns(), P.msquared(), 2.0 * C.GEV_TO_EV,
                           C.matter_potential_eV(2.8, 0.5))
    _w, V = jnp.linalg.eigh(h)
    Pavg = (jnp.abs(V) ** 2) @ (jnp.abs(V) ** 2).T
    assert float(jnp.max(jnp.abs(Pd - Pavg))) < 1e-12


def test_wavepacket_limits_and_unitarity():
    # huge sigma_x -> standard; small sigma_x -> damped but still unitary rows
    Pw = decoherence.probability(P, E, 1300.0, WavePacket(sigma_x_m=1.0), density=2.8)
    Pref = probability_constant(P, E, 1300.0, density=2.8, backend="eigh")
    assert float(jnp.max(jnp.abs(Pw - Pref))) < 1e-12
    Ps = decoherence.probability(P, E, 1300.0, WavePacket(sigma_x_m=1e-15))
    assert float(jnp.max(jnp.abs(Ps.sum(axis=-2) - 1.0))) < 1e-10


def test_decoherence_differentiable():
    f = lambda g: decoherence.probability(
        P, jnp.asarray(2.0), 1300.0,
        Decoherence(gamma21=g, gamma31=g, gamma32=g), density=2.8,
        flavor_in=Flavor.MU, flavor_out=Flavor.MU)
    g0 = jnp.asarray(1e-13)
    ad = float(jax.grad(f)(g0))
    h = 1e-16
    fd = (float(f(g0 + h)) - float(f(g0 - h))) / (2 * h)
    assert np.isfinite(ad) and abs(ad - fd) < 1e-3 * (abs(fd) + 1e-12)


# --- non-unitarity -----------------------------------------------------------

def test_nonunitarity_zero_alpha_is_standard():
    nu0 = NonUnitarity()
    for dens in (0.0, 2.8):
        Pn = nonunitarity.probability(P, nu0, E, 1300.0, density=dens)
        Pr = probability_constant(P, E, 1300.0, density=dens, backend="eigh")
        assert float(jnp.max(jnp.abs(Pn - Pr))) < 1e-12, dens


def test_nonunitarity_zero_distance_effect():
    nu = NonUnitarity(alpha11=0.01, alpha21=0.02 + 0.01j)
    Pzd = np.asarray(nonunitarity.probability(P, nu, jnp.asarray(1.0), 1e-6))
    NN = np.asarray(nu.N(P.pmns()) @ jnp.conj(nu.N(P.pmns())).T)
    dg = np.real(np.diag(NN))
    expect = np.abs(NN) ** 2 / np.outer(dg, dg)
    assert np.max(np.abs(Pzd - expect)) < 1e-9
    # flavor-violating at L=0 (the hallmark of non-unitarity)
    assert Pzd[0, 1] > 1e-5


def test_nonunitarity_differentiable():
    f = lambda a: nonunitarity.probability(
        P, NonUnitarity(alpha11=a), jnp.asarray(2.0), 1300.0, density=2.8,
        flavor_in=Flavor.MU, flavor_out=Flavor.E)
    a0 = jnp.asarray(0.01)
    ad = float(jax.grad(f)(a0))
    h = 1e-6
    fd = (float(f(a0 + h)) - float(f(a0 - h))) / (2 * h)
    assert np.isfinite(ad) and abs(ad - fd) < 1e-5 * (1 + abs(fd))
