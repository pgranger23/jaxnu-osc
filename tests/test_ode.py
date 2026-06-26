"""Continuous ODE backend cross-checks the piecewise-constant layer method."""

import importlib.util

import jax
import jax.numpy as jnp

from jaxnu import nufit_no, probability_earth
from jaxnu.ode import probability_earth_continuous

_HAS_DIFFRAX = importlib.util.find_spec("diffrax") is not None


def test_ode_matches_layers_mantle():
    # cz = -0.4 keeps the chord in the mantle (constant Y_e in both backends).
    p = nufit_no()
    cz = -0.4
    for E in (2.0, 5.0, 10.0):
        Pode = probability_earth_continuous(p, E, cz)
        Play = probability_earth(p, jnp.asarray(E), jnp.asarray(cz), n_sub=12)
        assert float(jnp.max(jnp.abs(Pode - Play))) < 2e-3


def test_ode_differentiable():
    p = nufit_no()
    g = jax.grad(lambda E: probability_earth_continuous(p, E, -0.4)[0, 1])(
        jnp.asarray(5.0))
    assert not bool(jnp.isnan(g))


def test_diffrax_backend_matches_odeint():
    if not _HAS_DIFFRAX:  # diffrax is an optional dependency
        return
    p = nufit_no()
    cz = -0.4
    for E in (2.0, 5.0):
        Pode = probability_earth_continuous(p, E, cz, backend="odeint")
        Pdfx = probability_earth_continuous(p, E, cz, backend="diffrax")
        assert float(jnp.max(jnp.abs(Pode - Pdfx))) < 1e-5
        assert float(jnp.max(jnp.abs(Pdfx.sum(axis=-2) - 1.0))) < 1e-6


def test_diffrax_backend_differentiable():
    if not _HAS_DIFFRAX:
        return
    p = nufit_no()
    g = jax.grad(lambda E: probability_earth_continuous(
        p, E, -0.4, backend="diffrax")[0, 1])(jnp.asarray(5.0))
    assert not bool(jnp.isnan(g))
