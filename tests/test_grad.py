"""Differentiability: autodiff must match central finite differences."""

import dataclasses

import jax
import jax.numpy as jnp

from jaxnu import nufit_no, probability_earth, probability_constant, Flavor


def _fd_param(fn, p, field, h):
    pp = dataclasses.replace(p, **{field: getattr(p, field) + h})
    pm = dataclasses.replace(p, **{field: getattr(p, field) - h})
    return (float(fn(pp)) - float(fn(pm))) / (2.0 * h)


def test_grad_params_constant():
    p = nufit_no()
    fn = lambda pr: probability_constant(
        pr, jnp.asarray(2.0), 1300.0, density=2.85,
        flavor_in=Flavor.MU, flavor_out=Flavor.E)
    g = jax.grad(fn)(p)
    for field, h in [("theta23", 1e-5), ("theta13", 1e-5),
                     ("deltacp", 1e-5), ("dm31", 1e-7)]:
        ad = float(getattr(g, field))
        fd = _fd_param(fn, p, field, h)
        assert abs(ad - fd) < 1e-3 * (1.0 + abs(fd)), (field, ad, fd)


def test_grad_params_earth():
    p = nufit_no()
    fn = lambda pr: probability_earth(
        pr, jnp.asarray(4.0), jnp.asarray(-1.0),
        flavor_in=Flavor.MU, flavor_out=Flavor.E)
    g = jax.grad(fn)(p)
    for field, h in [("theta23", 1e-5), ("dm31", 1e-7)]:
        ad = float(getattr(g, field))
        fd = _fd_param(fn, p, field, h)
        assert abs(ad - fd) < 1e-2 * (1.0 + abs(fd)), (field, ad, fd)
    # no NaNs anywhere in the gradient tree
    leaves = jax.tree_util.tree_leaves(g)
    assert not any(bool(jnp.any(jnp.isnan(x))) for x in leaves)


def test_grad_cos_zenith():
    p = nufit_no()
    f = lambda cz: probability_earth(
        p, jnp.asarray(4.0), cz, flavor_in=Flavor.MU, flavor_out=Flavor.E)
    for cz0 in (-0.9, -0.6, -0.3, -0.15):
        ad = float(jax.grad(f)(jnp.asarray(cz0)))
        h = 1e-5
        fd = (float(f(jnp.asarray(cz0 + h))) - float(f(jnp.asarray(cz0 - h)))) / (2 * h)
        assert abs(ad - fd) < 1e-3 * (1.0 + abs(fd)), (cz0, ad, fd)


def test_jit_compiles():
    p = nufit_no()
    f = jax.jit(lambda pr, E, cz: probability_earth(pr, E, cz))
    out = f(p, jnp.linspace(1.0, 15.0, 8), jnp.linspace(-1.0, -0.05, 6))
    assert out.shape == (6, 8, 3, 3)
