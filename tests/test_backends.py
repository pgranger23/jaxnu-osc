"""All three propagator backends must agree to ~float64 precision."""

import jax.numpy as jnp

from jaxnu import nufit_no, probability_constant, probability_earth


def _max(a, b):
    return float(jnp.max(jnp.abs(a - b)))


def test_backends_agree_constant():
    p = nufit_no()
    E = jnp.linspace(0.3, 15.0, 25)
    Pc = probability_constant(p, E, 2000.0, density=4.5, backend="cayley")
    Pe = probability_constant(p, E, 2000.0, density=4.5, backend="eigh")
    Px = probability_constant(p, E, 2000.0, density=4.5, backend="expm")
    Pn = probability_constant(p, E, 2000.0, density=4.5, backend="nufast")
    assert _max(Pc, Pe) < 1e-10
    assert _max(Pc, Px) < 1e-10
    assert _max(Pc, Pn) < 1e-10  # NuFast analytic == matrix-exponential (exact)


def test_nufast_backend_vacuum_anti_and_grad():
    import jax
    p = nufit_no()
    E = jnp.linspace(0.2, 5.0, 20)
    # vacuum and antineutrino agree with cayley
    assert _max(probability_constant(p, E, 1300.0, backend="nufast"),
                probability_constant(p, E, 1300.0, backend="cayley")) < 1e-10
    assert _max(probability_constant(p, E, 1300.0, density=2.8, anti=True, backend="nufast"),
                probability_constant(p, E, 1300.0, density=2.8, anti=True, backend="cayley")) < 1e-10
    # differentiable
    from jaxnu import Flavor
    g = jax.grad(lambda pr: probability_constant(
        pr, jnp.asarray(2.0), 1300.0, density=2.8,
        flavor_in=Flavor.MU, flavor_out=Flavor.E).sum())(p)
    assert not bool(jnp.isnan(g.theta23))


def test_backends_agree_earth():
    p = nufit_no()
    E = jnp.linspace(1.0, 15.0, 10)
    cz = jnp.linspace(-1.0, -0.05, 8)
    Pc = probability_earth(p, E, cz, backend="cayley")
    Pe = probability_earth(p, E, cz, backend="eigh")
    assert _max(Pc, Pe) < 1e-9


def test_zero_length_segment_is_identity():
    # cayley must survive degenerate (zero) eigenvalues from empty shells.
    p = nufit_no()
    P = probability_earth(p, jnp.asarray(5.0), jnp.asarray(-0.5), backend="cayley")
    assert not bool(jnp.any(jnp.isnan(P)))
