"""Tests for jaxnu.stat (Fisher information utilities)."""

import numpy as np
import jax.numpy as jnp

import jaxnu
from jaxnu import nufit_no, probability_constant, stat, Flavor


def test_fisher_counting_toy_analytic():
    # mu(theta) = theta * t  ->  F = sum(t)/theta  ->  sigma = sqrt(theta/sum t)
    t = jnp.asarray([10.0, 20.0, 30.0])
    theta = {"norm": jnp.asarray(2.0)}
    F = stat.fisher_matrix(lambda p: p["norm"] * t, theta)
    expect = float(jnp.sqrt(2.0 / t.sum()))
    assert abs(F.sigma("norm") - expect) < 1e-10
    assert abs(F.sigma_fixed("norm") - expect) < 1e-10  # 1 param: same


def _model(p):
    # simple DUNE-like nu_e appearance counting model
    E = jnp.linspace(0.75, 6.0, 40)
    flux = 4e3 * E**2 * jnp.exp(-E / 1.0)
    Pme = probability_constant(p, E, 1300.0, density=2.8,
                               flavor_in=Flavor.MU, flavor_out=Flavor.E)
    Pmebar = probability_constant(p, E, 1300.0, density=2.8, anti=True,
                                  flavor_in=Flavor.MU, flavor_out=Flavor.E)
    return {"fhc": flux * Pme, "rhc": 0.5 * flux * Pmebar}


def test_fisher_equals_asimov_chi2_curvature():
    import dataclasses
    p0 = nufit_no()
    F = stat.fisher_matrix(_model, p0)
    # numeric curvature of the Asimov Poisson chi2 in deltacp
    data = _model(p0)
    def chi2(d):
        m = _model(dataclasses.replace(p0, deltacp=jnp.asarray(d)))
        return sum(float(stat.poisson_chi2(m[k], data[k])) for k in m)
    h = 1e-3
    d0 = float(p0.deltacp)
    curv = (chi2(d0 + h) - 2 * chi2(d0) + chi2(d0 - h)) / h**2  # = 2 F_dd
    F_dd = float(F.matrix[F._idx("deltacp"), F._idx("deltacp")])
    assert abs(curv / 2.0 - F_dd) < 2e-3 * F_dd


def test_fisher_prior_and_fixed():
    p0 = nufit_no()
    F = stat.fisher_matrix(_model, p0)
    s_marg = F.sigma("deltacp")
    s_fix = F.fixed("theta23", "theta13", "dm31", "dm21", "theta12").sigma("deltacp")
    assert s_fix <= s_marg  # marginalization can only inflate
    # a tight theta13 prior must shrink the marginalized deltacp error
    s_prior = F.with_prior("theta13", 1e-4).sigma("deltacp")
    assert s_prior < s_marg
    labels = F.labels
    assert any("deltacp" in l for l in labels) and len(labels) == 6
