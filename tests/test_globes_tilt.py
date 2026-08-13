"""Spectral-tilt systematics in the GLoBES loader.

The tilt follows GLoBES manual v3.0.8 Eq. (11.26):

    s_i(a, b) = (1 + a) s_i + b s_i (E_i - Ebar) / (Emax - Emin),

with ``Ebar`` the midpoint of the *declared* energy range (``$emin``/``$emax``)
and the full width in the denominator -- so a tilt is not in general
rate-preserving. Each nuisance contributes a Gaussian pull ``(x/sigma)^2``,
signal and background independently.
"""

import os

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import mango.globes as G
from mango import nufit_no

GLB = os.path.join(os.path.dirname(__file__), "..", "analyses", "dune",
                   "globes", "DUNE_GLoBES.glb")

pytestmark = pytest.mark.skipif(not os.path.exists(GLB),
                                reason="DUNE GLoBES config not available")


@pytest.fixture(scope="module")
def exp():
    return G.load(GLB)


def test_tilt_basis_matches_globes_definition(exp):
    """(E_i - Ebar)/(Emax - Emin) over the declared range, midpoint pivot."""
    elo, ehi = float(exp.reco_edges[0]), float(exp.reco_edges[-1])
    want = (np.asarray(exp.reco_c) - 0.5 * (elo + ehi)) / (ehi - elo)
    assert np.max(np.abs(np.asarray(exp.tilt_basis) - want)) < 1e-12
    # pivot at the midpoint => basis spans about [-1/2, +1/2]
    assert -0.5 <= float(exp.tilt_basis.min()) < 0.0
    assert 0.0 < float(exp.tilt_basis.max()) <= 0.5


def test_tilt_none_reproduces_normalization_only(exp):
    """Default xi_tilt=None must leave the previous behaviour untouched."""
    p = nufit_no()
    n = len(exp.sys_labels)
    z = jnp.zeros(n)
    data = exp.spectra(p)
    assert abs(float(exp.chi2(p, z, data))
               - float(exp.chi2(p, z, data, xi_tilt=z))) < 1e-12


def test_zero_tilt_error_freezes_the_nuisance(exp):
    """A rule with no declared tilt error must be insensitive to xi_tilt."""
    p = nufit_no()
    n = len(exp.sys_labels)
    z = jnp.zeros(n)
    data = exp.spectra(p)
    saved = exp.sigma_tilt_vec
    try:
        exp.sigma_tilt_vec = jnp.zeros(n)
        base = float(exp.chi2(p, z, data, xi_tilt=z))
        huge = float(exp.chi2(p, z, data, xi_tilt=jnp.full(n, 1e3)))
        assert abs(huge - base) < 1e-9
    finally:
        exp.sigma_tilt_vec = saved


def test_pull_term_is_gaussian(exp):
    """With data equal to the tilted model, chi2 reduces to the pull alone."""
    p = nufit_no()
    n = len(exp.sys_labels)
    z = jnp.zeros(n)
    sigma, b = 0.05, 0.05
    saved = exp.sigma_tilt_vec
    try:
        exp.sigma_tilt_vec = jnp.full(n, sigma)
        lab = exp.sys_labels[3]
        j = exp.sys_index[lab]
        tb = np.asarray(exp.tilt_basis)
        xt = jnp.zeros(n).at[j].set(b)
        data_b = {r: sum(a * (1.0 + (b * tb if s == lab else 0.0))
                         for (s, _i, a) in exp.components(p)[r])
                  for r in exp.rules}
        assert abs(float(exp.chi2(p, z, data_b, xi_tilt=xt))
                   - (b / sigma) ** 2) < 1e-6
    finally:
        exp.sigma_tilt_vec = saved


def test_tilt_is_differentiable(exp):
    p = nufit_no()
    n = len(exp.sys_labels)
    z = jnp.zeros(n)
    data = exp.spectra(p)
    saved = exp.sigma_tilt_vec
    try:
        exp.sigma_tilt_vec = jnp.full(n, 0.05)
        g = np.asarray(jax.grad(
            lambda t: exp.chi2(p, z, data, xi_tilt=t))(jnp.full(n, 0.01)))
        assert np.all(np.isfinite(g))
        assert np.any(np.abs(g) > 0)
    finally:
        exp.sigma_tilt_vec = saved
