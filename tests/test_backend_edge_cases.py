"""Backend behaviour at the degenerate / null-parameter corners.

These are the points where an oscillation code is most likely to return NaN or
a silently wrong gradient: exactly degenerate mass eigenstates, a vanishing
mixing angle, the inverted ordering, and the horizon.  Every one of them is a
physically reasonable thing to ask for, and each was broken at some point.
"""

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mango import (nufit_no, probability_constant, probability_vacuum,
                   probability_earth)

BACKENDS = ("nufast", "cayley", "eigh", "expm")


def _p(**over):
    p = nufit_no()
    return dataclasses.replace(p, **{k: jnp.asarray(v) for k, v in over.items()})


# --- dm21 = 0 : the two lower matter eigenstates are exactly degenerate -------

def test_dm21_zero_all_backends_finite_and_consistent():
    """The Rosetta relations have |U^m_2|^2 ~ 1/Dl21, which diverges here; the
    probability stays finite only because it always appears times sin^2(Dl21 ...).
    The nufast backend used to return NaN in the forward pass."""
    p = _p(dm21=0.0)
    ref = None
    for bk in BACKENDS:
        pr = np.asarray(probability_constant(p, 2.0, 1300.0, density=2.8,
                                             backend=bk))
        assert np.all(np.isfinite(pr)), f"{bk} produced non-finite values"
        assert np.allclose(pr.sum(axis=-2), 1.0, atol=1e-10)
        if ref is None:
            ref = pr
        else:
            assert np.max(np.abs(pr - ref)) < 1e-10, f"{bk} disagrees at dm21=0"


def test_dm21_zero_limit_is_continuous():
    """Approaching degeneracy must not degrade: the floored Dl21 has to give the
    same answer as the general propagator all the way down to dm21 = 0."""
    for d21 in (1e-5, 1e-7, 1e-9, 1e-12, 0.0):
        p = _p(dm21=d21)
        a = float(probability_constant(p, 2.0, 1300.0, density=2.8)[1, 1])
        b = float(probability_constant(p, 2.0, 1300.0, density=2.8,
                                       backend="cayley")[1, 1])
        assert abs(a - b) < 1e-12, f"dm21={d21}: nufast {a} vs cayley {b}"


def test_dm21_zero_gradient_finite():
    f = lambda x: probability_constant(_p(dm21=x), 2.0, 1300.0,
                                       density=2.8)[1, 1]
    g = float(jax.grad(f)(jnp.asarray(0.0)))
    assert np.isfinite(g)


# --- theta13 = 0 : a standard physics null point ------------------------------

def test_theta13_zero_gradient_finite_and_matches_general_backend():
    """Jrr = sqrt(Um2sq_t * Ut2sq_t) with Ut2sq_t ~ theta13^2 made d/dtheta13
    NaN at zero.  The value was always fine; only the gradient broke."""
    for ch, idx in (("ee", (0, 0)), ("mue", (0, 1))):
        f_nf = lambda t: probability_constant(_p(theta13=t), 2.0, 1300.0,
                                              density=2.8)[idx]
        f_cy = lambda t: probability_constant(_p(theta13=t), 2.0, 1300.0,
                                              density=2.8,
                                              backend="cayley")[idx]
        g_nf = float(jax.grad(f_nf)(jnp.asarray(0.0)))
        g_cy = float(jax.grad(f_cy)(jnp.asarray(0.0)))
        assert np.isfinite(g_nf), f"{ch}: nufast gradient is NaN at theta13=0"
        # the two backends share no code; agreement pins the one-sided limit
        assert abs(g_nf - g_cy) < 1e-9, f"{ch}: {g_nf} vs {g_cy}"


def test_theta13_zero_forward_matches_all_backends():
    p = _p(theta13=0.0)
    vals = [np.asarray(probability_constant(p, 2.0, 1300.0, density=2.8,
                                            backend=bk)) for bk in BACKENDS]
    for v in vals[1:]:
        assert np.max(np.abs(v - vals[0])) < 1e-12


# --- inverted ordering: previously untested anywhere in the suite -------------

@pytest.mark.parametrize("bk", BACKENDS)
def test_inverted_ordering_unitary_and_consistent(bk):
    """Every pre-existing test used nufit_no(), i.e. normal ordering only.  A
    sign error specific to dm31 < 0 would have gone undetected."""
    p = _p(dm31=-2.498e-3)
    for anti in (False, True):
        pr = np.asarray(probability_constant(p, 2.0, 1300.0, density=2.8,
                                             backend=bk, anti=anti))
        assert np.all(np.isfinite(pr))
        assert np.allclose(pr.sum(axis=-2), 1.0, atol=1e-10)


def test_inverted_ordering_cross_backend_agreement():
    p = _p(dm31=-2.498e-3)
    E = jnp.linspace(0.5, 10.0, 12)
    ref = np.asarray(probability_constant(p, E, 1300.0, density=2.8,
                                          backend="cayley"))
    for bk in ("nufast", "eigh", "expm"):
        pr = np.asarray(probability_constant(p, E, 1300.0, density=2.8,
                                             backend=bk))
        assert np.max(np.abs(pr - ref)) < 1e-10, f"{bk} disagrees for IO"


def test_inverted_ordering_earth_matter_resonance_channel():
    """In matter the MSW resonance sits in the nu channel for NO and the nubar
    channel for IO; check the asymmetry actually flips with the ordering."""
    E = jnp.linspace(2.0, 12.0, 40)
    cz = jnp.full(40, -0.9)

    def asym(p):
        nu = np.asarray(probability_earth(p, E, cz, anti=False))[..., 0, 1]
        nb = np.asarray(probability_earth(p, E, cz, anti=True))[..., 0, 1]
        return float(np.max(nu) - np.max(nb))

    assert asym(_p()) > 0.0                     # normal ordering: nu enhanced
    assert asym(_p(dm31=-2.498e-3)) < 0.0       # inverted: nubar enhanced


# --- degenerate spectra in the propagator ------------------------------------

def test_exact_degeneracy_cayley_matches_eigh():
    """_DEGEN_EPS is deliberately left where the ordinary divided-difference
    branch runs at exact degeneracy; pin the resulting accuracy."""
    p = _p(dm21=0.0)
    a = np.asarray(probability_vacuum(p, 3.0, 1000.0, backend="cayley"))
    b = np.asarray(probability_vacuum(p, 3.0, 1000.0, backend="eigh"))
    assert np.max(np.abs(a - b)) < 1e-11


# --- the horizon and the poles -----------------------------------------------

@pytest.mark.parametrize("cz", [0.0, -1e-12, -0.05, -1.0])
def test_earth_near_horizon_and_poles_finite(cz):
    pr = np.asarray(probability_earth(nufit_no(), 3.0, jnp.asarray(cz)))
    assert np.all(np.isfinite(pr))
    assert np.allclose(pr.sum(axis=-2), 1.0, atol=1e-10)


def test_baseline_gradient_agrees_in_both_ad_modes():
    """jax.grad disagreeing with jacfwd is the signature of a where-NaN trap:
    the outer where hides a divergent sqrt'(0) forwards but reverse mode still
    pulls the NaN back through the unselected branch."""
    from mango import earth as E

    L = lambda cz: E.baseline_km(cz, det_depth_km=1.9)
    for cz in (-1.0, -0.5, 0.0, 1.0):
        gr = float(jax.grad(L)(jnp.asarray(cz)))
        gf = float(jax.jacfwd(L)(jnp.asarray(cz)))
        assert np.isfinite(gr), f"reverse-mode gradient is NaN at cz={cz}"
        assert np.isfinite(gf)
        assert abs(gr - gf) < 1e-9, f"cz={cz}: grad {gr} vs jacfwd {gf}"
