"""Input validation and API-defect tests.

Every check added to the public API must satisfy the package's central jit/vmap
constraint: it must raise ``ValueError`` on bad *concrete* input, but silently
skip when the same argument is a JAX tracer (traced under ``jit``/``vmap``/
``grad``) -- checks cannot branch on the *value* of a traced array. Each
validation below is therefore tested twice: once eagerly (must raise) and once
under ``jax.jit``/``jax.vmap`` with the same bad value smuggled in as a traced
argument (must NOT raise -- the check is a no-op while tracing, exactly as
documented).
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mango import (
    OscParams,
    NSI,
    NFlavorParams,
    Flavor,
    nufit_no,
    probability_vacuum,
    probability_constant,
    probability_profile,
    probability_earth,
    select,
    solar,
)
from mango.sterile import pmns_3plus1
from mango.oscillator import _is_concrete


# --- _is_concrete helper -------------------------------------------------

def test_is_concrete_true_for_plain_values():
    assert _is_concrete(1.0)
    assert _is_concrete(jnp.asarray(1.0))
    assert _is_concrete(np.asarray(1.0))


def test_is_concrete_false_under_jit_and_vmap_and_grad():
    seen = {}

    def f(x):
        seen["jit"] = _is_concrete(x)
        return x

    jax.jit(f)(jnp.asarray(1.0))
    assert seen["jit"] is False

    def g(x):
        seen["vmap"] = _is_concrete(x)
        return x

    jax.vmap(g)(jnp.asarray([1.0, 2.0]))
    assert seen["vmap"] is False

    def h(x):
        seen["grad"] = _is_concrete(x)
        return (x ** 2).sum()

    jax.grad(h)(jnp.asarray(1.0))
    assert seen["grad"] is False


# --- energy_GeV > 0 --------------------------------------------------------

@pytest.mark.parametrize("call", [
    lambda p, E: probability_vacuum(p, E, 1300.0),
    lambda p, E: probability_constant(p, E, 1300.0, density=2.8),
    lambda p, E: probability_profile(p, E, density_gcc=[2.8], ye=[0.5], length_km=[1300.0]),
    lambda p, E: probability_earth(p, E, jnp.asarray(-0.5)),
])
def test_negative_or_zero_energy_raises(call):
    p = nufit_no()
    with pytest.raises(ValueError, match="GeV"):
        call(p, jnp.asarray(-1.0))
    with pytest.raises(ValueError, match="GeV"):
        call(p, jnp.asarray(0.0))


def test_negative_energy_error_mentions_correct_units():
    p = nufit_no()
    with pytest.raises(ValueError) as exc:
        probability_vacuum(p, jnp.asarray(-1.0), 1300.0)
    assert "GeV" in str(exc.value) and "eV" in str(exc.value)


@pytest.mark.parametrize("call", [
    lambda p, E: probability_vacuum(p, E, 1300.0),
    lambda p, E: probability_constant(p, E, 1300.0, density=2.8),
    lambda p, E: probability_earth(p, E, jnp.asarray(-0.5)),
])
def test_energy_check_skips_under_jit_and_still_computes(call):
    # The most important test in the file: tracing must not be broken by the
    # check, even when the traced value would be rejected eagerly.
    p = nufit_no()
    f = jax.jit(lambda E: call(p, E))
    out = f(jnp.asarray(-1.0))  # bad value, but traced -> check is skipped
    assert out.shape[-2:] == (3, 3)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_energy_check_survives_vmap():
    p = nufit_no()
    f = jax.vmap(lambda E: probability_constant(p, E, 1300.0, density=2.8))
    out = f(jnp.linspace(0.5, 5.0, 6))
    assert out.shape == (6, 3, 3)


# --- baseline_km / length_km >= 0 ------------------------------------------

def test_negative_baseline_raises():
    p = nufit_no()
    with pytest.raises(ValueError, match="km"):
        probability_vacuum(p, jnp.asarray(2.0), -100.0)


def test_negative_baseline_skipped_under_jit():
    p = nufit_no()
    f = jax.jit(lambda L: probability_vacuum(p, jnp.asarray(2.0), L))
    out = f(jnp.asarray(-100.0))
    assert out.shape == (3, 3)


def test_negative_length_km_in_profile_raises():
    p = nufit_no()
    with pytest.raises(ValueError, match="km"):
        probability_profile(p, jnp.asarray(2.0), density_gcc=[2.0, 1.0],
                            ye=[0.5, 0.5], length_km=[100.0, -200.0])


def test_negative_length_km_in_profile_skipped_under_jit():
    p = nufit_no()
    f = jax.jit(lambda L: probability_profile(p, jnp.asarray(2.0),
                                              density_gcc=jnp.array([2.0, 1.0]),
                                              ye=jnp.array([0.5, 0.5]), length_km=L))
    out = f(jnp.array([100.0, -200.0]))
    assert out.shape == (3, 3)


# --- density / density_gcc >= 0 ---------------------------------------------

def test_negative_density_raises():
    p = nufit_no()
    with pytest.raises(ValueError, match="g/cm\\^3"):
        probability_constant(p, jnp.asarray(2.0), 1300.0, density=-1.0)


def test_negative_density_skipped_under_jit():
    p = nufit_no()
    f = jax.jit(lambda rho: probability_constant(p, jnp.asarray(2.0), 1300.0, density=rho))
    out = f(jnp.asarray(-1.0))
    assert out.shape == (3, 3)


def test_negative_density_gcc_in_profile_raises():
    p = nufit_no()
    with pytest.raises(ValueError, match="g/cm\\^3"):
        probability_profile(p, jnp.asarray(2.0), density_gcc=[2.0, -1.0],
                            ye=[0.5, 0.5], length_km=[100.0, 200.0])


# --- det_depth_km >= 0 -------------------------------------------------------

def test_negative_det_depth_raises():
    p = nufit_no()
    with pytest.raises(ValueError, match="km"):
        probability_earth(p, jnp.asarray(2.0), jnp.asarray(-0.5), det_depth_km=-5.0)


def test_negative_det_depth_skipped_under_jit():
    p = nufit_no()
    f = jax.jit(lambda d: probability_earth(p, jnp.asarray(2.0), jnp.asarray(-0.5), det_depth_km=d))
    out = f(jnp.asarray(-5.0))
    assert out.shape == (3, 3)


# --- |cos_zenith| <= 1 --------------------------------------------------------

@pytest.mark.parametrize("bad_cz", [1.5, -1.5])
def test_out_of_range_cos_zenith_raises(bad_cz):
    p = nufit_no()
    # From the bug report: this used to silently return the identity matrix.
    with pytest.raises(ValueError, match="cos_zenith"):
        probability_earth(p, jnp.asarray(5.0), jnp.asarray(bad_cz))


def test_cos_zenith_check_skipped_under_jit():
    p = nufit_no()
    f = jax.jit(lambda cz: probability_earth(p, jnp.asarray(5.0), cz))
    out = f(jnp.asarray(1.5))  # traced -> no error, even though unphysical
    assert out.shape == (3, 3)


def test_cos_zenith_check_survives_vmap():
    p = nufit_no()
    cz = jnp.linspace(-1.0, -0.05, 5)
    f = jax.vmap(lambda c: probability_earth(p, jnp.asarray(3.0), c))
    out = f(cz)
    assert out.shape == (5, 3, 3)
    assert float(jnp.max(jnp.abs(out.sum(axis=-2) - 1.0))) < 1e-8


def test_valid_cos_zenith_still_works():
    p = nufit_no()
    out = probability_earth(p, jnp.asarray(5.0), jnp.asarray(-0.5))
    assert out.shape == (3, 3)


# --- flavor index range (select + entry points) ------------------------------

def test_select_rejects_out_of_range_flavor_instead_of_clamping():
    p = nufit_no()
    P = probability_constant(p, jnp.asarray(2.0), 1300.0, density=2.8)
    # From the bug report: flavor_out=100 used to silently clamp to P[...,2,1].
    with pytest.raises(ValueError, match="flavor_out"):
        select(P, flavor_in=1, flavor_out=100)
    with pytest.raises(ValueError, match="flavor_in"):
        select(P, flavor_in=-1, flavor_out=0)


def test_select_valid_index_matches_manual_indexing():
    p = nufit_no()
    P = probability_constant(p, jnp.asarray(2.0), 1300.0, density=2.8)
    assert float(select(P, Flavor.MU, Flavor.E)) == pytest.approx(float(P[..., 0, 1]))


def test_probability_constant_flavor_index_out_of_range_raises():
    p = nufit_no()  # 3-flavor
    with pytest.raises(ValueError, match="flavor_out"):
        probability_constant(p, jnp.asarray(2.0), 1300.0, density=2.8,
                             flavor_in=0, flavor_out=5)


def test_flavor_index_out_of_range_raises_even_when_tracing():
    # flavor indices are always static Python ints (int() is called on them),
    # so this check is never gated on concreteness and fires even while a
    # surrounding jit trace is being built.
    p = nufit_no()
    with pytest.raises(ValueError, match="flavor_out"):
        jax.jit(lambda E: probability_constant(p, E, 1300.0, density=2.8,
                                               flavor_in=0, flavor_out=5))(jnp.asarray(2.0))


# --- NSI Hermiticity ----------------------------------------------------------

def test_raw_nsi_matrix_non_hermitian_raises():
    p = nufit_no()
    m = jnp.zeros((3, 3), dtype=jnp.complex128)
    m = m.at[0, 1].set(0.3 + 0.2j)  # eps_mue left at 0 -> not Hermitian
    with pytest.raises(ValueError, match="Hermitian"):
        probability_constant(p, jnp.asarray(2.0), 1300.0, density=2.8,
                             nsi=m, backend="cayley")


def test_raw_nsi_matrix_hermitian_is_accepted_and_unitary():
    p = nufit_no()
    m = jnp.zeros((3, 3), dtype=jnp.complex128)
    m = m.at[0, 1].set(0.3 + 0.2j)
    m = m.at[1, 0].set(jnp.conj(m[0, 1]))
    P = probability_constant(p, jnp.asarray(2.0), 1300.0, density=2.8,
                             nsi=m, backend="cayley")
    assert float(jnp.max(jnp.abs(P.sum(axis=-2) - 1.0))) < 1e-8


def test_nsi_dataclass_path_unaffected_by_hermiticity_check():
    p = nufit_no()
    P = probability_constant(p, jnp.asarray(2.0), 1300.0, density=2.8,
                             nsi=NSI(eps_emu=0.3 + 0.2j, eps_ee=0.05))
    assert float(jnp.max(jnp.abs(P.sum(axis=-2) - 1.0))) < 1e-8


def test_raw_nsi_hermiticity_check_skipped_under_jit():
    p = nufit_no()
    m = jnp.zeros((3, 3), dtype=jnp.complex128)
    m = m.at[0, 1].set(0.3 + 0.2j)  # non-Hermitian

    f = jax.jit(lambda mat: probability_constant(p, jnp.asarray(2.0), 1300.0,
                                                 density=2.8, nsi=mat, backend="cayley"))
    out = f(m)  # traced -> check is skipped, no exception
    assert out.shape == (3, 3)


# --- solar.adiabatic_mass_fractions accepts a scalar r_km -------------------

def test_adiabatic_mass_fractions_scalar_matches_one_element_array():
    p = nufit_no()
    prof = solar.exponential_profile()
    r_scalar = 0.5 * prof.R_sun_km
    r_emit = 0.05 * prof.R_sun_km

    F_scalar = solar.adiabatic_mass_fractions(p, 0.008, prof, r_scalar, r_emit, alpha=0)
    F_array = solar.adiabatic_mass_fractions(p, 0.008, prof, jnp.array([r_scalar]),
                                             r_emit, alpha=0)

    assert F_scalar.shape == (3,)
    assert F_array.shape == (1, 3)
    assert np.allclose(np.array(F_scalar), np.array(F_array[0]), atol=1e-10)
    assert float(jnp.sum(F_scalar)) == pytest.approx(1.0, abs=1e-9)


def test_adiabatic_mass_fractions_array_behaviour_unchanged():
    p = nufit_no()
    prof = solar.exponential_profile()
    r = np.geomspace(0.05 * prof.R_sun_km, prof.R_sun_km, 20)
    F = solar.adiabatic_mass_fractions(p, 0.008, prof, r, 0.05 * prof.R_sun_km, alpha=0)
    assert F.shape == (20, 3)
    assert np.allclose(np.array(F).sum(axis=1), 1.0, atol=1e-9)


# --- pmns_3plus1 delta14 ------------------------------------------------------

_ANGLES = dict(theta12=0.5836, theta13=0.1495, theta23=0.8587,
              theta14=0.2, theta24=0.15, theta34=0.05)


def test_pmns_3plus1_delta14_default_matches_no_delta14_call():
    U_no_arg = pmns_3plus1(**_ANGLES, delta13=0.3, delta24=0.4)
    U_explicit_zero = pmns_3plus1(**_ANGLES, delta13=0.3, delta14=0.0, delta24=0.4)
    assert np.allclose(np.array(U_no_arg), np.array(U_explicit_zero))


def test_pmns_3plus1_nonzero_delta14_changes_appearance_probability_and_unitary():
    U0 = pmns_3plus1(**_ANGLES, delta13=0.0, delta14=0.0, delta24=0.0)
    U1 = pmns_3plus1(**_ANGLES, delta13=0.0, delta14=1.3, delta24=0.0)

    for U in (U0, U1):
        gram = np.array(jnp.conj(U).T @ U)
        assert np.allclose(gram, np.eye(4), atol=1e-10)

    msq = jnp.array([0.0, 7.42e-5, 2.515e-3, 1.0])
    params0 = NFlavorParams(U=U0, msq=msq, n_active=3)
    params1 = NFlavorParams(U=U1, msq=msq, n_active=3)

    P0 = probability_vacuum(params0, jnp.asarray(1.0), 1000.0,
                            flavor_in=Flavor.E, flavor_out=Flavor.MU)
    P1 = probability_vacuum(params1, jnp.asarray(1.0), 1000.0,
                            flavor_in=Flavor.E, flavor_out=Flavor.MU)
    assert abs(float(P0) - float(P1)) > 1e-6
