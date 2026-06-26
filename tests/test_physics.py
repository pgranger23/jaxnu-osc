"""Physics correctness: vacuum limits, unitarity, MSW matter effects, Earth."""

import jax.numpy as jnp
import numpy as np

import jaxnu
from jaxnu import (
    OscParams,
    Flavor,
    nufit_no,
    probability_vacuum,
    probability_constant,
    probability_earth,
    earth,
)

R_E = jaxnu.constants.R_EARTH_KM


def _two_flavor_params(dm31=2.5e-3):
    # theta12, theta13 ~ 0 isolates the (2,3) sector; tiny non-zero values avoid
    # exact eigenvalue degeneracy.
    return OscParams(
        theta12=jnp.asarray(1e-7), theta13=jnp.asarray(0.0),
        theta23=jnp.asarray(np.pi / 4), deltacp=jnp.asarray(0.0),
        dm21=jnp.asarray(1e-12), dm31=jnp.asarray(dm31),
    )


def test_vacuum_two_flavor_limit():
    p = _two_flavor_params()
    L, E = 1000.0, 1.0
    Pmm = float(probability_vacuum(p, jnp.asarray(E), L,
                                   flavor_in=Flavor.MU, flavor_out=Flavor.MU))
    phase = 1.266932 * 2.5e-3 * L / E  # Delta m^2 L / 4E
    analytic = 1.0 - np.sin(phase) ** 2  # sin^2(2*pi/4)=1
    assert abs(Pmm - analytic) < 1e-6


def test_unitarity_vacuum_and_matter():
    p = nufit_no()
    for kind in ("vac", "mat"):
        E = jnp.linspace(0.5, 12.0, 20)
        if kind == "vac":
            P = probability_vacuum(p, E, 1300.0)
        else:
            P = probability_constant(p, E, 1300.0, density=2.85)
        assert float(jnp.max(jnp.abs(P.sum(axis=-2) - 1.0))) < 1e-10


def test_cp_asymmetry_sign_flip():
    # P(mu->e) for nu and nubar differ when delta_CP != 0, pi.
    p = nufit_no()
    E, L = jnp.asarray(0.8), 1300.0
    Pnu = float(probability_constant(p, E, L, density=2.85,
                                     flavor_in=Flavor.MU, flavor_out=Flavor.E))
    Pbar = float(probability_constant(p, E, L, density=2.85, anti=True,
                                      flavor_in=Flavor.MU, flavor_out=Flavor.E))
    assert abs(Pnu - Pbar) > 1e-3


def test_msw_matter_changes_probability():
    # Matter must change appearance relative to vacuum at GeV scale.
    p = nufit_no()
    E, L = jnp.asarray(2.0), 2000.0
    Pvac = float(probability_vacuum(p, E, L,
                                    flavor_in=Flavor.MU, flavor_out=Flavor.E))
    Pmat = float(probability_constant(p, E, L, density=4.5,
                                      flavor_in=Flavor.MU, flavor_out=Flavor.E))
    assert abs(Pvac - Pmat) > 1e-3


def test_earth_chord_geometry():
    tab = earth.shell_table(4)
    for cz in (-1.0, -0.7, -0.3, -0.05):
        _, _, L = earth.earth_segments(cz, tab)
        assert abs(float(jnp.sum(L)) - 2.0 * R_E * abs(cz)) < 1e-6
    # down-going: no Earth crossing
    _, _, L = earth.earth_segments(0.5, tab)
    assert float(jnp.sum(L)) == 0.0


def test_earth_unitarity_grid():
    p = nufit_no()
    E = jnp.linspace(1.0, 15.0, 12)
    cz = jnp.linspace(-1.0, -0.05, 10)
    P = probability_earth(p, E, cz)
    assert not bool(jnp.any(jnp.isnan(P)))
    assert float(jnp.max(jnp.abs(P.sum(axis=-2) - 1.0))) < 1e-10
