"""Regression test against NuFast reference values (external cross-code check).

Reference numbers were produced by compiling and running Peter Denton's NuFast
codes (https://github.com/PeterDenton) with parameters matching
``jaxnu.nufit_no()``:

* **NuFast-LBL** (constant density), exact mode (``N_Newton = -1``),
  ``L = 1300 km``, ``rhoYe = 1.425``  -> ``probs[alpha][beta] = P(a->b)``.
* **NuFast-Earth** (PREM), converged ``PREM_NDiscontinuityLayer(400,400,400,400)``,
  detector depth 0, production height 0, ``Y_e = 0.466`` (core) / ``0.494`` (mantle).

See ``validation/README.md`` for the full methodology and the finding that
NuFast's coarse ``PREM_Full`` model (one path-averaged shell per region) converges
onto jaxnu once subdivided.

Tolerances absorb NuFast's 6-significant-figure hard-coded constants (hbar*c and,
for LBL, the matter potential); jaxnu uses first-principles CODATA values.
"""

import jax.numpy as jnp
import numpy as np

from jaxnu import nufit_no, probability_constant, probability_earth, earth

P = nufit_no()

# NuFast-LBL exact: (tag, a, b, E_GeV, prob), constant density rhoYe=1.425, L=1300.
_LBL = [
    ("MAT", 0, 0, 2.25, 0.867961131823),
    ("MAT", 1, 0, 2.25, 0.083513139694),
    ("MAT", 1, 1, 2.25, 0.059029569630),
    ("VAC", 0, 0, 2.25, 0.916362188556),
    ("VAC", 1, 0, 2.25, 0.054133022060),
    ("ANT", 1, 0, 2.25, 0.024668127012),
    ("MAT", 0, 0, 4.6875, 0.939243153001),
    ("MAT", 1, 0, 4.6875, 0.032319875090),
    ("MAT", 1, 1, 4.6875, 0.432578198769),
    ("VAC", 1, 1, 4.6875, 0.432093792452),
    ("ANT", 0, 0, 4.6875, 0.960129890155),
]


def test_nufast_lbl_constant_density():
    for tag, a, b, e, ref in _LBL:
        anti = tag == "ANT"
        density = 0.0 if tag == "VAC" else 2.85
        Pmat = probability_constant(P, jnp.asarray(e), 1300.0,
                                    density=density, ye=0.5, anti=anti)
        mine = float(Pmat[b, a])  # P[beta, alpha] = P(a -> b)
        assert abs(mine - ref) < 1e-4, (tag, a, b, e, mine, ref)


# NuFast-Earth converged PREM, cz = -1, P(e->e).
_EARTH_PEE = {2.0: 0.79995640, 5.0: 0.07961404}


def test_nufast_earth_prem_converged():
    # Align Y_e to NuFast-Earth's convention (0.466 core / 0.494 mantle).
    for e, ref in _EARTH_PEE.items():
        mine = float(probability_earth(P, jnp.asarray(e), jnp.asarray(-1.0),
                                       n_sub=80, ye_core=0.466,
                                       ye_mantle=0.494)[0, 0])
        assert abs(mine - ref) < 1e-4, (e, mine, ref)
