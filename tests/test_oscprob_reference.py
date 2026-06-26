"""Regression test against OscProb reference values (external cross-code check).

OscProb (https://github.com/joaoabcoelho/OscProb), the widely used ROOT/Eigen
oscillation library, was compiled and run with parameters matching
``jaxnu.nufit_no()``.  Note OscProb conventions: ``SetMix(th12, th23, th13, dcp)``
and ``SetDeltaMsqrs(dm21, dm32)`` with ``dm32 = dm31 - dm21``.

* **Constant density** (``PMNS_Fast``, ``L = 1300 km``, ``rho = 2.85``,
  ``Z/A = 0.5``): jaxnu and OscProb agree to ~1e-9 -- the two codes share the same
  first-principles constants (and both differ from NuFast-LBL only by its
  6-figure rounding).
* **PREM Earth, identical path**: OscProb's ``PremModel`` path for ``cosT = -1``
  (85 segments incl. atmosphere) is fed verbatim into ``probability_profile``;
  agreement ~1e-7.

See ``validation/README.md`` for methodology.
"""

import os

import jax.numpy as jnp
import numpy as np

from jaxnu import nufit_no, probability_constant, probability_profile

P = nufit_no()

# OscProb constant density: (tag, a, b, E, prob).  P(a -> b).
_CONST = [
    ("MAT", 0, 0, 2.25, 0.867940740138),
    ("MAT", 1, 0, 2.25, 0.083525377422),
    ("MAT", 1, 1, 2.25, 0.059027441877),
    ("MAT", 0, 0, 4.6875, 0.939240339267),
    ("MAT", 1, 0, 4.6875, 0.032321482760),
    ("VAC", 0, 0, 2.25, 0.916362196309),
    ("VAC", 1, 0, 2.25, 0.054133020290),
    ("ANT", 0, 0, 2.25, 0.955835660280),
    ("ANT", 1, 0, 2.25, 0.024660479977),
    ("ANT", 0, 0, 4.6875, 0.960134937831),
]


def test_oscprob_constant_density():
    for tag, a, b, e, ref in _CONST:
        anti = tag == "ANT"
        density = 0.0 if tag == "VAC" else 2.85
        Pm = probability_constant(P, jnp.asarray(e), 1300.0,
                                  density=density, ye=0.5, anti=anti)
        mine = float(Pm[b, a])
        assert abs(mine - ref) < 1e-6, (tag, a, b, e, mine, ref)


def test_oscprob_earth_identical_path():
    # Feed OscProb's exact PREM path (cosT=-1) into jaxnu's layer propagator.
    data = os.path.join(os.path.dirname(__file__), "data",
                        "oscprob_earth_cosT-1.txt")
    segs, prob = [], {}
    with open(data) as f:
        for line in f:
            w = line.split()
            if w[0] == "SEG":
                segs.append((float(w[1]), float(w[2]), float(w[3])))
            elif w[0] == "PROB":
                prob[float(w[1])] = (float(w[2]), float(w[3]), float(w[4]))
    L = np.array([s[0] for s in segs])
    rho = np.array([s[1] for s in segs])
    ye = np.array([s[2] for s in segs])
    for e, (ee, me, mm) in prob.items():
        Pm = np.array(probability_profile(P, jnp.asarray(e), rho, ye, L))
        assert abs(Pm[0, 0] - ee) < 1e-6
        assert abs(Pm[0, 1] - me) < 1e-6  # P(mu->e)
        assert abs(Pm[1, 1] - mm) < 1e-6
