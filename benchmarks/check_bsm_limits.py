"""Measured BSM decoupling/limit checks behind the jaxnu paper's validation
table for the NSI and sterile front-ends (Table "Validation of the NSI and
sterile sectors through exact limits and gradient checks").

These front-ends have no established reference implementation to compare
against directly (unlike the standard 3-flavor core, which is cross-checked
against OscProb and NuFast -- see validation/README.md), so they are instead
validated through exact limits where the answer is known independently:
switching the extension off must reproduce the standard three-flavor result
through a different code path, the 3+1 vacuum survival probability must reach
the textbook depth at the mass-squared-splitting oscillation maximum,
probability must remain conserved (unitarity) in matter, and the gradients of
the new parameters must agree with central finite differences.

Standalone: only imports jaxnu / numpy / jax. Runs in well under a second.

Run from the repo root:
    JAX_PLATFORMS=cpu python benchmarks/check_bsm_limits.py
"""
import numpy as np
import jax
import jax.numpy as jnp
from jaxnu import (NSI, Sterile3plus1, nufit_no, probability_constant,
                   probability_vacuum, probability_earth, Flavor)

p = nufit_no()
E = jnp.linspace(0.5, 5.0, 16)


def st(**o):
    kw = dict(theta12=p.theta12, theta13=p.theta13, theta23=p.theta23,
              theta14=jnp.asarray(0.0), theta24=jnp.asarray(0.0),
              theta34=jnp.asarray(0.0), delta13=p.deltacp,
              delta24=jnp.asarray(0.0), dm21=p.dm21, dm31=p.dm31,
              dm41=jnp.asarray(1.0))
    kw.update({k: jnp.asarray(v) for k, v in o.items()})
    return Sterile3plus1(**kw)


# NSI, eps -> 0 reproduces standard oscillations.
P0 = probability_constant(p, E, 1300.0, density=2.8)
d1 = float(jnp.abs(P0 - probability_constant(p, E, 1300.0, density=2.8,
                                             nsi=NSI())).max())

# 3+1, theta_i4 -> 0 reproduces the three-flavor block.
P4 = np.array(probability_constant(st(), E, 1300.0, density=2.8))
d2 = float(np.abs(P4[:, :3, :3] - np.array(P0)).max())

# 3+1 vacuum P(nubar_e -> nubar_e) depth vs. the textbook 1 - sin^2(2 theta14)
# at the Delta m^2_41 oscillation maximum.
s2 = 0.1
th14 = 0.5 * np.arcsin(np.sqrt(s2))
Ee = 0.004
Pee = float(probability_vacuum(st(theta14=th14, dm41=1.0), jnp.asarray(Ee),
                               1.2369 * Ee, anti=True,
                               flavor_in=Flavor.E, flavor_out=Flavor.E))
d3 = abs(Pee - (1.0 - s2))

# 3+1 in PREM matter, unitarity.
Pm = probability_earth(st(theta14=0.15, theta24=0.1, dm41=0.5),
                       jnp.asarray(3.0), jnp.asarray(-1.0))
d4 = float(jnp.abs(Pm.sum(axis=-2) - 1.0).max())


# AD vs finite difference, d P / d theta14 (PREM, 3+1).
def f(x):
    return probability_earth(st(theta14=x, theta24=0.1, dm41=0.5),
                             jnp.asarray(3.0), jnp.asarray(-1.0),
                             flavor_in=Flavor.MU, flavor_out=Flavor.MU)


h = 1e-5
d5 = abs(float(jax.grad(f)(jnp.asarray(0.15)))
         - (float(f(0.15 + h)) - float(f(0.15 - h))) / (2 * h))


# AD vs finite difference, d P / d eps_ee (NSI, constant density).
def g(e):
    return probability_constant(p, jnp.asarray(2.0), 1300.0, density=2.8,
                                nsi=NSI(eps_ee=e), flavor_in=Flavor.MU,
                                flavor_out=Flavor.E)


d6 = abs(float(jax.grad(g)(jnp.asarray(0.05)))
         - (float(g(0.05 + h)) - float(g(0.05 - h))) / (2 * h))

print(f"NSI eps->0 recovers standard          : {d1:.2e}")
print(f"3+1 with theta_i4->0 recovers 3-flavor: {d2:.2e}")
print(f"3+1 RAA vacuum depth vs 1-sin^2(2t14) : {d3:.2e}")
print(f"3+1 in PREM matter, unitarity         : {d4:.2e}")
print(f"AD vs FD, d/dtheta14 (PREM, 3+1)      : {d5:.2e}")
print(f"AD vs FD, d/deps_ee (NSI, constant)   : {d6:.2e}")

# --- decoherence and non-unitary mixing (added with the v0.2.0 sectors) ------
from jaxnu import decoherence as dc, nonunitarity as nu
Egrid = jnp.linspace(0.5, 5.0, 12)
std = np.asarray(probability_constant(p, Egrid, 1300.0, density=2.8))
d7 = float(np.abs(np.asarray(dc.probability(p, Egrid, 1300.0, dc.Decoherence(),
                                            density=2.8)) - std).max())
d8 = float(np.abs(np.asarray(dc.probability(p, Egrid, 1300.0,
            dc.WavePacket(sigma_x_m=1e10), density=2.8)) - std).max())
d9 = float(np.abs(np.asarray(nu.probability(p, nu.NonUnitarity(), Egrid, 1300.0,
                                            density=2.8)) - std).max())
_big = dc.Decoherence(gamma21=1e-12, gamma31=1e-12, gamma32=1e-12)
d10 = float(np.abs(np.asarray(dc.probability(p, Egrid, 1300.0, _big,
                                             density=2.8)).sum(axis=-2) - 1.0).max())
print(f"Lindblad gamma->0 recovers standard   : {d7:.2e}")
print(f"wave packet sigma_x->inf recovers std : {d8:.2e}")
print(f"non-unitary alpha->0 recovers standard: {d9:.2e}")
print(f"strongly damped, unitarity preserved  : {d10:.2e}")
