"""Reproduction of nu-waves `vacuum_2flavors.jpg`: eV^2 sterile P_ee + smearing.

Demonstrates jaxnu's generic N-flavor core (here N=2) and Gaussian energy
smearing.  sin^2(2theta)=0.2, Delta m^2 = 1 eV^2, E = 3 MeV.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from _common import FIGDIR
from jaxnu import pmns, hamiltonian, propagator, constants as C

# 2-flavor mixing (electron <-> sterile)
sin2_2theta = 0.2
theta = 0.5 * np.arcsin(np.sqrt(sin2_2theta))
U = pmns.plane_rotation(2, 0, 1, jnp.asarray(theta))
msq = jnp.asarray([0.0, 1.0])  # Delta m^2 = 1 eV^2

E_fixed = 3e-3  # GeV (3 MeV)
L = np.linspace(1e-3, 20e-3, 400)  # km (1-20 m)


def Pee(E_GeV, L_km):
    H = hamiltonian.vacuum_hamiltonian(U, msq, E_GeV * C.GEV_TO_EV)
    S = propagator.propagator(H, L_km * C.KM_TO_INV_EV, backend="eigh")
    return jnp.abs(S[0, 0]) ** 2


# nominal
P = jax.vmap(lambda Lk: Pee(E_fixed, Lk))(jnp.asarray(L))

# Gaussian energy smearing sigma(E)/E = 10% (Gauss-Hermite quadrature)
sigma_rel = 0.10
nodes, weights = np.polynomial.hermite_e.hermegauss(64)  # exp(-x^2/2) weight
E_samples = E_fixed * (1.0 + sigma_rel * nodes)
w = weights / weights.sum()
P_grid = jax.vmap(lambda Lk: jax.vmap(lambda e: Pee(e, Lk))(jnp.asarray(E_samples)))(
    jnp.asarray(L))
P_smear = np.array(P_grid) @ w

plt.figure(figsize=(6.5, 4.0), dpi=150)
plt.plot(L * 1000, np.array(P), label=r"$P_{ee}$ disappearance", lw=2)
plt.plot(L * 1000, P_smear,
         label=r"$P_{ee}$ disappearance ($\sigma$(E)/E = 10.0%)", lw=2)
plt.plot(L * 1000, np.ones_like(L), "--", color="green",
         label="Total probability", lw=1.5)
plt.xlabel(r"$L_\nu$ [m]")
plt.ylabel("Probability")
plt.title(r"eV$^2$ sterile with $E_\nu$ = 3.0 MeV")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig(FIGDIR / "vacuum_2flavors.jpg", dpi=150)
print("saved", FIGDIR / "vacuum_2flavors.jpg")
