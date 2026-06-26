"""Demo of the NSI and sterile front-ends.

Left:  reactor antineutrino disappearance P_ee vs baseline for a 3+1 sterile
       (sin^2 2theta14 = 0.06, Delta m^2_41 = 1 eV^2) vs the 3-flavor case --
       reproduces the nu-waves `sterile_raa_plot` setup, with 5% energy smearing.
Right: DUNE-like nu_mu -> nu_e with and without matter NSI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

import jaxnu
from jaxnu import (OscParams, Sterile3plus1, NSI, probability_vacuum,
                   probability_constant, Flavor)

ang = dict(theta12=np.deg2rad(33.4), theta13=np.deg2rad(8.6),
           theta23=np.deg2rad(49.0))
DM21, DM31 = 7.42e-5, 7.42e-5 + 0.0024428
p3 = OscParams(deltacp=jnp.asarray(np.deg2rad(195.0)),
               dm21=jnp.asarray(DM21), dm31=jnp.asarray(DM31),
               **{k: jnp.asarray(v) for k, v in ang.items()})

# --- 3+1 sterile (RAA) -------------------------------------------------------
s2 = 0.06
th14 = np.arcsin(np.sqrt(s2)) / 2
st = Sterile3plus1(theta14=jnp.asarray(th14), theta24=jnp.asarray(0.0),
                   theta34=jnp.asarray(0.0), delta13=p3.deltacp,
                   delta24=jnp.asarray(0.0), dm21=jnp.asarray(DM21),
                   dm31=jnp.asarray(DM31), dm41=jnp.asarray(1.0),
                   **{k: jnp.asarray(v) for k, v in ang.items()})

E_fixed = 0.003  # 3 MeV
L = np.logspace(-2, 2, 400)

def pee(params, Lk, E):
    return probability_vacuum(params, jnp.asarray(E), Lk, anti=True,
                              flavor_in=Flavor.E, flavor_out=Flavor.E)

P_st = jax.vmap(lambda Lk: pee(st, Lk, E_fixed))(jnp.asarray(L))
P_3 = jax.vmap(lambda Lk: pee(p3, Lk, E_fixed))(jnp.asarray(L))

# 5% Gaussian energy smearing (Gauss-Hermite quadrature)
nodes, wts = np.polynomial.hermite_e.hermegauss(48)
Es = E_fixed * (1.0 + 0.05 * nodes)
w = wts / wts.sum()
P_st_sm = np.array(jax.vmap(
    lambda Lk: jax.vmap(lambda e: pee(st, Lk, e))(jnp.asarray(Es)))(jnp.asarray(L))) @ w

# --- NSI on DUNE -------------------------------------------------------------
E = np.linspace(0.2, 5.0, 400)
kw = dict(baseline_km=1300.0, density=2.8, ye=0.5,
          flavor_in=Flavor.MU, flavor_out=Flavor.E)
P_std = np.array(probability_constant(p3, E, **kw))
P_nsi = np.array(probability_constant(p3, E, nsi=NSI(eps_emu=0.15 + 0.1j), **kw))
P_nsi2 = np.array(probability_constant(p3, E, nsi=NSI(eps_etau=0.2), **kw))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), dpi=140)
ax1.semilogx(L, np.array(P_3), label="3-flavor", lw=2)
ax1.semilogx(L, np.array(P_st), label=r"3+1 ($\sin^2 2\theta_{14}=0.06$)", lw=2)
ax1.semilogx(L, P_st_sm, "--", label="3+1 ($\\sigma_E/E=5\\%$)", lw=2)
ax1.set_xlabel(r"$L$ [km]"); ax1.set_ylabel(r"$P(\bar\nu_e\to\bar\nu_e)$")
ax1.set_title(r"Reactor sterile (RAA), $E_\nu=3$ MeV, $\Delta m^2_{41}=1$ eV$^2$")
ax1.set_ylim(0.85, 1.005); ax1.legend(frameon=False)

ax2.plot(E, P_std, label="standard", lw=2)
ax2.plot(E, P_nsi, label=r"$\varepsilon_{e\mu}=0.15+0.1i$", lw=2)
ax2.plot(E, P_nsi2, label=r"$\varepsilon_{e\tau}=0.2$", lw=2)
ax2.set_xlabel(r"$E_\nu$ [GeV]"); ax2.set_ylabel(r"$P(\nu_\mu\to\nu_e)$")
ax2.set_title("DUNE-like (L=1300 km) with matter NSI")
ax2.set_ylim(0, 0.2); ax2.legend(frameon=False)

plt.tight_layout()
out = Path(__file__).resolve().parent / "nsi_sterile.jpg"
plt.savefig(out, dpi=140)
print("saved", out)
