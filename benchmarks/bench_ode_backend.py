"""Characterization of the continuous (ODE) backend: what it is for, measured.

Not a paper benchmark -- no number here appears in the accompanying paper,
which does not exercise this backend. It exists because the backend is shipped
and was otherwise unquantified, and because it documents a usability trap
(see (b) below) that is easy to walk into.

The layered path of :mod:`jaxnu.layers` requires the matter profile to be
piecewise constant. That is exact for PREM, which *is* a shell model, but a
density given as a smooth function of position -- a supernova profile, a
fitted/continuous Earth model, an analytic solar profile -- has to be
discretized first, and the discretization error is then set by how many layers
you are willing to pay for. The ODE backend integrates dS/ds = -i H(s) S
through the profile as given, with adaptive step control, so the error is set
by a solver tolerance instead of by a layer count.

Three measurements:

(a) THE MOTIVATION. On a genuinely smooth profile (a Gaussian density bump
    along the chord, no discontinuities anywhere), how many constant-density
    layers does the layered path need to reproduce the ODE answer? Second-order
    convergence is expected -- each doubling of the layer count should gain a
    factor ~4 -- so the layer count needed for a target accuracy is the honest
    statement of what the ODE route saves.

(b) THE CROSS-CHECK. Through the *PREM* Earth, where the layered path is the
    production route and is exact by construction, the two must agree. They do,
    but only once both are given the same electron fraction: ``ode.py``'s Earth
    helper takes Y_e constant (documented there as a smooth-profile
    demonstration) while the layered path uses the per-region value, and a
    core-crossing chord otherwise compares two different physics setups rather
    than two algorithms. Both comparisons are reported below so the difference
    between "the solver disagrees" and "the conventions differ" is visible.

(c) THE COST. The layered propagator is a product of exact unitary
    exponentials, so unitarity holds to machine precision. A Runge-Kutta step
    does not preserve the unitary group, so the ODE result drifts. That drift
    is the reason the layered route stays the production path.

Standalone: imports only jaxnu / numpy / jax. Writes raw numbers to
./benchmarks/output/jaxnu_bench_ode_backend.json.

Run from the repository root:
    JAX_PLATFORMS=cpu python benchmarks/bench_ode_backend.py
Takes well under a minute on CPU.
"""
import json
import os

import numpy as np
import jax
import jax.numpy as jnp

from jaxnu import nufit_no, probability_earth
from jaxnu import constants as C
from jaxnu.ode import probability_earth_continuous, propagate_continuous
import jaxnu.oscillator as _osc
import jaxnu.earth as earth

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTDIR, exist_ok=True)

P = nufit_no()
YE_ODE = 0.4957          # the constant Y_e ode.earth_potential_fn defaults to
results = {}

# ---------------------------------------------------------------------------
# (a) the motivation: a smooth profile the layer method must discretize
# ---------------------------------------------------------------------------
print("=" * 74)
print("(a) smooth profile: layers needed to reach the adaptive-ODE answer")
print("=" * 74)

E_TEST, CZ_TEST = 5.0, -0.9
L_KM = 2 * C.R_EARTH_KM * abs(CZ_TEST)
TOTAL = L_KM * C.KM_TO_INV_EV


def rho_smooth(s_km):
    """Smooth density [g/cm^3] along the chord: a Gaussian bump, C-infinity.

    Deliberately NOT PREM-like: no shell boundaries, so the layered path has
    nothing to key its segmentation to and pure discretization error is what
    is being measured.
    """
    x = 2.0 * s_km / L_KM - 1.0
    return 3.0 + 9.0 * jnp.exp(-4.0 * x ** 2)


def v_of_s(s_invEV):
    rho = rho_smooth(s_invEV / C.KM_TO_INV_EV)
    v_cc, _ = C.matter_potentials(rho, jnp.asarray(0.5))
    return v_cc


u, msq = P.pmns(), P.msquared()
E_eV = E_TEST * C.GEV_TO_EV
# tight tolerances: this is the reference the layer counts are measured against
S_ref = propagate_continuous(u, msq, E_eV, v_of_s, TOTAL, False,
                             rtol=1e-10, atol=1e-12)
P_ref = np.abs(np.asarray(S_ref)) ** 2

print(f"  Gaussian-bump profile, E={E_TEST} GeV, chord L={L_KM:.0f} km")
print(f"  {'n_layers':>9s} {'max|P_lay - P_ODE|':>22s} {'ratio to previous':>19s}")
conv = []
prev = None
for n in (5, 10, 20, 50, 100, 200, 500, 1000):
    edges = jnp.linspace(0.0, L_KM, n + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    v_cc, _ = C.matter_potentials(rho_smooth(mid), jnp.full(n, 0.5))
    Ls = jnp.full(n, L_KM / n) * C.KM_TO_INV_EV
    S = _osc.propagate_layers(u, msq, E_eV, v_cc, Ls, anti=False,
                              backend="cayley")
    d = float(np.abs(np.abs(np.asarray(S)) ** 2 - P_ref).max())
    r = "" if prev is None else f"{prev / d:19.2f}"
    print(f"  {n:9d} {d:22.2e} {r:>19s}")
    conv.append(dict(n_layers=n, max_abs_dev=d))
    prev = d
results["smooth_profile_convergence"] = conv
results["smooth_profile_config"] = dict(energy_GeV=E_TEST, cz=CZ_TEST,
                                        L_km=float(L_KM), rtol=1e-10, atol=1e-12)

# ---------------------------------------------------------------------------
# (b) the cross-check, through PREM, with and without matched Y_e conventions
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("(b) PREM cross-check: ODE vs layered, default vs matched Y_e")
print("=" * 74)
print(f"  {'cz':>6s} {'E [GeV]':>8s} {'default Y_e':>14s} {'matched Y_e':>14s}")
xcheck = []
for cz in (-0.4, -0.9, -0.99):
    for E in (2.0, 5.0, 10.0):
        Pode = np.asarray(probability_earth_continuous(P, E, cz, ye=YE_ODE))
        Pdef = np.asarray(probability_earth(P, jnp.asarray(E), jnp.asarray(cz),
                                            n_sub=12))
        rho, ye, L = earth.chord_segments(jnp.asarray(cz), earth.shell_table(12),
                                          h_atm_km=0.0, det_depth_km=0.0)
        v_cc, _ = C.matter_potentials(rho, jnp.full_like(ye, YE_ODE))
        S = _osc.propagate_layers(u, msq, E * C.GEV_TO_EV, v_cc,
                                  L * C.KM_TO_INV_EV, anti=False,
                                  backend="cayley")
        Pmatch = np.abs(np.asarray(S)) ** 2
        d_def = float(np.abs(Pode - Pdef).max())
        d_mat = float(np.abs(Pode - Pmatch).max())
        print(f"  {cz:6.2f} {E:8.1f} {d_def:14.2e} {d_mat:14.2e}")
        xcheck.append(dict(cz=cz, energy_GeV=E, dev_default_ye=d_def,
                           dev_matched_ye=d_mat))
results["prem_cross_check"] = xcheck
worst_matched = max(r["dev_matched_ye"] for r in xcheck)
worst_default = max(r["dev_default_ye"] for r in xcheck)
print(f"\n  worst with matched Y_e : {worst_matched:.1e}")
print(f"  worst with default Y_e : {worst_default:.1e}"
      f"   <- convention difference, not solver error")
results["worst_matched_ye"] = worst_matched
results["worst_default_ye"] = worst_default

# ---------------------------------------------------------------------------
# (c) the cost: unitarity drift
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("(c) unitarity: max |sum_beta P_(beta alpha) - 1|")
print("=" * 74)
print(f"  {'cz':>6s} {'E [GeV]':>8s} {'ODE':>12s} {'layered':>12s}")
unit = []
for cz in (-0.4, -0.9):
    for E in (2.0, 5.0, 10.0):
        Pode = np.asarray(probability_earth_continuous(P, E, cz, ye=YE_ODE))
        Play = np.asarray(probability_earth(P, jnp.asarray(E), jnp.asarray(cz),
                                            n_sub=12))
        u_ode = float(np.abs(Pode.sum(axis=-2) - 1.0).max())
        u_lay = float(np.abs(Play.sum(axis=-2) - 1.0).max())
        print(f"  {cz:6.2f} {E:8.1f} {u_ode:12.2e} {u_lay:12.2e}")
        unit.append(dict(cz=cz, energy_GeV=E, unitarity_ode=u_ode,
                         unitarity_layered=u_lay))
results["unitarity"] = unit
worst_u_ode = max(r["unitarity_ode"] for r in unit)
worst_u_lay = max(r["unitarity_layered"] for r in unit)
print(f"\n  worst ODE     : {worst_u_ode:.1e}")
print(f"  worst layered : {worst_u_lay:.1e}")
print("  The layered propagator is a product of exact unitary exponentials;")
print("  a Runge-Kutta step is not constrained to the unitary group.")
results["worst_unitarity_ode"] = worst_u_ode
results["worst_unitarity_layered"] = worst_u_lay

outpath = os.path.join(OUTDIR, "jaxnu_bench_ode_backend.json")
with open(outpath, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\nsaved {outpath}")
