# External cross-code validation

jaxnu is validated against two independent established codes: Peter Denton's
**NuFast** (C++, https://github.com/PeterDenton) and **OscProb** (ROOT/Eigen,
https://github.com/joaoabcoelho/OscProb). Reference values are embedded as
regression tests (`tests/test_nufast_reference.py`,
`tests/test_oscprob_reference.py`); this document records the methodology and
results.

**Summary:** three independent implementations agree. jaxnu matches **OscProb to
~1e-9** (constant density) and **~1e-7** (PREM Earth, identical path); it matches
**NuFast to ~1e-5**, with the residual fully explained by NuFast-LBL's
6-significant-figure constants (OscProb vs NuFast differ by the same ~1e-5, while
jaxnu vs OscProb is ~1e-9 — i.e. jaxnu and OscProb share the more precise
constants).

All comparisons use parameters matching `jaxnu.nufit_no()`
(s²θ from θ12=0.5836, θ13=0.1495, θ23=0.8587; δ=3.40; Δm²₂₁=7.42e-5;
Δm²₃₁=2.515e-3 eV²).

## 1. Constant density + vacuum + antineutrinos — NuFast-LBL

NuFast-LBL run in **exact** mode (`N_Newton = -1`) over a 25-point energy grid
(0.3–12 GeV), `L = 1300 km`, `rhoYe = 1.425`, for neutrinos, antineutrinos
(negative E), and vacuum. All nine channels compared.

| quantity            | max \|jaxnu − NuFast\| |
|---------------------|------------------------|
| vacuum              | 1.2e-6 |
| matter (ν)          | 2.5e-5 |
| matter (ν̄)          | 1.3e-5 |

These residuals are **entirely NuFast-LBL's 6-significant-figure constants**
(ħc = 1.97327e-7, matter potential `YerhoE2a = 1.52588e-4`). Overriding jaxnu's
first-principles constants with NuFast's literals collapses the agreement to:

| quantity | max \|jaxnu − NuFast\| (constants aligned) |
|----------|--------------------------------------------|
| vacuum / matter / antineutrino | **5e-13** |

i.e. the two independent codes agree to machine precision; jaxnu's CODATA-derived
constants are simply slightly more precise than NuFast-LBL's rounded literals.

## 2. PREM Earth — NuFast-Earth

NuFast-Earth (`Probability_Engine`) over up-going trajectories, detector depth 0,
production height 0, `Y_e = 0.466/0.494` (core/mantle). NuFast-Earth's matter
constant is `1.526493e-4`, **identical to jaxnu's first-principles value**, so
only the kinematic ħc rounding (~1e-6) remains.

* **Geometry isolation** — with a `Constant` Earth density, jaxnu at
  `L = baseline_km(cz)` matches NuFast-Earth to **~1e-7** at every zenith,
  confirming the chord geometry and `baseline_km(cz)`.
* **Full PREM** — vs the *converged* NuFast model
  `PREM_NDiscontinuityLayer(400,400,400,400)`, core-crossing `P(νe→νe)`:

  | E (GeV) | jaxnu (n_sub=100) | NuFast (n=400) | diff |
  |---------|-------------------|----------------|------|
  | 2.0     | 0.79995129        | 0.79995640     | 5e-6 |
  | 5.0     | 0.07961324        | 0.07961404     | 8e-7 |

### The `PREM_Full` subtlety

A first comparison against NuFast's default `PREM_Full` showed up to ~0.1
differences for **core-crossing** trajectories (mantle paths agreed to 1e-6).
Investigation showed `PREM_Full` collapses each PREM region to a **single
path-averaged constant-density shell** (10 km integration step) — fast but coarse
exactly where the density varies most (outer core, lower mantle). Subdividing
NuFast's own model converges it onto jaxnu:

```
NuFast PREM_NDiscontinuityLayer(n,n,n,n), cz=-1, E=2 GeV, P(ee):
  n=1   0.75149   (~ coarse PREM_Full)
  n=2   0.79336
  n=5   0.79914
  n=20  0.79997
  n=100 0.79997   ->  jaxnu = 0.79995
```

jaxnu's fine subdivision and its continuous **ODE backend** agree with each other
to 2e-5 and with converged NuFast to ~1e-5. jaxnu is the more accurate of the two
at fixed coarse settings.

## 3. OscProb (ROOT/Eigen)

OscProb's `PMNS_Fast` (Cayley-Hamilton-free; uses Eigen eigensolvers — a fully
independent algorithm from jaxnu).

* **Constant density** (`L = 1300`, `rho = 2.85`, `Z/A = 0.5`), three-way at
  E = 2.25 / 4.6875 GeV across ν/ν̄/vacuum channels:

  | comparison        | typical \|Δ\| |
  |-------------------|---------------|
  | jaxnu vs OscProb  | **1e-9**      |
  | jaxnu vs NuFast   | 1e-5          |
  | OscProb vs NuFast | 1e-5          |

  jaxnu and OscProb agree to ~1e-9 despite completely different propagators
  (analytic Cayley-Hamilton vs Eigen diagonalization), confirming both the
  algorithm and the matter potential. The ~1e-5 offset of both from NuFast is its
  rounded constants.

* **PREM Earth, identical path** — OscProb's `PremModel::FillPath(cosT=-1)`
  produces an 85-segment core-crossing path (incl. a 15 km atmosphere layer).
  Feeding that exact `(length, density, Z/A)` list into jaxnu's
  `probability_profile` reproduces OscProb's `P(νe→νe)`, `P(νμ→νe)`, `P(νμ→νμ)`
  to **~1e-7** at E = 2.25 and 4.6875 GeV. (Stored in
  `tests/data/oscprob_earth_cosT-1.txt`.)

## Reproducing

```bash
# NuFast-LBL (constant density / vacuum / antineutrino)
curl -LO https://raw.githubusercontent.com/PeterDenton/NuFast-LBL/main/NuFast_LBL.cpp
# strip its main(), add a driver calling NuFast::Probability_Matter_LBL / _Vacuum_LBL,
# compile with clang++ -O2 -std=c++17, run, compare to probability_constant/_vacuum.

# NuFast-Earth (PREM)
curl -LO https://github.com/PeterDenton/NuFast-Earth/archive/refs/heads/main.tar.gz
tar xzf main.tar.gz && cd NuFast-Earth-main
# drive Probability_Engine with PREM_NDiscontinuityLayer(N,N,N,N), make, run,
# compare to probability_earth (Y_e aligned to 0.466/0.494).
```

```bash
# OscProb (needs ROOT). Eigen is a submodule; drop 3.4.0 headers into ./eigen.
git clone --recursive https://github.com/joaoabcoelho/OscProb   # or add eigen/ by hand
source <root>/bin/thisroot.sh && cd OscProb && make
# compile a driver against -I inc -I eigen $(root-config --cflags --libs) -lOscProb:
#   PMNS_Fast: SetMix(th12,th23,th13,dcp); SetDeltaMsqrs(dm21, dm31-dm21);
#              SetPath(L,rho,zoa) -> Prob(flvi,flvf,E)
#   PremModel: FillPath(cosT); GetNuPath() -> feed segments to probability_profile.
```

The embedded regression tests (`tests/test_nufast_reference.py`,
`tests/test_oscprob_reference.py`) lock in these reference numbers without needing
the C++/ROOT toolchains.
