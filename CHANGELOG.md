# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.4] - 2026-08-04

### Fixed

- **`jaxnu.__version__` was never bumped past 0.2.2**, so a clone of the v0.2.3
  tag reported the wrong version. It now tracks `pyproject.toml`.

### Added

- **`degen` mode in `bench_backends_and_precision.py`.** The propagator error
  at an exactly degenerate spectrum was quoted from a source comment and could
  not be re-run. It is measured over 500 trials now, with the ensemble pinned
  (unit spectral radius, one radian of phase) because the error scales with the
  accumulated phase and an unnormalized ensemble does not define a number.
- **SVD pseudo-inverse cross-check** in `demo_tomography_fisher.py`. A small
  `F F^-1 - I` residual is a backward-error statement and does not on its own
  bound the forward error at this condition number; an independent inversion
  route does.
- **Sterile level-crossing gradient check** in `check_bsm_limits.py`,
  quantifying the one place a gradient is silently wrong.
- **`fig_chord_geometry.py`**, the chord-construction schematic.

## [0.2.3] - 2026-08-03

### Added

- **`benchmarks/bench_bsm_derivatives.py`** — dP/dgamma (Lindblad decoherence)
  and dP/dalpha (non-unitary mixing) at the standard-model point, the figure of
  BSM-parameter derivatives in the accompanying paper. This script and the
  second-order design derivative below post-dated the v0.2.2 tag, so v0.2.2
  could not reproduce two of the paper's exhibits; that is what this release
  fixes.
- **Second-order design derivative** in `demo_tomography_fisher.py`:
  `d2 sigma / d res_cz^2`, giving the range over which the first-order tangent
  can be extrapolated.
- **Solar closed-form MSW cross-check as a unit test.** The comparison against
  an independently written two-flavour adiabatic formula existed only as a
  benchmark; the paper's validation table describes every row as shipped in the
  test suite, which is now true.

### Fixed

- **`jaxnu.stat` terminology.** `1/sqrt(F_ii)` was described as the
  "profiled-out" bound. Profiling means minimizing over the other parameters
  and in the Gaussian limit coincides with *marginalizing*, i.e.
  `sqrt(inv(F)_ii)`. The fixed-others number is never the profiled result.
- **`NSI.matrix()` Hermiticity.** The off-diagonals were conjugate-paired but a
  complex diagonal was passed straight through, producing a non-Hermitian
  Hamiltonian and a silently wrong propagation. The diagonal is now taken real.

## [0.2.2] - 2026-07-30

### Fixed

- **`demo_tomography_fisher.py` bulk-core normalization.** The summary compared
  sigma of the common-mode core scaling against sigma of the *sum* of the two
  core zones -- quantities differing by a factor of two -- overstating the
  marginalization penalty. Corrected: 0.302 -> 0.151, and the design derivative
  2.371 -> 1.186. Per-zone numbers were unaffected.
- **`globes.py` silent zero baseline** for `$baseline` (only `$baselinelength`
  was recognised), which zeroed every appearance channel without error.
- Documented the `eigh` backend's degenerate-spectrum gradient hazard.

### Added

- Spectral-tilt systematics in the GLoBES loader (manual Eq. 11.26).
- Layered/Earth non-unitary propagation (previously single-layer only).
- `bench_derivative_classes.py`, `bench_bsm_comparison.py`, and a
  six-zone + antineutrino tomography demo.

## [0.2.1] - 2026-07-28

Review-round correctness fixes, applied on top of the v0.2.0 feature
release (decoherence, non-unitarity, GLoBES loader, `jaxnu.stat`).

### Added

- **`jaxnu.earth.critical_cos_zenith()`** — returns the `cos θ_z` values at
  which the neutrino chord grazes a PREM shell boundary. At those angles the
  entering shell contributes a length `~ sqrt(r_b² - r_min²)`, a genuine
  sqrt-cusp, so `dP/d cos θ_z` diverges like `1/sqrt(|Δ|)`: measured ~1.6e3 at
  1e-6 from the core–mantle boundary against ~1.6 nearby. This is inherent to
  piecewise-constant shells and is *not* smoothed away; the helper exists so
  that gradient-based optimizers which move `cos θ_z` bin edges continuously
  can avoid parking an edge on one of them.
- **`benchmarks/demo_tomography_fisher.py`** — a self-contained end-to-end
  example: PREM shell densities → chord geometry → oscillation probability →
  atmospheric flux → `(E, cos θ_z)` histogram → detector response matrix →
  marginalized Poisson Fisher information → σ(ln ρ_core), all from one
  `jacfwd`. It then differentiates that σ *through the matrix inverse* with
  respect to the detector's angular resolution, giving an experimental-design
  derivative (`∂σ/∂σ_cosθ = 0.1195`, matching central differences to five
  significant figures). Intended as a demonstration of the machinery, not a
  sensitivity forecast — the detector model is a placeholder.
- **`THIRD_PARTY_LICENSES.md`** — retains the upstream copyright and licence
  text for incorporated third-party material, as those licences require. In
  particular the NuFast-LBL MIT licence (Copyright (c) 2024 Peter B. Denton) is
  now reproduced verbatim; previously only a prose credit existed, which does
  not satisfy the MIT notice-retention condition. Also documents PREM, the BS05
  solar tables and nu-waves.
- **Marginalization breakdown in `demo_tomography_fisher.py`** — reports which
  nuisances actually drive the degradation in sigma(ln rho_core), and labels the
  per-bin information split explicitly as a decomposition of the nuisance-fixed
  F_00.
- **`benchmarks/`** — the timing, precision and BSM-limit scripts behind the
  numbers quoted in the accompanying paper, so they can be reproduced from a
  clone of this repository rather than only from the authors' analysis code.
- **`CITATION.cff`** and this changelog.
- **New test modules** `tests/test_geometry_grad.py`,
  `tests/test_backend_edge_cases.py` and `tests/test_api_validation.py`,
  covering the geometry/density gradients against finite differences, the
  degenerate and null-parameter corners, the inverted ordering (previously
  untested anywhere), and the new input validation under `jit`/`vmap`.

### Fixed

- **`cos(zenith) = ±1` gradients were silently halved.** The Earth-geometry
  code computed `rmin2 = r_det**2 * clip(1 - cz**2, 0, None)`, which landed
  exactly on the `clip` tie at the poles (`cz = ±1`); this halved
  `d(baseline_km)/d(cos θ_z)` and every downstream gradient (`dP/d cos θ_z`)
  evaluated exactly there. Straight-up/straight-down trajectories now get
  the correct one-sided derivative, matching finite differences.
- **The `nufast` backend could return `NaN` at `Δm²₂₁ = 0` or `θ₁₃ = 0`.**
  These are physically legitimate inputs (e.g. probing the two-flavor limit,
  or scanning `θ₁₃` down to zero), but the analytic constant-density formula
  hit a `0/0` in the eigenvalue-gap machinery there. It now returns the
  correct (finite) probability instead of `NaN`.
- **`jax.grad(baseline_km)` returned `NaN` at the horizon (`cos θ_z = 0`).**
  The ascending-leg length was computed as a bare `sqrt(clip(...))` whose
  argument is exactly zero there; the enclosing `jnp.where` hid the divergent
  `sqrt'(0)` in forward mode, but reverse mode still propagated `NaN` back
  through the unselected branch (`jax.grad` gave `NaN` while `jax.jacfwd` gave
  `0` — the tell-tale signature). It now uses the same guarded helper as the
  rest of the chord code, and the two AD modes agree everywhere.
- **`nufit_no()` was documented as NuFIT 5.2 but returns NuFIT 5.1 values.**
  The numbers match the NuFIT 5.1 (2021) normal-ordering best fit without SK
  atmospheric data digit for digit (5.2 quotes `dm31 = 2.511e-3`,
  `theta23 = 49.1 deg`, `deltacp = 197 deg`). Only the docstring changed; the
  returned parameters are unchanged.
- **`nsi=` accepted non-Hermitian matrices.** Passing a raw matter-NSI matrix
  (as opposed to a `jaxnu.nsi.NSI(...)` instance, which builds a Hermitian
  matrix by construction) with `eps_{alpha,beta} != conj(eps_{beta,alpha})`
  used to be accepted silently, which breaks unitarity of the propagated
  amplitude (injects unphysical gain/loss into some channels). A
  non-Hermitian raw `nsi=` matrix now raises `ValueError` on concrete input
  (the check is skipped under `jit`/`vmap`, consistent with the rest of the
  input validation below).
- **Missing input validation.** Public API calls with physically
  nonsensical concrete inputs (negative energy, negative baseline/density,
  `|cos θ_z| > 1`, out-of-range flavor indices, ...) used to either produce
  silently wrong probabilities or an opaque downstream `NaN`/shape error
  instead of a clear message. These now raise `ValueError` naming the bad
  argument and the expected units/range, when the value is concrete;
  as before, no check can run on a value being traced under `jit`/`vmap`/
  `grad`, so those code paths are unaffected and unchanged in behavior.
- **`jaxnu.solar.adiabatic_mass_fractions` raised on a scalar `r_km`.**
  Passing a single (scalar) radius, as shown in `README.md`'s solar
  snippet, used to raise a `vmap` rank error because the function assumed
  an array input internally. Scalar `r_km` is now accepted directly and
  returns a 1-D array of per-mass-state fractions (array `r_km` input and
  its output shape are unchanged).
- **`Sterile3plus1` / `pmns_3plus1` hardwired the `θ₁₄` CP phase (`δ₁₄`) to
  zero.** The 3+1 mixing-matrix builder had no way to set the Dirac phase
  associated with the `(0,3)` (`θ₁₄`) rotation, silently dropping a
  physical degree of freedom of the 3+1 model. `delta14` is now a
  constructor argument (defaulting to `0.0`, so existing code and saved
  parameter values are unaffected) that changes the active-sterile
  appearance probabilities as expected.

## [0.1.0] - TBD

(Version number is set in `pyproject.toml` / `jaxnu/__init__.py`; the release
date will be filled in when this version is actually tagged and published —
see "Releasing to PyPI" in `README.md`.)

Initial release: vacuum, constant-density matter, layered PREM Earth (with
atmospheric production height and detector depth), arbitrary user-supplied
density profiles, and adiabatic solar propagation; matter non-standard
interactions and 3+N sterile neutrinos. Four cross-checked propagation
backends (`cayley`, `eigh`, `expm`, `nufast`) and an optional continuous ODE
backend (`odeint` / `diffrax`). Validated against OscProb and NuFast and
against the nu-waves reference plots (see `validation/README.md`).
