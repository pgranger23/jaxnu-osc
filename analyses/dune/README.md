# DUNE long-baseline sensitivity with jaxnu

A reproduction of the **DUNE TDR oscillation analysis** — far-detector event
spectra and the CP-violation / mass-ordering sensitivities — using **jaxnu** for the
(differentiable) oscillation probabilities together with DUNE's official GLoBES
configuration. No binning optimization; the standard TDR analysis.

## What this is

- **Forward model** (`dune.py`): for each reconstructed bin and channel,
  `N = norm · post_eff · Σ_j migration[i,j] · flux[j] · xsec[j] · P_osc[j]`, summed
  over channels within each of the four rules (νe app, ν̄e app, νμ dis, ν̄μ dis) in
  neutrino (FHC) and antineutrino (RHC) beam modes. The migration matrices,
  efficiencies, flux, and cross-sections are DUNE's; **only the oscillation
  probabilities come from jaxnu**, so the entire spectrum is differentiable.
- **Sensitivity** (`dune.py`): an Asimov Poisson χ² with the nine GLoBES
  normalization systematics, profiled over θ₂₃ (both octants), θ₁₃ (prior), Δm²₃₁,
  δ_CP, and the matter density (2% prior). The minimization uses the **exact
  gradient from jaxnu** (`jax.value_and_grad` → L-BFGS) — this is the payoff of a
  differentiable simulator: fast, robust profiling over a ~15-parameter space.

## Configuration (from the DUNE GLoBES files)

- Baseline 1284.9 km, constant matter density 2.848 g/cm³.
- 40 kt fiducial, 1.2 MW, 6.5 + 6.5 years (FHC + RHC) = **624 kt·MW·yr**.
- NuFIT 4.0 normal-ordering parameters; normalization systematics 2% (νe signal),
  5% (νμ signal / bg), 5% (beam-νe bg), 20% (ντ bg), 10% (NC dis bg).
- Energy window 0.5–18 GeV.

## Results vs the DUNE reference

| observable | jaxnu | DUNE TDR GLoBES reference |
|---|---|---|
| νe-app signal peak | 275 / 0.25 GeV at 2.7 GeV | 275 / 0.25 GeV at 2.7 GeV ✓ |
| CPV peaks (√Δχ²) | 7.2 (δ<0), 7.5 (δ>0) | ~6.8, ~7.1 |
| CPV peak positions | δ_CP/π ≈ −0.4, +0.6 | −0.4, +0.6 ✓ |
| CPV zeros | δ_CP = 0, ±π | 0, ±π ✓ |
| MO √Δχ² range | 15 – 27 | ~13.5 – 24.5 |
| MO extrema positions | max −0.55, min +0.45 | max −0.6, min +0.4 ✓ |

The spectra match essentially exactly; the sensitivities reproduce the correct
shapes and peak positions, running ~6–10% high — expected, since the released
GLoBES config is an "example sensitivity" and this marginalization fixes θ₁₂ /
Δm²₂₁ (both tightly constrained). The event-rate overall normalization is
calibrated by a single factor (`CAL_NORM`) to the DUNE reference spectrum, as the
GLoBES internal unit convention is opaque; the shape and signal/background ratios
are pure physics from the migration matrices + jaxnu.

Reference figures are under `reference/` (`cpv_globes.png`, `mh_globes.png`,
`spec_app_nu_5yr.png`, …); the reproductions are `dune_spectra.png` and
`dune_sensitivity.png`.

## Run

```bash
python run_spectra.py       # -> dune_spectra.png  (4 far-detector spectra)
python run_sensitivity.py   # -> dune_sensitivity.png  (CPV + mass ordering; ~2.5 min)
```

```python
import dune
sig = dune.cpv_sensitivity([-0.5, 0.0, 0.5])          # sqrt(dChi2) for CP violation
mo  = dune.mass_ordering_sensitivity([-0.5, 0.0, 0.5])
# the forward model is differentiable: e.g. d(spectrum)/d(delta_CP)
```

## Why jaxnu here

Beyond reproducing the standard result, the differentiable forward model enables
what the C++ GLoBES pipeline cannot: exact gradients of the spectra and χ² w.r.t.
all parameters, so the profiling is gradient-based, and the same machinery extends
directly to **Fisher-information forecasts, optimal binning/design, NSI/sterile
fits, and HMC posteriors** — future directions this analysis is set up for.

## Provenance / attribution

The GLoBES inputs under `globes/` are the official DUNE configuration released as
ancillary files to **arXiv:2103.04797** ("Experiment Simulation Configurations
Approximating DUNE TDR", DUNE Collaboration), redistributed here for reproducibility
with attribution. The oscillation physics and the analysis code are jaxnu's.
