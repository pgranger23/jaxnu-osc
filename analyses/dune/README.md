# DUNE long-baseline sensitivity with jaxnu

A self-contained reproduction of the **DUNE TDR oscillation analysis** — far-detector
event spectra and the CP-violation / mass-ordering sensitivities — built on
[**jaxnu**](https://github.com/pgranger23/mango-osc) for the (differentiable)
oscillation probabilities and DUNE's official GLoBES configuration for everything
else. The entire forward model is differentiable, and the χ² accepts an arbitrary
**analysis binning as a matrix `W`** — the intended hook for a binning-optimization
study (see [Extension point](#extension-point-custom--optimized-binning)).

## Contents

```
dune.py              forward model + chi2 + CP-violation / mass-ordering sensitivity
run_spectra.py       -> dune_spectra.png       (the four far-detector spectra)
run_sensitivity.py   -> dune_sensitivity.png   (CPV + mass ordering; ~2.5 min)
example_binning.py   custom-binning demo + the optimizer hook
globes/              official DUNE GLoBES inputs (flux, xsec, migration matrices, ...)
reference/           DUNE reference figures for comparison
```

## Requirements

`jaxnu` (the parent package — this script adds the repo root to `sys.path`, or
`pip install jaxnu`), plus `numpy`, `scipy`, `matplotlib`. Float64 is required and
enabled automatically by mango.

```bash
python run_spectra.py        # four far-detector spectra
python run_sensitivity.py    # CP-violation + mass-ordering sensitivities
python example_binning.py    # sensitivity vs binning (extension demo)
```

## Physics configuration (from the DUNE GLoBES files)

- Baseline 1284.9 km, constant matter density 2.848 g/cm³.
- 40 kt fiducial, 1.2 MW, 6.5 + 6.5 years (FHC + RHC) = **624 kt·MW·yr**.
- NuFIT 4.0 normal-ordering parameters (`dune.NUFIT_NO`).
- Normalization systematics: 2 % (νe signal), 5 % (νμ signal / bg, beam-νe),
  20 % (ντ), 10 % (NC dis) — `dune.data().sys_sigma`.
- Energy window 0.5–18 GeV.

## Forward model

For each reconstructed bin `i` and channel `c`,

```
N_c[i] = norm · post_eff_c[i] · Σ_j  R_c[i,j] · flux_c[j] · xsec_c[j] · P_c[j]
```

with `R_c` the DUNE migration matrix (true→reco, full detector reconstruction),
summed over channels within each of the four rules (νe app, ν̄e app, νμ dis, ν̄μ dis)
in FHC and RHC beam modes. **Only the oscillation probabilities `P_c` come from
jaxnu**, so the whole spectrum is differentiable in the physics parameters.

## Validation vs the DUNE reference

| observable | jaxnu | DUNE reference |
|---|---|---|
| νe-app signal peak | 275 / 0.25 GeV at 2.7 GeV | 275 / 0.25 GeV at 2.7 GeV |
| CPV peaks (√Δχ²) | 7.2 / 7.5 | ~6.8 / 7.1 |
| CPV peak positions / zeros | δ_CP/π ≈ −0.4, +0.6 / 0, ±1 | same |
| mass-ordering √Δχ² range | 15 – 27 | ~13.5 – 24.5 |

Spectra match essentially exactly; the sensitivities reproduce the shapes and peak
positions, running ~6–10 % high (this marginalization fixes the tightly-constrained
θ₁₂ / Δm²₂₁, and the released GLoBES is an "example" config). The event-rate overall
normalization is set by a single factor `dune.CAL_NORM` calibrated to the DUNE
reference spectrum; the shape and signal/background ratios are pure physics.

## Public API

```python
import dune
d = dune.data()                       # parsed config: d.reco_c, d.reco_edges, d.win,
                                      #   d.sys_sigma, d.rules, ...
p  = dune.params_from([th23, th13, dm31, dcp])     # -> jaxnu OscParams (NuFIT th12/dm21)
sp = dune.total_spectra(p)            # dict rule -> total reco spectrum (fine bins)
cp = dune.rule_components(p, rho)     # dict rule -> [(sys_label, is_signal, spectrum)]
dat = dune.asimov(deltacp)            # Asimov 'data' at true NuFIT-NO + deltacp

# sensitivities (profiled over th23 both octants, th13/density priors, dm31, dcp,
#   and the 9 normalization systematics, via exact jaxnu gradients -> L-BFGS)
sig = dune.cpv_sensitivity(dcp_over_pi_array, W=None)          # sqrt(dChi2), CP violation
mo  = dune.mass_ordering_sensitivity(dcp_over_pi_array, W=None)

# the differentiable chi2 (jitted value_and_grad) used internally:
val, grad = dune.chi2_vg(free_vec, data_tuple, W)
#   free = [th23, th13, dm31, dcp, drho, xi_0..xi_8]  (drho = fractional density pull)
```

## Extension point: custom / optimized binning

Every sensitivity call takes a **rebinning matrix** `W` of shape `(K, nreco)` that
maps the native fine model to `K` analysis bins (`coarse = W @ fine`). `W = None`
uses the native fine binning; `dune.rebin_matrix(edges)` builds `W` for arbitrary
bin edges and **is differentiable in `edges`**. This is the hook for optimizing the
binning:

```python
import numpy as np, jax, jax.numpy as jnp, dune

# evaluate the sensitivity with any binning
edges = np.linspace(0.5, 8.0, 13)              # 12 uniform bins, say
W = dune.rebin_matrix(edges)                   # (12, nreco), dW/d(edges) exists
sigma = dune.cpv_sensitivity([-0.5, 0.5], W=W)

# ... or build a differentiable Fisher-information objective for delta_CP and
#     gradient-ascend the bin edges:
#   mu_k        = W @ (fine model at truth)                 # coarse expectations
#   dmu_k/dθ    = W @ jacrev(fine model)(θ)                 # via jaxnu autodiff
#   F(edges)    = Σ_k (dmu_k/dθ)(dmu_k/dθ)^T / mu_k + priors
#   FoM(edges)  = profiled Fisher for delta_CP = 1 / inv(F)[dcp,dcp]
#   grad(edges) = jax.grad(FoM)(edges)      # then Adam / projected gradient ascent
```

`dune.rule_components(params, rho)` gives the per-component fine spectra (each a JAX
array, differentiable in `params`), so `jax.jacrev` yields the per-bin
parameter-gradients needed for the Fisher matrix. `example_binning.py` shows the
uniform-vs-fine comparison and the hook.

## Provenance

The GLoBES inputs under `globes/` are the official DUNE configuration released as
ancillary files to **arXiv:2103.04797** ("Experiment Simulation Configurations
Approximating DUNE TDR", DUNE Collaboration), redistributed for reproducibility with
attribution. The oscillation physics and analysis code are jaxnu's.
