# Benchmarks

Standalone scripts that reproduce the performance and validation tables in the
accompanying paper (`jaxnu: an autodiff-first neutrino oscillation engine for
differentiable analysis pipelines`). Each imports only `jaxnu` / `numpy` /
`jax` (and `matplotlib` for the one script that makes a plot) — no path or
module from any private analysis repository. They write their output into
`benchmarks/output/` (created automatically) rather than the paper's own
build tree.

Run from the repository root, e.g.:

```bash
pip install -e ".[examples]"                       # matplotlib, for the timing/gradients script
JAX_PLATFORMS=cpu python benchmarks/check_bsm_limits.py
JAX_PLATFORMS=cpu python benchmarks/bench_backends_and_precision.py grad
JAX_PLATFORMS=cpu python benchmarks/bench_backends_and_precision.py timing
JAX_PLATFORMS=cpu python benchmarks/bench_backends_and_precision.py prec
JAX_PLATFORMS=cpu python benchmarks/bench_timing_and_gradients.py
```

Drop `JAX_PLATFORMS=cpu` to let JAX pick up a visible GPU/TPU.

## What each script measures

| script | measures | paper table / figure |
|---|---|---|
| `demo_tomography_fisher.py` | End-to-end composability demo: propagates the density derivatives through an atmospheric flux, an (E, cos theta) histogram, a Gaussian detector response and a marginalized Poisson Fisher matrix to a tomographic sigma(ln rho_core), then differentiates that sigma through the matrix inverse with respect to the angular resolution. A demonstration of the machinery, **not** a sensitivity forecast | Section "A worked example: tomographic sensitivity end to end" |
| `nsi_capability.py` | What the NSI derivatives are for, beyond the eps_ee/density degeneracy (which is a correctness check, not a capability demo): (a) wall-clock cost of a batched `jax.jacfwd` Jacobian vs. serialized finite differences as free NSI parameters are added to the fit; (b,c) per-bin profiled information on Re/Im eps_mutau via orthogonal projection against every other free parameter, which surfaces a ~290x real-vs-imaginary sensitivity asymmetry, cross-checked directly against dP/d(eps) away from the Fisher machinery | Section "What the NSI derivatives are for" (figure "Cost of extending the fit / per-bin profiled information") |
| `check_bsm_limits.py` | NSI/sterile decoupling limits, unitarity, and AD-vs-finite-difference checks for the beyond-standard-model front-ends (no independent reference code exists for these, so they are validated against exact analytic limits instead) | Table "Validation of the NSI and sterile sectors through exact limits and gradient checks" |
| `bench_backends_and_precision.py timing` | forward-evaluation throughput of the four propagation backends (`cayley`, `eigh`, `expm`, `nufast`) at constant density, plus the PREM Earth forward pass and its full 6-parameter gradient, run standalone (independent of the plotting script below) | Table "Backend performance per energy point" (the CPU column; run again on a GPU host for the A100 column) |
| `bench_backends_and_precision.py grad` | autodiff vs. central finite differences for all three derivative classes claimed in the paper: oscillation parameters (θ13, θ23, δCP, Δm²31), geometry (cos θ_z, atmospheric production height), and matter density (core/mantle log-density scale factors) | the "10⁻⁸ level or better" gradient-validation claim (Validation section) |
| `bench_backends_and_precision.py prec` | float32 vs float64 agreement for a PREM oscillogram, quantifying why float64 is mandatory | the "double precision is mandatory" claim (Units and precision, `README.md`) |
| `bench_derivative_classes.py` | reverse-mode gradient cost for each derivative class through the layered PREM Earth -- oscillation parameters (6), geometry (42) and the full parameter set of a 123-shell LayeredEarth (369) -- with mean and standard deviation over independent groups, and a finite-difference check of every class | Table "Cost of a reverse-mode gradient relative to one forward evaluation" |
| `bench_bsm_derivatives.py` | derivatives with respect to the BSM parameters themselves — dP/dgamma (Lindblad decoherence, Earth diameter) and dP/dalpha (non-unitary mixing, PREM), both at the standard-model point where a limit-setting analysis linearizes | the BSM-derivative figure |
| `bench_bsm_comparison.py` | the adiabatic solar sector against an independently written closed-form two-flavour MSW formula (the one beyond-standard-model sector for which a non-circular check is possible without a third-party code) | the solar cross-check quoted in the beyond-standard-model section |
| `bench_timing_and_gradients.py` | (a) the same backend timing rows as above but self-contained in one script including the NuFast port, and (b) `∂P/∂ln ρ` oscillograms for the core and mantle density — the matter-density tomography derivative that no analytic constant-density code (NuFast, Prob3++, ...) can supply | Table "Backend performance per energy point" (self-contained cross-check) and the density-derivative figure discussed in the Introduction/BSM sections |

`check_bsm_limits.py` takes about 35 s on CPU, almost all of it JIT warm-up rather than arithmetic; `demo_tomography_fisher.py` takes a couple of minutes on CPU (900 bins x 6 parameters through the layered Earth). The other two scripts do
real batched JAX work; see **Measured runtimes** below.

## Hardware dependence

**Absolute timings are hardware-dependent.** The paper's own numbers were
measured on 8 cores of an AMD EPYC 7542 (CPU) and one NVIDIA A100 (GPU), and
are reproducible to about 15% run-to-run on that hardware. Do not expect the
numbers from these scripts on a different machine to match the paper's table
values exactly — only the *relative* ordering of the backends and the
gradient/forward cost ratio (≈ 3×) should reproduce closely on any machine.

## Measured runtimes (this environment, CPU only)

Reproduced here on a shared CPU node (JAX 0.4+, float64, `JAX_PLATFORMS=cpu`,
no GPU used for this run); wall-clock times include JAX/XLA warm-up
compilation, which dominates for the smaller scripts:

| script (mode) | wall time | ran clean? |
|---|---|---|
| `check_bsm_limits.py` | ~35-70 s (almost all JIT warm-up) | yes |
| `bench_backends_and_precision.py timing` | ~2-3 min (most of it is `jit`-compiling the 6-parameter PREM gradient once) | yes |
| `bench_backends_and_precision.py grad` | ~1-2 min | yes |
| `bench_backends_and_precision.py prec` | ~15-20 s | yes |
| `bench_derivative_classes.py` | ~7 min CPU / ~1.5 min GPU | yes |
| `bench_bsm_comparison.py` | ~1 min | yes |
| `bench_bsm_derivatives.py` | ~1 min | yes |
| `bench_timing_and_gradients.py` | ~5-10 min (the `∂P/∂ln ρ` oscillogram is a 110×110 grid of PREM-Earth Jacobians, `NG=110`, on CPU) | yes, but slow — see note below |
| `nsi_capability.py` | ~10 min CPU (nine 900-bin Jacobian assemblies, k=0..8 free NSI parameters, plus the serialized finite-difference timings at each k); `FIGONLY=1` re-renders the figure from the saved `.npz` in seconds | yes |

None of these needed batch-size changes to finish in a reasonable time on
CPU **except** `bench_timing_and_gradients.py`'s density-derivative
oscillogram, which is the single most expensive thing in this directory (a
110×110 grid of `jax.jacfwd` PREM-Earth evaluations, `NG=110` in the script).
If you are only interested in the timing table and not the figure, run
`bench_backends_and_precision.py timing` instead, which reproduces the timing
rows without the oscillogram. If you do want the figure and CPU time is a
concern, lower `NG` (e.g. to 40-50) at the top of the "(b) density-derivative
oscillograms" section of `bench_timing_and_gradients.py` — the figure's
qualitative features (core vs. mantle sign structure, MSW resonance) are
already visible at `NG=40`.

## Output

- `output/jaxnu_bench_timing_and_gradients.npz` — raw timing rows + the
  `∂P/∂ln ρ` grids from `bench_timing_and_gradients.py`.
- `output/jaxnu_rho_grad.png` — the density-derivative oscillogram figure.
- `output/jaxnu_bench2_timing_<device>.json` — timing rows from
  `bench_backends_and_precision.py timing` (filename includes `cpu` or
  `gpu`/`tpu` depending on what JAX picked up).
- `output/jaxnu_bench2_grad.json` — AD-vs-finite-difference deviations from
  `bench_backends_and_precision.py grad`.
- `tomography_fisher.npz` — Fisher matrix, covariance and per-bin
  expected counts from `demo_tomography_fisher.py`.
- `output/nsi_capability.npz` / `.png` / `.pdf` — cost-vs-k timings and
  per-bin profiled NSI information maps from `nsi_capability.py`.
- `output/jaxnu_bench2_prec.json` — float32-vs-float64 deviations from
  `bench_backends_and_precision.py prec`.

`output/` is regenerated by re-running the scripts and is not meant to be
committed as a fixed artifact — it exists so you can inspect the raw numbers
behind the printed summary.
