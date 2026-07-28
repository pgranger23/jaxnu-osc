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
| `check_bsm_limits.py` | NSI/sterile decoupling limits, unitarity, and AD-vs-finite-difference checks for the beyond-standard-model front-ends (no independent reference code exists for these, so they are validated against exact analytic limits instead) | Table "Validation of the NSI and sterile sectors through exact limits and gradient checks" |
| `bench_backends_and_precision.py timing` | forward-evaluation throughput of the four propagation backends (`cayley`, `eigh`, `expm`, `nufast`) at constant density, plus the PREM Earth forward pass and its full 6-parameter gradient, run standalone (independent of the plotting script below) | Table "Backend performance per energy point" (the CPU column; run again on a GPU host for the A100 column) |
| `bench_backends_and_precision.py grad` | autodiff vs. central finite differences for all three derivative classes claimed in the paper: oscillation parameters (θ13, θ23, δCP, Δm²31), geometry (cos θ_z, atmospheric production height), and matter density (core/mantle log-density scale factors) | the "10⁻⁸ level or better" gradient-validation claim (Validation section) |
| `bench_backends_and_precision.py prec` | float32 vs float64 agreement for a PREM oscillogram, quantifying why float64 is mandatory | the "double precision is mandatory" claim (Units and precision, `README.md`) |
| `bench_timing_and_gradients.py` | (a) the same backend timing rows as above but self-contained in one script including the NuFast port, and (b) `∂P/∂ln ρ` oscillograms for the core and mantle density — the matter-density tomography derivative that no analytic constant-density code (NuFast, Prob3++, ...) can supply | Table "Backend performance per energy point" (self-contained cross-check) and the density-derivative figure discussed in the Introduction/BSM sections |

`check_bsm_limits.py` runs in well under a second. The other two scripts do
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
| `check_bsm_limits.py` | ~2 s | yes |
| `bench_backends_and_precision.py timing` | ~2-3 min (most of it is `jit`-compiling the 6-parameter PREM gradient once) | yes |
| `bench_backends_and_precision.py grad` | ~1-2 min | yes |
| `bench_backends_and_precision.py prec` | ~15-20 s | yes |
| `bench_timing_and_gradients.py` | ~5-10 min (the `∂P/∂ln ρ` oscillogram is a 110×110 grid of PREM-Earth Jacobians, `NG=110`, on CPU) | yes, but slow — see note below |

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
- `output/jaxnu_bench2_prec.json` — float32-vs-float64 deviations from
  `bench_backends_and_precision.py prec`.

`output/` is regenerated by re-running the scripts and is not meant to be
committed as a fixed artifact — it exists so you can inspect the raw numbers
behind the printed summary.
