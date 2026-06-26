# jaxnu — Differentiable Neutrino Oscillation Calculator in JAX

A JAX-native neutrino oscillation engine supporting **vacuum**, **constant-density
matter**, and **arbitrary (Earth / PREM) matter profiles**, fully differentiable
end-to-end and `jit`/`vmap`-friendly.

## 1. Goals & scope

- Compute `P(ν_α → ν_β)` (and antineutrinos) in vacuum, constant density, and
  non-constant density (Earth via PREM or a user-supplied profile).
- Differentiable w.r.t. all physics inputs: mixing angles, `δ_CP`, mass splittings,
  matter density / `Y_e`, baseline, energy, zenith angle.
- `jit` + `vmap` over energy / baseline / zenith grids.

### Confirmed decisions (v1)
- **3-flavor** standard oscillations for v1, but the numerical core is written
  generically in `N` flavors (the Hamiltonian takes a PMNS matrix `U` of any size and
  a mass-squared vector), so **sterile** neutrinos and **NSI** drop in by supplying a
  larger `U` / an NSI matrix — no core rewrite.
- **Atmospheric is the primary use case** → first-class PREM Earth path with
  `lax.scan` layer propagation; constant-density and vacuum also fully supported.
- **float64 / complex128 mandatory** (set on import; see §7).

## 2. Physics formalism

Flavor-basis Hamiltonian (natural units; `ħ = c = 1`):

```
H(x) = (1 / 2E) · U · diag(0, Δm²₂₁, Δm²₃₁) · U†  +  diag(V(x), 0, 0)
V(x) = √2 · G_F · N_e(x),   N_e = ρ(x) · Y_e · N_A
```

- `U` = PMNS (PDG parametrization `R₂₃ · U₁₃(δ) · R₁₂`).
- Antineutrinos: `U → U*`, `V → −V`.
- Segment evolution operator `S = exp(−i H L)`; `P_{αβ} = |S_{βα}|²`.
- Non-constant density: partition the path and multiply segment operators
  `S = S_N ··· S_2 S_1`.

The internal numerical core works **entirely in natural units** (energy in eV,
length in eV⁻¹); unit conversions (GeV, km, g/cm³) happen only at the API boundary,
so there are no magic factors inside the physics.

## 3. Module layout

```
jaxnu/
  constants.py     physical constants + unit conversions (first-principles V)
  pmns.py          generic N-flavor PMNS construction; 3-flavor helper
  hamiltonian.py   vacuum + matter Hamiltonian (generic N, NSI hook, anti flag)
  eigensolve.py    analytic 3x3 Hermitian eigenvalues (+ eigh fallback)
  propagator.py    exp(-iHL): cayley / eigh / expm backends
  earth.py         PREM model + chord geometry + fixed-shape path segmentation
  layers.py        piecewise-constant layer propagation via lax.scan
  oscillator.py    high-level API: vacuum / constant / earth; vmap+jit wrappers
  params.py        OscParams PyTree (differentiable leaves)
  nsi.py           matter-NSI parameters (epsilon matrix)            [implemented]
  sterile.py       3+N sterile front-end (PMNS builders + params)    [implemented]
  solar.py         solar SSM profile + adiabatic MSW                 [implemented]
```

Sterile/NSI are wired through the generic-N Hamiltonian: `matter_hamiltonian`
takes the NSI `epsilon` block and an `n_active` so sterile flavors receive only
the relative neutral-current potential; the propagator auto-falls back to `eigh`
for N != 3.

## 4. Numerical kernels — `exp(−iHL)` for Hermitian H

Three interchangeable backends:

- **`cayley`** (fast 3×3 default): `exp(−iHL) = a₀I + a₁H + a₂H²`, with `a` solving a
  Vandermonde system in the analytic eigenvalues with RHS `exp(−iλₖL)`. Eigenvalues
  only, no eigenvectors → fast and clean gradients.
- **`eigh`**: `H = VΛV†` → reconstruct. Generic for any `N` (steriles), validation
  oracle. Caveat: autodiff has `1/(λᵢ−λⱼ)` terms (fine for neutrinos — MSW is an
  avoided crossing).
- **`expm`**: `jax.scipy.linalg.expm(−iHL)`. Reference cross-check.

The 3×3 analytic eigensolver uses the stable trigonometric solution of the
characteristic cubic (real eigenvalues for Hermitian H), guarded with `where` to keep
gradients finite at the diagonal/degenerate limit.

## 5. Earth / non-constant density

**Primary — piecewise-constant layers (`lax.scan`).**
- `earth.py` implements **PREM** (density as piecewise polynomials in `x = r/R_E`,
  plus core/mantle `Y_e`).
- Chord geometry: from `cosθ_z` (and detector radius), closest-approach radius
  `r_min = R_det·√(1−cz²)`; the path crosses each shell with outer radius `> r_min`
  twice (descending + ascending leg). Segment length between radii `r_a>r_b` on a leg
  is `√(r_a²−r_min²) − √(r_b²−r_min²)`.
- **Fixed-shape for jit/vmap**: always emit all PREM shell edges; segments below
  `r_min` clamp to zero length → identity propagators. Shapes stay static across `cz`,
  and lengths are differentiable in `cz`.

**Optional — continuous ODE backend** (`i dψ/dx = H(x)ψ`) for arbitrary smooth
profiles and as an independent cross-check. Two solvers in `ode.py`: `odeint`
(default) and `diffrax` (optional install; stiff/implicit solvers + checkpointed
adjoints). The diffrax path evolves a real `[Re S, Im S]` state (diffrax complex
support is experimental). *Implemented.*

## 6. Differentiability

- Prefer `cayley` (eigenvalues only) for stable grads; `where`-guard the eigensolver.
- `vmap` over (E, cz), `lax.scan` over layers (reverse-mode AD flows through scan).
- Antineutrinos as a conjugation/sign flag, not a separate path.
- Validate every backend with finite-difference gradient checks.

## 7. Precision / performance

- `jax_enable_x64` set on import (Δm² ~ 1e−3 eV² + phase accuracy need float64).
- `jit` the probability fn; `vmap` over grids; `lax.scan` for the layer product.
- 3×3 analytic kernels avoid generic LAPACK overhead for batched small matrices.

## 8. Public API

```python
import jaxnu

params = jaxnu.OscParams(theta12, theta13, theta23, deltacp, dm21, dm31)

# vacuum / constant / earth, all batched + differentiable
P = jaxnu.prob_earth(params, energy_GeV, cos_zenith, flavor_in, flavor_out)
g = jax.grad(lambda p: jaxnu.prob_earth(p, E, cz, MU, E_).sum())(params)
```

## 9. Validation

- Unitarity (`Σ_β P = 1`), vacuum 2-/3-flavor analytic limits, MSW resonance.
- Cross-backend agreement (`cayley` vs `eigh` vs `expm`).
- Chord-length geometry test (`2·R_E·|cz|` for a surface detector).
- Cross-code numeric comparison vs OscProb / NuFast on benchmark points (manual).
- Finite-difference gradient checks.

## 10. Milestones

1. Foundations — constants, PMNS, vacuum H, vacuum probability, unitarity tests.
2. Constant density — matter H, all three backends, MSW + grad checks.
3. Earth — PREM, chord geometry, `lax.scan` propagation.
4. Performance — jit/vmap, benchmarks.
5. Extras — diffrax backend, NSI/sterile hooks, examples.

## 11. Roadmap (later)

**GPU-oriented NuFast-Earth propagator.** The constant-density `nufast` backend
reached parity with NuFast's C++; the Earth path did not, and a cost decomposition
showed the CPU gap is XLA per-op overhead on the sequential scan of tiny 3×3 ops —
not algorithmic — so restructuring buys ≤1.15× on CPU. NuFast-Earth's optimizations
(eigensystem caching across `cosθz`; reduced-basis **real** per-shell amplitudes
factoring θ₂₃/δ out — reuses the Rosetta code in `jaxnu.nufast`; symmetric-trajectory
halving; path-mean density) compound with the existing `vmap` structure and amortize
the overhead **on GPU/TPU**. Trigger: a GPU target. Start with the reduced-basis
real amplitude (highest flop saving, reuses existing code). Until then: for fast CPU
Earth probabilities, call NuFast-Earth directly; jaxnu's Earth value is
differentiability + validated physics.

**Supernova module** — the natural consumer of the diffrax (stiff/implicit)
backend; non-adiabatic / collective oscillations.
