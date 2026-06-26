# Reproducing the nu-waves reference plots with jaxnu

The [nu-waves](https://github.com/nadrino/nu-waves) library shows six reference
figures in its README. These scripts reproduce them with `jaxnu` using the same
physics inputs (θ12=33.4°, θ13=8.6°, θ23=49°, δ=195°, Δm²₂₁=7.42e-5,
Δm²₃₂=0.0024428 eV², normal ordering). Run any script directly:

```bash
python examples/nuwaves/vacuum_pmns.py        # -> figures/vacuum_pmns.jpg
python examples/nuwaves/matter_constant.py    # DUNE NO vs IO
python examples/nuwaves/vacuum_2flavors.py    # eV² sterile + energy smearing
python examples/nuwaves/vacuum_2d_pmns.py     # P over (E, L) grid
python examples/nuwaves/matter_prem.py        # 4-panel atmospheric oscillogram (~30 s)
python examples/nuwaves/adiabatic_sun_ssm.py  # solar adiabatic MSW
```

| plot | script | result |
|------|--------|--------|
| `vacuum_pmns` | `vacuum_pmns.py` | exact match |
| `matter_constant_test` | `matter_constant.py` | exact match |
| `vacuum_2flavors` | `vacuum_2flavors.py` | exact match (incl. 10% energy smearing) |
| `vacuum_2d_pmns` | `vacuum_2d_pmns.py` | exact match |
| `matter_prem_test` | `matter_prem.py` | match (MSW resonance, ν/ν̄ asymmetry, core-mantle, down-going atmosphere) |
| `adiabatic_sun_ssm_test` | `adiabatic_sun_ssm.py` | correct MSW physics (see note) |

## Features exercised (some added for these reproductions)

- **Configurable `Y_e`** (`probability_earth(..., ye_core=, ye_mantle=)`).
- **Atmospheric production height** (`probability_earth(..., h_atm_km=15.0)`,
  `earth.chord_segments`) — full `cos θ_z ∈ [-1, 1]` including down-going /
  near-horizon vacuum baselines, from the single unified Earth entrypoint.
- **Generic N-flavor core** — `vacuum_2flavors.py` builds a 2-flavor (sterile)
  system directly from `jaxnu.pmns` / `hamiltonian` / `propagator`.
- **Energy smearing** — Gauss-Hermite quadrature over a Gaussian energy resolution.
- **Solar adiabatic MSW** (`jaxnu.solar`) — BS05 SSM profile loader + averaged
  adiabatic vacuum mass-state fractions.

## Note on the solar plot

nu-waves' `adiabatic_sun_ssm.py` calls methods
(`adiabatic_mass_fractions_from_emission`, `make_torch_backend`) that are not
present in the published source, so its exact convention can't be matched. jaxnu
computes the **textbook averaged-adiabatic** mass-state fractions
`F_i(r) = Σ_k |<ν_i^vac|ν_k^m(r)>|² · |<ν_k^m(r_emit)|ν_e>|²`. This reproduces the
key physics — the ν_e emerges predominantly as ν₂ (the LMA-MSW solution) and
ν₃ ≈ sin²θ₁₃ throughout — matching the reference's direction and end state; the
reference's steeper starting split reflects its (single-eigenstate, θ₁₃=0)
convention.

Reference figures (for visual comparison) live under nu-waves'
[`figures/`](https://github.com/nadrino/nu-waves/tree/main/figures).
