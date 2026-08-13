"""Example: evaluating the DUNE sensitivity with a custom analysis binning.

This is the extension point for a binning-optimization study. The full forward model
and chi2 accept an arbitrary rebinning matrix ``W`` (K analysis bins x nreco fine
bins), and ``W`` is differentiable in the bin edges -- so an optimizer can maximize
the sensitivity (or a Fisher-information objective) with respect to the edges.

Here we simply compare uniform coarse binnings against the native fine binning.
"""
import numpy as np
import dune

dcp_grid = np.array([-0.5, -0.25, 0.25, 0.5])  # representative true delta_CP / pi

print("CP-violation sigma = sqrt(dChi2) at true delta_CP/pi:")
print("  binning                " + "  ".join(f"{x:+.2f}" for x in dcp_grid))

# native fine binning (the DUNE reference analysis)
fine = dune.cpv_sensitivity(dcp_grid, W=dune.fine_W())
print("  fine (native)          " + "  ".join(f"{v:5.2f}" for v in fine))

# uniform coarse binnings with K bins over 0.5-8 GeV
for K in (8, 12, 20):
    edges = np.linspace(0.5, 8.0, K + 1)
    W = dune.rebin_matrix(edges)
    s = dune.cpv_sensitivity(dcp_grid, W=W)
    print(f"  uniform K={K:<2d}            " + "  ".join(f"{v:5.2f}" for v in s))

print("""
--- extension hook for an optimizer -------------------------------------------
  import jax, jax.numpy as jnp, dune
  # differentiable model at the fine bins:
  comps = dune.rule_components(dune.params_from(vec), rho)   # per-component spectra
  # or the total:  dune.total_spectra(params)
  # build a (differentiable) binning and evaluate / optimize:
  W = dune.rebin_matrix(edges)                 # dW/d(edges) exists
  chi2, grad = dune.chi2_vg(free, data_tuple, W)   # exact gradient from mango
  # a Fisher objective for delta_CP:  F = sum_bins (W@dmu/dtheta)(...)^T / (W@mu)
  # then gradient-ascend the sensitivity / Fisher w.r.t. `edges`.
-------------------------------------------------------------------------------""")
