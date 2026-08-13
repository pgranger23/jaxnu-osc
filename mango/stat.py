"""Statistical utilities for differentiable sensitivity studies.

The core object is the **Fisher information matrix** of a binned Poisson
(counting) experiment,

    F_ij = sum_bins  (d mu_b / d theta_i)(d mu_b / d theta_j) / mu_b
           [ + prior terms 1/sigma_k^2 ],

computed *exactly* from the model Jacobian via jaxnu autodiff (no finite
differences). ``F`` equals the expected curvature of the Asimov chi2/2 at the
truth, so ``sqrt(inv(F)_ii)`` is the marginalized 1-sigma forecast for
parameter i and ``1/sqrt(F_ii)`` the bound with every other parameter held
fixed.

Note the terminology: *profiling* means minimizing over the other parameters,
which in the Gaussian limit gives the same answer as *marginalizing* -- i.e.
``sqrt(inv(F)_ii)``, not ``1/sqrt(F_ii)``. The fixed-others number is the
optimistic one, and is never the profiled result.

Works with any parameter PyTree (e.g. :class:`mango.OscParams`, or a dict mixing
oscillation and nuisance parameters) and any model returning a PyTree of
expected bin counts.

Example
-------
>>> model = lambda p: {"app": mango.probability_constant(p, E, L, density=2.8,
...                       flavor_in=Flavor.MU, flavor_out=Flavor.E) * flux_template}
>>> F = mango.stat.fisher_matrix(model, params)
>>> F = F.with_prior("theta13", 0.002)          # external constraint
>>> F.sigma("deltacp")                          # marginalized forecast
>>> F.fixed("dm31").sigma("deltacp")            # with dm31 held fixed
"""

from __future__ import annotations

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten_with_path, keystr


@dataclasses.dataclass
class FisherResult:
    """Fisher matrix with labeled parameters."""

    matrix: jnp.ndarray
    labels: list

    def _idx(self, name):
        matches = [i for i, l in enumerate(self.labels) if name in l]
        if len(matches) != 1:
            raise KeyError(f"parameter {name!r} matches {matches} in {self.labels}")
        return matches[0]

    def covariance(self):
        return jnp.linalg.inv(self.matrix)

    def sigmas(self):
        """Marginalized 1-sigma forecasts for every parameter."""
        s = jnp.sqrt(jnp.diag(self.covariance()))
        return {l: float(v) for l, v in zip(self.labels, s)}

    def sigma(self, name):
        """Marginalized 1-sigma forecast for one parameter."""
        return float(jnp.sqrt(self.covariance()[self._idx(name), self._idx(name)]))

    def sigma_fixed(self, name):
        """1-sigma bound with all other parameters held fixed."""
        return float(1.0 / jnp.sqrt(self.matrix[self._idx(name), self._idx(name)]))

    def with_prior(self, name, sigma):
        """Add a Gaussian prior of width ``sigma`` on ``name``."""
        i = self._idx(name)
        m = self.matrix.at[i, i].add(1.0 / sigma**2)
        return FisherResult(m, self.labels)

    def fixed(self, *names):
        """Remove (fix) parameters: drop their rows/columns."""
        drop = {self._idx(n) for n in names}
        keep = [i for i in range(len(self.labels)) if i not in drop]
        k = jnp.asarray(keep)
        return FisherResult(self.matrix[jnp.ix_(k, k)],
                            [self.labels[i] for i in keep])

    def submatrix(self, *names):
        """Marginalized covariance block for the named parameters (in order)."""
        idx = jnp.asarray([self._idx(n) for n in names])
        return self.covariance()[jnp.ix_(idx, idx)]


def _labels(params):
    leaves, _ = tree_flatten_with_path(params)
    out = []
    for path, leaf in leaves:
        base = keystr(path).lstrip(".") or "x"
        n = np.size(leaf)
        if n == 1:
            out.append(base)
        else:
            out.extend(f"{base}[{i}]" for i in range(n))
    return out


def fisher_matrix(model, params, priors=None, floor=1e-12) -> FisherResult:
    """Poisson Fisher matrix of ``model(params)`` at ``params``.

    ``model`` maps the parameter PyTree to expected bin counts (any PyTree of
    arrays; all leaves are concatenated). ``priors``: optional dict
    ``{label_substring: sigma}`` of Gaussian priors added on the diagonal.
    """
    flat, unravel = ravel_pytree(params)

    def flat_model(x):
        out = model(unravel(x))
        return ravel_pytree(out)[0]

    mu = flat_model(flat)
    J = jax.jacrev(flat_model)(flat)                       # (nbins, npar)
    w = 1.0 / jnp.clip(mu, floor, None)
    F = J.T @ (J * w[:, None])
    res = FisherResult(F, _labels(params))
    for name, sigma in (priors or {}).items():
        res = res.with_prior(name, sigma)
    return res


def poisson_chi2(mu, data):
    """Baker-Cousins Poisson chi2 ``2 sum (mu - n + n ln(n/mu))`` (safe at n=0)."""
    mu = jnp.clip(mu, 1e-12, None)
    n = jnp.clip(data, 0.0, None)
    return 2.0 * jnp.sum(mu - n + jnp.where(n > 0, n * jnp.log(n / mu), 0.0))
