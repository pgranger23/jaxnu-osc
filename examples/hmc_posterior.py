"""Bayesian posterior of the oscillation parameters via HMC (NUTS/blackjax).

The payoff of a differentiable oscillation model: the log-posterior gradient is
exact and cheap, so Hamiltonian Monte Carlo explores the (theta23, theta13,
dm31, deltacp) posterior of a DUNE-like counting experiment directly — no
random-walk tuning, no finite differences.

Model: nu_e appearance + nu_mu disappearance spectra in nu and nubar beam modes
at the DUNE baseline (constant density), Asimov data at NuFIT-4.0-like truth
with deltacp = -pi/2, Poisson likelihood, Gaussian prior on theta13 (reactor).
The HMC posterior is compared against the analytic Fisher forecast from
``jaxnu.stat`` — the two must (and do) agree in the Gaussian regime.

Requires: pip install blackjax
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import blackjax

import jaxnu
from jaxnu import OscParams, Flavor, probability_constant, stat

# --- DUNE-like counting model ------------------------------------------------
E = jnp.linspace(0.75, 6.0, 40)
FLUX = 4e3 * E**2 * jnp.exp(-E / 1.0)          # numu flux x xsec template
L, RHO = 1300.0, 2.8

TRUTH = dict(theta23=0.866, theta13=0.150, dm31=2.451e-3 + 7.39e-5,
             deltacp=-np.pi / 2)
TH13_PRIOR = 0.150 * 0.015                      # reactor constraint (1.5%)


def params_of(theta23, theta13, dm31, deltacp):
    return OscParams(theta12=jnp.asarray(0.5903), theta13=theta13,
                     theta23=theta23, deltacp=deltacp,
                     dm21=jnp.asarray(7.39e-5), dm31=dm31)


def model(p):
    out = {}
    for anti, w in ((False, 1.0), (True, 0.5)):
        Pme = probability_constant(p, E, L, density=RHO, anti=anti,
                                   flavor_in=Flavor.MU, flavor_out=Flavor.E)
        Pmm = probability_constant(p, E, L, density=RHO, anti=anti,
                                   flavor_in=Flavor.MU, flavor_out=Flavor.MU)
        out[f"app_{int(anti)}"] = w * FLUX * Pme
        out[f"dis_{int(anti)}"] = 0.3 * w * FLUX * Pmm
    return out


DATA = model(params_of(**{k: jnp.asarray(v) for k, v in TRUTH.items()}))
print("Asimov event totals:", {k: round(float(v.sum())) for k, v in DATA.items()})


def logpost(x):
    th23, th13, dm31, dcp = x
    mu = model(params_of(th23, th13, dm31, dcp))
    ll = 0.0
    for k, m in mu.items():
        m = jnp.clip(m, 1e-9, None)
        ll = ll + jnp.sum(DATA[k] * jnp.log(m) - m)
    ll = ll - 0.5 * ((th13 - TRUTH["theta13"]) / TH13_PRIOR) ** 2
    return ll


# --- NUTS via blackjax -------------------------------------------------------
x0 = jnp.asarray([TRUTH["theta23"], TRUTH["theta13"], TRUTH["dm31"],
                  TRUTH["deltacp"]])
rng = jax.random.PRNGKey(0)
t0 = time.time()
warm = blackjax.window_adaptation(blackjax.nuts, logpost)
(state, tuned), _ = warm.run(rng, x0, num_steps=500)
kernel = blackjax.nuts(logpost, **tuned).step

@jax.jit
def step(carry, key):
    st, = carry
    st, info = kernel(key, st)
    return (st,), st.position

keys = jax.random.split(jax.random.PRNGKey(1), 3000)
(_, ), samples = jax.lax.scan(step, (state,), keys)
samples = np.asarray(samples)
print(f"NUTS: 500 warmup + 3000 samples in {time.time()-t0:.0f}s")

# --- Fisher forecast for comparison -----------------------------------------
p0 = params_of(**{k: jnp.asarray(v) for k, v in TRUTH.items()})
F = stat.fisher_matrix(model, p0).with_prior("theta13", TH13_PRIOR)
F = F.fixed("theta12", "dm21")               # fixed in the sampling too
names = ["theta23", "theta13", "dm31", "deltacp"]
fisher_sig = {n: F.sigma(n) for n in names}
mcmc_sig = {n: samples[:, i].std() for i, n in enumerate(names)}
print(f"{'param':10s} {'Fisher':>12s} {'HMC':>12s}")
for n in names:
    print(f"{n:10s} {fisher_sig[n]:12.3e} {mcmc_sig[n]:12.3e}")

# --- plot --------------------------------------------------------------------
def ellipse(ax, cov, mean, n_sig, **kw):
    th = np.linspace(0, 2 * np.pi, 200)
    vals, vecs = np.linalg.eigh(np.asarray(cov))
    xy = (vecs * np.sqrt(np.maximum(vals, 0)) * n_sig) @ np.vstack(
        [np.cos(th), np.sin(th)])
    ax.plot(mean[0] + xy[0], mean[1] + xy[1], **kw)

pairs = [("deltacp", "theta23"), ("deltacp", "theta13"), ("theta23", "dm31")]
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=130)
for ax, (a, b) in zip(axs, pairs):
    ia, ib = names.index(a), names.index(b)
    ax.plot(samples[::3, ia], samples[::3, ib], ".", ms=1.5, alpha=0.25,
            color="C0", label="NUTS samples")
    cov = F.submatrix(a, b)
    mean = (TRUTH[a], TRUTH[b])
    for ns, ls in ((1, "-"), (2, "--")):
        ellipse(ax, cov, mean, ns, color="C3", lw=1.6, ls=ls,
                label=f"Fisher {ns}$\\sigma$" if ns == 1 else None)
    ax.plot(*mean, "k*", ms=10, label="truth")
    ax.set_xlabel(a); ax.set_ylabel(b)
    ax.legend(fontsize=8, loc="upper right")
fig.suptitle("HMC posterior (blackjax NUTS, exact jaxnu gradients) vs the "
             "analytic Fisher forecast (jaxnu.stat)", fontsize=12)
fig.tight_layout()
out = Path(__file__).resolve().parent / "hmc_posterior.jpg"
fig.savefig(out)
print("saved", out)
