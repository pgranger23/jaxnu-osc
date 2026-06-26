"""Piecewise-constant layer propagation.

Given per-segment matter potentials and (natural-unit) lengths, build each
segment evolution operator and chain them ``S = S_N ... S_1`` so that ``S`` acts
on an initial flavor state.  Two product strategies, auto-selected by device:

* ``parallel=False`` (CPU default): a sequential ``lax.scan``.  Minimal total
  work and memory -- fastest on CPU, where the parallel variant has no hardware
  to exploit.
* ``parallel=True`` (GPU/TPU default): build all per-layer propagators with
  ``vmap`` and combine with ``lax.associative_scan`` (parallel prefix, O(log N)
  depth), with the layer-independent vacuum Hamiltonian hoisted out.  Faster on
  accelerators; slower on CPU (it computes all prefixes).

Reverse-mode autodiff flows through both paths.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .hamiltonian import vacuum_hamiltonian, matter_hamiltonian, matter_term
from .propagator import propagator


def propagate_layers(u, msq, energy_eV, v_eV, length_invEV, anti=False,
                     backend="cayley", nsi=None, v_nc=None, n_active=None,
                     parallel=None):
    """Total evolution operator through a stack of constant-density segments.

    Parameters
    ----------
    v_eV, length_invEV : array_like, shape (n_seg,)
        Charged-current potential (eV) and length (eV^-1) of each segment, ordered
        from the *source* to the *detector* (segment 0 is applied first).
    v_nc : array_like, shape (n_seg,), optional
        Per-segment sterile-sector potential; defaults to zero (purely active).
    n_active : int, optional
        Number of active flavors (see :func:`jaxnu.hamiltonian.matter_hamiltonian`).
    parallel : bool, optional
        Product strategy (see module docstring).  Defaults to ``False`` on CPU and
        ``True`` on GPU/TPU.
    Returns the ``(N, N)`` evolution operator ``S``.
    """
    u = jnp.asarray(u, dtype=jnp.complex128)
    n = u.shape[0]
    if n_active is None:
        n_active = n
    if parallel is None:
        parallel = jax.default_backend() != "cpu"
    v_eV = jnp.asarray(v_eV)
    length_invEV = jnp.asarray(length_invEV)
    v_nc = jnp.zeros_like(v_eV) if v_nc is None else jnp.asarray(v_nc)

    if not parallel:
        def step(carry, seg):
            v_k, vnc_k, l_k = seg
            h = matter_hamiltonian(u, msq, energy_eV, v_k, anti=anti, nsi=nsi,
                                   v_nc_eV=vnc_k, n_active=n_active)
            return propagator(h, l_k, backend=backend) @ carry, None

        s0 = jnp.eye(n, dtype=jnp.complex128)
        s, _ = jax.lax.scan(step, s0, (v_eV, v_nc, length_invEV))
        return s

    # parallel (GPU/TPU): vmap-build with hoisted vacuum part + associative scan.
    if anti:
        u_eff = jnp.conj(u)
        v_eV, v_nc = -v_eV, -v_nc
        nsi_eff = jnp.conj(jnp.asarray(nsi)) if nsi is not None else None
    else:
        u_eff, nsi_eff = u, nsi
    h_vac = vacuum_hamiltonian(u_eff, msq, energy_eV)

    def make_s(v_k, vnc_k, l_k):
        h = h_vac + matter_term(n, v_k, vnc_k, nsi_eff, n_active)
        return propagator(h, l_k, backend=backend)

    s_all = jax.vmap(make_s)(v_eV, v_nc, length_invEV)
    prods = jax.lax.associative_scan(lambda a, b: jnp.matmul(b, a), s_all)
    return prods[-1]
