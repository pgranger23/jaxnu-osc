"""Continuous-density backend: direct integration of the Schroedinger equation.

Integrates ``dS/ds = -i H(s) S`` with an adaptive ODE solver.  This handles
*arbitrary smooth* density profiles and serves as an independent cross-check of
the piecewise-constant :mod:`jaxnu.layers` method.  Two backends:

* ``"odeint"`` (default) -- ``jax.experimental.ode.odeint`` (Dopri5); no extra
  dependency.
* ``"diffrax"`` -- the `diffrax <https://docs.kidger.site/diffrax>`_ library
  (optional install): a wider choice of solvers (incl. stiff/implicit), more
  robust checkpointed adjoints for long trajectories, dense output and events.
  Most useful for hard profiles (e.g. supernovae).  Imported lazily so diffrax is
  not required unless requested.

The layer method remains the production path (faster, ``jit``-friendly, exactly
unitary); these ODE backends are the "optional continuous backend" from the
design and are only approximately unitary (RK drifts off the unitary group).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.experimental.ode import odeint

from . import constants as C
from .earth import prem_density_jax
from .hamiltonian import matter_hamiltonian

R_E = C.R_EARTH_KM


def _propagate_odeint(u, msq, energy_eV, v_of_s, total, anti, rtol, atol):
    n = u.shape[0]

    def deriv(s_state, s):  # odeint signature is (y, t)
        h = matter_hamiltonian(u, msq, energy_eV, v_of_s(s), anti=anti)
        return -1j * (h @ s_state)

    s0 = jnp.eye(n, dtype=jnp.complex128)
    ts = jnp.stack([jnp.asarray(0.0), jnp.asarray(total)])
    return odeint(deriv, s0, ts, rtol=rtol, atol=atol)[-1]


def _make_diffrax_solver(solver):
    import diffrax
    if solver is None:
        return diffrax.Tsit5()
    if isinstance(solver, str):
        table = {
            "tsit5": diffrax.Tsit5,
            "dopri5": diffrax.Dopri5,
            "dopri8": diffrax.Dopri8,
            "kvaerno5": diffrax.Kvaerno5,  # implicit, for stiff profiles
        }
        try:
            return table[solver.lower()]()
        except KeyError:
            raise ValueError(
                f"unknown diffrax solver {solver!r}; choose from {sorted(table)}")
    return solver  # assume an instantiated diffrax solver


def _propagate_diffrax(u, msq, energy_eV, v_of_s, total, anti, rtol, atol,
                       solver, max_steps):
    try:
        import diffrax
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "backend='diffrax' requires the optional 'diffrax' package "
            "(pip install diffrax)") from exc

    n = u.shape[0]
    solver = _make_diffrax_solver(solver)

    # Evolve a real (2, N, N) state [Re S, Im S]: diffrax/optimistix complex
    # support is experimental, so we keep the solver, its error norm, and the
    # adjoint on real numbers (the standard recommendation).
    def vf(t, y, args):  # diffrax signature is (t, y, args)
        s = y[0] + 1j * y[1]
        h = matter_hamiltonian(u, msq, energy_eV, v_of_s(t), anti=anti)
        ds = -1j * (h @ s)
        return jnp.stack([ds.real, ds.imag])

    y0 = jnp.stack([jnp.eye(n), jnp.zeros((n, n))])
    total = jnp.asarray(total)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vf), solver, t0=0.0, t1=total, dt0=total * 1e-3, y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        saveat=diffrax.SaveAt(t1=True),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=max_steps,
    )
    yf = sol.ys[-1]
    return yf[0] + 1j * yf[1]


def propagate_continuous(u, msq, energy_eV, v_of_s, total_len_invEV,
                         anti=False, rtol=1e-9, atol=1e-9, backend="odeint",
                         solver=None, max_steps=100_000):
    """Evolution operator through a continuous potential ``v_of_s``.

    ``v_of_s(s)`` returns the matter potential (eV) at natural-unit path position
    ``s`` in ``[0, total_len_invEV]``.  ``backend`` is ``"odeint"`` or
    ``"diffrax"``; ``solver`` (diffrax only) is a solver name (``"tsit5"``,
    ``"dopri8"``, ``"kvaerno5"``, ...) or an instantiated diffrax solver.
    """
    u = jnp.asarray(u, dtype=jnp.complex128)
    if backend == "odeint":
        return _propagate_odeint(u, msq, energy_eV, v_of_s, total_len_invEV,
                                 anti, rtol, atol)
    if backend == "diffrax":
        return _propagate_diffrax(u, msq, energy_eV, v_of_s, total_len_invEV,
                                  anti, rtol, atol, solver, max_steps)
    raise ValueError(f"unknown backend {backend!r}; use 'odeint' or 'diffrax'")


def earth_potential_fn(cz, ye=0.4957, det_depth_km=0.0):
    """Build ``v_of_s`` for an Earth chord at ``cos(zenith) = cz``.

    Returns ``(v_of_s, total_len_invEV)``.  ``ye`` is taken constant here (a
    smooth-profile demonstration); the layered backend uses the per-region
    ``Y_e``.  Radius along the chord: ``r(s) = sqrt(r_min^2 + (s - s_turn)^2)``.
    """
    r_det = R_E - det_depth_km
    cz = jnp.asarray(cz, dtype=jnp.float64)
    r_min = r_det * jnp.sqrt(jnp.clip(1.0 - cz**2, 0.0, None))
    s_turn_km = jnp.sqrt(jnp.clip(R_E**2 - r_min**2, 0.0, None))  # entry->closest
    asc_km = jnp.sqrt(jnp.clip(r_det**2 - r_min**2, 0.0, None))
    total_km = s_turn_km + asc_km

    def v_of_s(s):
        s_km = s / C.KM_TO_INV_EV
        r = jnp.sqrt(r_min**2 + (s_km - s_turn_km) ** 2)
        rho = prem_density_jax(r)
        return C.matter_potential_eV(rho, ye)

    return v_of_s, total_km * C.KM_TO_INV_EV


def probability_earth_continuous(params, energy_GeV, cz, ye=0.4957,
                                 det_depth_km=0.0, anti=False, **kw):
    """``P[beta, alpha]`` through the Earth via the continuous ODE backend.

    Extra keywords are forwarded to :func:`propagate_continuous`, e.g.
    ``backend="diffrax"``, ``solver="dopri8"``, ``rtol=``, ``atol=``.
    """
    u = params.pmns()
    msq = params.msquared()
    v_of_s, total = earth_potential_fn(cz, ye=ye, det_depth_km=det_depth_km)
    s = propagate_continuous(u, msq, jnp.asarray(energy_GeV) * C.GEV_TO_EV,
                             v_of_s, total, anti=anti, **kw)
    return jnp.abs(s) ** 2
