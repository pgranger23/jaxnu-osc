"""NuFast-style direct constant-density probability (3-flavor, standard).

Ports the algorithm of Parke & Denton's NuFast-LBL (the DMP "Rosetta" relations):
the matter mixing-matrix magnitudes ``|U^m_{alpha i}|^2`` and the matter Jarlskog
are obtained *analytically* from the matter eigenvalues, and plugged straight into
the standard ``sin^2`` oscillation formula.  No eigenvectors, no complex matrix
exponential, no matmuls -- just scalar algebra, so it is several times faster than
the matrix-exponential backends while remaining exact and differentiable.

Single constant-density layer only (it returns probabilities, not the evolution
operator, so it cannot be chained across layers) and standard 3-flavor (no NSI /
steriles).  Use the matrix-exponential backends for those.
"""

from __future__ import annotations

import jax.numpy as jnp

_TWO_PI = 2.0 * jnp.pi


def prob_matrix(params, energy_eV, length_invEV, v_cc_eV, anti=False):
    """``P[beta, alpha] = P(nu_alpha -> nu_beta)`` (3x3) at one energy.

    Natural units: ``energy_eV`` in eV, ``length_invEV`` in eV^-1, ``v_cc_eV`` the
    charged-current potential in eV.
    """
    s12sq = jnp.sin(params.theta12) ** 2
    s13sq = jnp.sin(params.theta13) ** 2
    s23sq = jnp.sin(params.theta23) ** 2
    delta = params.deltacp
    dm21 = params.dm21
    dm31 = params.dm31
    if anti:
        delta = -delta
        v_cc_eV = -v_cc_eV

    c13sq = 1 - s13sq
    Ue2sq_t = c13sq * s12sq
    Ue3sq_t = s13sq
    Um3sq_t = c13sq * s23sq
    Ut2sq_t = s13sq * s12sq * s23sq
    Um2sq_t = (1 - s12sq) * (1 - s23sq)
    Jrr = jnp.sqrt(Um2sq_t * Ut2sq_t)
    sind, cosd = jnp.sin(delta), jnp.cos(delta)
    Um2sq_t = Um2sq_t + Ut2sq_t - 2 * Jrr * cosd
    Jmatter_t = 8 * Jrr * c13sq * sind
    A_tmp = dm21 + dm31
    See = A_tmp - dm21 * Ue2sq_t - dm31 * Ue3sq_t
    Tmm_tmp = dm21 * dm31
    Tee = Tmm_tmp * (1 - Ue3sq_t - Ue2sq_t)

    Amatter = 2.0 * energy_eV * v_cc_eV
    Cc = Amatter * Tee
    A = A_tmp + Amatter
    B = Tmm_tmp + Amatter * See

    # lambda3 from the exact cubic, then the other eigenvalue gaps analytically.
    rootAsqB = jnp.sqrt(A * A - 3.0 * B)
    arg = (A**3 - 4.5 * A * B + 13.5 * Cc) / rootAsqB**3
    ss0 = jnp.arccos(jnp.clip(arg, -1.0, 1.0))
    ss0 = jnp.where(dm31 < 0.0, ss0 + _TWO_PI, ss0)
    lambda3 = (A + 2.0 * rootAsqB * jnp.cos(ss0 / 3.0)) / 3.0

    tmp = A - lambda3
    Dl21 = jnp.sqrt(tmp * tmp - 4.0 * Cc / lambda3)
    lambda2 = 0.5 * (A - lambda3 + Dl21)
    Dl32 = lambda3 - lambda2
    Dl31 = Dl32 + Dl21
    PiInv = 1.0 / (Dl31 * Dl32 * Dl21)
    Xp3 = PiInv * Dl21
    Xp2 = -PiInv * Dl31

    # Rosetta: matter mixing magnitudes from eigenvalues + invariants.
    Ue3sq = (lambda3 * (lambda3 - See) + Tee) * Xp3
    Ue2sq = (lambda2 * (lambda2 - See) + Tee) * Xp2
    Smm = A - dm21 * Um2sq_t - dm31 * Um3sq_t
    Tmm = Tmm_tmp * (1 - Um3sq_t - Um2sq_t) + Amatter * (See + Smm - A)
    Um3sq = (lambda3 * (lambda3 - Smm) + Tmm) * Xp3
    Um2sq = (lambda2 * (lambda2 - Smm) + Tmm) * Xp2
    Jmatter = Jmatter_t * dm21 * dm31 * (dm31 - dm21) * PiInv

    Ue1sq = 1 - Ue3sq - Ue2sq
    Um1sq = 1 - Um3sq - Um2sq
    Ut3sq = 1 - Um3sq - Ue3sq
    Ut2sq = 1 - Um2sq - Ue2sq
    Ut1sq = 1 - Um1sq - Ue1sq

    Lover4E = length_invEV / (4.0 * energy_eV)
    D21 = Dl21 * Lover4E
    D32 = Dl32 * Lover4E
    sinD21 = jnp.sin(D21)
    sinD31 = jnp.sin(D32 + D21)
    sinD32 = jnp.sin(D32)
    triple = sinD21 * sinD31 * sinD32
    s21 = 2 * sinD21**2
    s31 = 2 * sinD31**2
    s32 = 2 * sinD32**2

    Pme_TC = ((Ut3sq - Um2sq * Ue1sq - Um1sq * Ue2sq) * s21
              + (Ut2sq - Um3sq * Ue1sq - Um1sq * Ue3sq) * s31
              + (Ut1sq - Um3sq * Ue2sq - Um2sq * Ue3sq) * s32)
    Pme_TV = -Jmatter * triple
    Pmm = 1 - 2 * (Um2sq * Um1sq * s21 + Um3sq * Um1sq * s31 + Um3sq * Um2sq * s32)
    Pee = 1 - 2 * (Ue2sq * Ue1sq * s21 + Ue3sq * Ue1sq * s31 + Ue3sq * Ue2sq * s32)
    Pem = Pme_TC - Pme_TV
    Pme = Pme_TC + Pme_TV
    Pet = 1 - Pee - Pem
    Pmt = 1 - Pme - Pmm
    Pte = 1 - Pee - Pme
    Ptm = 1 - Pem - Pmm
    Ptt = 1 - Pet - Pmt

    # NuFast indexes probs[alpha][beta] = P(alpha -> beta); transpose to P[beta, alpha].
    probs = jnp.array([[Pee, Pem, Pet], [Pme, Pmm, Pmt], [Pte, Ptm, Ptt]])
    return probs.T


def eligible(params, nsi, n_active):
    """True if the NuFast direct formula applies (standard 3-flavor, no NSI)."""
    return params.msquared().shape[0] == 3 and nsi is None and n_active == 3
