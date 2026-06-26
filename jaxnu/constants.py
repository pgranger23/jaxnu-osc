"""Physical constants and unit conversions.

The numerical core of :mod:`jaxnu` works entirely in **natural units**
(``hbar = c = 1``): energies in eV, lengths in eV^-1, masses^2 in eV^2.  Unit
conversions for the user-facing API (GeV, km, g/cm^3) live here so that no magic
factors leak into the physics modules.

All values are derived from first principles (CODATA / PDG) so the matter
potential is not an opaque "7.6e-14" literal.
"""

from __future__ import annotations

# --- fundamental constants ---------------------------------------------------
# Fermi coupling constant, G_F / (hbar c)^3 = 1.1663787e-5 GeV^-2.
GF_EV = 1.1663787e-23  # eV^-2
NA = 6.02214076e23  # Avogadro's number, mol^-1

# hbar * c = 197.3269804 MeV * fm  =>  1 m = 5.067730716e6 eV^-1.
HBARC_MEV_FM = 197.3269804
# hbar c in eV * m = (MeV->eV) * (fm->m):
HBARC_EV_M = HBARC_MEV_FM * 1.0e6 * 1.0e-15  # ~1.973269804e-7 eV*m
M_TO_INV_EV = 1.0 / HBARC_EV_M  # eV^-1 per metre (~5.06773e6)

# --- length conversions (to eV^-1) ------------------------------------------
KM_TO_INV_EV = M_TO_INV_EV * 1.0e3  # eV^-1 per km   (~5.06773e9)
CM_TO_INV_EV = M_TO_INV_EV * 1.0e-2  # eV^-1 per cm  (~5.06773e4)

# --- energy conversions (to eV) ---------------------------------------------
GEV_TO_EV = 1.0e9

# --- matter-potential prefactor ---------------------------------------------
# Electron number density N_e [eV^3] = rho[g/cm^3] * Y_e * NA * (1 cm)^-3,
# with (1 cm)^-3 expressed in eV^3 via CM_TO_INV_EV.
# Charged-current potential  V = sqrt(2) G_F N_e  [eV].
import math as _math

# N_e [eV^3] per (g/cm^3 * Y_e):
_NE_PER_RHO_YE = NA / CM_TO_INV_EV**3  # ~4.627e9 eV^3
# V [eV] per (g/cm^3 * Y_e):
V_FACTOR_EV = _math.sqrt(2.0) * GF_EV * _NE_PER_RHO_YE  # ~7.63e-14 eV

# --- Earth geometry ----------------------------------------------------------
R_EARTH_KM = 6371.0  # mean Earth radius

__all__ = [
    "GF_EV",
    "NA",
    "KM_TO_INV_EV",
    "CM_TO_INV_EV",
    "GEV_TO_EV",
    "V_FACTOR_EV",
    "R_EARTH_KM",
    "matter_potential_eV",
]


def matter_potential_eV(rho_gcc, ye):
    """Charged-current matter potential ``V = sqrt(2) G_F N_e`` in eV.

    Parameters
    ----------
    rho_gcc : array_like
        Matter mass density in g/cm^3.
    ye : array_like
        Electron fraction ``Y_e = N_e / (N_p + N_n)`` (~0.5 for the Earth).
    """
    return V_FACTOR_EV * rho_gcc * ye


def matter_potentials(rho_gcc, ye):
    """Charged-current and sterile-sector potentials ``(V_CC, V_NC)`` in eV.

    ``V_CC = sqrt(2) G_F N_e`` acts on nu_e.  Sterile neutrinos feel neither the
    CC nor the neutral-current potential; after subtracting the (flavor-universal)
    active NC term, the sterile diagonal entry of the matter Hamiltonian is
    ``V_NC = (1/2) sqrt(2) G_F N_n`` with ``N_n = rho N_A (1 - Y_e)``.  Returns 0
    for ``V_NC`` when ``Y_e = 1`` (no neutrons).
    """
    v_cc = V_FACTOR_EV * rho_gcc * ye
    v_nc = 0.5 * V_FACTOR_EV * rho_gcc * (1.0 - ye)
    return v_cc, v_nc
