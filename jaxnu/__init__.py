"""jaxnu — a differentiable neutrino oscillation calculator in JAX.

Float64 is required for oscillation phases (Delta m^2 ~ 1e-3 eV^2) and is enabled
on import.
"""

from __future__ import annotations

import jax as _jax

_jax.config.update("jax_enable_x64", True)

from . import constants  # noqa: E402
from .params import OscParams, nufit_no  # noqa: E402
from . import earth  # noqa: E402
from . import nufast as nufast  # noqa: E402  (fast analytic constant-density)
from . import solar  # noqa: E402  (solar profile + adiabatic MSW)
from . import ode  # noqa: E402  (continuous-density backend)
from . import nsi as nsi  # noqa: E402  (non-standard interactions)
from . import sterile as sterile  # noqa: E402  (3+N sterile front-end)
from .nsi import NSI  # noqa: E402
from .sterile import (  # noqa: E402
    Sterile3plus1,
    NFlavorParams,
    pmns_3plus1,
    pmns_nflavor,
)
from .oscillator import (  # noqa: E402
    Flavor,
    n_flavors,
    probability_constant,
    probability_vacuum,
    probability_profile,
    probability_earth,
    select,
)

__all__ = [
    "OscParams",
    "nufit_no",
    "Flavor",
    "n_flavors",
    "probability_constant",
    "probability_vacuum",
    "probability_profile",
    "probability_earth",
    "select",
    "NSI",
    "Sterile3plus1",
    "NFlavorParams",
    "pmns_3plus1",
    "pmns_nflavor",
    "constants",
    "earth",
    "nufast",
    "solar",
    "ode",
    "nsi",
    "sterile",
]

__version__ = "0.1.0"
