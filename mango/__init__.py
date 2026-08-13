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
from .earth import LayeredEarth, prem_layered  # noqa: E402
from . import globes as globes  # noqa: E402  (generic GLoBES/AEDL loader)
from . import nufast as nufast  # noqa: E402  (fast analytic constant-density)
from . import solar  # noqa: E402  (solar profile + adiabatic MSW)
from . import ode  # noqa: E402  (continuous-density backend)
from . import nsi as nsi  # noqa: E402  (non-standard interactions)
from . import sterile as sterile  # noqa: E402  (3+N sterile front-end)
from . import decoherence as decoherence  # noqa: E402  (Lindblad / wave packets)
from . import nonunitarity as nonunitarity  # noqa: E402  (alpha parametrization)
from . import stat as stat  # noqa: E402  (Fisher information & chi2 utilities)
from .decoherence import Decoherence, WavePacket  # noqa: E402
from .nonunitarity import NonUnitarity  # noqa: E402
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
    "Decoherence",
    "WavePacket",
    "NonUnitarity",
    "decoherence",
    "nonunitarity",
    "stat",
    "Sterile3plus1",
    "NFlavorParams",
    "pmns_3plus1",
    "pmns_nflavor",
    "LayeredEarth",
    "prem_layered",
    "constants",
    "earth",
    "globes",
    "nufast",
    "solar",
    "ode",
    "nsi",
    "sterile",
]

__version__ = "0.2.4"
