"""Shared setup for the nu-waves reference-plot reproductions.

Parameters match the nu-waves examples (PDG-2025-like): theta12=33.4 deg,
theta13=8.6 deg, theta23=49 deg, delta=195 deg, dm21=7.42e-5,
dm32=0.0024428 eV^2 (so dm31=0.0025170, normal ordering).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)
DATADIR = ROOT / "examples" / "data"

import jax.numpy as jnp  # noqa: E402
import jaxnu  # noqa: E402
from jaxnu import OscParams  # noqa: E402

DM21 = 7.42e-5
DM32 = 0.0024428
DM31_NO = DM21 + DM32  # 0.0025170


def nuwaves_params(ordering="NO"):
    dm31 = DM31_NO if ordering == "NO" else -DM32
    return OscParams(
        theta12=jnp.asarray(np.deg2rad(33.4)),
        theta13=jnp.asarray(np.deg2rad(8.6)),
        theta23=jnp.asarray(np.deg2rad(49.0)),
        deltacp=jnp.asarray(np.deg2rad(195.0)),
        dm21=jnp.asarray(DM21),
        dm31=jnp.asarray(dm31),
    )
