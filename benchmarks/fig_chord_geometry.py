"""jaxnu code paper: schematic of the chord construction.

The chord geometry is what makes the geometry derivatives possible at all:
cos(theta_z), the production height and the detector depth enter oscillation
physics only through this construction and the shell crossings it fixes, never
through the Hamiltonian. Three later sections lean on the picture (the
geometry-derivative figure, the core-crossing threshold in the tomography
example, and the shell-grazing cusps in the limitations), so it is worth
drawing once.

Schematic only: no jaxnu call, no data. Radii are to PREM scale, but h_atm and
the detector depth are exaggerated by ~50x to be visible at all.

Run:  python benchmarks/fig_chord_geometry.py
Artefacts go to benchmarks/output/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Arc

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTDIR, exist_ok=True)

R_E = 1.0                    # Earth radius, normalized
R_OUTER_CORE = 3480.0 / 6371.0
R_INNER_CORE = 1221.5 / 6371.0
H_ATM = 0.11                 # exaggerated
D_DET = 0.05                 # exaggerated

COS_TZ = -0.90               # a core-crossing trajectory
r_det = R_E - D_DET
r_prod = R_E + H_ATM
r_min = r_det * np.sqrt(1.0 - COS_TZ ** 2)


def x_at(r, sign):
    """Half-chord abscissa where the chord of impact parameter r_min meets r."""
    return sign * np.sqrt(max(r ** 2 - r_min ** 2, 0.0))


xP, xD = x_at(r_prod, -1), x_at(r_det, +1)
xS_in, xS_out = x_at(R_E, -1), x_at(R_E, +1)          # surface crossings
xC_in, xC_out = x_at(R_OUTER_CORE, -1), x_at(R_OUTER_CORE, +1)

plt.rcParams.update({"font.size": 19, "mathtext.fontset": "cm"})
fig, ax = plt.subplots(figsize=(7.2, 6.0))

for r, fc in ((R_E, "#e6ecf2"), (R_OUTER_CORE, "#c3d0de"),
              (R_INNER_CORE, "#9fb2c6")):
    ax.add_patch(Circle((0, 0), r, facecolor=fc, edgecolor="0.35", lw=1.1,
                        zorder=1))
ax.add_patch(Circle((0, 0), r_prod, facecolor="none", edgecolor="0.6",
                    lw=0.8, ls=":", zorder=1))

# the chord, coloured by the medium each leg traverses
seg = [((xP, xS_in), "0.45", ":", "atmosphere"),
       ((xS_in, xC_in), "tab:red", "-", "mantle"),
       ((xC_in, xC_out), "tab:orange", "-", "outer core"),
       ((xC_out, xD), "tab:red", "-", None)]
for (x0, x1), col, ls, lab in seg:
    ax.plot([x0, x1], [r_min, r_min], color=col, ls=ls, lw=3.0, zorder=4,
            label=lab, solid_capstyle="butt")

# closest approach and the impact parameter
ax.plot([0, 0], [0, r_min], color="0.25", ls="--", lw=1.1, zorder=4)
ax.plot([0], [r_min], "o", color="0.15", ms=5, zorder=6)
ax.plot([0], [0], "o", color="0.15", ms=4, zorder=6)
ax.annotate(r"$r_{\min}$", xy=(0.02, r_min / 2), fontsize=20, ha="left",
            va="center")
ax.annotate("$O$", xy=(0.03, -0.085), fontsize=19, ha="left", va="center")
ax.annotate("$C$", xy=(-0.02, r_min + 0.055), fontsize=19, ha="right",
            va="center")

# endpoints, labelled clear of the chord
ax.plot([xP], [r_min], "o", color="0.15", ms=5, zorder=6)
ax.plot([xD], [r_min], "o", color="0.15", ms=5, zorder=6)
ax.annotate("$P$", xy=(xP - 0.02, r_min + 0.12), fontsize=19, ha="center")
ax.annotate("$D$", xy=(xD + 0.10, r_min - 0.02), fontsize=19, ha="left", va="top")

# the local vertical at D, and the zenith angle between it and the chord
ux, uy = xD / r_det, r_min / r_det
ax.plot([xD, xD + 0.30 * ux], [r_min, r_min + 0.30 * uy], color="0.35",
        lw=1.0, zorder=4)
ang_vert = np.degrees(np.arctan2(uy, ux))
_R_ARC = 0.26
ax.add_patch(Arc((xD, r_min), _R_ARC, _R_ARC, angle=0, theta1=ang_vert,
                 theta2=180.0, color="0.25", lw=1.2, zorder=5))
_mid = np.radians(0.5 * (ang_vert + 180.0))
ax.annotate(r"$\theta_z$",
            xy=(xD + 1.30 * _R_ARC * np.cos(_mid),
                r_min + 1.30 * _R_ARC * np.sin(_mid)),
            fontsize=20, ha="center", va="center")

# exaggerated altitude and depth, annotated away from the chord
ax.annotate("", xy=(xP, r_min), xytext=(xS_in, r_min),
            arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.0))
ax.annotate(r"$h_{\rm atm}$", xy=((xP + xS_in) / 2 - 0.05, r_min - 0.21),
            fontsize=18, ha="center", color="0.25")
ax.annotate("", xy=(xD, r_min), xytext=(xS_out, r_min),
            arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.0))
ax.annotate("$d$", xy=((xD + xS_out) / 2, r_min - 0.21), fontsize=18,
            ha="center", color="0.25")

ax.set_xlim(-1.30, 1.42)
ax.set_ylim(-1.20, 1.30)
ax.set_aspect("equal")
ax.axis("off")
ax.legend(loc="lower center", frameon=False, ncol=3, fontsize=16,
          handlelength=1.6, columnspacing=1.4, bbox_to_anchor=(0.5, -0.02))

fp = os.path.join(OUTDIR, "jaxnu_chord_geometry")
fig.savefig(fp + ".png", dpi=200, bbox_inches="tight")
fig.savefig(fp + ".pdf", bbox_inches="tight")
print("saved", fp + ".pdf")
print(f"  cos(theta_z) = {COS_TZ}, r_min/R_E = {r_min:.3f}, "
      f"outer core at {R_OUTER_CORE:.3f} (chord crosses the core)")
