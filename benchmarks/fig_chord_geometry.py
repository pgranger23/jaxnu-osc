"""jaxnu code paper: the chord construction, drawn over the real shell table.

The chord geometry is what makes the geometry derivatives possible at all:
cos(theta_z), the production height and the detector depth enter oscillation
physics only through this construction and the shell crossings it fixes, never
through the Hamiltonian. Three later sections lean on the picture (the
geometry-derivative figure, the core-crossing threshold in the tomography
example, and the shell-grazing cusps in the limitations), so it is worth
drawing once.

The shells and their densities are the ACTUAL ``mango.earth.shell_table(4)``
used elsewhere in the paper (43 shells), not a three-zone cartoon: the point
that the propagation is an ordered product over many constant-density segments
is easier to see than to assert. The chord is coloured segment by segment with
the same density scale, so the segmentation the code performs is visible
directly -- at cos(theta_z) = -0.90 this chord is cut into ~50 segments.

Only the atmospheric height and the detector depth are exaggerated (by ~50x),
since at true scale neither would be a visible fraction of an Earth radius;
they are labelled as exaggerated in the figure.

Run:  python benchmarks/fig_chord_geometry.py
Artefacts go to benchmarks/output/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Arc
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe

# Interior annotations sit on saturated fills (the inner core is near-black in
# YlOrBr), where plain black text and a plain black marker are invisible. Every
# label and leader inside the disc gets a white halo instead of being recoloured,
# so the same styling works over both the pale mantle and the dark core.
HALO = [pe.withStroke(linewidth=3.2, foreground="white")]
HALO_THIN = [pe.withStroke(linewidth=2.6, foreground="white")]
MARKER = dict(marker="o", color="0.08", markeredgecolor="white",
              markeredgewidth=1.3, linestyle="none")

import mango.earth as earth

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTDIR, exist_ok=True)

# --- the real shell table, in units of the Earth radius ---------------------
N_SUB = 4
_T = earth.shell_table(N_SUB)
R_KM = earth.R_EARTH_KM
OUTER = np.asarray(_T.outer) / R_KM             # ascending, outer[-1] = 1
RHO = np.asarray(_T.rho)
N_SHELL = OUTER.size

R_E = 1.0
R_INNER_CORE = 1221.5 / R_KM
R_OUTER_CORE = earth.CORE_RADIUS_KM / R_KM
# Exaggerated (~50x): at true scale neither is a visible fraction of R_E. Both
# are drawn large enough that a double-headed arrow fits between the radii
# without its heads colliding, which is what sets these values.
H_ATM = 0.13
D_DET = 0.12

COS_TZ = -0.90               # a core-crossing trajectory
r_det = R_E - D_DET
r_prod = R_E + H_ATM
r_min = r_det * np.sqrt(1.0 - COS_TZ ** 2)


def x_at(r, sign):
    """Half-chord abscissa where the chord of impact parameter r_min meets r."""
    return sign * np.sqrt(max(r ** 2 - r_min ** 2, 0.0))


xP, xD = x_at(r_prod, -1), x_at(r_det, +1)
xS_in, xS_out = x_at(R_E, -1), x_at(R_E, +1)          # surface crossings

plt.rcParams.update({"font.size": 19, "mathtext.fontset": "cm"})
fig, ax = plt.subplots(figsize=(7.2, 7.2))

# --- the Earth, one filled disc per shell, coloured by density -------------
# Each shell gets a thin edge: with 43 shells and a continuous colour map the
# fills alone read as a smooth gradient, which is the opposite of the point.
norm = Normalize(vmin=RHO.min(), vmax=RHO.max())
cmap = plt.get_cmap("YlOrBr")
# outermost first so each inner shell paints over its parent
for i in range(N_SHELL - 1, -1, -1):
    ax.add_patch(Circle((0, 0), OUTER[i], facecolor=cmap(norm(RHO[i])),
                        edgecolor="white", lw=0.35,
                        zorder=1 + (N_SHELL - i) * 0.001))
# the three boundaries worth naming, drawn over the fills
for r, lw in ((R_INNER_CORE, 1.0), (R_OUTER_CORE, 1.3), (R_E, 1.5)):
    ax.add_patch(Circle((0, 0), r, facecolor="none", edgecolor="0.25", lw=lw,
                        zorder=3))
ax.add_patch(Circle((0, 0), r_prod, facecolor="none", edgecolor="0.55",
                    lw=1.0, ls=":", zorder=3))
# the detector sphere, so that d has a visible pair of surfaces to span
ax.add_patch(Circle((0, 0), r_det, facecolor="none", edgecolor="0.45",
                    lw=0.9, ls=(0, (5, 4)), zorder=3))

# --- the chord, cut at every shell crossing and coloured by that shell -----
# crossings: the chord of impact parameter r_min meets shell radius r at
# x = +-sqrt(r^2 - r_min^2), so every shell above r_min contributes two.
cuts = {xS_in, xS_out, xD}
for r in OUTER:
    if r > r_min:
        cuts.add(x_at(r, -1))
        cuts.add(x_at(r, +1))
cuts = np.array(sorted(c for c in cuts if xS_in <= c <= xD))

# A solid bar ruled at every shell crossing, NOT alternating bands: the bands
# read as a dash pattern whose rhythm looks arbitrary, because segment lengths
# here span a factor ~360. That spread is geometry, not decoration -- with
# x = sqrt(r^2 - r_min^2) the spacing dx/dr = r/sqrt(r^2 - r_min^2) diverges as
# r -> r_min, so near closest approach the chord runs nearly parallel to the
# boundaries and stays inside one shell for a long path, while near the surface
# the thin crust shells are crossed almost immediately. Rules say "divided
# here" without implying a periodic pattern.
# Not the density colour map either: shading the chord by the same scale as the
# shells it lies on makes it vanish into the background.
CHORD = "#1f5f8b"
n_seg = len(cuts) - 1

# white casing so the chord separates from the orange fills at any radius
ax.plot([xS_in, xD], [r_min, r_min], color="white", lw=8.0, zorder=4,
        solid_capstyle="butt")
ax.plot([xS_in, xD], [r_min, r_min], color=CHORD, lw=5.4, zorder=5,
        solid_capstyle="butt")
_TICK = 0.020                                  # just over the bar half-width
ax.add_collection(LineCollection(
    [[(x, r_min - _TICK), (x, r_min + _TICK)] for x in cuts[1:-1]],
    colors="white", linewidths=1.0, zorder=6))
# the atmospheric leg, which crosses no shell
ax.plot([xP, xS_in], [r_min, r_min], color="0.3", lw=2.0, ls=":", zorder=5,
        solid_capstyle="butt")

# --- closest approach and the impact parameter -----------------------------
ax.plot([0, 0], [0, r_min], color="0.12", ls="--", lw=1.4, zorder=6,
        path_effects=HALO_THIN)
ax.plot([0], [r_min], ms=6.5, zorder=8, **MARKER)
ax.plot([0], [0], ms=7.5, zorder=8, **MARKER)          # O: the centre
ax.annotate(r"$r_{\min}$", xy=(-0.055, r_min * 0.5), fontsize=19, ha="right",
            va="center", zorder=9, path_effects=HALO)
ax.annotate("$O$", xy=(0.075, -0.02), fontsize=19, ha="left", va="top",
            zorder=9, path_effects=HALO)
# C sits ON the chord, so push its label clear of the line
ax.annotate("$C$", xy=(-0.06, r_min + 0.05), fontsize=19, ha="right",
            va="bottom", zorder=9, path_effects=HALO)

# --- endpoints -------------------------------------------------------------
ax.plot([xP], [r_min], ms=6.5, zorder=8, **MARKER)
ax.plot([xD], [r_min], ms=6.5, zorder=8, **MARKER)
ax.annotate("$P$", xy=(xP - 0.015, r_min + 0.08), fontsize=19, ha="center",
            va="bottom", zorder=9, path_effects=HALO)
# D must clear the theta_z arc, which is drawn AROUND D with radius _R_ARC/2:
# anything within that radius in the upper half collides with it. Below-right
# of D the arc does not reach.
ax.annotate("$D$", xy=(xD + 0.085, r_min - 0.055), fontsize=19, ha="left",
            va="center", zorder=9, path_effects=HALO)

# --- local vertical at D and the zenith angle ------------------------------
ux, uy = xD / r_det, r_min / r_det
ax.plot([xD, xD + 0.32 * ux], [r_min, r_min + 0.32 * uy], color="0.2",
        lw=1.4, zorder=6, path_effects=HALO_THIN)
ang_vert = np.degrees(np.arctan2(uy, ux))
_R_ARC = 0.30
_arc = Arc((xD, r_min), _R_ARC, _R_ARC, angle=0, theta1=ang_vert,
           theta2=180.0, color="0.12", lw=1.5, zorder=7)
_arc.set_path_effects(HALO_THIN)
ax.add_patch(_arc)
_mid = np.radians(0.5 * (ang_vert + 180.0))
ax.annotate(r"$\theta_z$",
            xy=(xD + 0.85 * _R_ARC * np.cos(_mid),
                r_min + 0.85 * _R_ARC * np.sin(_mid)),
            fontsize=19, ha="center", va="center", zorder=9,
            path_effects=HALO)

# --- h_atm and d, annotated radially at the top where the gap is clear -----
# The previous version drew these as horizontal arrows along the chord, where
# the two radii differ by a few percent of the span and the arrowheads
# collided. Radially, near the top of the disc, the gap is the full
# (exaggerated) height and the leader lines have empty space to land in.
def _radial_pair(theta_deg, r0, r1, label, r_text, color="0.15"):
    """Double-headed arrow spanning [r0, r1] radially, labelled outside it."""
    th = np.radians(theta_deg)
    c, s = np.cos(th), np.sin(th)
    ax.annotate("", xy=(r1 * c, r1 * s), xytext=(r0 * c, r0 * s),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.4,
                                mutation_scale=11, shrinkA=0, shrinkB=0),
                zorder=8)
    ax.annotate(label, xy=(r_text * c, r_text * s), fontsize=17, color=color,
                ha="center", va="center", zorder=9,
                path_effects=HALO)


_radial_pair(108.0, R_E, r_prod, r"$h_{\rm atm}$", r_prod + 0.10)
_radial_pair(52.0, r_det, R_E, "$d$", R_E + 0.07)

ax.set_xlim(-1.42, 1.42)
ax.set_ylim(-1.30, 1.44)
ax.set_aspect("equal")
ax.axis("off")

# Region names go in the empty lower-left, as a legend rather than inline
# labels: inline, "inner core" has nowhere to sit that does not collide with
# the centre mark or the r_min line.
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles = [
    Patch(facecolor=cmap(norm(RHO[0])), edgecolor="0.4", label="inner core"),
    Patch(facecolor=cmap(norm(RHO[N_SHELL // 3])), edgecolor="0.4",
          label="outer core"),
    Patch(facecolor=cmap(norm(RHO[-4])), edgecolor="0.4", label="mantle"),
    Line2D([0], [0], color=CHORD, lw=4,
           label=f"chord, ruled at each of\nits {n_seg} shell crossings"),
    Line2D([0], [0], color="0.3", lw=2, ls=":", label="atmospheric leg"),
]
ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.92,
          edgecolor="0.75", fontsize=12, handlelength=1.4,
          borderpad=0.45, labelspacing=0.3,
          bbox_to_anchor=(-0.025, 0.0))

ax.annotate(f"PREM, {N_SHELL} shells ($n_{{\\rm sub}}={N_SUB}$)\n"
            r"$h_{\rm atm}$, $d$ exaggerated $\sim\!50\times$",
            xy=(0.995, 0.055), xycoords="axes fraction", fontsize=13,
            color="0.25", ha="right", va="center", linespacing=1.4)

cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                  orientation="horizontal", fraction=0.045, pad=0.02,
                  shrink=0.80)
cb.set_label(r"PREM density [g cm$^{-3}$]", fontsize=15)
cb.ax.tick_params(labelsize=13)

fp = os.path.join(OUTDIR, "jaxnu_chord_geometry")
fig.savefig(fp + ".png", dpi=200, bbox_inches="tight")
fig.savefig(fp + ".pdf", bbox_inches="tight")
print("saved", fp + ".pdf")
print(f"  shell table: n_sub={N_SUB} -> {N_SHELL} shells, "
      f"rho {RHO.min():.2f}-{RHO.max():.2f} g/cm^3")
print(f"  cos(theta_z) = {COS_TZ}, r_min/R_E = {r_min:.3f} "
      f"(outer core at {R_OUTER_CORE:.3f}: the chord crosses the core)")
print(f"  chord cut into {n_seg} constant-density segments")
