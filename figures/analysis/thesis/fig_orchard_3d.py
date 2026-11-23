"""F1 (fig:orchard-3d): the orchard in 3D from the LIO-SAM dense map.

Sources (all traceable):
  dense map   data_set/campinas_b/dense_map/dense_rep1own.npz (+ _height.npy = height above local ground)
  path        orchard-slam/data/thesis_jobs_20261007/for_thesis_20261007/trajectories/liosam_rep1_keyframes.csv
  trunks      orchard-weed-detection/trunk/v2_out/trunks_final_rep1.csv   (row_id 8,6,1,2 = R1..R4)
  north       data_set/campinas_b/maps/liosam_rep1/align_rep1.json        (rep1 xy -> ENU)
Views: orthographic oblique, presets f1a_block / f1b_alley in thesis_style.CAMERAS.
Row-aligned frame: u along the rows, v across (rotation by the mean row angle).
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from thesis_style import *

ROOT = Path("/home/afonso/thesis")
DENSE = ROOT / "data_set/campinas_b/dense_map"
TRAJ = ROOT / "orchard-slam/data/thesis_jobs_20261007/for_thesis_20261007/trajectories/liosam_rep1_keyframes.csv"
TRUNKS = ROOT / "orchard-weed-detection/trunk/v2_out/trunks_final_rep1.csv"
ALIGN = ROOT / "data_set/campinas_b/maps/liosam_rep1/align_rep1.json"
ROWNAME = {8: "R1", 6: "R2", 1: "R3", 2: "R4"}
H_CAN, H_TOP = 1.3, 3.2          # m: above H_CAN = canopy (faded); colour range top
VOX = 0.12


def load():
    P = np.load(DENSE / "dense_rep1own.npz")["xyz"]
    H = np.load(DENSE / "dense_rep1own_height.npy")
    tr = pd.read_csv(TRUNKS); tj = pd.read_csv(TRAJ)
    # row angle from the surveyed rows
    ang = []
    for r in ROWNAME:
        s = tr[tr.row_id == r]; ang.append(np.arctan(np.polyfit(s.x, s.y, 1)[0]))
    th = float(np.mean(ang))
    # rotate by -th, then flip so +u runs the same way as +x (rows run towards -x)
    c, s_ = np.cos(-th), np.sin(-th)
    Rm = np.array([[c, -s_], [s_, c]])
    # crop to the block
    x0, x1 = tr.x.min() - 10, tr.x.max() + 10
    y0, y1 = tr.y.min() - 10, tr.y.max() + 10
    m = (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1) & (H > -0.2) & (H < 5)
    P, H = P[m], H[m]
    print("points in block crop:", len(P))
    return P, H, tr, tj, Rm, th


def voxel(P, H, vox):
    k = np.floor(P / vox).astype(np.int64)
    key = k[:, 0] * 73856093 ^ k[:, 1] * 19349663 ^ k[:, 2] * 83492791
    _, idx = np.unique(key, return_index=True)
    return P[idx], H[idx]


def ground_fill(uv, P, H, win, step, k=10, maxd=0.5):
    """Light ground plane: z = median of the k nearest ground points, on a regular grid."""
    from scipy.spatial import cKDTree
    g = H < 0.12
    kd = cKDTree(uv[g]); zg = P[g, 2]
    uu, vv = np.meshgrid(np.arange(win[0], win[1], step), np.arange(win[2], win[3], step))
    q = np.c_[uu.ravel(), vv.ravel()]
    d, ii = kd.query(q, k=k)
    z = np.median(zg[ii], axis=1)
    ok = d[:, 0] < maxd
    return np.c_[q[ok], z[ok]]


def to_frame(xy, Rm, c0):
    return (xy - c0) @ Rm.T


def render(ax, UVZ, H, cam, fade, size, ground=None, gsize=None, keep=1.0, seed=0):
    if ground is not None:
        gX, gY, _ = project(ground, **cam)
        ax.scatter(gX, gY, s=gsize, c=COL["ground"], marker="s", linewidths=0, rasterized=True)
    if keep < 1.0:                                  # thin the canopy so it reads as a veil
        rng = np.random.default_rng(seed)
        m = (H <= H_CAN) | (rng.random(len(H)) < keep)
        UVZ, H = UVZ[m], H[m]
    X, Y, D = project(UVZ, **cam)
    order = np.argsort(-D)
    X, Y, H_, D = X[order], Y[order], H[order], D[order]
    cmap = HEIGHT_CMAP
    col = cmap(np.clip(H_ / H_TOP, 0, 1))
    ground = H_ < 0.10
    col[ground] = mpl.colors.to_rgba(COL["ground"])
    # canopy fade: opaque colour blended to white and lower alpha
    can = H_ > H_CAN
    col[can, 3] = fade
    col[~ground & ~can, 3] = 1.0
    # keep ground and trunk band opaque, draw canopy last so it is a veil
    base = ~can
    ax.scatter(X[base], Y[base], s=size, c=col[base], marker="s", linewidths=0, rasterized=True)
    ax.scatter(X[can], Y[can], s=size, c=col[can], marker="s", linewidths=0, rasterized=True)
    return X, Y


def main():
    P, H, tr, tj, Rm, th = load()
    Pfull, Hfull = P, H
    P, H = voxel(Pfull, Hfull, VOX)
    print("points after voxel:", len(P))
    c0 = np.array([tr.x.mean(), tr.y.mean()])
    uv = to_frame(P[:, :2], Rm, c0)
    UVZ = np.c_[uv, P[:, 2]]
    zref = np.median(P[H < 0.1, 2])
    tuv = to_frame(tr[["x", "y"]].values, Rm, c0)
    # ground z under each trunk
    tz = np.empty(len(tr))
    for i, (u, v) in enumerate(tuv):
        near = (np.abs(uv[:, 0] - u) < 0.5) & (np.abs(uv[:, 1] - v) < 0.5) & (H < 0.12)
        tz[i] = np.median(P[near, 2]) if near.sum() > 3 else zref
    puv = to_frame(tj[["x", "y"]].values, Rm, c0)
    pz = np.interp(np.arange(len(tj)), np.arange(len(tj)), np.zeros(len(tj)))
    # path height = ground z at nearest map ground point (cheap: grid median)
    from scipy.spatial import cKDTree
    g = H < 0.12
    kd = cKDTree(uv[g]); _, ii = kd.query(puv); pz = P[g][ii, 2]

    # north direction in the row frame (rep1 xy -> ENU via R)
    A = json.load(open(ALIGN)); R = np.array(A["R"])
    n_rep1 = R.T @ np.array([0.0, 1.0])          # ENU north expressed in rep1 xy
    n_uv = Rm @ n_rep1
    row_enu = R @ np.array([np.cos(th), np.sin(th)])
    hdg = np.degrees(np.arctan2(row_enu[0], row_enu[1])) % 180
    print(f"rows run {hdg:.1f} deg from true north (axis)")

    fig = plt.figure(figsize=(FIG_W, 13.2 * CM))
    gs = fig.add_gridspec(2, 1, height_ratios=[8.2, 4.4], hspace=0.07, left=0.01, right=0.99, top=0.985, bottom=0.01)

    # ---------------- (a) whole block -------------------------------------
    ax = fig.add_subplot(gs[0]); ax.set_axis_off(); ax.set_aspect("equal")
    cam = CAMERAS["f1a_block"]
    win = (UVZ[:, 0].min(), UVZ[:, 0].max(), UVZ[:, 1].min(), UVZ[:, 1].max())
    gfill = ground_fill(uv, P, H, win, 0.35)
    X, Y = render(ax, UVZ, H, cam, fade=0.35, size=0.7, ground=gfill, gsize=1.2, keep=0.12)
    # trunks as small dots, path as line
    tX, tY, _ = project(np.c_[tuv, tz], **cam)
    ax.scatter(tX, tY, s=4.5, c=COL["trunk"], edgecolors="white", linewidths=0.3, zorder=10)
    pX, pY, _ = project(np.c_[puv, pz + 0.05], **cam)
    ax.plot(pX, pY, color=COL["path_halo"], lw=2.4, zorder=11, solid_capstyle="round")
    ax.plot(pX, pY, color=COL["path"], lw=1.3, zorder=12, solid_capstyle="round")
    # row labels
    for r, nm in ROWNAME.items():
        s = np.where(tr.row_id.values == r)[0]
        i = s[np.argmin(tuv[s, 0])]
        ax.text(tX[i] - 1.6, tY[i], nm, fontsize=6.5, fontweight="bold", ha="right", va="center",
                zorder=15, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
    ax.set_xlim(X.min() - 2, X.max() + 2); ax.set_ylim(Y.min() - 2, Y.max() + 2)
    scale_bar(ax, 20, "20 m", xy=(0.03, 0.04), lw=1.6)
    nv = project(np.array([[n_uv[0], n_uv[1], 0.0]]), **cam)
    north_arrow(ax, nv[0][0], nv[1][0], xy=(0.90, 0.80), size=0.07)
    panel_label(ax, "(a)", x=0.0, y=0.93)
    # height colourbar (inset, horizontal)
    cax = ax.inset_axes([0.72, 0.03, 0.26, 0.018])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, H_TOP), cmap=HEIGHT_CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal"); cb.set_ticks([0, 1, 2, 3])
    cb.ax.tick_params(labelsize=6, length=2, width=0.4); cb.outline.set_linewidth(0.4)
    cax.set_title("height above ground (m)", fontsize=6.5, pad=1.5)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(fc=COL["ground"], ec="none", label="ground"),
                       Line2D([], [], color=COL["path"], lw=1.3, label="robot path"),
                       Line2D([], [], marker="o", ls="", ms=3, mfc=COL["trunk"], mec="white", mew=0.3, label="trunk")],
              loc="lower right", bbox_to_anchor=(1.0, 0.105), fontsize=6.5, handlelength=1.4,
              labelspacing=0.25, borderaxespad=0)

    # ---------------- (b) close-up of one alley ---------------------------
    ax2 = fig.add_subplot(gs[1]); ax2.set_axis_off(); ax2.set_aspect("equal")
    camb = CAMERAS["f1b_alley"]
    # alley between R3 and R4 (rows 1 and 2): window along the alley
    s1, s2 = tuv[tr.row_id.values == 1], tuv[tr.row_id.values == 2]
    vmid = 0.5 * (np.median(s1[:, 1]) + np.median(s2[:, 1]))
    umid = np.median(np.r_[s1[:, 0], s2[:, 0]])
    UW, VW = 7.5, 3.2           # half-window along rows, half-width across
    uvf = to_frame(Pfull[:, :2], Rm, c0)
    UVZf = np.c_[uvf, Pfull[:, 2]]
    vfar = max(np.median(s1[:, 1]), np.median(s2[:, 1]))
    VCUT = vmid - 0.3                      # near row and everything on the camera side is cut away
    sel = (np.abs(uvf[:, 0] - umid) < UW) & (uvf[:, 1] > VCUT) & (uvf[:, 1] < vfar + 2.2)
    Pb, Hb = voxel(Pfull[sel], Hfull[sel], 0.03)
    uvb = to_frame(Pb[:, :2], Rm, c0)
    gb = ground_fill(uvf, Pfull, Hfull, (umid - UW, umid + UW, vmid - 3.0, vfar + 2.2), 0.08, k=12)
    Xb, Yb = render(ax2, np.c_[uvb, Pb[:, 2]], Hb, camb, fade=0.28, size=2.2, ground=gb, gsize=2.6, keep=0.5)
    tm = (np.abs(tuv[:, 0] - umid) < UW) & (tuv[:, 1] > VCUT) & (tuv[:, 1] < vfar + 2.2)
    for (u, v), z in zip(tuv[tm], tz[tm]):
        # marker floats above the crown base so the scanned trunk stays visible
        top = project(np.array([[u, v, z + 1.55]]), **camb)
        ax2.plot(top[0], top[1], "v", color=COL["trunk"], ms=4.2, mec="white", mew=0.4, zorder=11)
    ax2.set_xlim(Xb.min(), Xb.max()); ax2.set_ylim(Yb.min(), Yb.max())
    scale_bar(ax2, 2, "2 m", xy=(0.03, 0.06), lw=1.6)
    panel_label(ax2, "(b)", x=0.0, y=0.93)
    save(fig, "orchard_3d")
    print("camera a", cam, "camera b", camb, "window u", umid, "v", vmid)


if __name__ == "__main__":
    main()
