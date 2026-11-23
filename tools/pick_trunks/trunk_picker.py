#!/usr/bin/env python3
"""
trunk_picker.py - pick trunk centres by hand on a point-cloud map (top view).

The map is cut to a horizontal slice above the LOCAL ground (default
0.30-0.80 m, ground = 5th percentile of z in 0.5 m cells, as in
trunk_detect.py), and shown from above. Each trunk appears as a ring or an arc.
You click its centre; the picks get IDs row by row, in the order of the field
sheets (R1-T1, R1-T2, ... then R2-T1, ...).

WINDOW
  left  : the slice seen from above (colour = height above the ground).
  right : zoom of the current pick, with a dashed circle of the expected
          trunk radius. Click here to correct the centre.
  bottom: height range slider - drag its ends to hide low or tall points.

MOUSE
  left click   (left panel)   new pick with the next ID
  left click   (right panel)  move the current pick to the clicked point
  scroll wheel                zoom in / out around the cursor (both panels)
  right-drag                  pan (left panel)

KEYS   (click once on the window so it has focus)
  r      start the next row (tree counter back to 1)
  s      skip one tree number (tree not visible / not in this map)
  u      undo the last pick
  n / p  select next / previous pick (to review or correct)
  c      centre the left view on the current pick
  f      full view
  w      save now  (it also saves after every change)
  h      print this help
  q      save and quit

INPUT  : .pcd (ascii or binary), .ply (via open3d if installed), .npz (key xyz),
         .txt/.csv/.xyz (x y z ...).
OUTPUT : CSV with id,row,tree,x,y,z_ground,n_points_near,map,time
         Re-running with the same --out continues where you stopped.

Examples
  python3 trunk_picker.py --map lio_sam_dense.pcd --out picks_liosam.csv --half top
  python3 trunk_picker.py --map map.pcd --out picks.csv --bbox -5 60 0 8
  python3 trunk_picker.py --map lio_sam_dense.pcd --out picks_liosam.csv
  python3 trunk_picker.py --map kiss_dense.pcd --out picks_kiss.csv \
          --detections trunks_kiss.csv --radius 0.10
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HELP = __doc__[__doc__.index("MOUSE"):__doc__.index("INPUT")]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def read_pcd(path):
    """Minimal PCD reader (ascii / binary). binary_compressed -> needs open3d."""
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if not line or line.startswith("#"):
                continue
            k, *v = line.split()
            header[k.upper()] = v
            if k.upper() == "DATA":
                break
        fields = header["FIELDS"]
        sizes = [int(s) for s in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
        n = int(header["POINTS"][0])
        kind = header["DATA"][0].lower()
        if kind == "ascii":
            arr = np.loadtxt(f, ndmin=2)
            cols = np.cumsum([0] + counts)
            idx = {name: cols[i] for i, name in enumerate(fields)}
            return arr[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float64)
        if kind == "binary":
            tmap = {("F", 4): "f4", ("F", 8): "f8", ("U", 1): "u1", ("U", 2): "u2",
                    ("U", 4): "u4", ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4"}
            dt = np.dtype([(name if name != "_" else f"_pad{i}",
                            tmap[(t, s)], (c,) if c > 1 else ())
                           for i, (name, s, t, c) in enumerate(zip(fields, sizes, types, counts))])
            data = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
            return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    raise ValueError("binary_compressed PCD: install open3d (pip install open3d)")


def load_cloud(path):
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".npz":
        return np.load(p)["xyz"].astype(np.float64)
    if ext in (".txt", ".csv", ".xyz"):
        delim = "," if ext == ".csv" else None
        arr = np.genfromtxt(p, delimiter=delim, skip_header=0, invalid_raise=False)
        if np.isnan(arr[0]).any():                       # header row
            arr = arr[1:]
        return arr[:, :3].astype(np.float64)
    if ext == ".pcd":
        try:
            return read_pcd(p)
        except ValueError:
            pass
    import open3d as o3d                                  # .ply or compressed pcd
    return np.asarray(o3d.io.read_point_cloud(str(p)).points, dtype=np.float64)


# ----------------------------------------------------------------------------
# Terrain-relative slice (same rule as trunk_detect.py)
# ----------------------------------------------------------------------------
def local_ground(xyz, cell=0.5, pct=5):
    ij = np.floor(xyz[:, :2] / cell).astype(np.int64)
    key = (ij[:, 0] << 32) ^ (ij[:, 1] & 0xFFFFFFFF)
    order = np.argsort(key, kind="stable")
    k_sorted = key[order]
    z_sorted = xyz[order, 2]
    starts = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
    ends = np.r_[starts[1:], len(k_sorted)]
    g = np.empty(len(xyz))
    for s, e in zip(starts, ends):
        g[order[s:e]] = np.percentile(z_sorted[s:e], pct)
    return g


def make_slice(xyz, h_lo, h_hi, cell):
    """Points with height above the local ground in [h_lo, h_hi]."""
    g = local_ground(xyz, cell)
    h = xyz[:, 2] - g
    m = (h >= h_lo) & (h <= h_hi)
    return xyz[m], g[m]


# ----------------------------------------------------------------------------
# Picker
# ----------------------------------------------------------------------------
class Picker:
    def __init__(self, pts, ground, out, map_name, radius, detections=None, zoom=0.75,
                 h_range=(0.30, 0.80), h_limits=(-0.2, 3.0), px=0.02):
        matplotlib.rcParams["toolbar"] = "None"          # own zoom / pan, no tool modes
        from matplotlib.widgets import RangeSlider
        from scipy.spatial import cKDTree

        self.pts, self.ground, self.out, self.map_name = pts, ground, Path(out), map_name
        self.h = pts[:, 2] - ground
        self.radius, self.zoom, self.det = radius, zoom, detections
        self.h_lo, self.h_hi = h_range
        self.picks, self.cur, self.row, self.tree = [], -1, 1, 1
        self._pan = None
        self._load_existing()
        self.kd = cKDTree(pts[:, :2])
        # the overview is an IMAGE (point count per px x px cell), not a scatter:
        # zoom and pan only move the view, nothing is re-drawn point by point
        lo, hi = pts[:, :2].min(0), pts[:, :2].max(0)
        px = max(px, np.sqrt((hi - lo).prod() / 25e6))          # cap at ~25 Mpx
        self.px, self.lo = px, lo
        self.shape = (int((hi[1] - lo[1]) / px) + 1, int((hi[0] - lo[0]) / px) + 1)
        self.cell = ((pts[:, 1] - lo[1]) / px).astype(np.int64) * self.shape[1] + \
                    ((pts[:, 0] - lo[0]) / px).astype(np.int64)
        print(f"overview image {self.shape[1]} x {self.shape[0]} px at {px * 100:.1f} cm/px")

        self.fig = plt.figure(figsize=(15, 8))
        self.fig.canvas.manager.set_window_title(f"trunk picker - {map_name}")
        self.ax = self.fig.add_axes([0.04, 0.12, 0.60, 0.80])
        self.az = self.fig.add_axes([0.67, 0.12, 0.31, 0.80])
        sax = self.fig.add_axes([0.20, 0.03, 0.38, 0.03])
        self.slider = RangeSlider(sax, "height above ground (m)", h_limits[0], h_limits[1],
                                  valinit=(self.h_lo, self.h_hi))
        self.slider.on_changed(self.on_slider)

        ext = (lo[0], lo[0] + self.shape[1] * px, lo[1], lo[1] + self.shape[0] * px)
        self.img = self.ax.imshow(np.zeros(self.shape, np.float32), origin="lower", extent=ext,
                                  cmap="Greys", interpolation="nearest", aspect="auto")
        if self.det is not None and len(self.det):
            self.ax.plot(self.det[:, 0], self.det[:, 1], "x", c="#D96C00", ms=6, mew=1.2)
        self.ax.set_aspect("auto")           # equal scale kept by set_view()
        self.ax.set_xlabel("x (m)"); self.ax.set_ylabel("y (m)")
        self.pick_artist, = self.ax.plot([], [], "o", mfc="none", mec="#0000B3", ms=8, mew=1.4)
        self.cur_artist, = self.ax.plot([], [], "o", mfc="none", mec="#E03C31", ms=13, mew=2)
        self.labels = []
        self.status = self.fig.suptitle("", fontsize=11)

        self.update_points()
        self.full_view()
        c = self.fig.canvas
        c.mpl_connect("button_press_event", self.on_press)
        c.mpl_connect("button_release_event", self.on_release)
        c.mpl_connect("motion_notify_event", self.on_motion)
        c.mpl_connect("scroll_event", self.on_scroll)
        c.mpl_connect("key_press_event", self.on_key)
        if self.picks:
            self.cur = len(self.picks) - 1
        self.redraw()
        print(HELP)

    # ---- persistence -------------------------------------------------------
    def _load_existing(self):
        if not self.out.exists():
            return
        with open(self.out) as f:
            for r in csv.DictReader(f):
                self.picks.append(dict(id=r["id"], row=int(r["row"]), tree=int(r["tree"]),
                                       x=float(r["x"]), y=float(r["y"]),
                                       z_ground=float(r["z_ground"]),
                                       n_near=int(r["n_points_near"]), time=r["time"]))
        if self.picks:
            last = self.picks[-1]
            self.row, self.tree = last["row"], last["tree"] + 1
            print(f"resuming {self.out}: {len(self.picks)} picks, next = R{self.row}-T{self.tree}")

    def save(self):
        tmp = self.out.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "row", "tree", "x", "y", "z_ground", "n_points_near", "map", "time"])
            for p in self.picks:
                w.writerow([p["id"], p["row"], p["tree"], f"{p['x']:.4f}", f"{p['y']:.4f}",
                            f"{p['z_ground']:.3f}", p["n_near"], self.map_name, p["time"]])
        os.replace(tmp, self.out)

    # ---- data ----------------------------------------------------------------
    def visible(self):
        return (self.h >= self.h_lo) & (self.h <= self.h_hi)

    def update_points(self):
        cnt = np.bincount(self.cell[self.visible()], minlength=self.shape[0] * self.shape[1])
        im = np.log1p(cnt.reshape(self.shape).astype(np.float32))
        self.img.set_data(im)
        self.img.set_clim(0, max(np.percentile(im[im > 0], 99), 1e-3) if (im > 0).any() else 1)

    def _info(self, x, y):
        idx = np.array(self.kd.query_ball_point([x, y], self.radius + 0.10), dtype=int)
        if len(idx):
            idx = idx[self.visible()[idx]]
        zg = float(np.median(self.ground[idx])) if len(idx) else float("nan")
        return zg, int(len(idx))

    def add_pick(self, x, y):
        zg, n = self._info(x, y)
        pid = f"R{self.row}-T{self.tree}"
        self.picks.append(dict(id=pid, row=self.row, tree=self.tree, x=x, y=y, z_ground=zg,
                               n_near=n, time=time.strftime("%Y-%m-%d %H:%M:%S")))
        self.cur = len(self.picks) - 1
        self.tree += 1
        self.save()
        print(f"pick {pid}: x={x:.3f} y={y:.3f}  ({n} points near)")

    def move_current(self, x, y):
        if self.cur < 0:
            return
        zg, n = self._info(x, y)
        p = self.picks[self.cur]
        p.update(x=x, y=y, z_ground=zg, n_near=n, time=time.strftime("%Y-%m-%d %H:%M:%S"))
        self.save()
        print(f"moved {p['id']}: x={x:.3f} y={y:.3f}")

    # ---- view ----------------------------------------------------------------
    def _px_ratio(self):
        bb = self.ax.get_window_extent()
        return bb.height / max(bb.width, 1)

    def set_view(self, x0, x1, y0, y1):
        """Set limits with equal metres per pixel in x and y (no distortion)."""
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hw, hh = (x1 - x0) / 2, (y1 - y0) / 2
        r = self._px_ratio()
        if hh > hw * r:
            hw = hh / r
        else:
            hh = hw * r
        self.ax.set_xlim(cx - hw, cx + hw); self.ax.set_ylim(cy - hh, cy + hh)

    def full_view(self):
        v = self.pts[self.visible(), :2]
        if not len(v):
            v = self.pts[:, :2]
        lo, hi = v.min(0), v.max(0)
        pad = 0.03 * (hi - lo).max()
        self.set_view(lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad)

    def zoom_at(self, ax, x, y, factor):
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        ax.set_xlim(x - (x - x0) * factor, x + (x1 - x) * factor)
        ax.set_ylim(y - (y - y0) * factor, y + (y1 - y) * factor)

    def redraw(self):
        self.pick_artist.set_data([p["x"] for p in self.picks], [p["y"] for p in self.picks])
        for t in self.labels:
            t.remove()
        self.labels = [self.ax.text(p["x"] + 0.12, p["y"] + 0.12, p["id"], fontsize=8,
                                    color="#0000B3", clip_on=True) for p in self.picks]
        if self.cur >= 0:
            c = self.picks[self.cur]
            self.cur_artist.set_data([c["x"]], [c["y"]])
        else:
            self.cur_artist.set_data([], [])

        az = self.az
        az.cla(); az.set_aspect("equal", adjustable="box")
        if self.cur >= 0:
            c = self.picks[self.cur]
            cx, cy, Z = c["x"], c["y"], self.zoom
            idx = np.array(self.kd.query_ball_point([cx, cy], Z * 1.5), dtype=int)
            if len(idx):
                idx = idx[self.visible()[idx]]
            P = self.pts[idx]
            az.scatter(P[:, 0], P[:, 1], s=6, c=self.h[idx], cmap="copper",
                       vmin=self.slider.valmin, vmax=1.5, lw=0)
            az.add_patch(Circle((cx, cy), self.radius, fill=False, ls="--", ec="#0000B3", lw=1.3))
            az.plot(cx, cy, "+", c="#E03C31", ms=16, mew=2)
            if self.det is not None and len(self.det):
                near = np.hypot(self.det[:, 0] - cx, self.det[:, 1] - cy) < Z
                az.plot(self.det[near, 0], self.det[near, 1], "x", c="#D96C00", ms=10, mew=2)
            az.set_xlim(cx - Z, cx + Z); az.set_ylim(cy - Z, cy + Z)
            az.set_title(f"{c['id']}  -  click to correct  ({c['n_near']} pts near)", fontsize=10)
        else:
            az.set_title("click a trunk in the left panel", fontsize=10)
        az.set_xticks([]); az.set_yticks([])

        self.status.set_text(
            f"{self.map_name}  |  {len(self.picks)} picks  |  next click = R{self.row}-T{self.tree}"
            f"  |  height {self.h_lo:.2f}-{self.h_hi:.2f} m  |  scroll = zoom, right-drag = pan,"
            f" r row, s skip, u undo, f full view, q quit")
        self.fig.canvas.draw_idle()

    # ---- events --------------------------------------------------------------
    def on_slider(self, val):
        self.h_lo, self.h_hi = float(val[0]), float(val[1])
        self.update_points()
        self.redraw()

    def on_press(self, ev):
        if ev.inaxes is self.ax and ev.button == 3:              # start pan
            self._pan = (ev.x, ev.y, self.ax.get_xlim(), self.ax.get_ylim())
            return
        if ev.button != 1 or ev.xdata is None:
            return
        if ev.inaxes is self.ax:
            self.add_pick(ev.xdata, ev.ydata)
        elif ev.inaxes is self.az:
            self.move_current(ev.xdata, ev.ydata)
        else:
            return                                               # slider etc.
        self.redraw()

    def on_motion(self, ev):
        if self._pan is None:
            return
        x0, y0, xl, yl = self._pan
        inv = self.ax.transData.inverted()
        (ax0, ay0), (ax1, ay1) = inv.transform([(x0, y0), (ev.x, ev.y)])
        dx, dy = ax1 - ax0, ay1 - ay0
        self.ax.set_xlim(xl[0] - dx, xl[1] - dx); self.ax.set_ylim(yl[0] - dy, yl[1] - dy)
        self.fig.canvas.draw_idle()

    def on_release(self, ev):
        self._pan = None

    def on_scroll(self, ev):
        if ev.xdata is None:
            return
        f = 1 / 1.3 if ev.button == "up" else 1.3
        if ev.inaxes is self.ax:
            self.zoom_at(self.ax, ev.xdata, ev.ydata, f)
            self.fig.canvas.draw_idle()
        elif ev.inaxes is self.az:
            self.zoom = float(np.clip(self.zoom * f, 0.2, 5.0))
            self.redraw()

    def on_key(self, ev):
        k = ev.key
        if k == "r":
            self.row += 1; self.tree = 1
        elif k == "s":
            self.tree += 1
        elif k == "u" and self.picks:
            last = self.picks.pop()
            self.row, self.tree = last["row"], last["tree"]
            self.cur = len(self.picks) - 1
            self.save(); print(f"undo {last['id']}")
        elif k in ("n", "p") and self.picks:
            self.cur = (self.cur + (1 if k == "n" else -1)) % len(self.picks)
        elif k == "c" and self.cur >= 0:
            c = self.picks[self.cur]
            x0, x1 = self.ax.get_xlim(); y0, y1 = self.ax.get_ylim()
            hw, hh = (x1 - x0) / 2, (y1 - y0) / 2
            self.ax.set_xlim(c["x"] - hw, c["x"] + hw); self.ax.set_ylim(c["y"] - hh, c["y"] + hh)
        elif k == "f":
            self.full_view()
        elif k == "w":
            self.save(); print(f"saved {self.out}")
        elif k == "h":
            print(HELP)
        elif k == "q":
            self.save(); print(f"saved {len(self.picks)} picks to {self.out}")
            plt.close(self.fig); return
        else:
            return
        self.redraw()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, help="point-cloud map (.pcd/.ply/.npz/.txt)")
    ap.add_argument("--out", required=True, help="CSV of picks (resumed if it exists)")
    ap.add_argument("--h-lo", type=float, default=0.30, help="slice bottom above ground (m)")
    ap.add_argument("--h-hi", type=float, default=0.80, help="slice top above ground (m)")
    ap.add_argument("--h-max-load", type=float, default=3.0,
                    help="points higher than this above the ground are never loaded (m)")
    ap.add_argument("--cell", type=float, default=0.5, help="ground cell size (m)")
    ap.add_argument("--radius", type=float, default=0.10, help="expected trunk radius (m)")
    ap.add_argument("--voxel", type=float, default=0.02, help="display voxel (m), 0 = off")
    ap.add_argument("--half", choices=["top", "bottom", "left", "right"], default=None,
                    help="keep only one half of the map (split at the median), e.g. --half top")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    default=None, help="keep only this box of the map (m)")
    ap.add_argument("--px", type=float, default=0.02, help="overview pixel size (m)")
    ap.add_argument("--detections", default=None,
                    help="optional CSV with detector centres (columns x,y or cx,cy)")
    ap.add_argument("--save-slice", default=None, help="also write the slice as .npz")
    args = ap.parse_args()

    t0 = time.time()
    xyz = load_cloud(args.map)
    print(f"loaded {len(xyz):,} points from {args.map} ({time.time() - t0:.1f}s)")
    pts, g = make_slice(xyz, -0.2, args.h_max_load, args.cell)
    if args.voxel > 0 and len(pts):
        v = np.floor(pts / args.voxel).astype(np.int64)
        _, keep = np.unique(v, axis=0, return_index=True)
        pts, g = pts[keep], g[keep]
    print(f"kept {len(pts):,} points between -0.2 and {args.h_max_load} m above the local ground")
    if args.bbox:
        x0, x1, y0, y1 = args.bbox
        m = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        pts, g = pts[m], g[m]
        print(f"bbox x[{x0},{x1}] y[{y0},{y1}]: {len(pts):,} points")
    if args.half:
        ax_i = 1 if args.half in ("top", "bottom") else 0
        mid = np.median(pts[:, ax_i])
        m = pts[:, ax_i] >= mid if args.half in ("top", "right") else pts[:, ax_i] <= mid
        pts, g = pts[m], g[m]
        print(f"kept the {args.half} half ({'y' if ax_i else 'x'} {'>=' if args.half in ('top','right') else '<='} {mid:.2f} m): {len(pts):,} points")
    if len(pts) == 0:
        sys.exit("empty slice: check the map units / frame (z up?) and --h-lo/--h-hi")
    if args.save_slice:
        np.savez_compressed(args.save_slice, xyz=pts, ground=g)

    det = None
    if args.detections:
        with open(args.detections) as f:
            rows = list(csv.DictReader(f))
        kx = "x" if "x" in rows[0] else "cx"; ky = "y" if "y" in rows[0] else "cy"
        det = np.array([[float(r[kx]), float(r[ky])] for r in rows])

    Picker(pts, g, args.out, Path(args.map).stem, args.radius, det,
           h_range=(args.h_lo, args.h_hi), h_limits=(-0.2, args.h_max_load), px=args.px)
    plt.show()


if __name__ == "__main__":
    main()