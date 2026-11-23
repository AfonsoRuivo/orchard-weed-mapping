"""Geometric prediction of the position scatter of a ground-projected detection (the 'plane-sensitivity' / 'sigma-loc'
model of Ch5/Ch8, derived here from the pipeline itself because the equation is not in the repo).

Position p = ray(u, v) intersected with the plane n.x + d = 0 (n unit normal in the LiDAR frame), then pose-transformed.
First-order error propagation, J = numeric Jacobian of weed_map_cached.pixel_to_ground:
    var(p) = sum_k J_k^2 sigma_k^2,   k in {plane pitch, plane roll, plane offset d, pixel v, pixel u}
plus a pose term (yaw error * range across track).  Closed-form small-angle intuition for the along-track part
    dx ~ (rho^2 / h) (d_theta + d_v / f) + (rho / h) d_h         (rho = ground range, h = sensor height)
so the plane tilt error enters with rho^2/h, the offset with rho/h, and the pixel row with rho^2/(h f).
"""
import numpy as np
import weed_map_cached as wm

def _rot(axis, ang):
    c, s = np.cos(ang), np.sin(ang)
    return {'x': np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), 'y': np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])}[axis]

def _ground(u, v, plane, T):
    pw, _ = wm.pixel_to_ground(u, v, *plane, T)
    return None if pw is None else pw[:2]

def jacobian(u, v, plane, T, d_ang=np.radians(0.5), d_off=0.01, d_pix=1.0):
    """Columns: d(p_world_xy)/d{pitch(about lidar y), roll(about lidar x), offset, v, u} per unit (rad, rad, m, px, px)."""
    a, b, c, d = plane; n = np.array([a, b, c]); nl = np.linalg.norm(n); n = n / nl; d = d / nl
    p0 = _ground(u, v, (*n, d), T)
    if p0 is None: return None
    foot = -d * n                                       # point of the plane below the sensor
    cols = []
    for ax in ('y', 'x'):
        n2 = _rot(ax, d_ang) @ n; d2 = -n2 @ foot
        p = _ground(u, v, (*n2, d2), T); cols.append(None if p is None else (p - p0) / d_ang)
    p = _ground(u, v, (*n, d + d_off), T); cols.append(None if p is None else (p - p0) / d_off)
    p = _ground(u, v + d_pix, (*n, d), T); cols.append(None if p is None else (p - p0) / d_pix)
    p = _ground(u + d_pix, v, (*n, d), T); cols.append(None if p is None else (p - p0) / d_pix)
    return None if any(c_ is None for c_ in cols) else np.array(cols).T   # 2 x 5

def predict_axes(J, ux, uy, sig):
    """sig = dict(pitch, roll, off, v, u) 1-sigma; returns (var_along, var_across) for travel direction (ux, uy)."""
    S = np.diag([sig['pitch'] ** 2, sig['roll'] ** 2, sig['off'] ** 2, sig['v'] ** 2, sig['u'] ** 2])
    C = J @ S @ J.T; e1 = np.array([ux, uy]); e2 = np.array([-uy, ux])
    return float(e1 @ C @ e1), float(e2 @ C @ e2)
