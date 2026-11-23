#!/usr/bin/env python3
"""Satellite background for the CA-10 bundle, made as the CB-11 one (viz_bundle_20261014/satellite/):
Esri World Imagery tiles via contextily at zoom 19 (EPSG:3857 mosaic), an affine ENU -> EPSG:3857 fitted on the CA-10
GNSS samples (east/north vs lon/lat), and a 0.1 m/px resample on the CA-10 ENU grid covering the GNSS track + 10 m.
Writes viz_bundle_ca10/satellite/{esri_world_imagery_mercator.png, esri_world_imagery_enu.png, georeference.json}
with the same keys as CB-11.   Usage: satellite_ca10.py <ca10_run dir>"""
import json, sys
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd
import contextily as cx
from PIL import Image
from scipy.ndimage import map_coordinates

R = Path(sys.argv[1]); O = R / 'viz_bundle_ca10/satellite'; O.mkdir(parents=True, exist_ok=True)
g = pd.read_csv('/home/afonso/ca10_gap/gnss_enu.csv'); fx = pd.read_csv('/home/afonso/ca10_gap/gnss_fix.csv')
g = g.merge(fx[['timestamp', 'latitude', 'longitude']], on='timestamp')
RE = 6378137.0
mx = np.radians(g.longitude.values) * RE
my = np.log(np.tan(np.pi / 4 + np.radians(g.latitude.values) / 2)) * RE
X = np.column_stack([g.e.values, g.n.values, np.ones(len(g))])
M, *_ = np.linalg.lstsq(X, np.column_stack([mx, my]), rcond=None)
res = np.abs(X @ M - np.column_stack([mx, my])) / M[0, 0]
e0, e1 = g.e.min() - 10, g.e.max() + 10; n0, n1 = g.n.min() - 10, g.n.max() + 10
corners = np.array([[e0, n0, 1], [e1, n0, 1], [e0, n1, 1], [e1, n1, 1]]) @ M
w, s, e, n = corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()
img, ext = cx.bounds2img(w, s, e, n, zoom=19, source=cx.providers.Esri.WorldImagery, ll=False)   # ext = (l, r, b, t)
img = img[..., :3]
Image.fromarray(img).save(O / 'esri_world_imagery_mercator.png')
# ENU grid at 0.1 m/px, row 0 = north
res_m = 0.1
E = np.arange(e0, e1, res_m) + res_m / 2; N = np.arange(n1, n0, -res_m) - res_m / 2
EE, NN = np.meshgrid(E, N)
Mx = EE * M[0, 0] + NN * M[1, 0] + M[2, 0]; My = EE * M[0, 1] + NN * M[1, 1] + M[2, 1]
H, W = img.shape[:2]
col = (Mx - ext[0]) / (ext[1] - ext[0]) * W - 0.5; row = (ext[3] - My) / (ext[3] - ext[2]) * H - 0.5
enu = np.stack([map_coordinates(img[..., c].astype(float), [row, col], order=1, mode='nearest') for c in range(3)], -1)
Image.fromarray(np.clip(enu, 0, 255).astype(np.uint8)).save(O / 'esri_world_imagery_enu.png')
geo = {
    'source': f'Esri World Imagery tiles via contextily (attribution required: Esri, Maxar, Earthstar Geographics, and the GIS User Community); downloaded {date.today().isoformat()}',
    'zoom': 19,
    'enu_png': {'file': 'esri_world_imagery_enu.png', 'size_px': [int(len(E)), int(len(N))], 'metres_per_px': res_m,
                'extent_east_north': [float(e0), float(e0 + len(E) * res_m), float(n1 - len(N) * res_m), float(n1)],
                'use': 'imshow(img, extent=[e0,e1,n0,n1], origin="upper")',
                'frame': 'CA-10 GNSS ENU: x=east, y=north, origin = first GNSS fix (lat %.8f, lon %.8f)' % (fx.latitude.iloc[0], fx.longitude.iloc[0])},
    'mercator_png': {'file': 'esri_world_imagery_mercator.png', 'size_px': [int(W), int(H)],
                     'extent_epsg3857_left_right_bottom_top': [float(v) for v in ext]},
    'enu_to_epsg3857': {'formula': '[merc_x, merc_y] = [east, north, 1] @ M', 'M': M.tolist(),
                        'max_fit_resid_m': float(res.max()), 'fitted_on': 'CA-10 GNSS samples (east/north vs lon/lat)'},
    'caveat': 'GNSS is ~1 m off in absolute terms on CA-10 (merged SLAM vs GNSS RMSE 1.00 m); the imagery registration to ENU inherits that. '
              'Imagery date unknown (Esri mosaic).'}
json.dump(geo, open(O / 'georeference.json', 'w'), indent=1)
print(json.dumps({k: geo[k] for k in ('zoom', 'enu_png', 'mercator_png')}, indent=1), 'resid', res.max())
