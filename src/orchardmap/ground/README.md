# ground/

The per-frame RANSAC ground plane has no file of its own. It lives in `../weeds/weed_map_cached.py`,
which is staged whole so the paper's code is not split:

- constants, lines 76-87: `RADIUS = 8.0`, `PERCENTILE = 15`, `CANDIDATE_FRACTION = 0.20`, `RANSAC_DIST = 0.06`,
  `MAX_PLANE_TILT_DEG = 3.0`
- `ransac_plane()`, line 172: seeded RANSAC with the tilt limit
- `fit_ground()`, line 204: lowest 20 % within 8 m, then the plane lowered to the 15th percentile
