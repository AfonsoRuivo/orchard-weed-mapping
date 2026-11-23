# gate/

The LiDAR height gate has no file of its own. It lives in `../weeds/weed_map_cached.py` (staged whole):

- constants, lines 79-90: `H_WEED = 0.40` (band 0.25-0.55 m), `N_STRIPS = 40`, `HORIZON_MODEL = 'line'`
- `get_horizon()`, line 226: projects the band into the image, takes the 40 strip maxima, and draws a straight
  boundary at their median, parallel to the ground plane's vanishing line
- where it is applied: `get_weed_boxes()` (YOLO, box bottom-centre) and `get_colour_boxes()` (mask, pixel by pixel)
