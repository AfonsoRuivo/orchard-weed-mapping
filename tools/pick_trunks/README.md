# pick_trunks

`trunk_picker.py` shows a point-cloud map as a horizontal slice above the local
ground (ground = 5th percentile of z in 0.5 m cells) and lets you click the
centre of each trunk. Picks are numbered row by row in the order of the field
sheets, R1-T1, R1-T2, …, and written to CSV as you go, so a session can be
stopped and resumed.

```bash
python3 trunk_picker.py --map lio_sam_dense.pcd --out picks.csv \
        --h-lo 0.08 --h-hi 0.30 --radius 0.10
```

## What this version does not do

The 106 surveyed centres released with the dataset (`trunk_survey/`) carry four
columns this script does not write: `ladder`, `set_by`, `r_fit_m` and
`picked_as`. Of those 106 centres, **37 are `set_by = hand` and 69 are
`set_by = circle_fit`** with a fitted radius, so for most of them the centre
came from a circle fit that is not in this file. The method is described in the
survey notes: a slice 0.08–0.30 m above the local ground, RANSAC on three
points with the centre held near the seed, then a geometric circle refit on the
inliers.

This file is therefore an earlier version of the tool, kept because it is the
picking interface that produced the identifiers and the hand-placed centres.
The fitting stage is missing and has not been found.
