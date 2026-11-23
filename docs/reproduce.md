# Reproducing the paper

Figure and table numbers refer to the submitted manuscript.

## In this repository

| Output | Script |
|---|---|
| Fig. 8, trunk detection method | `figures/paper/trunk_method_col.py` |
| Fig. 11, detected trunks over imagery | `figures/paper/sat_maps.py` (needs the true GNSS fixes and the imagery; not reproducible from the public data) |
| Fig. 14, weed map and priority | `figures/paper/paper_map.py` with `figures/paper/heatlib.py` |
| Fig. 15, YOLO detection funnel | `figures/paper/stairs.py` |
| Fig. 16, detections to the map | `figures/paper/seq.py` |
| Fig. 17, single-view spread | `figures/paper/sig_eps.py`, `figures/analysis/fig_sigma_loc.py` |
| Fig. 18, CA-10 map | `figures/paper/ca10_3d.py` |
| Fig. 19, joining the CA-10 segments | `figures/paper/ca10_merge.py` |
| Fig. 20, CA-10 priority density | `figures/paper/ca10_heat.py` |
| Table 8, gating interval | `figures/analysis/s3b_gating.py` |
| Table 9, weed map summary | `figures/analysis/s4_map_summary.py` |
| Priority and its stability | `figures/analysis/s5_priority.py`, `figures/analysis/fig_priority_stability.py` |
| Merge-radius sweep | `figures/analysis/s3_eps_sweep.py` |
| Per-frame export used by several figures | `runs/appendix_export/export_appendix.py` |
| The whole CA-10 run | `runs/ca10/` (see the scripts in order: `extract_frames_ca10.py`, `run_liosam_half.py`, `merge.py`, `trunks.py`, `run_weed_ca10.py`, `post_weed.py`, `build_bundle.py`) |

## Not yet in this repository

These come from the machine the pipeline runs on and still have to be added:

| Output | What is missing |
|---|---|
| Fig. 4, session CB-11 overview | the overview script for the released session |
| Fig. 5, labelled examples | the label-rendering script |
| Fig. 7, the height gate on one frame | the gate figure script |
| Fig. 9, trajectories; Fig. 10, maps of CB-11 | the estimator comparison scripts |
| Fig. 12, trunk detection on CB-11 | the trunk result figure |
| Fig. 13, detector examples | the detector example script |
| Tables 4 and 5, estimators | ATE/RPE/MME evaluation |
| Table 6, accuracy against the survey | the survey comparison |
| Table 7, weed detection | the detection scoring, and `map/fp_audit.py` with its `audit_summary.json` (which also holds the confidence range of the audited precision) |

Figs. 1 and 6 are drawn in LaTeX in the manuscript; Figs. 2 and 3 are
photographs.
