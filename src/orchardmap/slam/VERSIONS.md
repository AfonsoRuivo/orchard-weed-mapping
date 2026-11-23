# SLAM estimators: upstream versions and local changes

No third-party source is included here. Each estimator is built from the upstream commit below, with the patch in
`patches/` applied where one is listed. Patches are `git diff HEAD` of the local checkout in `~/ros2_ws/src`, so they
cover tracked files only. Our own config files that were new in those checkouts are staged separately.

| Estimator | Upstream repository | Commit / version | Local patch | Our config |
|---|---|---|---|---|
| LIO-SAM | https://github.com/TixiaoShan/LIO-SAM (branch `ros2`) | `08af3f32f01725372d4269838dc44c19c6d9e76b` (2025-02-14) | `patches/lio_sam_08af3f3.patch`: `config/params.yaml`, `launch/run.launch.py`, `src/mapOptmization.cpp` (save-map path no longer prefixed with $HOME; edited 2026-05-31, before the paper runs) | `as_run/liosam_rep1_params_used.yaml` (exact params of the reference run rep1) |
| DLIO | https://github.com/vectr-ucla/direct_lidar_inertial_odometry (branch `feature/ros2`) | `c8acc37100e349d70a9d8432d656cbce7e5072cd` (2024-11-21) | `patches/dlio_c8acc37_publish_mutex.patch`: two mutexes around path/keyframe publishing (race fix, 2026-10-07). **The paper's DLIO run `base_full` (2026-10-06 17:24) ran on unpatched c8acc37**; only the repeat J5b used the patch. The estimate itself is not touched. | `dlio/dlio_orchard.yaml`, `dlio/dlio_orchard.launch.py`, `as_run/dlio_base_full_params_override.yaml` |
| KISS-ICP | https://github.com/PRBonn/kiss-icp (branch `main`) | `b16835283aee62f7d5e2bdf6c1c3bb2930de74ff` (v1.3.0, 2026-04-26) | none (clean checkout) | `as_run/kiss_tuned_rep1_params.txt` (parameters plus the ground filter z_min = -0.45 m) |
| FAST-LIO | https://github.com/Caltech-AMBER/FAST_LIO_ROS2 (branch `ros2`) | `60b11c70bfed36d4afebc60acf99fd1bb5213c65` (2025-07-01) | `patches/fast_lio_ros2_60b11c7.patch`: `config/mid360.yaml`, `launch/mapping.launch.py`, `src/IMU_Processing.hpp` | `fast_lio/velodyne_orchard.yaml` |
| RTAB-Map | https://github.com/introlab/rtabmap_ros | Ubuntu package `ros-humble-rtabmap-ros 0.22.1-1jammy.20260328.175427` (core `ros-humble-rtabmap 0.22.1-1jammy.20260307.204116`); installed from apt, so there is no git commit | none | `rtabmap/rtabmap_kissicp_loop.launch.py` (byte-identical to the `launch_used.py` saved with the paper's run `kisstuned_rtabloop_r1.0_20261004_195621`), `rtabmap/rtabmap_kissicp.launch.py` |

Runs reported in the thesis (`thesis-document/figures/data/estimator_runs.csv`), with the build and config each used

| Estimator | Run | Run directory (orchard-slam/data/...) | Build | Config |
|---|---|---|---|---|
| LIO-SAM | r08 | lio_sam/sweeps/…_rli_calib_20261004_142635/baseline | 08af3f3 + patch | `params_used.yaml`, identical to rep1's except the save path |
| LIO-SAM | rep1, rep2 | lio_sam/sweeps/…_6run_r05_20261004_220758/baseline_rep{1,2} | 08af3f3 + patch | `as_run/liosam_rep1_params_used.yaml` (rep2 the same except the save path) |
| LIO-SAM | J1 | lio_sam/runs/liosam_baseline_loops_20261007_103848 | 08af3f3 + patch | the same, except the save path |
| KISS-ICP | ng_p20, rep1, rep2, rep3 | kiss_icp/runs/…_ng_p20_nodeskew_20261004_192110, …_tuned_rep{1,2}_20261005_*, kiss_tuned_rep3_20261008_103812 | b168352 | `as_run/kiss_tuned_rep1_params.txt` (all four have the same parameters) |
| DLIO | base_full | dlio/runs/…_base_full_20261006_172420 | **c8acc37, unpatched** (the patch is from 2026-10-07 14:00) | `dlio/dlio_orchard.yaml` + `as_run/dlio_base_full_params_override.yaml` |
| DLIO | J5b (rep4) | dlio/runs/dlio_rep4_20261007_145324 | **c8acc37 + publish-mutex patch** | the same; the override is identical |
| RTAB-Map + KISS-ICP | loop run (trajectory pack) | rtabmap/runs/…_kisstuned_rtabloop_r1.0_20261004_195621 | apt 0.22.1 + KISS b168352 | `rtabmap/rtabmap_kissicp_loop.launch.py` |

Notes
- FAST-LIO: the installed `fast_lio` package is a symlink install of `FAST_LIO_ROS2`. A second checkout,
  `~/ros2_ws/src/FAST_LIO` (Ericsii fork, `2fffc57`), has `COLCON_IGNORE` and was not built.
- FAST-LIO never completed the CB-11 bag. No FAST-LIO trajectory is in the thesis trajectory pack, so the run the paper
  reports, if any, is not identified here.
- Every run used ROS 2 Humble on Ubuntu 22.04.
