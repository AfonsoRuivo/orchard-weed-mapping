#!/usr/bin/env python3
"""
LIO-SAM one-factor-at-a-time (OFAT) parameter sweep on a single bag.

Runs the exact same 5-terminal manual recipe the user normally drives by hand:

  1. ros2 launch lio_sam run.launch.py params_file:=<per-run params.yaml>
  2. python3 liosam/imu_fix_quat_node.py
  3. ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link velodyne
  4. python3 trajectories/recorders/liosam_trajectory_recorder.py <per-run trajectory.csv>
  5. ros2 bag play <bag> --clock --rate <rate>

...once per parameter variant, each writing its trajectory CSV, saved map
(PCD via the /lio_sam/save_map service), the exact params.yaml used, and a
log file into its own directory, so nothing gets overwritten between runs.

Baseline params file: config/params.yaml from the installed 'lio_sam' package
share directory (resolved via ament_index_python at runtime - the generic,
non-orchard-tuned one, as opposed to params_orchard.yaml) - this is
intentional: the sweep is meant to explore sensitivity around the generic
baseline, not the already-tuned orchard config.

Usage:
    python3 liosam_param_sweep.py --bag /path/to/bag_dir [options]

Run from a terminal where the LIO-SAM ROS2 workspace is already sourced
(same precondition as the manual recipe - this script does not source
anything for you).
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit(
        "PyYAML is required (pip install pyyaml / pip3 install pyyaml --break-system-packages) "
        "and was not found in this Python environment."
    )

try:
    from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
except ImportError:
    sys.exit(
        "ament_index_python is required and was not found - source your ROS2 workspace "
        "(e.g. `source ~/ros2_ws/install/setup.bash`) before running this script."
    )

# ---------------------------------------------------------------------------
# Paths (derived relative to this script's location; do not hardcode a user)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ORCHARD_SLAM = SCRIPT_DIR.parent
try:
    LIO_SAM_SHARE = Path(get_package_share_directory("lio_sam"))
except PackageNotFoundError:
    sys.exit(
        "ROS2 package 'lio_sam' not found - source your ROS2 workspace "
        "(e.g. `source ~/ros2_ws/install/setup.bash`) before running this script."
    )
BASE_PARAMS = LIO_SAM_SHARE / "config" / "params.yaml"
RECORDER_SCRIPT = ORCHARD_SLAM / "trajectories" / "recorders" / "liosam_trajectory_recorder.py"
RUN_RECORDER_SCRIPT = ORCHARD_SLAM / "trajectories" / "recorders" / "liosam_run_recorder.py"
PROC_SAMPLER_SCRIPT = SCRIPT_DIR / "proc_sampler.py"
EXPORT_SCRIPT = SCRIPT_DIR / "liosam_export.py"
IMU_FIX_SCRIPT = ORCHARD_SLAM / "liosam" / "imu_fix_quat_node.py"
DEFAULT_SWEEP_ROOT = ORCHARD_SLAM / "data" / "lio_sam" / "sweeps"

# ---------------------------------------------------------------------------
# Timing (seconds). The manual recipe's waits are human-paced - the user
# naturally waits for each terminal's output to settle before typing the
# next command. Firing fixed sleeps back-to-back doesn't give that same
# slack, and can race LIO-SAM's node/subscription startup (seen in practice:
# early IMU/point-cloud messages dropped during DDS discovery lag caused
# immediate pose divergence). So instead of guessing longer fixed sleeps,
# each step polls `ros2 node list` until the node(s) it depends on are
# actually up, with a generous timeout and a short settle buffer after.
# ---------------------------------------------------------------------------
LAUNCH_READY_TIMEOUT = 30
IMU_FIX_READY_TIMEOUT = 15
STATIC_TF_READY_TIMEOUT = 10
RECORDER_READY_TIMEOUT = 10
READY_POLL_INTERVAL = 0.5
POST_READY_SETTLE = 2
TEARDOWN_GRACE = 6

LAUNCH_EXPECTED_NODES = [
    "/robot_state_publisher",
    "/lio_sam_imuPreintegration",
    "/lio_sam_imageProjection",
    "/lio_sam_featureExtraction",
    "/lio_sam_mapOptimization",
]

# foxglove_bridge is launched with use_foxglove:=false for every sweep run (see
# run_variant): its websocket serialization of every advertised topic,
# including the deskewed cloud and global map, is a variable CPU/memory load
# that can starve the bag player independent of the parameter under test
# (this is suspected to be part of what --read-ahead-queue-size's help text
# describes). It's deliberately left out of LAUNCH_EXPECTED_NODES above
# since a sweep never starts it.


def wait_for_nodes(expected=(), prefix_counts=None, timeout=30.0, poll_interval=READY_POLL_INTERVAL):
    """Poll `ros2 node list` until every name in `expected` is present and every
    prefix in `prefix_counts` has at least that many matching nodes, or until
    `timeout` elapses. Returns True if satisfied, False on timeout (caller
    should warn and proceed rather than hang forever - a genuinely stuck
    launch will still surface as a broken run downstream)."""
    prefix_counts = prefix_counts or {}
    deadline = time.time() + timeout
    while True:
        try:
            result = subprocess.run(
                ["ros2", "node", "list"], capture_output=True, text=True, timeout=5,
            )
            nodes = result.stdout.splitlines()
        except Exception:
            nodes = []
        if all(name in nodes for name in expected) and all(
            sum(1 for n in nodes if n.startswith(p)) >= c for p, c in prefix_counts.items()
        ):
            return True
        if time.time() >= deadline:
            return False
        time.sleep(poll_interval)

# ---------------------------------------------------------------------------
# OFAT variants. lidarMaxRange=100 is not repeated as its own variant since
# it is already the baseline's default value.
# ---------------------------------------------------------------------------
VARIANTS = [
    {"id": "baseline", "label": "Baseline (params.yaml as-is)", "overrides": {}},
    # --- 2026-10-04 set: calibration / vibration / IMU trust / map density ---
    # "imu_fix_params" are extra ROS params for liosam/imu_fix_quat_node.py.
    {
        "id": "calib_ablation",
        "label": "IMU rotation = plain 90-deg permutation (no calibrated yaw/tilt)",
        "overrides": {},
        "imu_fix_params": ["rotation:=permutation"],
    },
    {
        "id": "imu_lowpass",
        "label": "4 Hz Butterworth low-pass on accel + gyro (vibration)",
        "overrides": {},
        "imu_fix_params": ["acc_lowpass_hz:=4.0", "gyr_lowpass_hz:=4.0"],
    },
    {
        "id": "imu_noise_high",
        "label": "imuAccNoise 0.1, imuGyrNoise 0.01 (trust the vibrating IMU less)",
        "overrides": {"imuAccNoise": 0.1, "imuGyrNoise": 0.01},
    },
    {
        "id": "imu_time_offset_30ms",
        "label": "IMU stamps +30 ms (calibrated IMU-LiDAR time offset)",
        "overrides": {},
        "imu_fix_params": ["time_offset:=0.03"],
    },
    {
        "id": "rtabmap_on_liosam",
        "label": "Baseline LIO-SAM + RTAB-Map (loop closure) on LIO-SAM odometry",
        "overrides": {},
        "extra_launch": "rtabmap/rtabmap_liosam.launch.py",
    },
    {"id": "baseline_rep1", "label": "Baseline repeat 1 (repeatability)", "overrides": {}},
    # --- 2026-10-07 thesis job list (J1-J4) ---
    {"id": "baseline_loops", "label": "Baseline (= baseline_rep1), recording loop closures (J1)", "overrides": {}},
    {
        "id": "imu_rot_perm_tilt",
        "label": "IMU rotation = permutation + calibrated roll/pitch, yaw removed (J3)",
        "overrides": {},
        "imu_fix_params": ["rotation:=custom",
                           "rotation_matrix:=[-0.0015,-0.0482,0.9988,0.9995,-0.0307,0.0,0.0306,0.9984,0.0482]"],
    },
    {
        "id": "imu_rot_camlidar",
        "label": "IMU rotation implied by camera-LiDAR calibration, R_LC*diag(-1,-1,1) (J4)",
        "overrides": {},
        "imu_fix_params": ["rotation:=custom",
                           "rotation_matrix:=[0.0072,0.0319,0.9995,0.9999,-0.0043,-0.0070,0.0041,0.9995,-0.0320]"],
    },
    {
        # J10 held-out bag (2026-05-25): the camera IMU sat ~6 deg differently. Calibrated R_LI plus the tilt that
        # aligns R*accel with the LiDAR ground normal on that bag (median over 749 scans, yaw kept).
        # NOT a CB-11 configuration; only for the labelled J10 follow-up.
        "id": "imu_rot_j10_tiltfix",
        "label": "J10 only: calibrated R_LI + 6.2 deg tilt from LiDAR ground normal of the held-out bag",
        "overrides": {},
        "imu_fix_params": ["rotation:=custom",
                           "rotation_matrix:=[0.1820,-0.1279,0.9749,0.9828,0.0553,-0.1763,-0.0314,0.9903,0.1357]"],
    },
    {"id": "baseline_rep2", "label": "Baseline repeat 2 (repeatability)", "overrides": {}},
    {
        "id": "keyframe_dist_05",
        "label": "surroundingkeyframeAddingDistThreshold = 0.5 m (baseline 1.0 m)",
        "overrides": {"surroundingkeyframeAddingDistThreshold": 0.5},
    },
    {
        "id": "leaf_finer_moderate",
        "label": "Finer leaf sizes: corner 0.1, surf 0.2, odom surf 0.2",
        "overrides": {"mappingCornerLeafSize": 0.1, "mappingSurfLeafSize": 0.2, "odometrySurfLeafSize": 0.2},
    },
    {
        "id": "loop_closure_off",
        "label": "Loop closure disabled",
        "overrides": {"loopClosureEnableFlag": False},
    },
    {
        "id": "leaf_finer",
        "label": "Finer voxel leaf sizes (denser map)",
        "overrides": {
            "mappingCornerLeafSize": 0.1,
            "mappingSurfLeafSize": 0.2,
            "odometrySurfLeafSize": 0.025,
        },
    },
    {
        "id": "leaf_coarser",
        "label": "Coarser voxel leaf sizes (sparser map)",
        "overrides": {
            "mappingCornerLeafSize": 0.4,
            "mappingSurfLeafSize": 0.8,
            "odometrySurfLeafSize": 0.1,
        },
    },
    {
        "id": "lidar_range_30",
        "label": "lidarMaxRange = 30 m",
        "overrides": {"lidarMaxRange": 30.0},
    },
    {
        "id": "lidar_range_50",
        "label": "lidarMaxRange = 50 m",
        "overrides": {"lidarMaxRange": 50.0},
    },
    {
        "id": "features_stricter",
        "label": "Stricter edge/surf feature thresholds (fewer, higher-quality features)",
        "overrides": {"edgeThreshold": 2.0, "surfThreshold": 0.2},
    },
    {
        "id": "features_looser",
        "label": "Looser edge/surf feature thresholds (more, noisier features)",
        "overrides": {"edgeThreshold": 0.5, "surfThreshold": 0.05},
    },
    {
        "id": "history_radius_30",
        "label": "historyKeyframeSearchRadius = 30 m (baseline 15 m)",
        "overrides": {"historyKeyframeSearchRadius": 30.0},
    },
    {
        "id": "history_radius_50",
        "label": "historyKeyframeSearchRadius = 50 m (baseline 15 m)",
        "overrides": {"historyKeyframeSearchRadius": 50.0},
    },
    {
        "id": "surrounding_radius_20",
        "label": "surroundingKeyframeSearchRadius = 20 m (baseline 50 m)",
        "overrides": {"surroundingKeyframeSearchRadius": 20.0},
    },
    {
        "id": "surrounding_radius_10",
        "label": "surroundingKeyframeSearchRadius = 10 m (baseline 50 m)",
        "overrides": {"surroundingKeyframeSearchRadius": 10.0},
    },
    {
        "id": "odom_leaf_03",
        "label": "odometrySurfLeafSize = 0.3 (baseline 0.4)",
        "overrides": {"odometrySurfLeafSize": 0.3},
    },
    {
        "id": "odom_leaf_02",
        "label": "odometrySurfLeafSize = 0.2 (baseline 0.4)",
        "overrides": {"odometrySurfLeafSize": 0.2},
    },
    {
        "id": "mapping_interval_005",
        "label": "mappingProcessInterval = 0.05 (baseline 0.15)",
        "overrides": {"mappingProcessInterval": 0.05},
    },
    {
        "id": "mapping_interval_03",
        "label": "mappingProcessInterval = 0.3 (baseline 0.15)",
        "overrides": {"mappingProcessInterval": 0.3},
    },
]


def load_base_params():
    with open(BASE_PARAMS) as f:
        return yaml.safe_load(f)


def build_params_for_variant(base, overrides, map_dir):
    data = deepcopy(base)
    ros_params = data["/**"]["ros__parameters"]
    for key, value in overrides.items():
        if key not in ros_params:
            raise KeyError(f"Override key '{key}' not found in base params.yaml - typo?")
        ros_params[key] = value
    # Always point PCD export at this run's own map directory.
    ros_params["savePCD"] = True
    ros_params["savePCDDirectory"] = str(map_dir) + "/"
    return data


class RunHandle:
    """Tracks one subprocess for later graceful teardown."""

    def __init__(self, name, proc, logfile):
        self.name = name
        self.proc = proc
        self.logfile = logfile

    def alive(self):
        return self.proc.poll() is None

    def stop(self, grace=TEARDOWN_GRACE):
        if not self.alive():
            return
        try:
            pgid = os.getpgid(self.proc.pid)
            os.killpg(pgid, signal.SIGINT)
        except ProcessLookupError:
            return
        deadline = time.time() + grace
        while time.time() < deadline and self.alive():
            time.sleep(0.3)
        if self.alive():
            try:
                pgid = os.getpgid(self.proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait(timeout=5)


def spawn(name, args, run_dir, extra_env=None):
    log_path = run_dir / f"{name}.log"
    log_f = open(log_path, "w")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(
        args,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    return RunHandle(name, proc, log_f)


def run_variant(variant, bag_path, sweep_root, rate, settle, resolution, dry_run, read_ahead_queue_size=None, start_offset=None, flat=False):
    run_dir = sweep_root if flat else sweep_root / variant["id"]
    map_dir = run_dir / "map"
    run_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_params()
    params_data = build_params_for_variant(base, variant["overrides"], map_dir)
    params_path = run_dir / "params_used.yaml"
    with open(params_path, "w") as f:
        yaml.safe_dump(params_data, f, default_flow_style=False, sort_keys=False)

    trajectory_csv = run_dir / "trajectory.csv"
    manifest_entry = {
        "id": variant["id"],
        "label": variant["label"],
        "overrides": variant["overrides"],
        "imu_fix_params": variant.get("imu_fix_params", []),
        "params_used": str(params_path),
        "trajectory_csv": str(trajectory_csv),
        "map_dir": str(map_dir),
        "run_dir": str(run_dir),
        "start_offset": start_offset,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "save_map_success": None,
        "error": None,
    }

    if dry_run:
        print(f"[DRY-RUN] {variant['id']}: would write {params_path} and run bag {bag_path}")
        manifest_entry["dry_run"] = True
        return manifest_entry

    print(f"\n=== Running variant '{variant['id']}' - {variant['label']} ===")
    print(f"    run dir: {run_dir}")

    handles = []
    t_start = time.time()
    try:
        launch = spawn(
            "launch",
            ["ros2", "launch", "lio_sam", "run.launch.py",
             f"params_file:={params_path}", "use_foxglove:=false"],
            run_dir,
        )
        handles.append(launch)
        if not wait_for_nodes(LAUNCH_EXPECTED_NODES, timeout=LAUNCH_READY_TIMEOUT):
            print(f"    WARNING: not all lio_sam nodes came up within {LAUNCH_READY_TIMEOUT}s, proceeding anyway")
        time.sleep(POST_READY_SETTLE)

        imu_fix = spawn(
            "imu_fix",
            ["python3", str(IMU_FIX_SCRIPT), "--ros-args", "-p", "use_sim_time:=true"]
            + [a for prm in variant.get("imu_fix_params", []) for a in ("-p", prm)],
            run_dir,
        )
        handles.append(imu_fix)
        if not wait_for_nodes(["/imu_fix_quat_node"], timeout=IMU_FIX_READY_TIMEOUT):
            print(f"    WARNING: imu_fix_quat_node did not come up within {IMU_FIX_READY_TIMEOUT}s, proceeding anyway")
        time.sleep(POST_READY_SETTLE)

        static_tf = spawn(
            "static_tf",
            ["ros2", "run", "tf2_ros", "static_transform_publisher",
             "--ros-args", "-p", "use_sim_time:=true", "--",
             "0", "0", "0", "0", "0", "0", "base_link", "velodyne"],
            run_dir,
        )
        handles.append(static_tf)
        if not wait_for_nodes(prefix_counts={"/static_transform_publisher": 2}, timeout=STATIC_TF_READY_TIMEOUT):
            print(f"    WARNING: static_transform_publisher nodes did not both come up within {STATIC_TF_READY_TIMEOUT}s, proceeding anyway")
        time.sleep(POST_READY_SETTLE)

        recorder = spawn(
            "recorder", ["python3", str(RECORDER_SCRIPT), str(trajectory_csv)], run_dir
        )
        handles.append(recorder)
        if not wait_for_nodes(["/trajectory_recorder"], timeout=RECORDER_READY_TIMEOUT):
            print(f"    WARNING: trajectory_recorder did not come up within {RECORDER_READY_TIMEOUT}s, proceeding anyway")
        time.sleep(POST_READY_SETTLE)

        # full-rate odometry, live keyframes and loop edges (only subscribes) + CPU/RAM sampling
        handles.append(spawn("run_recorder", ["python3", str(RUN_RECORDER_SCRIPT), str(run_dir)], run_dir))
        handles.append(spawn("proc_sampler", ["python3", str(PROC_SAMPLER_SCRIPT), r"lio_sam_\w+", str(run_dir)], run_dir))
        time.sleep(POST_READY_SETTLE)

        if variant.get("extra_launch"):
            extra = spawn("extra_launch", ["ros2", "launch", str(ORCHARD_SLAM / variant["extra_launch"]),
                                           f"out_dir:={run_dir}"], run_dir)
            handles.append(extra)
            time.sleep(15)

        bag_cmd = ["ros2", "bag", "play", str(bag_path), "--clock", "--rate", str(rate)]
        if read_ahead_queue_size:
            bag_cmd += ["--read-ahead-queue-size", str(read_ahead_queue_size)]
        if start_offset:
            bag_cmd += ["--start-offset", str(start_offset)]
        print(f"    playing bag ({bag_path}, rate={rate}, read_ahead_queue_size={read_ahead_queue_size}, start_offset={start_offset}) ...")
        bag_log = open(run_dir / "bag_play.log", "w")
        bag_proc = subprocess.run(
            bag_cmd,
            stdout=bag_log,
            stderr=subprocess.STDOUT,
        )
        bag_log.close()
        if bag_proc.returncode != 0:
            manifest_entry["error"] = f"ros2 bag play exited {bag_proc.returncode}"
            print(f"    WARNING: bag play exited {bag_proc.returncode}, see bag_play.log")

        print(f"    bag finished, settling {settle}s before save_map ...")
        time.sleep(settle)

        save_call = subprocess.run(
            [
                "ros2", "service", "call", "/lio_sam/save_map", "lio_sam/srv/SaveMap",
                f"{{resolution: {resolution}, destination: '{map_dir}/'}}",
            ],
            capture_output=True, text=True, timeout=120,
        )
        (run_dir / "save_map.log").write_text(save_call.stdout + save_call.stderr)
        # ros2 service call prints the response as repr, e.g.
        # "lio_sam.srv.SaveMap_Response(success=True)" - match both that and a
        # plain "success: true" just in case the CLI's formatting ever changes.
        manifest_entry["save_map_success"] = bool(
            re.search(r"success\s*[:=]\s*true", save_call.stdout.lower())
        )
        if not manifest_entry["save_map_success"]:
            print("    WARNING: save_map did not report success - check save_map.log")

    except Exception as exc:  # keep sweeping even if one variant blows up
        manifest_entry["error"] = f"{type(exc).__name__}: {exc}"
        print(f"    ERROR in variant '{variant['id']}': {manifest_entry['error']}")

    finally:
        print("    tearing down ...")
        # Reverse start order; recorder first so its SIGINT-triggered save runs
        # while the topics it depends on are still alive.
        for h in reversed(handles):
            h.stop()
            h.logfile.close()

    if not dry_run:
        try:
            log = (run_dir / "bag_play.log").read_text(errors="ignore")
            manifest_entry["starvation_warnings"] = log.count("starved")
        except FileNotFoundError:
            manifest_entry["starvation_warnings"] = None
        try:
            manifest_entry["resources"] = json.loads((run_dir / "resources.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            manifest_entry["resources"] = None
        exp = subprocess.run(["python3", str(EXPORT_SCRIPT), str(run_dir)], capture_output=True, text=True)
        (run_dir / "export.log").write_text(exp.stdout + exp.stderr)
        manifest_entry["export"] = exp.stdout.strip() or exp.stderr.strip()[-300:]
        (run_dir / "run_entry.json").write_text(json.dumps(manifest_entry, indent=2))

    manifest_entry["elapsed_seconds"] = round(time.time() - t_start, 1)
    manifest_entry["finished_at"] = datetime.now().isoformat(timespec="seconds")
    return manifest_entry


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True, help="Path to the rosbag2 directory to play")
    ap.add_argument("--out", default=None, help="Sweep output root (default: data/lio_sam/sweeps/<bag>_<timestamp>)")
    ap.add_argument("--only", default=None, help="Comma-separated variant ids to run (default: all)")
    ap.add_argument("--list", action="store_true", help="List available variants and exit")
    ap.add_argument("--rate", default="0.3", help="ros2 bag play --rate (default 0.3, matches manual recipe)")
    ap.add_argument("--settle", type=float, default=8.0, help="Seconds to wait after bag finishes before calling save_map")
    ap.add_argument("--resolution", default="0.2", help="save_map PCD resolution (default 0.2)")
    ap.add_argument("--dry-run", action="store_true", help="Only write params_used.yaml per variant, don't launch anything")
    ap.add_argument("--read-ahead-queue-size", default=None, help="ros2 bag play --read-ahead-queue-size (default: unset, uses ros2's own default). Bump this if bag_play.log shows 'Message queue starved' warnings - seen to correlate with LIO-SAM divergence unrelated to any actual parameter under test (e.g. session_20260706_140610(best): 0 warnings in the original Aug run, 34-55 in later re-runs of the same config).")
    ap.add_argument("--flat", action="store_true", help="Write the (single) variant directly into --out instead of --out/<variant id> (used by the 2026-10-07 job queue's <method>_<variant>_<stamp> naming)")
    ap.add_argument("--start-offset", default=None, help="ros2 bag play --start-offset in bag-seconds, applied to every variant in this sweep")
    args = ap.parse_args()

    if args.list:
        for v in VARIANTS:
            print(f"  {v['id']:<20} {v['label']}")
        return

    if not BASE_PARAMS.exists():
        sys.exit(f"Base params file not found: {BASE_PARAMS}")
    if not RECORDER_SCRIPT.exists():
        sys.exit(f"Recorder script not found: {RECORDER_SCRIPT}")
    if not IMU_FIX_SCRIPT.exists():
        sys.exit(f"IMU fix node not found: {IMU_FIX_SCRIPT}")

    bag_path = Path(args.bag).expanduser().resolve()
    if not args.dry_run and not bag_path.exists():
        sys.exit(f"Bag path not found: {bag_path}")

    variants = VARIANTS
    if args.only:
        wanted = set(args.only.split(","))
        variants = [v for v in VARIANTS if v["id"] in wanted]
        missing = wanted - {v["id"] for v in variants}
        if missing:
            sys.exit(f"Unknown variant id(s): {', '.join(sorted(missing))}")

    if args.out:
        sweep_root = Path(args.out).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_root = DEFAULT_SWEEP_ROOT / f"{bag_path.name}_{stamp}"
    if args.flat and len(variants) != 1:
        sys.exit("--flat needs exactly one variant (--only <id>)")
    if args.flat and sweep_root.exists() and any(sweep_root.iterdir()):
        sys.exit(f"--flat output dir {sweep_root} is not empty - refusing to overwrite a run")
    sweep_root.mkdir(parents=True, exist_ok=True)

    print(f"Bag:         {bag_path}")
    print(f"Sweep root:  {sweep_root}")
    print(f"Variants:    {', '.join(v['id'] for v in variants)}")
    print(f"Base params: {BASE_PARAMS}")
    print(f"Start offset: {args.start_offset}")

    manifest = {
        "bag": str(bag_path),
        "sweep_root": str(sweep_root),
        "base_params": str(BASE_PARAMS),
        "rate": args.rate,
        "settle": args.settle,
        "resolution": args.resolution,
        "start_offset": args.start_offset,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "runs": [],
    }

    for i, variant in enumerate(variants):
        entry = run_variant(
            variant, bag_path, sweep_root, args.rate, args.settle, args.resolution, args.dry_run,
            read_ahead_queue_size=args.read_ahead_queue_size,
            start_offset=args.start_offset, flat=args.flat,
        )
        manifest["runs"].append(entry)
        (sweep_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if i < len(variants) - 1 and not args.dry_run:
            print("    cooling down before next run ...")
            time.sleep(5)

    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    (sweep_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nSweep complete. Manifest: {sweep_root / 'manifest.json'}")
    n_ok = sum(1 for r in manifest["runs"] if r.get("save_map_success"))
    print(f"{n_ok}/{len(manifest['runs'])} variants saved a map successfully.")


if __name__ == "__main__":
    main()
