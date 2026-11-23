#!/usr/bin/env python3
"""
Single FAST-LIO2 test run on a bag, with a live drift watchdog.

Runs the same manual recipe used for LIO-SAM (see liosam_param_sweep.py),
adapted for FAST-LIO2:

  1. ros2 launch fast_lio mapping.launch.py config_file:=<config> use_sim_time:=true
     (this also brings up the camera_init/body static TFs; foxglove_bridge is
     kept off with use_foxglove:=false to avoid playback delays)
  2. ros2 run fast_imu_utils imu_fix_quat_node
  3. python3 fastlio_state_logger.py -p imu_topic:=<imu_topic> -p out_dir:=<run_dir>
  4. ros2 bag play <bag> --clock --rate <rate>

The IMU-fix relay IS spawned (unlike an earlier version of this script): FAST-LIO's
"body" frame is the raw IMU axis frame with no rotation ever applied to it - the
config's old extrinsic_R only rotated LiDAR points into that raw frame for internal
fusion, never the published /Odometry pose itself, so state.pos/state.rot came out
tilted ~90 degrees off the robot/velodyne axis convention (confirmed both visually
in Foxglove and by comparing recorded runs against real GPS - see run history).
velodyne_orchard.yaml now consumes the pre-corrected /camera/camera/imu_fixed with
extrinsic_R set to identity, exactly like LIO-SAM already does, so "body" comes out
natively robot-frame-aligned and the launch file's identity static TFs are correct.

Every --check-interval seconds (default 30s) this tails the state logger's
odom CSV and checks two physically-grounded divergence signals against the
latest row:
  - implied speed from position deltas (speed_from_pos) exceeding --max-speed
  - |z - z_at_start| exceeding --max-z-drift
If either trips, the bag playback is killed immediately (so the run doesn't
burn the rest of its ~40min wall-clock budget at rate=0.3 on a diverged
trajectory) and everything is torn down.

Run from a terminal where the ROS2 workspace is already sourced (same
precondition as liosam_param_sweep.py - this script does not source
anything for you).
"""
import argparse
import csv
import io
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
except ImportError:
    sys.exit(
        "ament_index_python is required and was not found - source your ROS2 workspace "
        "(e.g. `source ~/ros2_ws/install/setup.bash`) before running this script."
    )

SCRIPT_DIR = Path(__file__).resolve().parent
ORCHARD_SLAM = SCRIPT_DIR.parent
STATE_LOGGER_SCRIPT = Path.home() / "ros2_ws" / "fastlio_state_logger.py"
DEFAULT_RUNS_ROOT = ORCHARD_SLAM / "data" / "fast_lio" / "runs"
DEFAULT_BAG = Path.home() / "bags" / "campinas_full_run"

ODOM_COLS = [
    "timestamp", "x", "y", "z", "qx", "qy", "qz", "qw", "roll", "pitch", "yaw",
    "vx", "vy", "vz", "speed", "speed_from_pos", "wx", "wy", "wz",
    "cov0", "cov1", "cov2", "cov3", "cov4", "cov5", "cov_trace",
]

LAUNCH_READY_TIMEOUT = 30
LOGGER_READY_TIMEOUT = 15
READY_POLL_INTERVAL = 0.5
POST_READY_SETTLE = 2
TEARDOWN_GRACE = 6


def wait_for_nodes(expected=(), timeout=30.0, poll_interval=READY_POLL_INTERVAL):
    deadline = time.time() + timeout
    while True:
        try:
            result = subprocess.run(["ros2", "node", "list"], capture_output=True, text=True, timeout=5)
            nodes = result.stdout.splitlines()
        except Exception:
            nodes = []
        if all(name in nodes for name in expected):
            return True
        if time.time() >= deadline:
            return False
        time.sleep(poll_interval)


class RunHandle:
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
    proc = subprocess.Popen(args, stdout=log_f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    return RunHandle(name, proc, log_f)


def tail_last_row(csv_path, chunk=8192):
    """Read just the last non-empty line of csv_path without loading it all."""
    try:
        with open(csv_path, "rb") as f:
            f.seek(0, io.SEEK_END)
            size = f.tell()
            data = b""
            while size > 0 and data.count(b"\n") < 2:
                step = min(chunk, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
            lines = [l for l in data.splitlines() if l.strip()]
    except FileNotFoundError:
        return None
    if not lines:
        return None
    last = lines[-1].decode("utf-8", errors="replace")
    row = next(csv.reader([last]))
    if row == ODOM_COLS or len(row) != len(ODOM_COLS):
        return None
    return dict(zip(ODOM_COLS, row))


def watch_and_wait(bag_proc, odom_csv, check_interval, max_speed, max_z_drift, log):
    baseline_z = None
    while True:
        if bag_proc.poll() is not None:
            return "COMPLETED", None
        time.sleep(check_interval)
        if bag_proc.poll() is not None:
            return "COMPLETED", None
        row = tail_last_row(odom_csv)
        if row is None:
            log("  [watchdog] no odom rows yet")
            continue
        t = float(row["timestamp"])
        x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
        speed = float(row["speed_from_pos"])
        cov_trace = float(row["cov_trace"])
        if baseline_z is None:
            baseline_z = z
        z_drift = abs(z - baseline_z)
        log(f"  [watchdog] t={t:.1f}s x={x:.2f} y={y:.2f} z={z:.2f} "
            f"speed={speed:.2f}m/s z_drift={z_drift:.2f}m cov_trace={cov_trace:.4g}")

        reasons = []
        if speed > max_speed:
            reasons.append(f"implied speed {speed:.2f} m/s > --max-speed {max_speed} m/s")
        if z_drift > max_z_drift:
            reasons.append(f"|z - z0| {z_drift:.2f} m > --max-z-drift {max_z_drift} m")
        if reasons:
            log("  [watchdog] DIVERGENCE DETECTED: " + "; ".join(reasons))
            return "DIVERGED", "; ".join(reasons)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", default=str(DEFAULT_BAG), help="Path to the rosbag2 directory to play")
    ap.add_argument("--config", default="velodyne_orchard.yaml", help="FAST-LIO config file name (resolved against --config-path)")
    ap.add_argument("--config-path", default=None, help="Directory containing --config (default: fast_lio's installed config dir) - lets a sweep driver point at a per-run params_used.yaml instead of the installed configs")
    ap.add_argument("--imu-topic", default="/camera/camera/imu_fixed", help="IMU topic for the state logger (must match the config's common.imu_topic)")
    ap.add_argument("--out", default=None, help="Run output dir (default: data/fast_lio/runs/<bag>_<timestamp>)")
    ap.add_argument("--rate", default="0.3", help="ros2 bag play --rate (default 0.3, matches the LIO-SAM manual recipe / sweep script)")
    ap.add_argument("--start-offset", default=None, help="ros2 bag play --start-offset in bag-seconds, to skip a rough opening segment")
    ap.add_argument("--check-interval", type=float, default=30.0, help="Drift watchdog polling interval in seconds (default 30)")
    ap.add_argument("--max-speed", type=float, default=3.0, help="Kill run if implied speed exceeds this (m/s, default 3.0)")
    ap.add_argument("--max-z-drift", type=float, default=3.0, help="Kill run if |z - z_start| exceeds this (m, default 3.0)")
    ap.add_argument("--imu-fix-param", action="append", default=[], help="Extra ROS param for the Python IMU-fix node, e.g. acc_lowpass_hz:=4.0 (repeatable)")
    ap.add_argument("--cloud-fix-script", default=None, help="Also run this Python point-cloud relay (e.g. fastlio/cloud_time_fix_node.py); the config's lid_topic must point at its output")
    ap.add_argument("--imu-fix-script", default=None, help="Run this Python IMU-fix node (e.g. liosam/imu_fix_quat_node.py) instead of the C++ fast_imu_utils one, whose rotation is hardcoded to Rx(+90deg)")
    args = ap.parse_args()

    if not STATE_LOGGER_SCRIPT.exists():
        sys.exit(f"State logger script not found: {STATE_LOGGER_SCRIPT}")
    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.exists():
        sys.exit(f"Bag path not found: {bag_path}")
    try:
        fast_lio_share = Path(get_package_share_directory("fast_lio"))
    except PackageNotFoundError:
        sys.exit("ROS2 package 'fast_lio' not found - source your ROS2 workspace before running this script.")
    config_dir = Path(args.config_path).expanduser().resolve() if args.config_path else fast_lio_share / "config"
    config_path = config_dir / args.config
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    if args.out:
        run_dir = Path(args.out).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = DEFAULT_RUNS_ROOT / f"{bag_path.name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "run.log"
    log_f = open(log_path, "a")

    def log(msg):
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line)
        log_f.write(line + "\n")
        log_f.flush()

    log(f"bag={bag_path}")
    log(f"config={config_path}")
    log(f"run_dir={run_dir}")
    log(f"rate={args.rate} check_interval={args.check_interval}s max_speed={args.max_speed}m/s max_z_drift={args.max_z_drift}m")

    odom_csv = run_dir / "fastlio_odom_state.csv"
    result = {
        "bag": str(bag_path), "config": str(config_path), "run_dir": str(run_dir),
        "rate": args.rate, "start_offset": args.start_offset, "check_interval": args.check_interval,
        "max_speed": args.max_speed, "max_z_drift": args.max_z_drift,
        "imu_fix_script": args.imu_fix_script, "cloud_fix_script": args.cloud_fix_script,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "status": None, "reason": None,
    }

    handles = []
    t_start = time.time()
    try:
        launch = spawn(
            "launch",
            ["ros2", "launch", "fast_lio", "mapping.launch.py",
             f"config_path:={config_dir}", f"config_file:={args.config}",
             "use_sim_time:=true", "rviz:=false", "use_foxglove:=false"],
            run_dir,
        )
        handles.append(launch)
        if not wait_for_nodes(["/laser_mapping"], timeout=LAUNCH_READY_TIMEOUT):
            log(f"WARNING: /laser_mapping not up within {LAUNCH_READY_TIMEOUT}s, proceeding anyway")
        time.sleep(POST_READY_SETTLE)

        if args.imu_fix_script:
            imu_fix_cmd = ["python3", str(Path(args.imu_fix_script).expanduser().resolve()),
                           "--ros-args", "-p", "use_sim_time:=true"]
            for prm in args.imu_fix_param:
                imu_fix_cmd += ["-p", prm]
        else:
            imu_fix_cmd = ["ros2", "run", "fast_imu_utils", "imu_fix_quat_node"]
        log(f"imu_fix={' '.join(imu_fix_cmd)}")
        imu_fix = spawn("imu_fix", imu_fix_cmd, run_dir)
        if args.cloud_fix_script:
            cloud_fix_cmd = ["python3", str(Path(args.cloud_fix_script).expanduser().resolve()),
                             "--ros-args", "-p", "use_sim_time:=true"]
            log(f"cloud_fix={' '.join(cloud_fix_cmd)}")
            handles.append(spawn("cloud_fix", cloud_fix_cmd, run_dir))
        handles.append(imu_fix)
        if not wait_for_nodes(["/imu_fix_quat_node"], timeout=LOGGER_READY_TIMEOUT):
            log(f"WARNING: /imu_fix_quat_node not up within {LOGGER_READY_TIMEOUT}s, proceeding anyway")
        time.sleep(POST_READY_SETTLE)

        logger = spawn(
            "state_logger",
            ["python3", str(STATE_LOGGER_SCRIPT), "--ros-args",
             "-p", "use_sim_time:=true",
             "-p", f"imu_topic:={args.imu_topic}",
             "-p", f"out_dir:={run_dir}"],
            run_dir,
        )
        handles.append(logger)
        if not wait_for_nodes(["/fastlio_state_logger"], timeout=LOGGER_READY_TIMEOUT):
            log(f"WARNING: /fastlio_state_logger not up within {LOGGER_READY_TIMEOUT}s, proceeding anyway")
        time.sleep(POST_READY_SETTLE)

        bag_cmd = ["ros2", "bag", "play", str(bag_path), "--clock", "--rate", str(args.rate)]
        if args.start_offset is not None:
            bag_cmd += ["--start-offset", str(args.start_offset)]
        log(f"playing bag (rate={args.rate}, start_offset={args.start_offset}) ...")
        bag_log = open(run_dir / "bag_play.log", "w")
        bag_proc = subprocess.Popen(
            bag_cmd, stdout=bag_log, stderr=subprocess.STDOUT, start_new_session=True,
        )

        status, reason = watch_and_wait(
            bag_proc, odom_csv, args.check_interval, args.max_speed, args.max_z_drift, log
        )
        result["status"] = status
        result["reason"] = reason

        try:
            pgid = os.getpgid(bag_proc.pid)
            os.killpg(pgid, signal.SIGINT if status == "COMPLETED" else signal.SIGKILL)
        except ProcessLookupError:
            pass
        bag_log.close()

        log(f"run finished: status={status} reason={reason}")

    except Exception as exc:
        result["status"] = "ERROR"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        log(f"ERROR: {result['reason']}")

    finally:
        log("tearing down ...")
        for h in reversed(handles):
            h.stop()
            h.logfile.close()

    result["elapsed_seconds"] = round(time.time() - t_start, 1)
    result["finished_at"] = datetime.now().isoformat(timespec="seconds")
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    log(f"result: {json.dumps(result)}")
    log_f.close()


if __name__ == "__main__":
    main()
