#!/usr/bin/env python3
"""LIO-SAM on one half of CA-10, played from the original bag (read-only view), CB-11 rep1 configuration.

Same node sequence and settings as orchard-slam/tools/liosam_param_sweep.py run_variant (the driver behind
liosam_rep1): lio_sam run.launch.py with rep1's params_used.yaml (only savePCDDirectory changed),
imu_fix_quat_node.py (working tree: calibrated R_LI, default params), static TF base_link->velodyne,
liosam_trajectory_recorder.py, liosam_run_recorder.py (full-rate odometry), proc_sampler.py, then
save_map (resolution 0.2) after an 8 s settle, and liosam_export.py.

Differences, all forced by playing from the original bag instead of a written half:
  --topics      only /velodyne_points /camera/camera/imu /gnss/fix /gnss/vel /tf_static are played
  --start-offset for B (bag seconds)
  --abort-on-starved (default on): a thread reads bag_play.log every 0.5 s; on the first "starved" line it
                sends SIGINT to `ros2 bag play` and the run is marked aborted (user rule, 2026-10-27)
  --stop-at     a /clock watcher sends SIGINT to `ros2 bag play` once the bag clock passes this bag time;
                save_map is called after that (before LIO-SAM is stopped)
The watcher also records the last /velodyne_points and /camera/camera/imu_fixed header stamps that were
published, the last /lio_sam/mapping/odometry stamp, and the bag clock at the stop (watcher.json).

Usage: run_liosam_half.py --name A --out DIR [--start-offset S] [--stop-at S] [--rate 1.0]
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

T0 = 1782577258.896306          # bag start (first message, receive time), unix s
ORCHARD_SLAM = Path.home() / "thesis" / "orchard-slam"
REP1 = ORCHARD_SLAM / "data/lio_sam/sweeps/session_20260706_140610_best_6run_r05_20261004_220758/baseline_rep1"
BAG = Path.home() / "ca10_run" / "bag_view"
TOPICS = ["/velodyne_points", "/camera/camera/imu", "/gnss/fix", "/gnss/vel", "/tf_static"]
sys.path.insert(0, str(ORCHARD_SLAM / "tools"))
import liosam_param_sweep as sw   # noqa: E402  (helpers only: spawn, wait_for_nodes, timing constants)


class Watcher:
    def __init__(self, stop_at):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import PointCloud2, Imu
        from nav_msgs.msg import Odometry
        rclpy.init()
        self.rclpy = rclpy
        self.node = Node("ca10_watcher")
        self.stop_at, self.bag_pid, self.fired = stop_at, None, None
        self.last = dict(clock=None, lidar_hdr=None, imu_fixed_hdr=None, odom_hdr=None, n_lidar=0, n_odom=0)
        be = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
        st = lambda h: h.stamp.sec + h.stamp.nanosec * 1e-9 - T0
        self.node.create_subscription(Clock, "/clock", self.on_clock, be)

        def on_lidar(m):
            self.last["lidar_hdr"] = st(m.header); self.last["n_lidar"] += 1
        def on_imu(m):
            self.last["imu_fixed_hdr"] = st(m.header)
        def on_odom(m):
            self.last["odom_hdr"] = st(m.header); self.last["n_odom"] += 1
        self.node.create_subscription(PointCloud2, "/velodyne_points", on_lidar, be)
        self.node.create_subscription(Imu, "/camera/camera/imu_fixed", on_imu, 200)
        self.node.create_subscription(Odometry, "/lio_sam/mapping/odometry", on_odom, be)
        threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True).start()

    def on_clock(self, m):
        t = m.clock.sec + m.clock.nanosec * 1e-9 - T0
        self.last["clock"] = t
        if self.stop_at is not None and self.fired is None and t >= self.stop_at and self.bag_pid:
            os.kill(self.bag_pid, signal.SIGINT)
            self.fired = dict(clock_at_sigint=t, wall=time.time(), last_at_sigint=dict(self.last))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start-offset", type=float, default=None)
    ap.add_argument("--stop-at", type=float, default=None)
    ap.add_argument("--rate", default="1.0")
    ap.add_argument("--read-ahead-queue-size", default=None, help="ros2 bag play --read-ahead-queue-size (player buffer)")
    ap.add_argument("--no-abort-on-starved", dest="abort_on_starved", action="store_false")
    ap.add_argument("--settle", type=float, default=8.0)
    ap.add_argument("--resolution", default="0.2")
    ap.add_argument("--bag", default=str(BAG))
    ap.add_argument("--extra-bag", default=None, help="second bag played at the same time (IMU re-order fallback)")
    a = ap.parse_args()
    run_dir = Path(a.out); map_dir = run_dir / "map"; map_dir.mkdir(parents=True, exist_ok=True)

    params = yaml.safe_load(open(REP1 / "params_used.yaml"))
    params["/**"]["ros__parameters"]["savePCDDirectory"] = str(map_dir) + "/"
    params_path = run_dir / "params_used.yaml"
    yaml.safe_dump(params, open(params_path, "w"), sort_keys=False)

    entry = dict(name=a.name, bag=a.bag, extra_bag=a.extra_bag, topics=TOPICS, rate=a.rate,
                 start_offset=a.start_offset, stop_at=a.stop_at, params_from=str(REP1 / "params_used.yaml"),
                 started_at=datetime.now().isoformat(timespec="seconds"))
    w = Watcher(a.stop_at)
    handles = []
    t_start = time.time()
    try:
        handles.append(sw.spawn("launch", ["ros2", "launch", "lio_sam", "run.launch.py",
                                           f"params_file:={params_path}", "use_foxglove:=false"], run_dir))
        sw.wait_for_nodes(sw.LAUNCH_EXPECTED_NODES, timeout=sw.LAUNCH_READY_TIMEOUT); time.sleep(sw.POST_READY_SETTLE)
        handles.append(sw.spawn("imu_fix", ["python3", str(sw.IMU_FIX_SCRIPT), "--ros-args", "-p",
                                            "use_sim_time:=true"], run_dir))
        sw.wait_for_nodes(["/imu_fix_quat_node"], timeout=sw.IMU_FIX_READY_TIMEOUT); time.sleep(sw.POST_READY_SETTLE)
        handles.append(sw.spawn("static_tf", ["ros2", "run", "tf2_ros", "static_transform_publisher",
                                              "--ros-args", "-p", "use_sim_time:=true", "--",
                                              "0", "0", "0", "0", "0", "0", "base_link", "velodyne"], run_dir))
        sw.wait_for_nodes(prefix_counts={"/static_transform_publisher": 2}, timeout=sw.STATIC_TF_READY_TIMEOUT)
        time.sleep(sw.POST_READY_SETTLE)
        handles.append(sw.spawn("recorder", ["python3", str(sw.RECORDER_SCRIPT), str(run_dir / "trajectory.csv")], run_dir))
        sw.wait_for_nodes(["/trajectory_recorder"], timeout=sw.RECORDER_READY_TIMEOUT); time.sleep(sw.POST_READY_SETTLE)
        handles.append(sw.spawn("run_recorder", ["python3", str(sw.RUN_RECORDER_SCRIPT), str(run_dir)], run_dir))
        handles.append(sw.spawn("proc_sampler", ["python3", str(sw.PROC_SAMPLER_SCRIPT), r"lio_sam_\w+", str(run_dir)], run_dir))
        time.sleep(sw.POST_READY_SETTLE)

        cmd = ["ros2", "bag", "play", a.bag, "--clock", "--rate", a.rate, "--topics", *TOPICS]
        if a.extra_bag:
            cmd = ["ros2", "bag", "play", a.bag, "--clock", "--rate", a.rate, "--topics",
                   *[t for t in TOPICS if t != "/camera/camera/imu"]]
        if a.start_offset is not None:
            cmd += ["--start-offset", str(a.start_offset)]
        if a.read_ahead_queue_size:
            cmd += ["--read-ahead-queue-size", str(a.read_ahead_queue_size)]
        entry["bag_cmd"] = cmd
        print("playing:", " ".join(cmd), flush=True)
        log = open(run_dir / "bag_play.log", "w")
        bag = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        w.bag_pid = bag.pid
        aborted = {}

        def starve_guard():
            while bag.poll() is None:
                txt = (run_dir / "bag_play.log").read_text(errors="ignore")
                if "starved" in txt:
                    aborted.update(wall=time.time(), bag_clock=w.last["clock"],
                                   line=next(l for l in txt.splitlines() if "starved" in l))
                    bag.send_signal(signal.SIGINT)
                    print("ABORT: queue starved ->", aborted, flush=True)
                    return
                time.sleep(0.5)
        if a.abort_on_starved:
            threading.Thread(target=starve_guard, daemon=True).start()
        extra = None
        if a.extra_bag:
            ecmd = ["ros2", "bag", "play", a.extra_bag, "--rate", a.rate]
            if a.start_offset is not None:
                ecmd += ["--start-offset", "0"]
            entry["extra_bag_cmd"] = ecmd
            extra = subprocess.Popen(ecmd, stdout=open(run_dir / "bag_play_extra.log", "w"), stderr=subprocess.STDOUT)
        rc = bag.wait()
        if extra:
            extra.send_signal(signal.SIGINT); extra.wait(timeout=20)
        log.close()
        entry["bag_returncode"] = rc
        entry["watcher_at_stop"] = w.fired
        entry["aborted_starved"] = aborted or None
        print(f"bag play ended rc={rc}, watcher={w.fired}; settling {a.settle}s", flush=True)
        time.sleep(a.settle)
        entry["watcher_before_save"] = dict(w.last)
        sc = subprocess.run(["ros2", "service", "call", "/lio_sam/save_map", "lio_sam/srv/SaveMap",
                             f"{{resolution: {a.resolution}, destination: '{map_dir}/'}}"],
                            capture_output=True, text=True, timeout=180)
        (run_dir / "save_map.log").write_text(sc.stdout + sc.stderr)
        entry["save_map_success"] = bool(re.search(r"success\s*[:=]\s*true", sc.stdout.lower()))
        entry["watcher_after_save"] = dict(w.last)
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
        print("ERROR", entry["error"], flush=True)
    finally:
        for h in reversed(handles):
            h.stop(); h.logfile.close()
    blog = (run_dir / "bag_play.log").read_text(errors="ignore")
    entry["starvation_warnings"] = blog.count("starved")
    exp = subprocess.run(["python3", str(sw.EXPORT_SCRIPT), str(run_dir)], capture_output=True, text=True)
    (run_dir / "export.log").write_text(exp.stdout + exp.stderr)
    entry["export"] = exp.stdout.strip() or exp.stderr.strip()[-300:]
    entry["elapsed_s"] = round(time.time() - t_start, 1)
    entry["finished_at"] = datetime.now().isoformat(timespec="seconds")
    (run_dir / "run_entry.json").write_text(json.dumps(entry, indent=2))
    print(json.dumps(entry, indent=1), flush=True)
    w.rclpy.shutdown()


if __name__ == "__main__":
    main()
