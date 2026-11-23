#!/usr/bin/env python3
"""
Records /lio_sam/mapping/path to a CSV.

Usage:
    python3 liosam_trajectory_recorder.py [output_csv_path]

If output_csv_path is omitted, falls back to the original hardcoded default
so existing manual workflows are unaffected.
"""
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Path
import csv
from pathlib import Path as FPath

DEFAULT_OUTPUT = FPath.home() / "thesis" / "orchard-slam" / "data" / "lio_sam" / "trajectory_20260706.csv"
OUTPUT = FPath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT


class TrajectoryRecorder(Node):
    def __init__(self):
        super().__init__('trajectory_recorder')

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)

        self.create_subscription(Path, '/lio_sam/mapping/path', self.cb, qos)
        self.poses = {}
        self.cb_count = 0
        self.get_logger().info(f'Recording → {OUTPUT}')

    def cb(self, msg):
        if not msg.poses:
            return
        for p in msg.poses:
            ts = p.header.stamp.sec + p.header.stamp.nanosec * 1e-9
            self.poses[ts] = (
                ts,
                p.pose.position.x,
                p.pose.position.y,
                p.pose.position.z,
                p.pose.orientation.x,
                p.pose.orientation.y,
                p.pose.orientation.z,
                p.pose.orientation.w,
            )
        self.cb_count += 1
        if self.cb_count % 5 == 0:
            self.save(silent=True)
        last = msg.poses[-1].pose.position
        self.get_logger().info(
            f'{len(self.poses)} poses | x={last.x:.2f} y={last.y:.2f} z={last.z:.2f}')

    def save(self, silent=False):
        if not self.poses:
            if not silent:
                self.get_logger().warn('No poses recorded!')
            return
        rows = sorted(self.poses.values(), key=lambda r: r[0])
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
            w.writerows(rows)
        if not silent:
            self.get_logger().info(f'Saved {len(rows)} poses → {OUTPUT}')


def main():
    rclpy.init()
    node = TrajectoryRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
