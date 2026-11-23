#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
import csv
from pathlib import Path as FPath

import sys
OUTPUT = FPath(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith('--') \
    else FPath.home() / "ros2_ws" / "maps" / "kissicp_gps_trajectory.csv"


class TrajectoryRecorder(Node):
    def __init__(self):
        super().__init__('trajectory_recorder')
        for topic in ['/mapPath', '/rtabmap/mapPath', '/global_path']:
            self.create_subscription(Path, topic, self.cb, 10)
        # Use a dict keyed by timestamp to accumulate poses across messages
        self.poses = {}
        self.cb_count = 0
        self.get_logger().info(f'Recording → {OUTPUT}')

    def cb(self, msg):
        if not msg.poses:
            return
        # Merge incoming poses — never delete existing ones
        for p in msg.poses:
            ts = p.header.stamp.sec + p.header.stamp.nanosec * 1e-9
            self.poses[ts] = (ts, p.pose.position.x,
                                  p.pose.position.y,
                                  p.pose.position.z)
        self.cb_count += 1
        if self.cb_count % 2 == 0:
            self.save(silent=True)
        self.get_logger().info(
            f'  {len(self.poses)} total poses accumulated')

    def save(self, silent=False):
        if not self.poses:
            if not silent:
                self.get_logger().warn("No poses recorded!")
            return
        rows = sorted(self.poses.values(), key=lambda r: r[0])
        with open(OUTPUT, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['timestamp', 'x', 'y', 'z'])
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
