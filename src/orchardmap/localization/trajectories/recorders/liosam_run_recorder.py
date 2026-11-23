#!/usr/bin/env python3
"""Record what a LIO-SAM run needs for the thesis, in one node.

    python3 liosam_run_recorder.py <run_dir>

Writes into <run_dir>:
  odometry.csv       /lio_sam/mapping/odometry at full rate: t,x,y,z,qx,qy,qz,qw
  keyframes_live.csv every keyframe of /lio_sam/mapping/path the first time it appears
                     (its pose when created, before any later loop correction):
                     kf,t,x,y,z,qx,qy,qz,qw
  loops_raw.csv      one row per new loop edge in /lio_sam/mapping/loop_closure_constraints:
                     t,edge_idx,x_from,y_from,z_from,x_to,y_to,z_to
                     (t = stamp of the first message containing the edge, i.e. within ~1 s of
                     the loop being accepted; positions as of that moment)
  loops_final_edges.csv  all edges with their final positions, written at shutdown;
                     tools/liosam_export.py matches them to final keyframes for kf indices.
The node only subscribes; LIO-SAM's estimation does not depend on subscribers.
"""
import os
import sys

import rclpy
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import MarkerArray


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


class RunRecorder(Node):
    def __init__(self, out):
        super().__init__('liosam_run_recorder')
        # LIO-SAM publishes mapping/odometry BEST_EFFORT (utility.hpp qos_profile); a RELIABLE
        # subscriber would receive nothing. BEST_EFFORT matches either publisher.
        qos = QoSProfile(depth=1000, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self.f_odom = open(os.path.join(out, 'odometry.csv'), 'w')
        self.f_odom.write('t,x,y,z,qx,qy,qz,qw\n')
        self.f_kf = open(os.path.join(out, 'keyframes_live.csv'), 'w')
        self.f_kf.write('kf,t,x,y,z,qx,qy,qz,qw\n')
        self.f_loop = open(os.path.join(out, 'loops_raw.csv'), 'w')
        self.f_loop.write('t,edge_idx,x_from,y_from,z_from,x_to,y_to,z_to\n')
        self.kf_seen = set()
        self.out = out
        self.last_pairs = []
        self.edges = []  # list of (from, to) point tuples already written (live positions)
        self.n = 0
        self.create_subscription(Odometry, '/lio_sam/mapping/odometry', self.on_odom, qos)
        self.create_subscription(Path, '/lio_sam/mapping/path', self.on_path, 10)
        self.create_subscription(MarkerArray, '/lio_sam/mapping/loop_closure_constraints', self.on_loops, 10)

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.f_odom.write(f'{stamp(m.header):.9f},{p.x},{p.y},{p.z},{q.x},{q.y},{q.z},{q.w}\n')
        self.n += 1
        if self.n % 100 == 0:
            self.flush()

    def on_path(self, m):
        # path poses are keyframes in creation order; a loop correction rebuilds the whole
        # path, so only the FIRST time a keyframe stamp is seen is its live pose
        for i, ps in enumerate(m.poses):
            t = stamp(ps.header)
            if t in self.kf_seen:
                continue
            self.kf_seen.add(t)
            p, q = ps.pose.position, ps.pose.orientation
            self.f_kf.write(f'{i},{t:.9f},{p.x},{p.y},{p.z},{q.x},{q.y},{q.z},{q.w}\n')

    def on_loops(self, m):
        edge = next((k for k in m.markers if k.ns == 'loop_edges'), None)
        if edge is None:
            return
        pts = [(p.x, p.y, p.z) for p in edge.points]
        pairs = list(zip(pts[0::2], pts[1::2]))
        t = stamp(edge.header)
        # loopIndexContainer is a std::map keyed by the current keyframe, and a new loop always
        # attaches to the newest keyframe, so new edges are appended at the end of the list
        for k in range(len(self.edges), len(pairs)):
            a, b = pairs[k]
            self.f_loop.write(f'{t:.9f},{k},{a[0]},{a[1]},{a[2]},{b[0]},{b[1]},{b[2]}\n')
            self.edges.append(pairs[k])
        self.last_pairs = pairs
        self.f_loop.flush()

    def flush(self):
        for f in (self.f_odom, self.f_kf, self.f_loop):
            f.flush()
        # all edges with their latest (final, after every correction) endpoint positions
        with open(os.path.join(self.out, 'loops_final_edges.csv'), 'w') as f:
            f.write('edge_idx,x_from,y_from,z_from,x_to,y_to,z_to\n')
            for k, (a, b) in enumerate(self.last_pairs):
                f.write(f'{k},{a[0]},{a[1]},{a[2]},{b[0]},{b[1]},{b[2]}\n')


def main():
    rclpy.init()
    node = RunRecorder(sys.argv[1])
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.flush()


if __name__ == '__main__':
    main()
