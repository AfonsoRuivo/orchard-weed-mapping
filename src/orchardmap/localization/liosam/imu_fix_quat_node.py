#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Imu
import numpy as np

class ImuFixNode(Node):
    def __init__(self):
        super().__init__('imu_fix_quat_node')

        # Candidate under test (2026-10-04, second run): R_LI estimated from
        # session_20260627_160540_calib (gyro vs LiDAR-ICP hand-eye for
        # roll/pitch, travel direction for yaw) = Rz(-10.8) Ry(-2.8) Rx(+1.8)
        # applied on top of [[0,0,1],[1,0,0],[0,1,0]]. Applied directly here
        # with extrinsicRot kept at identity in params.yaml. Prior candidates:
        # [[0,0,1],[1,0,0],[0,1,0]] (diverged ~80 s in, 2026-10-04 09:57),
        # Rx(+90deg) (Setup A, the only one with a clean full-bag run so far)
        # and [[0,0,-1],[-1,0,0],[0,1,0]] (diverged ~55-60% regardless of
        # noise settings).
        self.R = np.array([
            [0.1858, -0.0531,  0.9811],
            [0.9821, -0.0211, -0.1872],
            [0.0306,  0.9984,  0.0482]
        ], dtype=float)
        # 'permutation' = the plain 90-degree axis mapping without the calibrated
        # ~11 deg yaw / 2-3 deg tilt (for ablation runs only).
        # Seconds added to every IMU stamp. +0.03 aligns the camera IMU with the
        # LiDAR (estimated on session_20260627_160540_calib by maximising the
        # correlation of |rotation| between gyro and LiDAR ICP).
        self.declare_parameter('time_offset', 0.0)
        self.time_offset_ns = int(round(float(self.get_parameter('time_offset').value) * 1e9))
        self.declare_parameter('rotation', 'calibrated')
        # 'custom' = rotation_matrix param, 9 values row-major (ablations: perm+tilt only,
        # camera-LiDAR-implied rotation).
        self.declare_parameter('rotation_matrix', [0.0] * 9)
        if self.get_parameter('rotation').value == 'permutation':
            self.R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
            self.get_logger().info('using plain permutation rotation (ablation)')
        elif self.get_parameter('rotation').value == 'custom':
            self.R = np.array(self.get_parameter('rotation_matrix').value, dtype=float).reshape(3, 3)
            dev = np.abs(self.R @ self.R.T - np.eye(3)).max()
            if dev > 0.01 or abs(np.linalg.det(self.R) - 1) > 0.01:
                raise ValueError(f'rotation_matrix is not a rotation (|RR^T-I|max={dev:.3g}, det={np.linalg.det(self.R):.4f})')
            self.get_logger().info(f'using custom rotation {self.R.tolist()}')
        self.get_logger().info(f'IMU rotation R_LI = {self.R.tolist()}')

        qos_in = QoSProfile(depth=1000,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)
        qos_out = QoSProfile(depth=1000,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE)

        # Optional 2nd-order Butterworth low-pass on the accelerometer (Hz, 0 = off).
        # The tractor shakes the IMU at ~7 Hz (vertical std ~3 m/s^2 while driving).
        self.declare_parameter('acc_lowpass_hz', 0.0)
        self.declare_parameter('gyr_lowpass_hz', 0.0)   # same filter for the gyro (0 = off)
        self.declare_parameter('imu_rate_hz', 200.0)
        fc = float(self.get_parameter('acc_lowpass_hz').value)
        self.lp = None
        if fc > 0.0:
            fs = float(self.get_parameter('imu_rate_hz').value)
            k = np.tan(np.pi * fc / fs); q = np.sqrt(2.0); n = 1.0 / (1.0 + q * k + k * k)
            self.lp = (np.array([k * k * n, 2 * k * k * n, k * k * n]),        # b0 b1 b2
                       np.array([2 * (k * k - 1) * n, (1 - q * k + k * k) * n]))  # a1 a2
            self.lp_x = None   # last two inputs/outputs per axis, primed on first sample
            self.get_logger().info(f'accelerometer low-pass {fc} Hz at {fs} Hz')
        fg = float(self.get_parameter('gyr_lowpass_hz').value)
        self.lpg = None
        if fg > 0.0:
            fs = float(self.get_parameter('imu_rate_hz').value)
            k = np.tan(np.pi * fg / fs); q = np.sqrt(2.0); n = 1.0 / (1.0 + q * k + k * k)
            self.lpg = (np.array([k * k * n, 2 * k * k * n, k * k * n]),
                        np.array([2 * (k * k - 1) * n, (1 - q * k + k * k) * n]))
            self.lpg_x = None
            self.get_logger().info(f'gyro low-pass {fg} Hz at {fs} Hz')

        self.sub = self.create_subscription(
            Imu, '/camera/camera/imu', self.callback, qos_in)
        self.pub = self.create_publisher(
            Imu, '/camera/camera/imu_fixed', qos_out)

    def callback(self, msg):
        out = Imu()
        out.header = msg.header
        if self.time_offset_ns:
            ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec + self.time_offset_ns
            out.header.stamp.sec, out.header.stamp.nanosec = divmod(ns, 1_000_000_000)
        out.header.frame_id = 'imu_link'   # rename from optical frame

        # Rotate acceleration
        a = np.array([msg.linear_acceleration.x,
                      msg.linear_acceleration.y,
                      msg.linear_acceleration.z])
        ar = self.R @ a
        if self.lp is not None:
            b, a_ = self.lp
            if self.lp_x is None:
                self.lp_x = [ar.copy(), ar.copy(), ar.copy(), ar.copy()]  # x1 x2 y1 y2
            x1, x2, y1, y2 = self.lp_x
            y = b[0] * ar + b[1] * x1 + b[2] * x2 - a_[0] * y1 - a_[1] * y2
            self.lp_x = [ar.copy(), x1, y, y1]
            ar = y
        out.linear_acceleration.x = ar[0]
        out.linear_acceleration.y = ar[1]
        out.linear_acceleration.z = ar[2]

        # Rotate gyroscope
        g = np.array([msg.angular_velocity.x,
                      msg.angular_velocity.y,
                      msg.angular_velocity.z])
        gr = self.R @ g
        if self.lpg is not None:
            b, a_ = self.lpg
            if self.lpg_x is None:
                self.lpg_x = [gr.copy(), gr.copy(), gr.copy(), gr.copy()]
            x1, x2, y1, y2 = self.lpg_x
            y = b[0] * gr + b[1] * x1 + b[2] * x2 - a_[0] * y1 - a_[1] * y2
            self.lpg_x = [gr.copy(), x1, y, y1]
            gr = y
        out.angular_velocity.x = gr[0]
        out.angular_velocity.y = gr[1]
        out.angular_velocity.z = gr[2]

        # Valid identity quaternion (orientation unobservable on 6-DoF)
        out.orientation.x = 0.0
        out.orientation.y = 0.0
        out.orientation.z = 0.0
        out.orientation.w = 1.0
        out.orientation_covariance[0] = 0.01

        self.pub.publish(out)

def main(args=None):
    rclpy.init(args=args)
    node = ImuFixNode()
    rclpy.spin(node)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ImuFixNode())

if __name__ == '__main__':
    main()
