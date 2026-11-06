"""Loopback simulator: /cmd_vel -> ground truth + noisy wheel odometry + drifting IMU.

Stands in for the motor controller (MCU) + IMU driver. On the real robot this node
is replaced by the serial/micro-ROS bridge that publishes the same two topics:
  /wheel/odom  (nav_msgs/Odometry, odom -> base_link, NO tf: the EKF owns that)
  /imu/data    (sensor_msgs/Imu, base_link)
Ground truth (/sim/ground_truth, tf world -> base_truth) only exists in simulation.
"""
import math
import random

import rclpy
import rclpy.executors
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty
from tf2_ros import TransformBroadcaster

from .common import START_X, START_Y, START_YAW, clamp, wrap, yaw_to_quat


class LoopbackSim(Node):
    def __init__(self):
        super().__init__('loopback_sim')
        params = self.declare_parameters('', [
            ('rate', 50.0),
            ('max_lin_vel', 0.6), ('max_ang_vel', 2.0),
            ('max_lin_acc', 0.8), ('max_ang_acc', 4.0),
            ('cmd_timeout', 0.5),
            ('wheel_scale_error', 0.03),   # encoders over-report distance by 3 %
            ('wheel_yaw_bias', 0.04),      # rad heading error per metre (wheel diameter mismatch, ~2 deg/m)
            ('wheel_noise', 0.03),
            ('gyro_noise', 0.01),
            # MEMS gyro after a short still-stand calibration at boot. Raise these to make
            # the IMU worse (e.g. gyro_bias_init:=0.005) and watch the EKF path drift.
            ('gyro_bias_init', 0.0005),    # rad/s residual bias (~0.03 deg/s)
            ('gyro_bias_walk', 0.00005),   # rad/s/sqrt(s) bias instability
            ('gyro_scale_error', 0.01),    # 1 % -> ~2 deg error per half turn
        ])
        self.p = {prm.name: prm.value for prm in params}

        self.cmd = Twist()
        self.last_cmd = self.get_clock().now()
        self.reset_state()

        self.odom_pub = self.create_publisher(Odometry, '/wheel/odom', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self.truth_pub = self.create_publisher(Odometry, '/sim/ground_truth', 10)
        self.set_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/set_pose', 10)
        self.truth_path_pub = self.create_publisher(Path, '/sim/path/truth', 10)
        self.odom_path_pub = self.create_publisher(Path, '/sim/path/wheel_odom', 10)
        self.tf_pub = TransformBroadcaster(self)

        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)
        self.create_subscription(Empty, '/sim/reset', self.on_reset, 10)

        self.dt = 1.0 / self.p['rate']
        self.create_timer(self.dt, self.step)
        self.create_timer(0.2, self.publish_paths)
        self.get_logger().info('loopback sim ready: /cmd_vel -> /wheel/odom + /imu/data')

    def reset_state(self):
        self.x, self.y, self.yaw = START_X, START_Y, START_YAW   # ground truth
        self.v = self.w = 0.0
        self.ox, self.oy, self.oyaw = START_X, START_Y, START_YAW  # raw wheel dead-reckoning
        self.imu_yaw = START_YAW
        self.gyro_bias = random.gauss(0.0, self.p['gyro_bias_init'])
        self.truth_path, self.odom_path = [], []

    def on_cmd(self, msg):
        self.cmd = msg
        self.last_cmd = self.get_clock().now()

    def on_reset(self, _):
        self.p = {k: self.get_parameter(k).value for k in self.p}   # pick up `ros2 param set`
        self.reset_state()
        self.cmd = Twist()
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'odom'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x, msg.pose.pose.position.y = START_X, START_Y
        msg.pose.pose.orientation = yaw_to_quat(START_YAW)
        msg.pose.covariance[0] = msg.pose.covariance[7] = msg.pose.covariance[35] = 1e-4
        self.set_pose_pub.publish(msg)
        self.get_logger().info('sim reset -> robot back in Startfeld, EKF re-initialised')

    def step(self):
        p, dt = self.p, self.dt
        now = self.get_clock().now()
        stale = (now - self.last_cmd).nanoseconds * 1e-9 > p['cmd_timeout']
        tv = 0.0 if stale else clamp(self.cmd.linear.x, -p['max_lin_vel'], p['max_lin_vel'])
        tw = 0.0 if stale else clamp(self.cmd.angular.z, -p['max_ang_vel'], p['max_ang_vel'])

        # frictionless plane with acceleration limits
        v_prev = self.v
        self.v += clamp(tv - self.v, -p['max_lin_acc'] * dt, p['max_lin_acc'] * dt)
        self.w += clamp(tw - self.w, -p['max_ang_acc'] * dt, p['max_ang_acc'] * dt)
        yaw_mid = self.yaw + 0.5 * self.w * dt
        self.x += self.v * math.cos(yaw_mid) * dt
        self.y += self.v * math.sin(yaw_mid) * dt
        self.yaw = wrap(self.yaw + self.w * dt)

        # what the wheel encoders report (systematic + random error)
        v_m = self.v * (1.0 + p['wheel_scale_error']) + random.gauss(0, p['wheel_noise'] * abs(self.v))
        w_m = (self.w + p['wheel_yaw_bias'] * self.v
               + random.gauss(0, p['wheel_noise'] * (abs(self.w) + abs(self.v))))
        yaw_mid = self.oyaw + 0.5 * w_m * dt
        self.ox += v_m * math.cos(yaw_mid) * dt
        self.oy += v_m * math.sin(yaw_mid) * dt
        self.oyaw = wrap(self.oyaw + w_m * dt)

        # gyro with slowly wandering bias; orientation = integrated gyro (no magnetometer indoors)
        self.gyro_bias += random.gauss(0, p['gyro_bias_walk'] * math.sqrt(dt))
        gz = self.w * (1.0 + p['gyro_scale_error']) + self.gyro_bias + random.gauss(0, p['gyro_noise'])
        self.imu_yaw = wrap(self.imu_yaw + gz * dt)

        stamp = now.to_msg()
        self.publish_wheel_odom(stamp, v_m, w_m)
        self.publish_imu(stamp, gz, (self.v - v_prev) / dt)
        self.publish_truth(stamp)

    def publish_wheel_odom(self, stamp, v_m, w_m):
        o = Odometry()
        o.header.stamp, o.header.frame_id, o.child_frame_id = stamp, 'odom', 'base_link'
        o.pose.pose.position.x, o.pose.pose.position.y = self.ox, self.oy
        o.pose.pose.orientation = yaw_to_quat(self.oyaw)
        o.pose.covariance[0] = o.pose.covariance[7] = o.pose.covariance[35] = 1e3
        o.twist.twist.linear.x = v_m
        o.twist.twist.angular.z = w_m
        o.twist.covariance[0] = 0.002
        o.twist.covariance[7] = 0.001
        o.twist.covariance[35] = 0.02
        self.odom_pub.publish(o)

    def publish_imu(self, stamp, gz, ax):
        m = Imu()
        m.header.stamp, m.header.frame_id = stamp, 'base_link'
        m.orientation = yaw_to_quat(self.imu_yaw)
        m.orientation_covariance = [1e-3, 0.0, 0.0, 0.0, 1e-3, 0.0, 0.0, 0.0, 2e-3]
        m.angular_velocity.z = gz
        m.angular_velocity_covariance = [1e-4, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, 1e-4]
        m.linear_acceleration.x = ax
        m.linear_acceleration.y = self.v * self.w
        m.linear_acceleration.z = 9.81
        m.linear_acceleration_covariance = [0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05]
        self.imu_pub.publish(m)

    def publish_truth(self, stamp):
        o = Odometry()
        o.header.stamp, o.header.frame_id, o.child_frame_id = stamp, 'world', 'base_truth'
        o.pose.pose.position.x, o.pose.pose.position.y = self.x, self.y
        o.pose.pose.orientation = yaw_to_quat(self.yaw)
        o.twist.twist.linear.x, o.twist.twist.angular.z = self.v, self.w
        self.truth_pub.publish(o)

        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, 'world', 'base_truth'
        t.transform.translation.x, t.transform.translation.y = self.x, self.y
        t.transform.rotation = yaw_to_quat(self.yaw)
        self.tf_pub.sendTransform(t)

    def publish_paths(self):
        for path, frame, x, y, pub in (
            (self.truth_path, 'world', self.x, self.y, self.truth_path_pub),
            (self.odom_path, 'odom', self.ox, self.oy, self.odom_path_pub),
        ):
            if not path or math.hypot(path[-1][0] - x, path[-1][1] - y) > 0.02:
                path.append((x, y))
                del path[:-3000]
            msg = Path()
            msg.header.frame_id = frame
            msg.header.stamp = self.get_clock().now().to_msg()
            for px, py in path:
                ps = PoseStamped()
                ps.header.frame_id = frame
                ps.pose.position.x, ps.pose.position.y = px, py
                ps.pose.orientation.w = 1.0
                msg.poses.append(ps)
            pub.publish(msg)


def main():
    rclpy.init()
    try:
        rclpy.spin(LoopbackSim())
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
