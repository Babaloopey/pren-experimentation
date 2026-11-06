"""Robot-side map: /map, map -> odom anchor, fused path.

map -> odom is static identity: the dog always starts at the known Startfeld pose,
so odom (EKF, wheel + IMU) *is* the map estimate. A later absolute correction
(e.g. pylon detection) would replace this with a map-frame filter.
"""
import math

import rclpy
import rclpy.executors
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Empty
from tf2_ros import StaticTransformBroadcaster

from .common import LATCHED, in_segment


class Field(Node):
    def __init__(self):
        super().__init__('field')
        self.static_tf = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = 'map', 'odom'
        t.transform.rotation.w = 1.0
        self.static_tf.sendTransform(t)

        self.map_pub = self.create_publisher(OccupancyGrid, '/map', LATCHED)
        self.path_pub = self.create_publisher(Path, '/path/fused', 10)
        self.create_subscription(Odometry, '/odometry/filtered', self.on_fused, 10)
        self.create_subscription(Empty, '/sim/reset', lambda _: self.path.clear(), 10)
        self.path = []

        self.map_pub.publish(self.occupancy_map())
        self.create_timer(0.2, self.publish_path)
        self.get_logger().info('field published (/map, map->odom)')

    def occupancy_map(self):
        """Coarse 2D map of the hall: free floor, tape lines marked 50 (not obstacles)."""
        res, w, h = 0.05, 200, 140          # 10 m x 7 m
        ox, oy = -3.5, -3.5
        grid = OccupancyGrid()
        grid.header.frame_id = 'map'
        grid.info.resolution = res
        grid.info.width, grid.info.height = w, h
        grid.info.origin.position.x, grid.info.origin.position.y = ox, oy
        grid.info.origin.orientation.w = 1.0
        data = []
        for j in range(h):
            for i in range(w):
                x, y = ox + (i + 0.5) * res, oy + (j + 0.5) * res
                data.append(30 if in_segment(x, y) else 0)
        grid.data = data
        return grid

    def on_fused(self, msg):
        p = msg.pose.pose.position
        if not self.path or math.hypot(self.path[-1][0] - p.x, self.path[-1][1] - p.y) > 0.02:
            self.path.append((p.x, p.y))
            del self.path[:-3000]

    def publish_path(self):
        msg = Path()
        msg.header.frame_id = 'odom'
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.path:
            ps = PoseStamped()
            ps.header.frame_id = 'odom'
            ps.pose.position.x, ps.pose.position.y = x, y
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)


def main():
    rclpy.init()
    try:
        rclpy.spin(Field())
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
