"""Fake bone detector: places artificial bones and "sees" them from the ground-truth pose.

No camera images are simulated. Instead: camera pose + FOV + range + bone position
-> detection with range/bearing noise and distance-dependent miss rate.
Output is exactly what the real vision node will publish:
  /bone_detections (vision_msgs/Detection3DArray, frame camera_link)
The search node converts these into the map frame with tf2.

Bones: N random ones inside the search segment, plus any point clicked with the
RViz "Publish Point" tool (moves the bone, or adds one with click_adds:=true). Contact between the robot's 50 x 20 cm footprint and a bone
-> /bumper (std_msgs/Bool, published continuously like a real switch).
"""
import math
import random

import rclpy
import rclpy.executors
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Empty
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from .common import (LATCHED, SEG_HALF_ANGLE, SEG_R_MAX, marker, quat_to_yaw, wrap)

BONE_LEN = 0.21


class BoneDetector(Node):
    def __init__(self):
        super().__init__('bone_detector')
        params = self.declare_parameters('', [
            ('n_bones', 1), ('seed', -1),
            ('click_adds', False),          # RViz click: False = move the bone there, True = add one
            ('rate', 10.0),
            ('fov_deg', 62.0), ('range_min', 0.2), ('range_max', 3.0),
            ('p_detect_near', 0.95), ('p_detect_far', 0.4),
            ('range_noise', 0.04),          # relative (4 % of distance)
            ('bearing_noise_deg', 1.5),
            ('cam_x', 0.2), ('cam_z', 0.25),
            ('robot_length', 0.5), ('robot_width', 0.2),
            ('contact_margin', 0.03),       # bumper travel / shell around the footprint
        ])
        self.p = {prm.name: prm.value for prm in params}
        if self.p['seed'] >= 0:
            random.seed(self.p['seed'])

        self.truth = None
        self.bones = []

        self.det_pub = self.create_publisher(Detection3DArray, '/bone_detections', 10)
        self.bone_pub = self.create_publisher(MarkerArray, '/sim/bones', LATCHED)
        self.bumper_pub = self.create_publisher(Bool, '/bumper', 10)
        self.viz_pub = self.create_publisher(MarkerArray, '/detector/markers', 10)

        self.create_subscription(Odometry, '/sim/ground_truth', self.on_truth, 10)
        self.create_subscription(PointStamped, '/clicked_point', self.on_click, 10)
        self.create_subscription(Empty, '/sim/reset', lambda _: self.spawn(), 10)

        self.spawn()
        self.create_timer(1.0 / self.p['rate'], self.step)
        self.get_logger().info('bone detector ready (RViz "Publish Point" places a bone)')

    def spawn(self):
        self.bones = []
        for _ in range(self.p['n_bones']):
            r = random.uniform(1.6, SEG_R_MAX - 0.3)
            a = random.uniform(-SEG_HALF_ANGLE + 0.05, SEG_HALF_ANGLE - 0.05)
            self.bones.append({'x': r * math.cos(a), 'y': r * math.sin(a),
                               'yaw': random.uniform(-math.pi, math.pi), 'touched': False})
        self.publish_bones()

    def on_click(self, msg):
        if msg.header.frame_id not in ('world', 'map'):
            self.get_logger().warn(f'clicked point in {msg.header.frame_id}, set RViz fixed frame to world')
            return
        if not self.get_parameter('click_adds').value:
            self.bones = []
        self.bones.append({'x': msg.point.x, 'y': msg.point.y,
                           'yaw': random.uniform(-math.pi, math.pi), 'touched': False})
        self.get_logger().info(f'bone at ({msg.point.x:.2f}, {msg.point.y:.2f}), {len(self.bones)} total')
        self.publish_bones()

    def on_truth(self, msg):
        p = msg.pose.pose
        self.truth = (p.position.x, p.position.y, quat_to_yaw(p.orientation))

    def step(self):
        if self.truth is None:
            return
        p = self.p
        tx, ty, tyaw = self.truth
        cx, cy = tx + p['cam_x'] * math.cos(tyaw), ty + p['cam_x'] * math.sin(tyaw)
        half_fov = math.radians(p['fov_deg']) / 2

        arr = Detection3DArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.header.frame_id = 'camera_link'
        rays = []
        contact = False
        for i, b in enumerate(self.bones):
            if self.in_contact(b, tx, ty, tyaw):
                contact = True
                if not b['touched']:
                    b['touched'] = True
                    self.get_logger().info(f'contact with bone_{i}')
                    self.publish_bones()

            r = math.hypot(b['x'] - cx, b['y'] - cy)
            bearing = wrap(math.atan2(b['y'] - cy, b['x'] - cx) - tyaw)
            if not (p['range_min'] <= r <= p['range_max'] and abs(bearing) <= half_fov):
                continue
            frac = (r - p['range_min']) / (p['range_max'] - p['range_min'])
            p_det = p['p_detect_near'] + (p['p_detect_far'] - p['p_detect_near']) * frac
            if random.random() > p_det:
                continue
            rn = r * (1.0 + random.gauss(0, p['range_noise']))
            bn = bearing + random.gauss(0, math.radians(p['bearing_noise_deg']))
            x, y, z = rn * math.cos(bn), rn * math.sin(bn), -p['cam_z']

            det = Detection3D()
            det.header = arr.header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = 'bone'
            hyp.hypothesis.score = round(p_det * random.uniform(0.85, 1.0), 2)
            hyp.pose.pose.position.x, hyp.pose.pose.position.y, hyp.pose.pose.position.z = x, y, z
            hyp.pose.pose.orientation.w = 1.0
            det.results.append(hyp)
            det.bbox.center = hyp.pose.pose
            det.bbox.size.x, det.bbox.size.y, det.bbox.size.z = BONE_LEN, 0.05, 0.05
            arr.detections.append(det)
            rays.append((x, y, z))
        self.det_pub.publish(arr)
        self.bumper_pub.publish(Bool(data=contact))
        self.publish_viz(rays, half_fov)

    def in_contact(self, b, tx, ty, tyaw):
        """Any point of the bone (shaft + knobs) inside the robot footprint + margin."""
        p = self.p
        hx = p['robot_length'] / 2 + p['contact_margin']
        hy = p['robot_width'] / 2 + p['contact_margin']
        c, s = math.cos(b['yaw']), math.sin(b['yaw'])
        cy, sy = math.cos(tyaw), math.sin(tyaw)
        for k in range(9):
            along = (k / 8 - 0.5) * BONE_LEN
            for side in (-0.025, 0.0, 0.025):
                px = b['x'] + along * c - side * s - tx
                py = b['y'] + along * s + side * c - ty
                lx, ly = px * cy + py * sy, -px * sy + py * cy   # into robot frame
                if abs(lx) <= hx and abs(ly) <= hy:
                    return True
        return False

    def publish_viz(self, rays, half_fov):
        z = -self.p['cam_z'] + 0.01
        rmax = self.p['range_max']
        fan = []
        n = 16
        for k in range(n):
            a0 = -half_fov + 2 * half_fov * k / n
            a1 = -half_fov + 2 * half_fov * (k + 1) / n
            fan += [(0, 0, z), (rmax * math.cos(a0), rmax * math.sin(a0), z),
                    (rmax * math.cos(a1), rmax * math.sin(a1), z)]
        fov = marker('camera_truth', 'fov', 0, Marker.TRIANGLE_LIST, (1, 1, 1), (0.2, 0.9, 1.0, 0.12),
                     points=fan, locked=True)
        line_pts = []
        for x, y, zz in rays:
            line_pts += [(0, 0, 0), (x, y, zz)]
        ray_m = marker('camera_truth', 'rays', 0, Marker.LINE_LIST, (0.015, 0, 0), (1.0, 0.2, 0.2, 0.9),
                       points=line_pts or [(0, 0, 0), (0, 0, 0)], lifetime=0.3, locked=True)
        self.viz_pub.publish(MarkerArray(markers=[fov, ray_m]))

    def publish_bones(self):
        ms = [marker('world', '', 0, Marker.DELETEALL)]
        ms[0].action = Marker.DELETEALL
        for i, b in enumerate(self.bones):
            rgba = (1.0, 0.8, 0.1, 1.0) if b['touched'] else (0.97, 0.97, 0.92, 1.0)
            c, s = math.cos(b['yaw']), math.sin(b['yaw'])
            ms.append(marker('world', 'bone_shaft', i, Marker.CUBE, (BONE_LEN, 0.035, 0.035), rgba,
                             xyz=(b['x'], b['y'], 0.025), yaw=b['yaw']))
            for k, (dl, dw) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
                ex = b['x'] + dl * BONE_LEN / 2 * c - dw * 0.025 * s
                ey = b['y'] + dl * BONE_LEN / 2 * s + dw * 0.025 * c
                ms.append(marker('world', 'bone_knob', i * 4 + k, Marker.SPHERE, (0.045, 0.045, 0.045),
                                 rgba, xyz=(ex, ey, 0.025)))
            ms.append(marker('world', 'bone_label', i, Marker.TEXT_VIEW_FACING, (0, 0, 0.07),
                             (1, 1, 1, 0.8), xyz=(b['x'], b['y'], 0.2),
                             text=f'bone_{i}' + ('!' if b['touched'] else '')))
        self.bone_pub.publish(MarkerArray(markers=ms))


def main():
    rclpy.init()
    try:
        rclpy.spin(BoneDetector())
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
