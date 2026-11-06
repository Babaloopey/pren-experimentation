"""Search mission: SUCH -> turn -> sweep segment -> touch bone -> report -> return -> sit.

Uses only what the real robot will have:
  pose       tf2 map -> base_link (EKF estimate, never ground truth)
  detections /bone_detections in camera_link, converted to map with tf2
  contact    /bumper (std_msgs/Bool; on the robot: bumper microswitch)
Output: /cmd_vel_search, /mission/state, /mission/markers
Competition rules: 2 min hard limit, report before returning, end sitting in Startfeld.
"""
import math

import rclpy
import rclpy.executors
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool, Empty, String
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from .common import (SEG_HALF_ANGLE, START_X, START_Y, START_YAW, go_to, lookup_pose, marker,
                     turn_to)

ACTIVE = ('TURN', 'SWEEP', 'APPROACH', 'CREEP', 'REPORT', 'RETURN', 'ALIGN')


def sweep_waypoints():
    """Zig-zag through the segment; camera range 3 m covers the rest."""
    a = SEG_HALF_ANGLE - 0.06
    wps, flip = [], 1
    for r in (1.6, 2.7, 3.8, 4.6):
        wps += [(r * math.cos(flip * a), r * math.sin(flip * a)),
                (r * math.cos(-flip * a), r * math.sin(-flip * a))]
        flip = -flip
    return wps


class SearchNode(Node):
    def __init__(self):
        super().__init__('search_node')
        params = self.declare_parameters('', [
            ('time_limit', 120.0), ('min_hits', 3), ('merge_dist', 0.4),
            ('min_score', 0.3), ('report_time', 2.0),
        ])
        self.p = {prm.name: prm.value for prm in params}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.waypoints = sweep_waypoints()
        self.reset()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_search', 10)
        self.state_pub = self.create_publisher(String, '/mission/state', 10)
        self.viz_pub = self.create_publisher(MarkerArray, '/mission/markers', 10)

        self.create_subscription(String, '/mission/command', self.on_command, 10)
        self.create_subscription(Detection3DArray, '/bone_detections', self.on_detections, 10)
        self.create_subscription(Bool, '/bumper', self.on_bumper, 10)
        self.create_subscription(Empty, '/sim/reset', lambda _: self.reset(), 10)

        self.create_timer(0.05, self.step)
        self.create_timer(0.5, self.publish_status)
        self.get_logger().info('search node ready, waiting for SUCH')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def reset(self):
        self.state = 'IDLE'
        self.t_start = self.t_state = self.now()
        self.wp_idx = 0
        self.cands = []          # {'x','y','hits','rejected'}
        self.target = None
        self.touched = False
        self.result = ''

    def set_state(self, s, note=''):
        self.get_logger().info(f'{self.state} -> {s} {note}')
        self.state, self.t_state = s, self.now()
        self.publish_status()

    # ---------------------------------------------------------------- inputs
    def on_command(self, msg):
        if msg.data == 'START' and self.state not in ACTIVE:
            self.reset()
            self.t_start = self.now()
            self.set_state('TURN')
        elif msg.data == 'ABORT' and self.state in ACTIVE:
            self.cmd_pub.publish(Twist())
            self.result = 'abgebrochen'
            self.set_state('ABORTED')

    def on_detections(self, arr):
        if self.state not in ACTIVE:
            return
        # tf2: camera_link -> map at the detection timestamp (fallback: latest)
        try:
            tf = self.tf_buffer.lookup_transform('map', arr.header.frame_id, Time.from_msg(arr.header.stamp))
        except TransformException:
            try:
                tf = self.tf_buffer.lookup_transform('map', arr.header.frame_id, Time())
            except TransformException as e:
                self.get_logger().warn(f'tf2: {e}', throttle_duration_sec=2.0)
                return
        for det in arr.detections:
            if not det.results or det.results[0].hypothesis.score < self.p['min_score']:
                continue
            ps = PoseStamped()
            ps.header = arr.header
            ps.pose = det.results[0].pose.pose
            p = do_transform_pose_stamped(ps, tf).pose.position
            self.add_observation(p.x, p.y)

    def add_observation(self, x, y):
        for c in self.cands:
            if math.hypot(c['x'] - x, c['y'] - y) < self.p['merge_dist']:
                n = min(c['hits'], 20)                # running mean, slowly forgetting
                c['x'] = (c['x'] * n + x) / (n + 1)
                c['y'] = (c['y'] * n + y) / (n + 1)
                c['hits'] += 1
                return
        self.cands.append({'x': x, 'y': y, 'hits': 1, 'rejected': False})

    def on_bumper(self, msg):
        if msg.data and self.state in ('APPROACH', 'CREEP'):
            self.touched = True

    # ---------------------------------------------------------------- FSM
    def confirmed(self):
        ok = [c for c in self.cands if c['hits'] >= self.p['min_hits'] and not c['rejected']]
        return max(ok, key=lambda c: c['hits']) if ok else None

    def step(self):
        if self.state not in ACTIVE:
            return
        if self.now() - self.t_start > self.p['time_limit']:
            self.cmd_pub.publish(Twist())
            self.result = 'Zeitlimit 2 min'
            return self.set_state('ABORTED')
        pose = lookup_pose(self.tf_buffer)
        if pose is None:
            return
        x, y, yaw = pose
        v = w = 0.0
        t = self.now() - self.t_state

        if self.state == 'TURN':                   # "umdrehen in den Suchbereich"
            w, done = turn_to(yaw, 0.0)
            if done:
                self.set_state('SWEEP')
        elif self.state == 'SWEEP':
            c = self.confirmed()
            if c:
                self.target = c
                return self.set_state('APPROACH', f'bone at ({c["x"]:.2f}, {c["y"]:.2f})')
            wx, wy = self.waypoints[self.wp_idx]
            v, w, done = go_to(x, y, yaw, wx, wy, v_max=0.45, tol=0.15)
            if done:
                self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)
        elif self.state == 'APPROACH':
            if self.touched:
                return self.set_state('REPORT')
            v, w, done = go_to(x, y, yaw, self.target['x'], self.target['y'], v_max=0.3, tol=0.2)
            if done:
                self.set_state('CREEP')
        elif self.state == 'CREEP':                # estimate may be off: creep on, sweep the nose
            if self.touched:
                return self.set_state('REPORT')
            v = 0.1
            w = 0.8 * math.sin(2 * math.pi * 0.5 * t)
            if t > 4.0:
                self.target['rejected'] = True
                self.set_state('SWEEP', '(no contact, candidate rejected)')
        elif self.state == 'REPORT':               # "Fund melden" before returning
            if t > self.p['report_time']:
                self.set_state('RETURN')
        elif self.state == 'RETURN':
            v, w, done = go_to(x, y, yaw, START_X, START_Y, v_max=0.55, tol=0.06)
            if done:
                self.set_state('ALIGN')
        elif self.state == 'ALIGN':
            w, done = turn_to(yaw, START_YAW)
            if done:
                self.cmd_pub.publish(Twist())
                self.result = f'{self.now() - self.t_start:.1f} s'
                return self.set_state('DONE')

        cmd = Twist()
        cmd.linear.x, cmd.angular.z = v, w
        self.cmd_pub.publish(cmd)

    # ---------------------------------------------------------------- output
    def publish_status(self):
        elapsed = (self.now() - self.t_start) if self.state in ACTIVE else 0.0
        self.state_pub.publish(String(data=f'{self.state} {elapsed:.1f}s {self.result}'.strip()))

        txt = {'IDLE': 'warte auf SUCH', 'DONE': f'Suche FERTIG: {self.result}',
               'ABORTED': f'Suche ABGEBROCHEN: {self.result}',
               'REPORT': 'WUFF WUFF! Knochen gefunden!'}.get(self.state, f'Suche: {self.state}')
        if self.state in ACTIVE:
            txt += f'   {elapsed:.0f}/{self.p["time_limit"]:.0f}s'
        ms = [marker('map', 'status', 0, Marker.TEXT_VIEW_FACING, (0, 0, 0.16), (1, 1, 0.4, 1),
                     xyz=(2.5, -2.3, 0.3), text=txt)]
        ms.append(marker('map', 'plan', 0, Marker.LINE_STRIP, (0.02, 0, 0), (0.6, 0.6, 1.0, 0.5),
                         points=[(wx, wy, 0.01) for wx, wy in self.waypoints]))
        ms.append(marker('map', 'clear', 0, Marker.DELETEALL))
        ms[-1].action = Marker.DELETEALL
        for i, c in enumerate(self.cands):
            if c['rejected']:
                rgba = (0.5, 0.5, 0.5, 0.6)
            elif c['hits'] >= self.p['min_hits']:
                rgba = (0.2, 1.0, 0.3, 0.9)
            else:
                rgba = (1.0, 0.5, 0.0, 0.7)
            ms.append(marker('map', 'cands', i, Marker.CYLINDER, (0.12, 0.12, 0.02), rgba,
                             xyz=(c['x'], c['y'], 0.06)))
            ms.append(marker('map', 'cands', 1000 + i, Marker.TEXT_VIEW_FACING, (0, 0, 0.08),
                             rgba[:3] + (1.0,), xyz=(c['x'], c['y'], 0.35),
                             text=f'estimate:{c["hits"]}'))
        self.viz_pub.publish(MarkerArray(markers=ms))


def main():
    rclpy.init()
    try:
        rclpy.spin(SearchNode())
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
