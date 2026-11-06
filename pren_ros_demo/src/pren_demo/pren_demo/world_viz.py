"""Simulated real world (sim only): the hall as it *is*, not as the robot believes.

  tf world -> map    static identity: the Startfeld is known, so map starts aligned with
                     the world. Localisation error lives in odom -> base_link.
  /world/field       tape, pylons, person, legend (frame world)
  /world/dog         the real dog at base_truth, shaped by /dog/state, speaking /dog/ack
"""
import math

import rclpy
import rclpy.executors
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from .common import (INTERACT_RADIUS, LATCHED, PERSON_X, PERSON_Y, SEG_HALF_ANGLE,
                     SEG_R_MAX, SEG_R_MIN, START_BOX, marker, rpy_to_quat)

TAPE = (1.0, 0.85, 0.1, 1.0)
PYLON = (1.0, 0.45, 0.0, 1.0)
BODY = (0.55, 0.38, 0.2)


class WorldViz(Node):
    def __init__(self):
        super().__init__('world_viz')
        self.static_tf = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = 'world', 'map'
        t.transform.rotation.w = 1.0
        self.static_tf.sendTransform(t)

        self.posture, self.behavior = 'STAND', 'IDLE'
        self.field_pub = self.create_publisher(MarkerArray, '/world/field', LATCHED)
        self.dog_pub = self.create_publisher(MarkerArray, '/world/dog', 10)
        self.create_subscription(String, '/dog/state', self.on_state, LATCHED)
        self.create_subscription(String, '/dog/ack', self.on_ack, 10)

        self.field_pub.publish(self.field_markers())
        self.create_timer(0.1, self.publish_dog)
        self.get_logger().info('world published (/world/field, /world/dog, world->map)')

    def on_state(self, msg):
        self.posture, _, self.behavior = msg.data.partition('|')

    def on_ack(self, msg):
        nack = 'ignoriert' in msg.data
        m = marker('base_truth', 'ack', 0, Marker.TEXT_VIEW_FACING, (0, 0, 0.12),
                   (1.0, 0.6, 0.2, 1.0) if nack else (0.3, 1.0, 0.4, 1.0),
                   xyz=(0, 0, 0.75), text=msg.data, lifetime=3.0, locked=True)
        self.dog_pub.publish(MarkerArray(markers=[m]))

    def field_markers(self):
        ms, mid = [], 0

        def add(m):
            nonlocal mid
            m.id = mid
            mid += 1
            ms.append(m)

        hx, hy = START_BOX[0] / 2, START_BOX[1] / 2
        add(marker('world', 'startfeld', 0, Marker.LINE_STRIP, (0.03, 0, 0), TAPE,
                   points=[(-hx, -hy, 0.005), (hx, -hy, 0.005), (hx, hy, 0.005),
                           (-hx, hy, 0.005), (-hx, -hy, 0.005)]))
        add(marker('world', 'labels', 0, Marker.TEXT_VIEW_FACING, (0, 0, 0.1), (1, 1, 1, 0.9),
                   xyz=(0.0, -0.4, 0.05), text='Startfeld'))

        # 30 deg search segment between SEG_R_MIN and SEG_R_MAX
        arc = lambda r: [(r * math.cos(a), r * math.sin(a), 0.005) for a in
                         [-SEG_HALF_ANGLE + i * 2 * SEG_HALF_ANGLE / 30 for i in range(31)]]
        outline = arc(SEG_R_MIN) + list(reversed(arc(SEG_R_MAX))) + [arc(SEG_R_MIN)[0]]
        add(marker('world', 'segment', 0, Marker.LINE_STRIP, (0.04, 0, 0), TAPE, points=outline))
        add(marker('world', 'labels', 0, Marker.TEXT_VIEW_FACING, (0, 0, 0.13), (1, 1, 1, 0.9),
                   xyz=(5.4, 0.0, 0.05), text='Suchbereich'))
        for r in (SEG_R_MIN, SEG_R_MAX):
            for s in (-1, 1):
                add(marker('world', 'pylons', 0, Marker.CYLINDER, (0.15, 0.15, 0.3), PYLON,
                           xyz=(r * math.cos(SEG_HALF_ANGLE), s * r * math.sin(SEG_HALF_ANGLE), 0.15)))

        # interaction area + referee
        circle = [(PERSON_X + INTERACT_RADIUS * math.cos(a), PERSON_Y + INTERACT_RADIUS * math.sin(a), 0.005)
                  for a in [i * 2 * math.pi / 48 for i in range(49)]]
        add(marker('world', 'interaction', 0, Marker.LINE_STRIP, (0.02, 0, 0), (0.4, 0.8, 1.0, 0.8),
                   points=circle))
        add(marker('world', 'person', 0, Marker.CYLINDER, (0.35, 0.45, 1.5), (0.3, 0.5, 0.9, 0.9),
                   xyz=(PERSON_X, PERSON_Y, 0.75)))
        add(marker('world', 'person', 0, Marker.SPHERE, (0.25, 0.25, 0.25), (0.95, 0.8, 0.65, 1.0),
                   xyz=(PERSON_X, PERSON_Y, 1.65)))
        add(marker('world', 'labels', 0, Marker.TEXT_VIEW_FACING, (0, 0, 0.13), (1, 1, 1, 0.9),
                   xyz=(PERSON_X, PERSON_Y, 2.0), text='Person'))
        # legend for the three paths (colours match demo.rviz)
        for row, (label, rgb) in enumerate([('truth', (0.2, 0.86, 0.31)),
                                            ('wheels', (0.9, 0.24, 0.24)),
                                            ('EKF', (0.24, 0.55, 1.0))]):
            lx = 0.9 - row * 0.35
            add(marker('world', 'legend', 0, Marker.LINE_LIST, (0.05, 0, 0), (*rgb, 1.0),
                       points=[(lx, -1.7, 0.01), (lx, -2.1, 0.01)]))
            add(marker('world', 'legend', 0, Marker.TEXT_VIEW_FACING, (0, 0, 0.16), (*rgb, 1.0),
                       xyz=(lx, -2.5, 0.05), text=label))
        add(marker('world', 'legend', 0, Marker.TEXT_VIEW_FACING, (0, 0, 0.13), (1, 1, 1, 0.9),
                   xyz=(1.3, -2.2, 0.05), text='Paths'))
        return MarkerArray(markers=ms)

    def publish_dog(self):
        """Cartoon dog on the TRUE pose. Shape shows the posture."""
        if self.posture == 'SIT':
            body = dict(xyz=(-0.04, 0, 0.17), quat=rpy_to_quat(0, -0.55, 0))
            head, legs = (0.15, 0, 0.42), [(0.1, 0.07, 0.12), (0.1, -0.07, 0.12)]
        elif self.posture == 'PLATZ':
            body = dict(xyz=(0, 0, 0.07), quat=rpy_to_quat(0, 0, 0))
            head, legs = (0.25, 0, 0.12), []
        else:
            body = dict(xyz=(0, 0, 0.2), quat=rpy_to_quat(0, 0, 0))
            head, legs = (0.24, 0, 0.32), [(sx, sy, 0.08) for sx in (0.13, -0.13) for sy in (0.07, -0.07)]

        f = 'base_truth'
        ms = [marker(f, 'body', 0, Marker.CUBE, (0.4, 0.17, 0.13), (*BODY, 1.0), locked=True, **body),
              marker(f, 'body', 1, Marker.SPHERE, (0.14, 0.13, 0.13), (*BODY, 1.0), xyz=head, locked=True),
              marker(f, 'body', 2, Marker.SPHERE, (0.05, 0.05, 0.05), (0.1, 0.1, 0.1, 1),
                     xyz=(head[0] + 0.07, 0, head[2]), locked=True),
              marker(f, 'body', 3, Marker.CUBE, (0.5, 0.2, 0.01), (0.2, 0.9, 0.3, 0.3),
                     xyz=(0, 0, 0.005), locked=True),        # 50 x 20 cm footprint
              marker(f, 'body', 4, Marker.TEXT_VIEW_FACING, (0, 0, 0.09), (1, 1, 1, 0.9),
                     xyz=(0, 0, 0.55), locked=True,
                     text=self.posture + ('' if self.behavior == 'IDLE' else f'|{self.behavior}'))]
        for i in range(4):
            if i < len(legs):
                ms.append(marker(f, 'legs', i, Marker.CYLINDER, (0.04, 0.04, 0.16), (*BODY, 1.0),
                                 xyz=legs[i], locked=True))
            else:
                d = marker(f, 'legs', i, Marker.CYLINDER)
                d.action = Marker.DELETE
                ms.append(d)
        self.dog_pub.publish(MarkerArray(markers=ms))


def main():
    rclpy.init()
    try:
        rclpy.spin(WorldViz())
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
