"""Fake voice / gesture recognizer with a clickable command panel in RViz.

Replaces the real Vosk (voice) and MediaPipe (gesture) nodes. Inputs:
  - RViz: click a button of the "command panel" (Interact tool, InteractiveMarkers display)
  - text: /dog/inject (std_msgs/String) "voice:SUCH", "gesture:PLATZ", "SUCH" (= voice), "ctl:reset"
    e.g. from keyboard_teleop or `ros2 topic pub --once /dog/inject std_msgs/String "data: voice:HIER"`
Output (same as the real recognizers will publish):
  /dog/command  std_msgs/String JSON {"modality", "command", "confidence"}

Realism: gestures are only recognised if the person is 0.5-2 m in front of the camera,
unknown words are rejected, and an optional misrecognition rate drops voice commands.
"""
import json
import math
import random

import rclpy
import rclpy.executors
from interactive_markers import InteractiveMarkerServer
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, String
from visualization_msgs.msg import (InteractiveMarker, InteractiveMarkerControl,
                                    InteractiveMarkerFeedback, Marker, MarkerArray)

from .common import LATCHED, PERSON_X, PERSON_Y, marker, quat_to_yaw, wrap

VOCAB = {
    'voice': {'SUCH', 'HIER', 'BRAV', 'SITZ', 'TRICK'},
    'gesture': {'SITZ', 'PLATZ', 'AUF'},
    'button': {'SUCH'},
}

# (label, action, rgba)
BUTTONS = [
    ('Sprache SUCH', 'voice:SUCH', (0.2, 0.45, 0.9)),
    ('Sprache HIER', 'voice:HIER', (0.2, 0.45, 0.9)),
    ('Sprache BRAV', 'voice:BRAV', (0.2, 0.45, 0.9)),
    ('Sprache SITZ', 'voice:SITZ', (0.2, 0.45, 0.9)),
    ('Geste SITZ', 'gesture:SITZ', (0.6, 0.3, 0.85)),
    ('Geste PLATZ', 'gesture:PLATZ', (0.6, 0.3, 0.85)),
    ('Geste AUF', 'gesture:AUF', (0.6, 0.3, 0.85)),
    ('Kunststueck', 'voice:TRICK', (0.15, 0.65, 0.55)),
    ('Taster SUCH', 'button:SUCH', (0.15, 0.65, 0.55)),
    ('Mission ABORT', 'ctl:abort', (0.55, 0.55, 0.55)),
    ('Zurueck zum Start', 'ctl:home', (0.85, 0.55, 0.1)),
    ('NOT-AUS', 'ctl:estop', (0.85, 0.1, 0.1)),
    ('Reset Sim', 'ctl:reset', (0.55, 0.55, 0.55)),
]
PANEL_X, PANEL_Y, PANEL_STEP = 3.6, 2.4, 0.42


class CommandRecognizer(Node):
    def __init__(self):
        super().__init__('command_recognizer')
        params = self.declare_parameters('', [
            ('voice_miss_rate', 0.0),       # e.g. 0.2 -> every 5th word "not understood"
            ('gesture_min_dist', 0.5), ('gesture_max_dist', 2.2),
            ('gesture_max_angle_deg', 35.0),
        ])
        self.p = {prm.name: prm.value for prm in params}
        self.truth = None
        self.estop = False

        self.cmd_pub = self.create_publisher(String, '/dog/command', 10)
        self.mission_pub = self.create_publisher(String, '/mission/command', 10)
        self.reset_pub = self.create_publisher(Empty, '/sim/reset', 10)
        self.estop_pub = self.create_publisher(Bool, '/estop', LATCHED)
        self.viz_pub = self.create_publisher(MarkerArray, '/recognizer/markers', 10)

        self.create_subscription(String, '/dog/inject', lambda m: self.process(m.data), 10)
        self.create_subscription(Odometry, '/sim/ground_truth', self.on_truth, 10)
        self.create_subscription(Bool, '/estop', self.on_estop, LATCHED)

        self.server = InteractiveMarkerServer(self, 'command_panel')
        for i, (label, action, rgba) in enumerate(BUTTONS):
            self.server.insert(self.button(i, label, action, rgba), feedback_callback=self.on_click)
        self.server.applyChanges()
        self.get_logger().info('command panel ready: RViz -> Interact tool -> click a button')

    def button(self, i, label, action, rgba):
        im = InteractiveMarker()
        im.header.frame_id = 'world'
        im.name = action
        im.description = ''
        im.scale = 0.4
        im.pose.position.x = PANEL_X - i * PANEL_STEP
        im.pose.position.y = PANEL_Y
        im.pose.orientation.w = 1.0
        ctl = InteractiveMarkerControl()
        ctl.interaction_mode = InteractiveMarkerControl.BUTTON
        ctl.always_visible = True
        # empty frame_id: control markers are placed relative to the interactive marker
        ctl.markers.append(marker('', '', 0, Marker.CUBE, (0.34, 1.3, 0.04), (*rgba, 0.95)))
        ctl.markers.append(marker('', '', 1, Marker.TEXT_VIEW_FACING, (0, 0, 0.12),
                                  (1, 1, 1, 1), xyz=(0.0, 0.0, 0.12), text=label))
        im.controls.append(ctl)
        return im

    def on_click(self, fb):
        if fb.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            self.process(fb.marker_name)

    def on_truth(self, msg):
        p = msg.pose.pose
        self.truth = (p.position.x, p.position.y, quat_to_yaw(p.orientation))

    def on_estop(self, msg):
        self.estop = msg.data

    def process(self, raw):
        raw = raw.strip()
        modality, _, word = raw.partition(':') if ':' in raw else ('voice', '', raw)
        modality, word = modality.strip().lower(), word.strip().upper()

        if modality == 'ctl':
            if word == 'RESET':
                self.reset_pub.publish(Empty())
            elif word == 'ABORT':
                self.mission_pub.publish(String(data='ABORT'))
            elif word == 'ESTOP':
                self.estop_pub.publish(Bool(data=not self.estop))
            elif word == 'HOME':      # handled by dog_brain (aborts a running search first)
                self.cmd_pub.publish(String(data=json.dumps(
                    {'modality': 'ctl', 'command': 'HOME', 'confidence': 1.0})))
            self.feedback(f'[ctl] {word}', (0.8, 0.8, 0.8))
            return

        if word not in VOCAB.get(modality, ()):
            return self.reject(modality, word, 'unbekannt')
        if modality == 'gesture':
            why = self.gesture_blocked()
            if why:
                return self.reject(modality, word, why)
        if modality == 'voice' and random.random() < self.p['voice_miss_rate']:
            return self.reject(modality, word, 'nicht verstanden')

        conf = round(random.uniform(0.75, 0.99), 2)
        self.cmd_pub.publish(String(data=json.dumps(
            {'modality': modality, 'command': word, 'confidence': conf})))
        self.get_logger().info(f'recognised [{modality}] {word} ({conf})')
        self.feedback(f'[{modality}] {word}  {conf:.2f}', (0.3, 1.0, 0.4))

    def gesture_blocked(self):
        """Camera must see the person: right distance and roughly facing them."""
        if self.truth is None:
            return 'keine Pose'
        x, y, yaw = self.truth
        d = math.hypot(PERSON_X - x, PERSON_Y - y)
        ang = abs(wrap(math.atan2(PERSON_Y - y, PERSON_X - x) - yaw))
        if not self.p['gesture_min_dist'] <= d <= self.p['gesture_max_dist']:
            return f'Person {d:.1f} m entfernt'
        if ang > math.radians(self.p['gesture_max_angle_deg']):
            return 'Person nicht im Bild'
        return ''

    def reject(self, modality, word, why):
        self.get_logger().warn(f'rejected [{modality}] {word}: {why}')
        self.feedback(f'[{modality}] {word} ? {why}', (1.0, 0.35, 0.3))

    def feedback(self, text, rgb):
        m = marker('world', 'recognizer', 0, Marker.TEXT_VIEW_FACING, (0, 0, 0.13), (*rgb, 1.0),
                   xyz=(PERSON_X, PERSON_Y, 2.3), text=text, lifetime=3.0)
        self.viz_pub.publish(MarkerArray(markers=[m]))


def main():
    rclpy.init()
    try:
        rclpy.spin(CommandRecognizer())
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
