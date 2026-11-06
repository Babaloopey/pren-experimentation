"""Dog brain: turns recognised commands into behaviour + acknowledgements.

  /dog/command (JSON)  -> posture (STAND/SIT/PLATZ), short behaviours, ack
  SUCH                 -> /mission/command START (search_node owns the mission)
  HIER                 -> drive to the person's feet, wait, return to Startfeld
  HOME (ctl)           -> abort search if running, return to Startfeld
  BRAV                 -> wag + "WUFF"
  TRICK                -> pirouette (Kunststueck)
  RViz "2D Goal Pose"  -> drive there (handy for manual testing)
Every accepted command is acknowledged on /dog/ack (requirement 1.6) - on the
robot that topic drives the buzzer / LEDs. Posture/behaviour go out on /dog/state.
Motion goes out on /cmd_vel_brain, the cmd_mux decides who drives.
"""
import json
import math

import rclpy
import rclpy.executors
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from std_msgs.msg import Empty, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .common import (LATCHED, PERSON_X, PERSON_Y, START_X, START_Y, START_YAW, go_to, lookup_pose,
                     marker, quat_to_yaw, turn_to)

class DogBrain(Node):
    def __init__(self):
        super().__init__('dog_brain')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.posture = 'STAND'
        self.behavior = 'IDLE'
        self.t0 = self.now()
        self.goal = None
        self.mission_state = 'IDLE'

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_brain', 10)
        self.ack_pub = self.create_publisher(String, '/dog/ack', 10)
        self.mission_pub = self.create_publisher(String, '/mission/command', 10)
        self.viz_pub = self.create_publisher(MarkerArray, '/dog/markers', 10)
        self.state_pub = self.create_publisher(String, '/dog/state', LATCHED)
        self.last_state = ''

        self.create_subscription(String, '/dog/command', self.on_command, 10)
        self.create_subscription(String, '/mission/state', self.on_mission_state, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 10)
        self.create_subscription(Empty, '/sim/reset', self.on_reset, 10)

        self.create_timer(0.05, self.step)
        self.create_timer(0.1, self.publish_body)
        self.get_logger().info('dog brain ready')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------------------------------------------------------------- commands
    def on_command(self, msg):
        c = json.loads(msg.data)
        cmd, mod = c['command'], c['modality']
        if cmd == 'HOME':                          # works always, also mid-search
            if self.mission_state not in ('IDLE', 'DONE', 'ABORTED'):
                self.mission_pub.publish(String(data='ABORT'))
                self.mission_state = 'ABORTED'
            self.posture = 'STAND'
            self.set_behavior('HIER_BACK')
            return self.ack('Zurueck zum Start')
        busy = self.mission_state not in ('IDLE', 'DONE', 'ABORTED')
        if busy:
            return self.ack(f'{cmd}: ignoriert, Suche laeuft', ok=False)

        if cmd == 'SUCH':
            self.set_behavior('IDLE')
            self.posture = 'STAND'
            self.mission_pub.publish(String(data='START'))
            self.ack('SUCH -> Suche startet')
        elif cmd == 'SITZ':
            self.set_behavior('IDLE')
            self.posture = 'SIT'
            self.ack(f'SITZ ({mod})')
        elif cmd == 'PLATZ':
            self.set_behavior('IDLE')
            self.posture = 'PLATZ'
            self.ack('PLATZ (gesture)')
        elif cmd == 'AUF':
            self.posture = 'STAND'
            self.ack('AUF (gesture)')
        elif cmd == 'HIER':
            self.posture = 'STAND'
            self.set_behavior('HIER_GO')
            self.ack('HIER -> komme!')
        elif cmd == 'BRAV':
            self.set_behavior('BRAV')
            self.ack('BRAV -> WUFF! *wedel*')
        elif cmd == 'TRICK':
            self.posture = 'STAND'
            self.set_behavior('TRICK')
            self.ack('Kunststueck: Pirouette')

    def on_goal(self, msg):
        if msg.header.frame_id not in ('map', 'odom', 'world'):   # world == map in sim
            return
        self.posture = 'STAND'
        self.goal = (msg.pose.position.x, msg.pose.position.y, quat_to_yaw(msg.pose.orientation))
        self.set_behavior('GOAL')
        self.ack(f'Ziel ({self.goal[0]:.1f}, {self.goal[1]:.1f})')

    def on_mission_state(self, msg):
        state = msg.data.split()[0]
        if state == 'DONE' and self.mission_state != 'DONE':
            self.posture = 'SIT'
            self.ack('Zurueck im Startfeld -> SITZ *beep beep*')
        self.mission_state = state

    def on_reset(self, _):
        self.posture, self.mission_state = 'STAND', 'IDLE'
        self.set_behavior('IDLE')

    def set_behavior(self, b):
        if self.behavior != 'IDLE' and b == 'IDLE':
            self.cmd_pub.publish(Twist())
        self.behavior, self.t0 = b, self.now()

    def ack(self, text, ok=True):
        self.ack_pub.publish(String(data=text))
        if ok:   # rclpy forbids one call site logging at two severities
            self.get_logger().info(f'ACK: {text}')
        else:
            self.get_logger().warn(f'NACK: {text}')

    # ---------------------------------------------------------------- motion
    def step(self):
        if self.behavior == 'IDLE':
            return
        pose = lookup_pose(self.tf_buffer)
        if pose is None:
            return
        x, y, yaw = pose
        t = self.now() - self.t0
        v = w = 0.0

        if self.behavior == 'HIER_GO':
            d = math.hypot(x - PERSON_X, y - PERSON_Y) or 1.0
            tx, ty = PERSON_X + 0.45 * (x - PERSON_X) / d, PERSON_Y + 0.45 * (y - PERSON_Y) / d
            v, w, done = go_to(x, y, yaw, tx, ty, v_max=0.4)
            if done:
                self.set_behavior('HIER_WAIT')
        elif self.behavior == 'HIER_WAIT':
            if t > 1.5:
                self.set_behavior('HIER_BACK')
        elif self.behavior == 'HIER_BACK':
            v, w, done = go_to(x, y, yaw, START_X, START_Y, v_max=0.4, tol=0.06)
            if done:
                self.set_behavior('HIER_ALIGN')
        elif self.behavior == 'HIER_ALIGN':
            w, done = turn_to(yaw, START_YAW)
            if done:
                self.set_behavior('IDLE')
                self.ack('Im Startfeld')
                return
        elif self.behavior == 'BRAV':
            w = 1.6 * math.sin(2 * math.pi * 1.5 * t)
            if t > 2.0:
                return self.set_behavior('IDLE')
        elif self.behavior == 'TRICK':
            w = 2.0
            if t > 2 * math.pi / 2.0:
                return self.set_behavior('IDLE')
        elif self.behavior == 'GOAL':
            gx, gy, gyaw = self.goal
            v, w, done = go_to(x, y, yaw, gx, gy, v_max=0.5)
            if done:
                w, aligned = turn_to(yaw, gyaw)
                if aligned:
                    return self.set_behavior('IDLE')

        cmd = Twist()
        cmd.linear.x, cmd.angular.z = v, w
        self.cmd_pub.publish(cmd)

    # ---------------------------------------------------------------- visual
    def publish_body(self):
        """Robot side: publish own state; draw only the *belief* (EKF pose) as an outline.
        The real-looking dog is drawn by the sim (world_viz) at the true pose."""
        state = f'{self.posture}|{self.behavior}'
        if state != self.last_state:
            self.last_state = state
            self.state_pub.publish(String(data=state))
        ms = [marker('base_link', 'belief', 0, Marker.CUBE, (0.5, 0.2, 0.3), (0.25, 0.55, 1.0, 0.25),
                     xyz=(0, 0, 0.15), locked=True),
              marker('base_link', 'belief', 1, Marker.ARROW, (0.45, 0.04, 0.04), (0.25, 0.55, 1.0, 0.9),
                     xyz=(0, 0, 0.32), locked=True),
              marker('base_link', 'belief', 2, Marker.TEXT_VIEW_FACING, (0, 0, 0.08), (0.4, 0.7, 1.0, 1),
                     xyz=(0, 0, 0.42), text='EKF', locked=True)]
        self.viz_pub.publish(MarkerArray(markers=ms))

def main():
    rclpy.init()
    try:
        rclpy.spin(DogBrain())
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
