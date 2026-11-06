"""Velocity multiplexer (like twist_mux): manual > brain > search, plus latched NOT-AUS.

The highest-priority source that published within `timeout` wins. /estop=true forces
zero velocity (software mirror of the hardware Not-Aus, which cuts the motors anyway).
"""
import rclpy
import rclpy.executors
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, String

from .common import LATCHED

SOURCES = [('manual', '/cmd_vel_manual'), ('brain', '/cmd_vel_brain'), ('search', '/cmd_vel_search')]


class CmdMux(Node):
    def __init__(self):
        super().__init__('cmd_mux')
        self.timeout = self.declare_parameter('timeout', 0.3).value
        self.last = {name: (None, 0.0) for name, _ in SOURCES}
        self.estop = False
        self.active = None

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.active_pub = self.create_publisher(String, '/cmd_mux/active', LATCHED)
        for name, topic in SOURCES:
            self.create_subscription(Twist, topic, lambda m, n=name: self.on_cmd(n, m), 10)
        self.create_subscription(Bool, '/estop', self.on_estop, LATCHED)
        self.create_timer(1.0 / 30.0, self.step)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_cmd(self, name, msg):
        self.last[name] = (msg, self.now())

    def on_estop(self, msg):
        self.estop = msg.data
        self.get_logger().warn('NOT-AUS AKTIV' if msg.data else 'Not-Aus geloest')

    def step(self):
        out, active = Twist(), 'none'
        if self.estop:
            active = 'ESTOP'
        else:
            for name, _ in SOURCES:
                msg, t = self.last[name]
                if msg is not None and self.now() - t < self.timeout:
                    out, active = msg, name
                    break
        if active != self.active:
            self.active = active
            self.active_pub.publish(String(data=active))
            self.get_logger().info(f'driving: {active}')
        self.pub.publish(out)


def main():
    rclpy.init()
    try:
        rclpy.spin(CmdMux())
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
