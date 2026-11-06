"""Keyboard driver: manual driving + typing commands (run in its own terminal).

Driving publishes /cmd_vel_manual (highest mux priority, overrides autonomy).
Command keys go through /dog/inject -> command_recognizer, i.e. exactly the same
path as a click in RViz or a real recognised word/gesture.
"""
import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, String

from .common import LATCHED

HELP = r"""
 PREN dog - keyboard driver
 --------------------------------------------------------------
  drive        w / s : faster / slower      a / d : turn left / right
               arrows work too               space : stop (hand back to autonomy)
  voice        1 SUCH   2 HIER   3 BRAV   4 SITZ   8 Kunststueck
  gesture      5 SITZ   6 PLATZ  7 AUF
  other        t Taster SUCH   x abort mission   z zurueck zum Start   e NOT-AUS   r reset sim
  type         :  then a word, e.g.  :such  or  :gesture:platz  (Enter)
               h help    q quit
 --------------------------------------------------------------
"""

KEYMAP = {
    '1': 'voice:SUCH', '2': 'voice:HIER', '3': 'voice:BRAV', '4': 'voice:SITZ', '8': 'voice:TRICK',
    '5': 'gesture:SITZ', '6': 'gesture:PLATZ', '7': 'gesture:AUF',
    't': 'button:SUCH', 'x': 'ctl:abort', 'z': 'ctl:home', 'e': 'ctl:estop', 'r': 'ctl:reset',
}
ARROWS = {'\x1b[A': 'w', '\x1b[B': 's', '\x1b[C': 'd', '\x1b[D': 'a'}
LIN_STEP, ANG_STEP, LIN_MAX, ANG_MAX = 0.1, 0.4, 0.6, 2.0


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_manual', 10)
        self.inject_pub = self.create_publisher(String, '/dog/inject', 10)
        self.create_subscription(String, '/dog/ack', lambda m: self.say(f'ACK  {m.data}'), 10)
        self.create_subscription(String, '/cmd_mux/active', lambda m: self.say(f'mux  {m.data}'), LATCHED)
        self.create_subscription(Bool, '/estop', lambda m: self.say(f'NOT-AUS {"AN" if m.data else "aus"}'), LATCHED)
        self.v = self.w = 0.0
        self.zero_burst = 0

    def say(self, text):
        sys.stdout.write(f'\r  > {text}\033[K\r\n')
        sys.stdout.flush()

    def key(self, k):
        k = ARROWS.get(k, k)
        if k == 'w':
            self.v = min(LIN_MAX, self.v + LIN_STEP)
        elif k == 's':
            self.v = max(-LIN_MAX, self.v - LIN_STEP)
        elif k == 'a':
            self.w = min(ANG_MAX, self.w + ANG_STEP)
        elif k == 'd':
            self.w = max(-ANG_MAX, self.w - ANG_STEP)
        elif k == ' ':
            self.v = self.w = 0.0
            self.zero_burst = 3
        elif k in KEYMAP:
            self.inject_pub.publish(String(data=KEYMAP[k]))
            return
        else:
            return
        self.status()

    def status(self):
        sys.stdout.write(f'\r  manual v={self.v:+.1f} m/s  w={self.w:+.1f} rad/s\033[K')
        sys.stdout.flush()

    def publish(self):
        if self.v or self.w or self.zero_burst:
            t = Twist()
            t.linear.x, t.angular.z = self.v, self.w
            self.cmd_pub.publish(t)
            self.zero_burst = max(0, self.zero_burst - 1)


def main():
    rclpy.init()
    node = KeyboardTeleop()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    print(HELP)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            if select.select([sys.stdin], [], [], 0.1)[0]:
                k = sys.stdin.read(1)
                if k == '\x1b':
                    k += sys.stdin.read(2)
                if k == 'q':
                    break
                if k == 'h':
                    print(HELP)
                elif k == ':':
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
                    try:
                        word = input('\r\n  say> ').strip()
                    finally:
                        tty.setcbreak(fd)
                    if word:
                        node.inject_pub.publish(String(data=word))
                else:
                    node.key(k.lower() if len(k) == 1 else k)
            node.publish()
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
