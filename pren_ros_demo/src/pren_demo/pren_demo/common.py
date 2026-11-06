"""Field geometry (map frame) and small helpers shared by all nodes.

map origin = centre of the Startfeld, +x points into the search segment.
The dog starts facing the person (-x) and has to turn around for "SUCH".
"""
import math

from geometry_msgs.msg import Point, Quaternion
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformException
from visualization_msgs.msg import Marker

START_X, START_Y, START_YAW = 0.0, 0.0, math.pi
START_BOX = (0.5, 0.2)                 # Startfeld 50 x 20 cm, long side along x
SEG_R_MIN, SEG_R_MAX = 1.0, 5.0        # search segment radii
SEG_HALF_ANGLE = math.radians(15.0)    # 30 deg segment
PERSON_X, PERSON_Y = -1.5, 0.0         # referee stands in the interaction area
INTERACT_RADIUS = 0.75                 # D = 1.5 m

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def yaw_to_quat(yaw):
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def rpy_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def in_segment(x, y):
    r = math.hypot(x, y)
    return SEG_R_MIN <= r <= SEG_R_MAX and abs(math.atan2(y, x)) <= SEG_HALF_ANGLE


def go_to(x, y, yaw, tx, ty, v_max=0.4, w_max=1.5, tol=0.08):
    """P-controller: turn towards target, then drive. Returns (v, w, reached)."""
    dx, dy = tx - x, ty - y
    dist = math.hypot(dx, dy)
    if dist < tol:
        return 0.0, 0.0, True
    err = wrap(math.atan2(dy, dx) - yaw)
    w = clamp(2.5 * err, -w_max, w_max)
    v = 0.0 if abs(err) > 0.6 else min(v_max, 0.8 * dist) * math.cos(err)
    return v, w, False


def turn_to(yaw, target_yaw, w_max=1.5, tol=0.05):
    """Returns (w, reached)."""
    err = wrap(target_yaw - yaw)
    if abs(err) < tol:
        return 0.0, True
    w = clamp(2.5 * err, -w_max, w_max)
    if abs(w) < 0.2:
        w = math.copysign(0.2, w)
    return w, False


def lookup_pose(tf_buffer, target='map', source='base_link'):
    """Latest (x, y, yaw) of `source` in `target`, or None."""
    try:
        t = tf_buffer.lookup_transform(target, source, Time())
    except TransformException:
        return None
    tr = t.transform
    return tr.translation.x, tr.translation.y, quat_to_yaw(tr.rotation)


def color(r, g, b, a=1.0):
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


def pt(x, y, z=0.0):
    return Point(x=float(x), y=float(y), z=float(z))


def marker(frame, ns, mid, mtype, scale=(0.1, 0.1, 0.1), rgba=(1, 1, 1, 1),
           xyz=(0.0, 0.0, 0.0), yaw=0.0, quat=None, text='', points=None,
           lifetime=0.0, locked=False):
    m = Marker()
    m.header.frame_id = frame
    m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
    m.scale.x, m.scale.y, m.scale.z = (float(s) for s in scale)
    m.color = color(*rgba)
    m.pose.position = pt(*xyz)
    m.pose.orientation = quat if quat is not None else yaw_to_quat(yaw)
    m.text = text.replace(' ', '_')   # rviz2 renders every space ~2 glyphs wide
    if points:
        m.points = [pt(*p) for p in points]
    m.lifetime.sec = int(lifetime)
    m.lifetime.nanosec = int((lifetime - int(lifetime)) * 1e9)
    m.frame_locked = locked
    return m
