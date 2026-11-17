#!/usr/bin/env python3
import os
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO

# --- Konfiguration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GESTURE_MODEL = os.path.join(BASE_DIR, "gesture_recognizer.task")
POSE_MODEL = "yolo26n-pose.pt"

SITZ, PLATZ, AUF = "SITZ", "PLATZ", "AUF"
NONE = "KEIN BEFEHL"

CAM_WIDTH, CAM_HEIGHT = 1280, 720
POSE_EVERY_N = 2
POSE_IMGSZ = 320
KP_CONF = 0.5
TORSO_MARGIN = 0.25

EXT_ON, EXT_OFF = 0.88, 0.78
CURL_ON, CURL_OFF = 0.62, 0.72
POINT_OTHERS_MAX = 0.74
POINT_OTHERS_MIN_COUNT = 2
COS_DOWN = np.cos(np.radians(70))
COS_UP = np.cos(np.radians(60))
MIN_2D_LEN = 0.35

ARM_DOWN_COS = np.cos(np.radians(65))
ARM_MATCH_FRAC = 0.8
ARM_WRIST_CONF = 0.3
POINT_INDEX_LOOSE = 0.80
HAND_DETECTION_CONF = 0.5
EMA_ALPHA = 0.5
DOM_MIN = 1.35

OPEN_PALM_MIN = 0.5
POINTING_UP_MIN = 0.5

HISTORY_LENGTH = 10
ENTER_VOTES = 6
KEEP_VOTES = 4

FINGERS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


class GestureNode(Node):

  def __init__(self):
    super().__init__("gesture_recognition_node")
    self.publisher_ = self.create_publisher(String, "/command/gesture", 10)

    if not os.path.exists(GESTURE_MODEL):
      self.get_logger().error(f"'{GESTURE_MODEL}' nicht gefunden!")
      raise SystemExit(1)

    self.recognizer = vision.GestureRecognizer.create_from_options(
        vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=GESTURE_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=HAND_DETECTION_CONF,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    self.pose_model = YOLO(POSE_MODEL)

    self.cap = cv2.VideoCapture(0)
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    self.history = deque(maxlen=HISTORY_LENGTH)
    self.stable_gesture = NONE
    self.last_ts = -1
    self.sm_worlds, self.sm_imgs, self.states = {}, {}, {}
    self.torso, self.arms = None, []
    self.frame_no = 0

    # Timer for processing loop matching camera framerate (~30fps)
    self.timer = self.create_timer(0.033, self.process_frame)
    self.get_logger().info("Gesture Recognition Node gestartet.")

  def process_frame(self):
    ok, frame = self.cap.read()
    if not ok:
      self.get_logger().warn("Fehler beim Zugriff auf die Webcam.")
      return

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    self.frame_no += 1

    # Pose (gedrosselt)
    if self.frame_no % POSE_EVERY_N == 0:
      pose_res = self.pose_model(frame, imgsz=POSE_IMGSZ, verbose=False)[0]
      self.torso = self.torso_box(pose_res, w, h)
      self.arms = self.arm_segments(pose_res)

    # Hände
    ts = max(int(time.monotonic() * 1000), self.last_ts + 1)
    self.last_ts = ts
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    )
    result = self.recognizer.recognize_for_video(mp_image, ts)

    current = NONE
    seen = set()

    if result.hand_landmarks and result.hand_world_landmarks:
      for i, (hand_lms, world_lms) in enumerate(
          zip(result.hand_landmarks, result.hand_world_landmarks)
      ):
        label = (
            result.handedness[i][0].category_name
            if result.handedness and i < len(result.handedness)
            else str(i)
        )
        key = label if label not in seen else f"{label}_2"
        seen.add(key)

        g_name, g_score = "", 0.0
        if result.gestures and i < len(result.gestures) and result.gestures[i]:
          g_name = result.gestures[i][0].category_name
          g_score = result.gestures[i][0].score

        world = np.array([[p.x, p.y, p.z] for p in world_lms])
        img = np.array([[p.x, p.y] for p in hand_lms])

        if key not in self.sm_worlds:
          self.sm_worlds[key], self.sm_imgs[key] = world, img
          self.states[key] = {n: "mid" for n in FINGERS}
        else:
          self.sm_worlds[key] = (
              EMA_ALPHA * world + (1 - EMA_ALPHA) * self.sm_worlds[key]
          )
          self.sm_imgs[key] = (
              EMA_ALPHA * img + (1 - EMA_ALPHA) * self.sm_imgs[key]
          )

        values = {}
        for name, ids in FINGERS.items():
          values[name] = self.straightness(self.sm_worlds[key], ids)
          self.states[key][name] = self.update_state(
              self.states[key][name], values[name]
          )

        idx_v = self.vertical(self.sm_worlds[key], self.sm_imgs[key], 5, 8, w, h)
        others = ("middle", "ring", "pinky")
        n_curled = sum(values[n] < POINT_OTHERS_MAX for n in others)
        n_ext = sum(self.states[key][n] == "ext" for n in others)
        pointing = (
            self.states[key]["index"] == "ext"
            and n_curled >= POINT_OTHERS_MIN_COUNT
            and n_ext == 0
        )

        palm = np.mean(
            [[hand_lms[j].x * w, hand_lms[j].y * h] for j in (0, 5, 9, 13, 17)],
            axis=0,
        )
        in_torso = self.point_in_box(palm, self.torso)

        arm_v = self.arm_down_value(palm, self.arms)
        arm_down = arm_v is not None and arm_v > ARM_DOWN_COS
        w3d = self.sm_worlds[key]
        palm_len = max(np.linalg.norm(w3d[9] - w3d[0]), 1e-6)
        reach_idx = np.linalg.norm(w3d[8] - w3d[0]) / palm_len
        reach_oth = max(
            np.linalg.norm(w3d[t] - w3d[0]) / palm_len for t in (12, 16, 20)
        )
        dominance = reach_idx / max(reach_oth, 1e-6)

        pointing_loose = (
            values["index"] >= POINT_INDEX_LOOSE and dominance >= DOM_MIN
        )
        finger_down = pointing and idx_v > COS_DOWN
        arm_pointing_down = pointing_loose and arm_down and idx_v > 0.0

        cmd = NONE
        if g_name == "Open_Palm" and g_score >= OPEN_PALM_MIN and in_torso:
          cmd = SITZ
        elif (g_name == "Pointing_Up" and g_score >= POINTING_UP_MIN) or (
            pointing and idx_v < -COS_UP
        ):
          cmd = AUF
        elif finger_down or arm_pointing_down:
            cmd = PLATZ

        if cmd != NONE:
          current = cmd

    for k in [k for k in self.sm_worlds if k not in seen]:
      self.sm_worlds.pop(k), self.sm_imgs.pop(k), self.states.pop(k)

    self.history.append(current)
    self.stable_gesture = self.stabilize(self.history, self.stable_gesture)

    # Publish recognized gesture
    msg = String()
    msg.data = self.stable_gesture
    self.publisher_.publish(msg)

  # --- Helper Methods ---
  def straightness(self, pts, ids):
    mcp, pip, dip, tip = (pts[i] for i in ids)
    path = (
        np.linalg.norm(pip - mcp)
        + np.linalg.norm(dip - pip)
        + np.linalg.norm(tip - dip)
    )
    return np.linalg.norm(tip - mcp) / max(path, 1e-6)

  def update_state(self, prev, s):
    if prev == "ext" and s >= EXT_OFF:
      return "ext"
    if prev == "curl" and s <= CURL_OFF:
      return "curl"
    if s > EXT_ON:
      return "ext"
    if s < CURL_ON:
      return "curl"
    return "mid"

  def vertical(self, world, img, a, b, w, h):
    ref_len = np.linalg.norm(
        np.array([(img[9][0] - img[0][0]) * w, (img[9][1] - img[0][1]) * h])
    )
    v2 = np.array([(img[b][0] - img[a][0]) * w, (img[b][1] - img[a][1]) * h])
    if ref_len > 1e-6 and np.linalg.norm(v2) > MIN_2D_LEN * ref_len:
      return (v2 / np.linalg.norm(v2))[1]
    v3 = world[b] - world[a]
    return (v3 / np.linalg.norm(v3))[1]

  def stabilize(self, history, stable):
    if len(history) < HISTORY_LENGTH:
      return stable
    counts = Counter(history)
    candidate, votes = counts.most_common(1)[0]
    if candidate != stable and candidate != NONE and votes >= ENTER_VOTES:
      return candidate
    if counts[stable] < KEEP_VOTES:
      return candidate if (candidate != NONE and votes >= ENTER_VOTES) else NONE
    return stable

  def torso_box(self, result, w, h):
    if (
        result is None
        or result.keypoints is None
        or result.boxes is None
        or len(result.boxes) == 0
    ):
      return None
    xyxy = result.boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    i = int(np.argmax(areas))
    xy = result.keypoints.xy[i].cpu().numpy()
    conf = result.keypoints.conf
    conf_v = conf[i].cpu().numpy() if conf is not None else np.ones(len(xy))
    if len(xy) < 13 or conf_v[5] < KP_CONF or conf_v[6] < KP_CONF:
      return None
    ls, rs = xy[5], xy[6]
    sh_w = abs(ls[0] - rs[0])
    sh_y = (ls[1] + rs[1]) / 2
    x1 = min(ls[0], rs[0]) - TORSO_MARGIN * sh_w
    x2 = max(ls[0], rs[0]) + TORSO_MARGIN * sh_w
    y1 = sh_y - 0.3 * sh_w
    y2 = (
        (xy[11][1] + xy[12][1]) / 2
        if conf_v[11] >= KP_CONF and conf_v[12] >= KP_CONF
        else sh_y + 1.6 * sh_w
    )
    return (max(0, x1), max(0, y1), min(w, x2), min(h, y2))

  def point_in_box(self, pt, box):
    return (
        box is not None
        and box[0] <= pt[0] <= box[2]
        and box[1] <= pt[1] <= box[3]
    )

  def arm_segments(self, result):
    if (
        result is None
        or result.keypoints is None
        or result.boxes is None
        or len(result.boxes) == 0
    ):
      return []
    xyxy = result.boxes.xyxy.cpu().numpy()
    i = int(np.argmax((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])))
    xy = result.keypoints.xy[i].cpu().numpy()
    conf = result.keypoints.conf
    conf_v = conf[i].cpu().numpy() if conf is not None else np.ones(len(xy))
    if len(xy) < 11 or conf_v[5] < KP_CONF or conf_v[6] < KP_CONF:
      return []
    sh_w = abs(xy[5][0] - xy[6][0])
    return [
        (xy[s], xy[wr], sh_w)
        for s, wr in ((5, 9), (6, 10))
        if conf_v[wr] >= ARM_WRIST_CONF
    ]

  def arm_down_value(self, palm, arms):
    best, best_d = None, 1e9
    for sh, wr, sh_w in arms:
      d = np.linalg.norm(wr - palm)
      if d < best_d and d < ARM_MATCH_FRAC * max(sh_w, 1.0):
        best, best_d = (sh, wr), d
    if best is None:
      return None
    v = best[1] - best[0]
    return float((v / np.linalg.norm(v))[1])

  def destroy_node(self):
    self.cap.release()
    self.recognizer.close()
    super().destroy_node()


def main(args=None):
  rclpy.init(args=args)
  node = GestureNode()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
  main()