import os
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "gesture_recognizer.task"
)

if not os.path.exists(MODEL_PATH):
  print(
      "[FEHLER] Die Datei 'gesture_recognizer.task' wurde im Ordner nicht"
      " gefunden!"
  )
  exit(1)

# Schwellenwerte für die präzise Zeigefinger-nach-unten-Logik («PLATZ»)
EXT_ON, EXT_OFF = 0.88, 0.78
CURL_ON, CURL_OFF = 0.62, 0.72
POINT_OTHERS_MAX = 0.74
POINT_OTHERS_MIN_COUNT = 2
COS_DOWN = np.cos(np.radians(70))
MIN_2D_LEN = 0.35
EMA_ALPHA = 0.5

# Entprellung
HISTORY_LENGTH = 10
ENTER_VOTES = 6
KEEP_VOTES = 4
NONE = "Keine Geste"

FINGERS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


# ---------------------------------------------------------------------------
# Hilfsfunktionen für die Geometrie-Logik («PLATZ»)
# ---------------------------------------------------------------------------
def unit(v):
  n = np.linalg.norm(v)
  return v / n if n > 1e-6 else v


def straightness(pts, ids):
  mcp, pip, dip, tip = (pts[i] for i in ids)
  path = (
      np.linalg.norm(pip - mcp)
      + np.linalg.norm(dip - pip)
      + np.linalg.norm(tip - dip)
  )
  return np.linalg.norm(tip - mcp) / max(path, 1e-6)


def update_state(prev, s):
  if prev == "ext" and s >= EXT_OFF:
    return "ext"
  if prev == "curl" and s <= CURL_OFF:
    return "curl"
  if s > EXT_ON:
    return "ext"
  if s < CURL_ON:
    return "curl"
  return "mid"


def vertical(world, img, a, b, w, h):
  ref_len = np.linalg.norm(
      np.array([(img[9][0] - img[0][0]) * w, (img[9][1] - img[0][1]) * h])
  )
  v2 = np.array([(img[b][0] - img[a][0]) * w, (img[b][1] - img[a][1]) * h])
  if ref_len > 1e-6 and np.linalg.norm(v2) > MIN_2D_LEN * ref_len:
    return unit(v2)[1], "2D"
  return unit(world[b] - world[a])[1], "3D"


def stabilize(history, stable):
  if len(history) < HISTORY_LENGTH:
    return stable
  counts = Counter(history)
  candidate, votes = counts.most_common(1)[0]
  if candidate != stable and candidate != NONE and votes >= ENTER_VOTES:
    return candidate
  if counts[stable] < KEEP_VOTES:
    return candidate if (candidate != NONE and votes >= ENTER_VOTES) else NONE
  return stable


# ---------------------------------------------------------------------------
# Setup MediaPipe Gesture Recognizer
# ---------------------------------------------------------------------------
options = vision.GestureRecognizerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
recognizer = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)
print(
    "[*] Hybride Gestenerkennung (Recognizer + Landmarker) gestartet. 'q'"
    " Beenden."
)

history = deque(maxlen=HISTORY_LENGTH)
stable_gesture = NONE
last_ts = -1
sm_world = None
sm_img = None
states = {name: "mid" for name in FINGERS}

# ---------------------------------------------------------------------------
# Hauptschleife
# ---------------------------------------------------------------------------
while cap.isOpened():
  ok, frame = cap.read()
  if not ok:
    print("Fehler beim Zugriff auf die Webcam.")
    break

  frame = cv2.flip(frame, 1)
  h, w, _ = frame.shape

  mp_image = mp.Image(
      image_format=mp.ImageFormat.SRGB,
      data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
  )

  ts = int(time.monotonic() * 1000)
  if ts <= last_ts:
    ts = last_ts + 1
  last_ts = ts

  # Ausführen mit Video-Modus für zeitliche Konsistenz
  result = recognizer.recognize_for_video(mp_image, ts)

  current = NONE

  if result.gestures and result.hand_landmarks and result.hand_world_landmarks:
    # 1. KI-Klassifikation für SITZ & AUF auswerten
    top_gesture = result.gestures[0][0]
    gesture_name = top_gesture.category_name
    gesture_score = top_gesture.score

    # 2. Landmarks für die präzise PLATZ-Logik vorbereiten
    world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]])
    img = np.array([[p.x, p.y] for p in result.hand_landmarks[0]])

    if sm_world is None:
      sm_world, sm_img = world, img
    else:
      sm_world = EMA_ALPHA * world + (1 - EMA_ALPHA) * sm_world
      sm_img = EMA_ALPHA * img + (1 - EMA_ALPHA) * sm_img

    # Finger-Streckung für PLATZ berechnen
    values = {}
    for name, ids in FINGERS.items():
      values[name] = straightness(sm_world, ids)
      states[name] = update_state(states[name], values[name])

    idx_v, _ = vertical(sm_world, sm_img, 5, 8, w, h)
    idx_down = idx_v > COS_DOWN

    others = ["middle", "ring", "pinky"]
    n_curled = sum(values[n] < POINT_OTHERS_MAX for n in others)
    n_ext = sum(states[n] == "ext" for n in others)
    pointing = (
        states["index"] == "ext"
        and n_curled >= POINT_OTHERS_MIN_COUNT
        and n_ext == 0
    )

    # --- ENTSCHEIDUNG ---
    if gesture_name == "Open_Palm" and gesture_score >= 0.5:
      current = "SITZ"
    elif gesture_name == "Pointing_Up" and gesture_score >= 0.6:
      current = "AUF"
    elif pointing and idx_down:
      current = "PLATZ"

    for p in result.hand_landmarks[0]:
      cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (0, 0, 255), -1)
  else:
    sm_world = sm_img = None
    states = {name: "mid" for name in FINGERS}

  history.append(current)
  stable_gesture = stabilize(history, stable_gesture)

  # Anzeige im Live-Bild
  cv2.putText(
      frame,
      f"Befehl: {stable_gesture}",
      (30, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (0, 0, 255) if stable_gesture == NONE else (0, 255, 0),
      2,
      cv2.LINE_AA,
  )

  cv2.imshow("PREN Gestenerkennung - Hybrid v4", frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

recognizer.close()
cap.release()
cv2.destroyAllWindows()