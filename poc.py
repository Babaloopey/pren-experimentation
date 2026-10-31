"""
PREN - Befehlserkennung (Proof of Concept)

Vereint:
  * MediaPipe Gesture Recognizer (Hand-Gesten, bis zu 2 Hände)
  * YOLO26 Pose (Hand muss vor dem Oberkörper sein -> SITZ)
  * OpenWakeWord (Sprachbefehl "Sitz")

Befehle:
  SITZ  : Handfläche vor der Brust (Geste) ODER Sprachbefehl "Sitz"
  PLATZ : Zeigefinger zeigt zum Boden
  AUF   : Zeigefinger oder «L» (Daumen + Zeigefinger) zeigt nach oben

Steuerung:  q / ESC = Beenden,  f = Vollbild an/aus
Optionen:   --camera N, --no-voice
"""

import argparse
import os
import threading
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GESTURE_MODEL = os.path.join(BASE_DIR, "gesture_recognizer.task")
POSE_MODEL = "yolo26n-pose.pt"
WAKEWORD_MODEL = os.path.join(BASE_DIR, "Sitz_20261014_145414.onnx")

# Befehle
SITZ, PLATZ, AUF = "SITZ", "PLATZ", "AUF"
NONE = "KEIN BEFEHL"

# Kamera
CAM_WIDTH, CAM_HEIGHT = 1280, 720

# Pose (läuft nur jedes N-te Frame, Box wird dazwischen wiederverwendet)
POSE_EVERY_N = 2
POSE_IMGSZ = 320
KP_CONF = 0.5  # min. Konfidenz der Schulter-/Hüft-Keypoints
TORSO_MARGIN = 0.25  # Torso-Box seitlich um x * Schulterbreite erweitern

# Geometrie-Logik (Zeigefinger)
EXT_ON, EXT_OFF = 0.88, 0.78
CURL_ON, CURL_OFF = 0.62, 0.72
POINT_OTHERS_MAX = 0.74
POINT_OTHERS_MIN_COUNT = 2
COS_DOWN = np.cos(np.radians(70))  # PLATZ: max. 70° Abweichung von "unten"
COS_UP = np.cos(np.radians(60))  # AUF: max. 60° Abweichung von "oben"
MIN_2D_LEN = 0.35
EMA_ALPHA = 0.5

# Klassifikator-Schwellen
OPEN_PALM_MIN = 0.5 #0.5
POINTING_UP_MIN = 0.5 #0.6

# Entprellung
HISTORY_LENGTH = 10
ENTER_VOTES = 6
KEEP_VOTES = 4

# Sprache
VOICE_COMMAND = SITZ
VOICE_THRESHOLD = 0.2
VOICE_VAD = 0.3
VOICE_REQUIRED_CONSECUTIVE = 2
VOICE_COOLDOWN_S = 1.5
VOICE_HOLD_S = 1.5  # so lange bleibt der Sprachbefehl im UI sichtbar
AUDIO_RATE = 16000
AUDIO_CHUNK = 1280  # 80 ms

FINGERS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

WINDOW = "PREN Befehlserkennung"


# ---------------------------------------------------------------------------
# Sprachbefehl (eigener Thread, blockiert die Kamera nicht)
# ---------------------------------------------------------------------------
class VoiceListener(threading.Thread):
  def __init__(self, model_path):
    super().__init__(daemon=True)
    self.model_path = model_path
    self.stop_event = threading.Event()
    self.lock = threading.Lock()
    self.last_hit = -1e9
    self.score = 0.0
    self.ready = False
    self.error = None

  def seconds_since_hit(self):
    with self.lock:
      return time.time() - self.last_hit

  def current_score(self):
    with self.lock:
      return self.score

  def run(self):
    try:
      import pyaudio
      from openwakeword.model import Model

      model = Model(
          wakeword_models=[self.model_path],
          inference_framework="onnx",
          vad_threshold=VOICE_VAD,
      )
      pa = pyaudio.PyAudio()
      stream = pa.open(
          format=pyaudio.paInt16,
          channels=1,
          rate=AUDIO_RATE,
          input=True,
          frames_per_buffer=AUDIO_CHUNK,
      )
    except Exception as e:  # Modell/Mikrofon fehlt -> Gesten laufen weiter
      self.error = str(e)
      print(f"[WARNUNG] Sprachbefehl deaktiviert: {e}")
      print(
          "          Falls melspectrogram.onnx fehlt: python -c"
          " 'import openwakeword; openwakeword.utils.download_models()'"
      )
      return

    key = os.path.splitext(os.path.basename(self.model_path))[0]
    hits, last_trigger = 0, 0.0
    self.ready = True
    print(f"[*] Sprachbefehl aktiv (Modell: {key})")
    try:
      while not self.stop_event.is_set():
        data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
        pred = model.predict(np.frombuffer(data, dtype=np.int16))
        score = pred.get(key, max(pred.values(), default=0.0))
        with self.lock:
          self.score = float(score)

        hits = hits + 1 if score > VOICE_THRESHOLD else 0
        now = time.time()
        if (
            hits >= VOICE_REQUIRED_CONSECUTIVE
            and now - last_trigger > VOICE_COOLDOWN_S
        ):
          last_trigger, hits = now, 0
          with self.lock:
            self.last_hit = now
    finally:
      stream.stop_stream()
      stream.close()
      pa.terminate()

  def stop(self):
    self.stop_event.set()


# ---------------------------------------------------------------------------
# Geometrie-Helfer
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
  """Hysterese: verhindert Flackern zwischen gestreckt/gebeugt."""
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
  """y-Komponente (Bild: +y = unten) des Vektors a->b. 2D bevorzugt, sonst 3D."""
  ref_len = np.linalg.norm(
      np.array([(img[9][0] - img[0][0]) * w, (img[9][1] - img[0][1]) * h])
  )
  v2 = np.array([(img[b][0] - img[a][0]) * w, (img[b][1] - img[a][1]) * h])
  if ref_len > 1e-6 and np.linalg.norm(v2) > MIN_2D_LEN * ref_len:
    return unit(v2)[1]
  return unit(world[b] - world[a])[1]


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
# Pose: Oberkörper-Box der (grössten) Person
# ---------------------------------------------------------------------------
def torso_box(result, w, h):
  """Gibt (x1, y1, x2, y2) in Pixeln zurück oder None."""
  if (
      result is None
      or result.keypoints is None
      or result.boxes is None
      or len(result.boxes) == 0
  ):
    return None

  xyxy = result.boxes.xyxy.cpu().numpy()
  areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
  i = int(np.argmax(areas))  # grösste Person = Person vor dem Roboter

  xy = result.keypoints.xy[i].cpu().numpy()
  conf_t = result.keypoints.conf
  conf = conf_t[i].cpu().numpy() if conf_t is not None else np.ones(len(xy))
  if len(xy) < 13 or conf[5] < KP_CONF or conf[6] < KP_CONF:
    return None

  ls, rs = xy[5], xy[6]
  sh_w = abs(ls[0] - rs[0])
  sh_y = (ls[1] + rs[1]) / 2
  x1 = min(ls[0], rs[0]) - TORSO_MARGIN * sh_w
  x2 = max(ls[0], rs[0]) + TORSO_MARGIN * sh_w
  y1 = sh_y - 0.3 * sh_w

  if conf[11] >= KP_CONF and conf[12] >= KP_CONF:
    y2 = (xy[11][1] + xy[12][1]) / 2
  else:  # Hüfte nicht im Bild (z.B. Person sitzt am Tisch) -> schätzen
    y2 = sh_y + 1.6 * sh_w

  return (
      max(0, x1),
      max(0, y1),
      min(w, x2),
      min(h, y2),
  )


def point_in_box(pt, box):
  return box is not None and box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def put_centered(frame, text, cy, scale, color, thickness):
  h, w = frame.shape[:2]
  (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
  cv2.putText(
      frame,
      text,
      ((w - tw) // 2, cy + th // 2),
      cv2.FONT_HERSHEY_SIMPLEX,
      scale,
      color,
      thickness,
      cv2.LINE_AA,
  )


def draw_ui(frame, command, source, info):
  h, w = frame.shape[:2]
  active = command != NONE

  # Halbtransparentes Banner oben
  banner_h = int(h * 0.26)
  color = (50, 170, 50) if active else (50, 50, 190)  # BGR
  overlay = frame.copy()
  cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
  cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

  put_centered(frame, command, int(banner_h * 0.5), 3.2 if active else 2.4,
               (255, 255, 255), 8 if active else 5)
  sub = f"Quelle: {source}" if active else "Warte auf Geste oder Sprachbefehl"
  put_centered(frame, sub, int(banner_h * 0.85), 1.0, (255, 255, 255), 2)

  # Statuszeile unten
  cv2.rectangle(frame, (0, h - 40), (w, h), (30, 30, 30), -1)
  cv2.putText(frame, info, (15, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
              (220, 220, 220), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--camera", type=int, default=0)
  parser.add_argument("--no-voice", action="store_true")
  args = parser.parse_args()

  if not os.path.exists(GESTURE_MODEL):
    print(f"[FEHLER] '{GESTURE_MODEL}' nicht gefunden!")
    raise SystemExit(1)

  recognizer = vision.GestureRecognizer.create_from_options(
      vision.GestureRecognizerOptions(
          base_options=mp_python.BaseOptions(model_asset_path=GESTURE_MODEL),
          running_mode=vision.RunningMode.VIDEO,
          num_hands=2,
          min_hand_detection_confidence=0.6,
          min_hand_presence_confidence=0.5,
          min_tracking_confidence=0.5,
      )
  )
  pose_model = YOLO(POSE_MODEL)

  voice = None
  if not args.no_voice and os.path.exists(WAKEWORD_MODEL):
    voice = VoiceListener(WAKEWORD_MODEL)
    voice.start()
  elif not args.no_voice:
    print(f"[WARNUNG] Wakeword-Modell fehlt: {WAKEWORD_MODEL} -> ohne Sprache")

  cap = cv2.VideoCapture(args.camera)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

  cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
  fullscreen = True
  cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

  history = deque(maxlen=HISTORY_LENGTH)
  stable_gesture = NONE
  last_ts = -1
  sm_worlds, sm_imgs, states = {}, {}, {}
  torso = None
  frame_no = 0
  fps, t_prev = 0.0, time.perf_counter()

  print("[*] Läuft. 'q'/ESC beenden, 'f' Vollbild umschalten.")

  try:
    while cap.isOpened():
      ok, frame = cap.read()
      if not ok:
        print("Fehler beim Zugriff auf die Webcam.")
        break

      frame = cv2.flip(frame, 1)
      h, w, _ = frame.shape
      frame_no += 1

      # --- Pose (gedrosselt) ---
      if frame_no % POSE_EVERY_N == 0:
        pose_res = pose_model(frame, imgsz=POSE_IMGSZ, verbose=False)[0]
        torso = torso_box(pose_res, w, h)

      # --- Hände ---
      ts = max(int(time.monotonic() * 1000), last_ts + 1)
      last_ts = ts
      mp_image = mp.Image(
          image_format=mp.ImageFormat.SRGB,
          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
      )
      result = recognizer.recognize_for_video(mp_image, ts)

      current = NONE
      seen = set()

      if result.hand_landmarks and result.hand_world_landmarks:
        for i, (hand_lms, world_lms) in enumerate(
            zip(result.hand_landmarks, result.hand_world_landmarks)
        ):
          # Hand-Schlüssel (links/rechts) statt Index -> stabiler Zustand
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

          if key not in sm_worlds:
            sm_worlds[key], sm_imgs[key] = world, img
            states[key] = {n: "mid" for n in FINGERS}
          else:
            sm_worlds[key] = EMA_ALPHA * world + (1 - EMA_ALPHA) * sm_worlds[key]
            sm_imgs[key] = EMA_ALPHA * img + (1 - EMA_ALPHA) * sm_imgs[key]

          values = {}
          for name, ids in FINGERS.items():
            values[name] = straightness(sm_worlds[key], ids)
            states[key][name] = update_state(states[key][name], values[name])

          idx_v = vertical(sm_worlds[key], sm_imgs[key], 5, 8, w, h)
          others = ("middle", "ring", "pinky")
          n_curled = sum(values[n] < POINT_OTHERS_MAX for n in others)
          n_ext = sum(states[key][n] == "ext" for n in others)
          pointing = (
              states[key]["index"] == "ext"
              and n_curled >= POINT_OTHERS_MIN_COUNT
              and n_ext == 0
          )

          # Handzentrum in Pixeln (Handgelenk + Fingerwurzeln)
          palm = np.mean([[hand_lms[j].x * w, hand_lms[j].y * h]
                          for j in (0, 5, 9, 13, 17)], axis=0)
          in_torso = point_in_box(palm, torso)

          # --- Entscheidung für diese Hand (Priorität: SITZ > AUF > PLATZ) ---
          cmd = NONE
          if g_name == "Open_Palm" and g_score >= OPEN_PALM_MIN and in_torso:
            cmd = SITZ
          elif (g_name == "Pointing_Up" and g_score >= POINTING_UP_MIN) or (
              pointing and idx_v < -COS_UP  # deckt auch die «L»-Haltung ab
          ):
            cmd = AUF
          elif pointing and idx_v > COS_DOWN:
            cmd = PLATZ

          if cmd != NONE:
            current = cmd

          for p in hand_lms:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (0, 0, 255), -1)
          if cmd != NONE:
            cv2.putText(frame, cmd, (int(palm[0]) - 30, int(palm[1]) - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
                        cv2.LINE_AA)

      # Zustände verschwundener Hände aufräumen
      for k in [k for k in sm_worlds if k not in seen]:
        sm_worlds.pop(k), sm_imgs.pop(k), states.pop(k)

      history.append(current)
      stable_gesture = stabilize(history, stable_gesture)

      # --- Fusion: Sprache (kurz gehalten) hat Vorrang vor Geste ---
      if voice and voice.ready and voice.seconds_since_hit() < VOICE_HOLD_S:
        command, source = VOICE_COMMAND, "Sprache"
      elif stable_gesture != NONE:
        command, source = stable_gesture, "Geste"
      else:
        command, source = NONE, "-"

      # Torso-Box (Debug)
      if torso is not None:
        cv2.rectangle(frame, (int(torso[0]), int(torso[1])),
                      (int(torso[2]), int(torso[3])), (255, 165, 0), 2)

      now = time.perf_counter()
      fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
      t_prev = now
      if voice is None:
        v_txt = "Sprache: aus"
      elif voice.error:
        v_txt = "Sprache: Fehler"
      elif not voice.ready:
        v_txt = "Sprache: lädt..."
      else:
        v_txt = f"Sprache: Score {voice.current_score():.2f}"
      info = (
          f"Pose: {'OK' if torso else 'keine Person'}  |  {v_txt}  |  "
          f"Geste: {stable_gesture}  |  {fps:.0f} FPS"
      )
      draw_ui(frame, command, source, info)
      cv2.imshow(WINDOW, frame)

      k = cv2.waitKey(1) & 0xFF
      if k in (ord("q"), 27):
        break
      if k == ord("f"):
        fullscreen = not fullscreen
        cv2.setWindowProperty(
            WINDOW,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
        )
  finally:
    if voice:
      voice.stop()
    recognizer.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  main()