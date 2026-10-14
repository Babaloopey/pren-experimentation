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
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# Streckungs-Mass pro Finger: Luftlinie(MCP->Spitze) / Fingerlänge entlang der Gelenke
#   gerade ~ 0.95-1.0 | locker/hängend ~ 0.8-0.9 | Faust ~ 0.35-0.55
EXT_ON, EXT_OFF = 0.88, 0.78      # gestreckt (mit Hysterese)
CURL_ON, CURL_OFF = 0.62, 0.72    # eingerollt (mit Hysterese)

# SITZ: Ring-/Kleinfinger sind kürzer und bei geschlossenen Fingern ungenauer
SITZ_SOFT_EXT = 0.84

# Zeigegesten (AUF / PLATZ): "Rest eingerollt" als weicher Wert (auch bei verdeckten Fingern)
POINT_OTHERS_MAX = 0.74           # Mittel-/Ring-/Kleinfinger: Wert darunter = eingerollt
POINT_OTHERS_MIN_COUNT = 2        # mind. 2 von 3 müssen eingerollt sein

# Richtungs-Kegel (max. Winkel zur Senkrechten)
COS_UP = np.cos(np.radians(55))     # AUF / SITZ
COS_DOWN = np.cos(np.radians(70))   # PLATZ

# 2D-Richtung gilt als zuverlässig, wenn sie mind. so lang ist wie dieser Anteil der Handlänge
MIN_2D_LEN = 0.35

EMA_ALPHA = 0.5

# Entprellung (asymmetrisch)
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
# Hilfsfunktionen
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
    """Vertikale Komponente (+ = nach unten) der Richtung a->b.
    Primär aus dem sichtbaren 2D-Bild, Fallback auf 3D bei zu kurzem 2D-Vektor
    (Finger zeigt zur Kamera)."""
    ref_len = np.linalg.norm(
        np.array([(img[9][0] - img[0][0]) * w, (img[9][1] - img[0][1]) * h])
    )
    v2 = np.array([(img[b][0] - img[a][0]) * w, (img[b][1] - img[a][1]) * h])
    if ref_len > 1e-6 and np.linalg.norm(v2) > MIN_2D_LEN * ref_len:
        return unit(v2)[1], "2D"
    return unit(world[b] - world[a])[1], "3D"


def classify(world, img, states, w, h):
    values = {}
    for name, ids in FINGERS.items():
        values[name] = straightness(world, ids)
        states[name] = update_state(states[name], values[name])

    idx_v, idx_src = vertical(world, img, 5, 8, w, h)     # + = unten
    palm_v, _ = vertical(world, img, 0, 9, w, h)

    idx_up = (-idx_v) > COS_UP
    idx_down = idx_v > COS_DOWN
    palm_up = (-palm_v) > COS_UP

    # SITZ: ALLE vier Finger gestreckt (Ring/Klein mit weicherer Schwelle), Hand aufrecht
    sitz = (
        states["index"] == "ext"
        and states["middle"] == "ext"
        and values["ring"] > SITZ_SOFT_EXT
        and values["pinky"] > SITZ_SOFT_EXT
        and palm_up
    )

    # Zeigegeste: Zeigefinger gestreckt, Rest überwiegend eingerollt, keiner gestreckt
    others = ["middle", "ring", "pinky"]
    n_curled = sum(values[n] < POINT_OTHERS_MAX for n in others)
    n_ext = sum(states[n] == "ext" for n in others)
    pointing = states["index"] == "ext" and n_curled >= POINT_OTHERS_MIN_COUNT and n_ext == 0

    info = {
        "idx_v": idx_v, "src": idx_src, "idx_up": idx_up, "idx_down": idx_down,
        "pointing": pointing, "n_curled": n_curled, "sitz": sitz,
    }

    if sitz:
        return "SITZ", values, info
    if pointing and idx_up:
        return "AUF", values, info
    if pointing and idx_down:
        return "PLATZ", values, info
    return NONE, values, info


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
# Setup
# ---------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    print("[FEHLER] 'hand_landmarker.task' wurde nicht gefunden!")
    raise SystemExit(1)

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
print("[*] Gestenerkennung v3 gestartet. 'q' = Beenden, 'd' = Debug an/aus.")

history = deque(maxlen=HISTORY_LENGTH)
stable_gesture = NONE
debug = False
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

    result = detector.detect_for_video(mp_image, ts)

    current = NONE
    values = None
    info = None

    if result.hand_landmarks and result.hand_world_landmarks:
        world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]])
        img = np.array([[p.x, p.y] for p in result.hand_landmarks[0]])

        if sm_world is None:
            sm_world, sm_img = world, img
        else:
            sm_world = EMA_ALPHA * world + (1 - EMA_ALPHA) * sm_world
            sm_img = EMA_ALPHA * img + (1 - EMA_ALPHA) * sm_img

        current, values, info = classify(sm_world, sm_img, states, w, h)

        for p in result.hand_landmarks[0]:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (0, 0, 255), -1)
    else:
        sm_world = sm_img = None
        states = {name: "mid" for name in FINGERS}

    history.append(current)
    stable_gesture = stabilize(history, stable_gesture)

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

    if debug and values:
        y = 90
        lines = [f"Frame: {current}"]
        lines += [f"{n}: {v:.2f} ({states[n]})" for n, v in values.items()]
        lines += [
            f"Zeigefinger vertikal: {info['idx_v']:+.2f} [{info['src']}] (+ = unten)",
            f"up={info['idx_up']} down={info['idx_down']}",
            f"pointing={info['pointing']} (eingerollt: {info['n_curled']}/3)",
            f"sitz={info['sitz']}",
        ]
        for line in lines:
            cv2.putText(frame, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 1, cv2.LINE_AA)
            y += 25

    cv2.imshow("PREN Gestenerkennung v3", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("d"):
        debug = not debug

detector.close()
cap.release()
cv2.destroyAllWindows()