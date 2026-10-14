import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Konfiguration für den modernen PoseLandmarker
base_options = python.BaseOptions(
    model_asset_path="pose_landmarker_lite.task"
)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
print(
    "[*] Moderner MediaPipe PoseLandmarker gestartet. Drücke 'q' zum Beenden."
)

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break

  frame = cv2.flip(frame, 1)
  h, w, _ = frame.shape

  # OpenCV BGR zu MediaPipe Image konvertieren
  mp_image = mp.Image(
      image_format=mp.ImageFormat.SRGB,
      data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
  )

  # Inferenz ausführen
  detection_result = detector.detect(mp_image)

  status_text = "Keine Person im Sichtfeld"
  status_color = (0, 0, 255)

  if detection_result.pose_landmarks:
    status_text = "Person erkannt"
    status_color = (0, 255, 0)

    for landmarks in detection_result.pose_landmarks:
      # Relevante Indizes für den Oberkörper:
      # 11: Linke Schulter, 12: Rechte Schulter
      # 23: Linke Hüfte, 24: Rechte Hüfte
      # 15: Linkes Handgelenk, 16: Rechtes Handgelenk
      l_shoulder = landmarks[11]
      r_shoulder = landmarks[12]
      l_hip = landmarks[23]
      r_hip = landmarks[24]

      # Beide Hände prüfen
      wrists = [landmarks[15], landmarks[16]]

      # Horizontale und vertikale Grenzen des Torso-Bereichs (Brust/Bauch) bestimmen
      min_x = min(l_shoulder.x, r_shoulder.x, l_hip.x, r_hip.x) * w
      max_x = max(l_shoulder.x, r_shoulder.x, l_hip.x, r_hip.x) * w
      min_y = min(l_shoulder.y, r_shoulder.y) * h  # Oben (Schultern)
      max_y = max(l_hip.y, r_hip.y) * h  # Unten (Hüften)

      hand_in_chest_area = False
      for wrist in wrists:
        wx, wy = wrist.x * w, wrist.y * h
        # Prüfen, ob das Handgelenk innerhalb der Torso-Box liegt und gut sichtbar ist
        if (
            (min_x <= wx <= max_x)
            and (min_y <= wy <= max_y)
            and (wrist.visibility > 0.5)
        ):
          hand_in_chest_area = True
          break

      if hand_in_chest_area:
        status_text = "Hand ist vor der Brust!"
        status_color = (255, 165, 0)  # Orange

      # Keypoints ins Bild einzeichnen
      for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

  # Status-Text im Live-Bild anzeigen
  cv2.putText(
      frame,
      status_text,
      (30, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      status_color,
      2,
      cv2.LINE_AA,
  )

  cv2.imshow("MediaPipe Pose - Hand vor der Brust", frame)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()