import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

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
    "[*] Moderner MediaPipe Distanz-Test gestartet. Drücke 'q' zum Beenden."
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

  distance_text = "Keine Person"

  if detection_result.pose_landmarks:
    for landmarks in detection_result.pose_landmarks:
      # Linke Schulter (Index 11) und rechte Schulter (Index 12)
      left_shoulder = landmarks[11]
      right_shoulder = landmarks[12]

      # Berechne den Pixelabstand der Schultern
      x1, y1 = int(left_shoulder.x * w), int(left_shoulder.y * h)
      x2, y2 = int(right_shoulder.x * w), int(right_shoulder.y * h)
      pixel_width = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

      # Grobe empirische Formel zur Distanz-Abschätzung in Zentimetern
      if pixel_width > 0:
        estimated_distance = 60.0 / (pixel_width / 100.0)
        distance_text = (
            f"Distanz ca.: {estimated_distance:.2f} cm (Px:"
            f" {int(pixel_width)})"
        )

      # Keypoints ins Bild einzeichnen
      for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

  cv2.putText(
      frame,
      distance_text,
      (30, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.7,
      (255, 0, 0),
      2,
      cv2.LINE_AA,
  )
  cv2.imshow("MediaPipe Distance Modern", frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()