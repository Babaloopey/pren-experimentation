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
    "[*] Bounding-Box Distanz-Test gestartet. Drücke 'q' zum Beenden."
)

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break

  frame = cv2.flip(frame, 1)
  h, w, _ = frame.shape

  mp_image = mp.Image(
      image_format=mp.ImageFormat.SRGB,
      data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
  )
  detection_result = detector.detect(mp_image)

  distance_text = "Keine Person erkannt"
  box_color = (0, 0, 255)

  if detection_result.pose_landmarks:
    for landmarks in detection_result.pose_landmarks:
      # Alle x- und y-Koordianten der Keypoints sammeln, um eine Bounding Box um die Person zu legen
      x_coords = [lm.x * w for lm in landmarks]
      y_coords = [lm.y * h for lm in landmarks]

      xmin, xmax = int(min(x_coords)), int(max(x_coords))
      ymin, ymax = int(min(y_coords)), int(max(y_coords))

      # Höhe und Breite der Box in Pixeln
      box_height = ymax - ymin
      box_width = xmax - xmin
      box_area = box_height * box_width

      # Empirischer Ansatz: Je grösser die Fläche (Box-Area), desto näher ist die Person.
      # Diese Werte müssen für die reale Kamera im Raum kalibriert werden.
      if box_area > 0:
        # Beispiel-Heuristik für den relativen "Nähe-Wert"
        # Je kleiner die Fläche, desto weiter weg.
        # Definiere z.B. einen Schwellenwert für den Stopp-Bereich (z.B. bei 1.5m)
        distance_text = f"Box-Fläche: {box_area} px (Höhe: {box_height}px)"
        box_color = (0, 255, 0)

        # Visuelle Bounding Box um die Person zeichnen
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)

      # Keypoints einzeichnen
      for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 2, (255, 0, 0), -1)

  cv2.putText(
      frame,
      distance_text,
      (30, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.7,
      box_color,
      2,
      cv2.LINE_AA,
  )
  cv2.imshow("Bounding Box Distance Test", frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()