import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Konfiguration für den modernen MediaPipe Object Detector
# Benötigt die Modelldatei 'efficientdet_lite0.tflite' im selben Ordner
base_options = python.BaseOptions(
    model_asset_path="efficientdet_lite0.tflite"
)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    max_results=5,  # Maximale Anzahl gleichzeitiger Objekte
    score_threshold=0.5,  # Konfidenz-Schwellenwert (50%)
)
detector = vision.ObjectDetector.create_from_options(options)

cap = cv2.VideoCapture(0)
print(
    "[*] MediaPipe Objekterkennung gestartet. Drücke 'q' zum Beenden."
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

  # Objekterkennung ausführen
  detection_result = detector.detect(mp_image)

  # Gefundene Objekte im Bild einzeichnen
  for detection in detection_result.detections:
    # Bounding Box auslesen
    bbox = detection.bounding_box
    start_point = (bbox.origin_x, bbox.origin_y)
    end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)

    # Rechteck um das Objekt zeichnen
    cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

    # Klassenname und Konfidenz-Score auslesen
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score * 100, 1)
    result_text = f"{category_name} ({probability}%)"

    # Text über die Bounding Box schreiben
    cv2.putText(
        frame,
        result_text,
        (bbox.origin_x, max(30, bbox.origin_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

  # Statusanzeige, falls keine Objekte erkannt wurden
  if not detection_result.detections:
    cv2.putText(
        frame,
        "Keine bekannten Objekte im Sichtfeld",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

  cv2.imshow("MediaPipe Object Detector", frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()