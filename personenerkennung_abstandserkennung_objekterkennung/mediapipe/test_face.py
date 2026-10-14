import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Konfiguration für den modernen FaceDetector
base_options = python.BaseOptions(
    model_asset_path="blaze_face_short_range.tflite"
)
options = vision.FaceDetectorOptions(
    base_options=base_options, min_detection_confidence=0.5
)
detector = vision.FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0)
print(
    "[*] Moderner MediaPipe FaceDetector gestartet. Drücke 'q' zum Beenden."
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

  if detection_result.detections:
    for detection in detection_result.detections:
      bbox = detection.bounding_box
      start_point = (bbox.origin_x, bbox.origin_y)
      end_point = (
          bbox.origin_x + bbox.width,
          bbox.origin_y + bbox.height,
      )
      cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"Gesichter: {len(detection_result.detections)}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
  else:
    cv2.putText(
        frame,
        "Kein Gesicht gefunden",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

  cv2.imshow("MediaPipe Face Modern", frame)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()