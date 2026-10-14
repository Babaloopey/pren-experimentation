import cv2
from ultralytics import YOLO

# 1. YOLOv11 Pose-Modell laden (z.B. die Nano-Variante für maximale Geschwindigkeit)
model = YOLO("yolo11n-pose.pt")

# Webcam starten
cap = cv2.VideoCapture(0)
print("[*] YOLO11 Hand-Pose-Erkennung gestartet. Drücke 'q' zum Beenden.")

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break

  frame = cv2.flip(frame, 1)

  # 2. Inferenz auf das Live-Bild durchführen
  results = model(frame, verbose=False)

  # 3. Ergebnisse visualisieren (YOLO zeichnet Skelett und Keypoints automatisch ein)
  annotated_frame = results[0].plot()

  cv2.imshow("YOLO11 Pose Estimation", annotated_frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()