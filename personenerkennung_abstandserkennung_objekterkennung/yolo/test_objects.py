import cv2
from ultralytics import YOLO

# YOLO26 Standard-Objektdetektor laden
model = YOLO("yolo26n.pt")

cap = cv2.VideoCapture(0)
print(
    "[*] YOLO26 Objekterkennung gestartet. Drücke 'q' zum Beenden."
)

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break

  frame = cv2.flip(frame, 1)

  # Inferenz ausführen
  results = model(frame, verbose=False)

  # Integrierte Visualisierung von YOLO26
  annotated_frame = results[0].plot()

  cv2.imshow("YOLO26 Object Detector", annotated_frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()