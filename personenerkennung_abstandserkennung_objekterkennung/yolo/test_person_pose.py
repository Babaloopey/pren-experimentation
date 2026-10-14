import cv2
from ultralytics import YOLO

# YOLO26 Pose-Modell laden (Nano-Variante für maximale Performance)
model = YOLO("yolo26n-pose.pt")

cap = cv2.VideoCapture(0)
print(
    "[*] YOLO26 Pose & Hand-vor-der-Brust Test gestartet. Drücke 'q' zum"
    " Beenden."
)

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break

  frame = cv2.flip(frame, 1)
  h, w, _ = frame.shape

  # Inferenz ausführen
  results = model(frame, verbose=False)

  status_text = "Keine Person im Sichtfeld"
  status_color = (0, 0, 255)

  if results and results[0].keypoints is not None:
    kpts = results[0].keypoints.xy  # (x, y) Koordinaten der Keypoints

    if len(kpts) > 0:
      status_text = "Person erkannt"
      status_color = (0, 255, 0)

      # COCO-Keypoint Indizes bei YOLO Pose:
      # 5: Linke Schulter, 6: Rechte Schulter
      # 11: Linke Hüfte, 12: Rechte Hüfte
      # 9: Linkes Handgelenk, 10: Rechtes Handgelenk
      person_kpts = kpts[0]

      if len(person_kpts) >= 12:
        l_shoulder = person_kpts[5]
        r_shoulder = person_kpts[6]
        l_hip = person_kpts[11]
        r_hip = person_kpts[12]
        wrists = [person_kpts[9], person_kpts[10]]

        if l_shoulder[0] > 0 and r_shoulder[0] > 0 and l_hip[0] > 0:
          min_x = min(l_shoulder[0], r_shoulder[0], l_hip[0], r_hip[0])
          max_x = max(l_shoulder[0], r_shoulder[0], l_hip[0], r_hip[0])
          min_y = min(l_shoulder[1], r_shoulder[1])
          max_y = max(l_hip[1], r_hip[1])

          hand_in_chest = False
          for wrist in wrists:
            wx, wy = wrist[0], wrist[1]
            if (min_x <= wx <= max_x) and (min_y <= wy <= max_y):
              hand_in_chest = True
              break

          if hand_in_chest:
            status_text = "Hand ist vor der Brust (YOLO26)"
            status_color = (255, 165, 0)

  # Skelett und Boxen automatisch von YOLO26 zeichnen lassen
  annotated_frame = results[0].plot()

  cv2.putText(
      annotated_frame,
      status_text,
      (30, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      status_color,
      2,
      cv2.LINE_AA,
  )
  cv2.imshow("YOLO26 Pose Test", annotated_frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()