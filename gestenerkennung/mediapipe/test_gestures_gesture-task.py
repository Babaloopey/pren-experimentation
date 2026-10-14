import os
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Pfad zum Gesture Recognizer Modell (.task)
model_path = os.path.join(
    os.path.dirname(__file__), "gesture_recognizer.task"
)

if not os.path.exists(model_path):
  print(
      "[FEHLER] Die Datei 'gesture_recognizer.task' wurde nicht gefunden!"
  )
  print(
      "Bitte lade sie von der offiziellen Google MediaPipe Doku herunter und"
      " lege sie hier ab."
  )
  exit(1)

# 2. GestureRecognizer Options konfigurieren (gemäss offizieller Doku)
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)

recognizer = vision.GestureRecognizer.create_from_options(options)

# Webcam starten
cap = cv2.VideoCapture(0)
print(
    "[*] MediaPipe Gesture Recognizer gestartet. Drücke 'q' zum Beenden."
)

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("Fehler beim Zugriff auf die Webcam.")
    break

  # Bild spiegeln und in RGB konvertieren
  frame = cv2.flip(frame, 1)
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # MediaPipe Image Objekt erstellen
  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

  # Gestenerkennung ausführen
  recognition_result = recognizer.recognize(mp_image)

  detected_command = "Keine Geste"

  # Auswertung der erkannten Gesten und Labels
  if recognition_result.gestures:
    # Die oberste erkannte Geste abrufen
    top_gesture = recognition_result.gestures[0][0]
    gesture_name = top_gesture.category_name
    score = top_gesture.score

    # Abgleich mit den Roboter-Befehlen aus der Aufgabenstellung
    if gesture_name == "Pointing_Up" and score > 0.6:
      detected_command = f"AUF (Erkannt: {gesture_name}, {score:.2f})"
    elif gesture_name == "Open_Palm" and score > 0.6:
      detected_command = f"SITZ (Erkannt: {gesture_name}, {score:.2f})"
    else:
      detected_command = f"{gesture_name} ({score:.2f})"

  # Text-Ausgabe im Live-Bild
  cv2.putText(
      frame,
      f"Befehl: {detected_command}",
      (30, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (0, 255, 0),
      2,
      cv2.LINE_AA,
  )

  cv2.imshow("PREN Gestenerkennung - Gesture Recognizer", frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()