import cv2
import numpy as np
from ultralytics import YOLO

# 1. YOLO26 Depth-Modell laden (lädt das vortrainierte Nano-Tiefenmodell automatisch herunter)
model = YOLO("yolo26n-depth.pt")

cap = cv2.VideoCapture(0)
print(
    "[*] YOLO26 Depth Estimation gestartet. Drücke 'q' zum Beenden."
)

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("Fehler beim Zugriff auf die Webcam.")
    break

  # Bild spiegeln für intuitive Interaktion
  frame = cv2.flip(frame, 1)
  h, w, _ = frame.shape

  # 2. Inferenz mit einer festen Trainingsauflösung (z.B. 640 oder 768) ausführen
  results = model(frame, imgsz=640, verbose=False)

  if results and hasattr(results[0], "depth") and results[0].depth is not None:
    # Die Tiefenkarte direkt als NumPy-Array abgreifen (Einheit: Meter pro Pixel)
    depth_map = results[0].depth.data.cpu().numpy()

    # Tiefenkarte für die Visualisierung normalisieren (0 bis 255)
    # Nahe Bereiche hell/farbig, ferne Bereiche dunkel (oder umgekehrt via colormap)
    depth_normalized = cv2.normalize(
        depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    depth_normalized = np.uint8(depth_normalized)

    # Farbpalette anwenden (z.B. JET oder TURBO für schöne Tiefen-Heatmaps)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)

    # Skaliere die Tiefenkarte auf die Original-Webcam-Grösse zurück, falls nötig
    depth_colormap = cv2.resize(depth_colormap, (w, h))

    # Beispiel: Extrahiere den metrischen Abstand in der Bildmitte (z.B. für Hindernis-Check)
    center_y, center_x = h // 2, w // 2
    center_distance = depth_map[
        int(center_y * depth_map.shape[0] / h),
        int(center_x * depth_map.shape[1] / w),
    ]

    # Text-Ausgabe ins Live-Bild einfügen
    cv2.putText(
        depth_colormap,
        f"Abstand Zentrum: {center_distance:.2f} m",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Originalbild und Tiefen-Heatmap nebeneinander oder kombiniert anzeigen
    combined_view = cv2.addWeighted(frame, 0.5, depth_colormap, 0.5, 0)
    cv2.imshow("YOLO26 Depth Estimation (Overlay)", combined_view)
  else:
    cv2.imshow("YOLO26 Depth Estimation (Overlay)", frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()