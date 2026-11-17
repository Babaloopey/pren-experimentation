#!/usr/bin/env python3
import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO

# --- Konfiguration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "bone_recognition_yolov8.pt")

CAM_WIDTH, CAM_HEIGHT = 1280, 720
CONF_THRESHOLD = 0.5  # Mindest-Konfidenz für Detektionen


class BoneRecognitionNode(Node):

    def __init__(self):
        super().__init__("bone_recognition_node")

        # Publisher für erkannte Knochen (z.B. Status-Text oder JSON/CSV mit Koordinaten)
        self.publisher_ = self.create_publisher(String, "/perception/bone_detected", 10)

        if not os.path.exists(MODEL_PATH):
            self.get_logger().error(f"Modell nicht gefunden unter: {MODEL_PATH}")
            raise SystemExit(1)

        self.model = YOLO(MODEL_PATH)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

        # Timer für die Verarbeitungs-Loop (~30fps)
        self.timer = self.create_timer(0.033, self.process_frame)
        self.get_logger().info("Bone Recognition Node gestartet.")

    def process_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("Fehler beim Zugriff auf die Webcam für Knochen-Erkennung.")
            return

        frame = cv2.flip(frame, 1)

        # YOLO Inferenz ausführen
        results = self.model(frame, verbose=False)

        bone_found = False
        detection_summary = "KEIN KNOCHEN"

        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]

                if conf >= CONF_THRESHOLD:
                    bone_found = True
                    # Optional: Bounding Box Koordinaten (xyxy) extrahieren
                    xyxy = box.xyxy[0].cpu().numpy()
                    detection_summary = f"KNOCHEN_GEFUNDEN ({cls_name}, Conf: {conf:.2f})"
                    break
            if bone_found:
                break

        # Status auf ROS 2 Topic veröffentlichen
        msg = String()
        msg.data = detection_summary
        self.publisher_.publish(msg)

        if bone_found:
            self.get_logger().info(f"Erkannt: {detection_summary}", throttle_duration_sec=1.0)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BoneRecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()