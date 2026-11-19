#!/usr/bin/env python3
"""Knochen-/Pylon-Erkennung als ROS-2-Node – ohne Ultralytics/PyTorch.

Das YOLO-Modell läuft als ONNX über onnxruntime (siehe yolo_onnx_detector.py).
Vorhandene Modelle in models/: bone_recognition[_yolov8]_{320,640}.onnx
"""
import os
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from yolo_onnx_detector import YoloOnnxDetector

# --- Konfiguration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "bone_recognition_640.onnx")
CLASS_NAMES = {0: "Knochen", 1: "Pylon"}

CAM_WIDTH, CAM_HEIGHT = 1280, 720
CONF_THRESHOLD = 0.5   # Mindest-Konfidenz für Detektionen
IOU_THRESHOLD = 0.45   # Überlappung für Non-Maximum-Suppression
PROCESS_EVERY_N = 3    # nur jedes N-te Frame auswerten (CPU sparen)
NUM_THREADS = 2        # CPU-Threads für onnxruntime (Pi: 2–4)


class BoneRecognitionNode(Node):

    def __init__(self):
        super().__init__("bone_recognition_node")

        # Publisher für erkannte Knochen (z.B. Status-Text oder JSON/CSV mit Koordinaten)
        self.publisher_ = self.create_publisher(String, "/perception/bone_detected", 10)

        if not os.path.exists(MODEL_PATH):
            self.get_logger().error(f"Modell nicht gefunden unter: {MODEL_PATH}")
            raise SystemExit(1)

        self.detector = YoloOnnxDetector(MODEL_PATH, num_threads=NUM_THREADS)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.frame_count = 0

        # Timer für die Verarbeitungs-Loop (~30fps Kamera, Inferenz nur jedes N-te Frame)
        self.timer = self.create_timer(0.033, self.process_frame)
        self.get_logger().info(
            f"Bone Recognition Node gestartet (ONNX, {self.detector.input_w}x{self.detector.input_h})."
        )

    def process_frame(self):
        ok, frame = self.cap.read()  # immer lesen, damit der Kamera-Puffer aktuell bleibt
        if not ok:
            self.get_logger().warn("Fehler beim Zugriff auf die Webcam für Knochen-Erkennung.")
            return

        self.frame_count += 1
        if self.frame_count % PROCESS_EVERY_N != 0:
            return

        frame = cv2.flip(frame, 1)
        detections = self.detector.detect(frame, CONF_THRESHOLD, IOU_THRESHOLD)

        bone_found = False
        detection_summary = "KEIN KNOCHEN"
        for x1, y1, x2, y2, conf, cls_id in detections:
            if cls_id == 0:  # Knochen
                bone_found = True
                cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
                detection_summary = f"KNOCHEN_GEFUNDEN ({cls_name}, Conf: {conf:.2f})"
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
