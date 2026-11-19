"""Live-Vergleich der Knochen-/Pylon-Modelle: PyTorch (Ultralytics) vs. ONNX 320 vs. ONNX 640.

Aufruf (aus diesem Ordner):
    python test_bone_recognition.py                      # startet mit ONNX 320
    python test_bone_recognition.py --model bone_recognition --backend onnx640

Tasten im Fenster:
    1 = PyTorch (.pt, 640)   2 = ONNX 320   3 = ONNX 640
    m = Modell wechseln (bone_recognition_yolov8 <-> bone_recognition)
    q = Beenden

Angezeigt werden die Inferenzzeit und die CPU-Zeit pro Frame (Mittel der letzten 30 Frames).
CPU-Zeit > Inferenzzeit heisst: mehrere Kerne werden genutzt. Die CPU-Zeit ist der relevante Wert
für die Auslastung auf dem Raspberry Pi. Ultralytics wird nur geladen, wenn Backend 1 gewählt wird.
"""
import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PT_DIR = os.path.join(BASE_DIR, "models")
NODES_DIR = os.path.join(BASE_DIR, "..", "..", "ros-recognition-nodes")
ONNX_DIR = os.path.join(NODES_DIR, "models")
sys.path.insert(0, NODES_DIR)
from yolo_onnx_detector import YoloOnnxDetector  # noqa: E402

MODELS = ["bone_recognition_yolov8", "bone_recognition"]
BACKENDS = {ord("1"): "pt", ord("2"): "onnx320", ord("3"): "onnx640"}
CLASS_NAMES = {0: "Knochen", 1: "Pylon"}
COLORS = {0: (0, 200, 255), 1: (255, 120, 0)}
CAM_WIDTH, CAM_HEIGHT = 1280, 720


class Runner:
    """Lädt Modelle bei Bedarf und liefert einheitlich (x1, y1, x2, y2, conf, cls_id)."""

    def __init__(self, conf, threads):
        self.conf, self.threads, self.cache = conf, threads, {}

    def get(self, model, backend):
        key = (model, backend)
        if key not in self.cache:
            if backend == "pt":
                from ultralytics import YOLO
                yolo = YOLO(os.path.join(PT_DIR, f"{model}.pt"))

                def run(frame, yolo=yolo):
                    r = yolo(frame, conf=self.conf, verbose=False)[0]
                    return [(*b.xyxy[0].tolist(), float(b.conf), int(b.cls)) for b in r.boxes]
            else:
                size = backend.removeprefix("onnx")
                det = YoloOnnxDetector(os.path.join(ONNX_DIR, f"{model}_{size}.onnx"), self.threads)

                def run(frame, det=det):
                    return det.detect(frame, conf_thres=self.conf)
            run(np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), np.uint8))  # Aufwärmen
            self.cache[key] = run
        return self.cache[key]


def draw(frame, detections, lines):
    for x1, y1, x2, y2, conf, cls_id in detections:
        color = COLORS.get(cls_id, (0, 255, 0))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        cv2.putText(frame, f"{CLASS_NAMES.get(cls_id, cls_id)} {conf:.2f}", (p1[0], max(15, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for i, text in enumerate(lines):
        y = 28 + i * 26
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, default=MODELS[0])
    ap.add_argument("--backend", choices=list(BACKENDS.values()), default="onnx320")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=2, help="CPU-Threads für onnxruntime")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    runner = Runner(args.conf, args.threads)
    model, backend = args.model, args.backend
    infer_ms, cpu_ms = deque(maxlen=30), deque(maxlen=30)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    print("[*] Knochen-Erkennung gestartet. Tasten: 1/2/3 = Backend, m = Modell, q = Beenden.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        run = runner.get(model, backend)
        t_wall, t_cpu = time.perf_counter(), time.process_time()
        detections = run(frame)
        infer_ms.append((time.perf_counter() - t_wall) * 1000)
        cpu_ms.append((time.process_time() - t_cpu) * 1000)

        avg_inf = sum(infer_ms) / len(infer_ms)
        avg_cpu = sum(cpu_ms) / len(cpu_ms)
        draw(frame, detections, [
            f"{model} | {backend}",
            f"Inferenz: {avg_inf:5.1f} ms   CPU: {avg_cpu:5.1f} ms",
            "1=pt 2=onnx320 3=onnx640 m=Modell q=Ende",
        ])
        cv2.imshow("Knochen-Erkennung Vergleich", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in BACKENDS or key == ord("m"):
            if key == ord("m"):
                model = MODELS[(MODELS.index(model) + 1) % len(MODELS)]
            else:
                backend = BACKENDS[key]
            infer_ms.clear()
            cpu_ms.clear()
            print(f"[*] {model} | {backend}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
