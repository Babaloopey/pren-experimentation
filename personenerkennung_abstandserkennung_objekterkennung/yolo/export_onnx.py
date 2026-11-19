"""Exportiert die Knochen-/Pylon-Modelle (.pt) nach ONNX für die Ausführung ohne Ultralytics/PyTorch.

Quelle: models/<name>.pt (dieser Ordner)
Ziel:   ros-recognition-nodes/models/<name>_<imgsz>.onnx

Aufruf (aus diesem Ordner, Ultralytics + onnx + onnxslim nötig – nur auf dem PC):
    python export_onnx.py                          # alle Modelle, 320 und 640
    python export_onnx.py --imgsz 640              # nur 640
    python export_onnx.py --models bone_recognition --imgsz 640

Export-Einstellungen:
    format="onnx"   ONNX-Modell, ausführbar mit onnxruntime (siehe ros-recognition-nodes/yolo_onnx_detector.py)
    imgsz           feste quadratische Eingabe [1, 3, imgsz, imgsz]; Bild wird per Letterbox angepasst
    simplify=True   Graph mit onnxslim vereinfachen (kleiner/schneller, gleiche Ausgabe)
    opset=17        ONNX-Opset, von aktuellen onnxruntime-Versionen (auch ARM/Raspberry Pi) unterstützt
    dynamic=False   feste Eingabegrösse (Standard) – schneller als dynamische Shapes
    nms=False       NMS nicht im Modell (Standard); wird in yolo_onnx_detector.py gemacht

Ausgabe des Modells: [1, 4 + Anzahl Klassen, N] = (cx, cy, w, h, Score Knochen, Score Pylon) pro Kandidat,
N = 2100 bei 320 px bzw. 8400 bei 640 px.
"""
import argparse
import os
import shutil

from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PT_DIR = os.path.join(BASE_DIR, "models")
ONNX_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "ros-recognition-nodes", "models"))

MODELS = ["bone_recognition_yolov8", "bone_recognition"]
IMG_SIZES = [320, 640]
OPSET = 17


def export(name, imgsz):
    pt_path = os.path.join(PT_DIR, f"{name}.pt")
    model = YOLO(pt_path)
    print(f"[*] {name}.pt  (Klassen: {model.names})  -> imgsz {imgsz}")

    # Ultralytics legt die Datei neben der .pt ab (<name>.onnx) – danach umbenennen/verschieben
    exported = model.export(format="onnx", imgsz=imgsz, simplify=True, opset=OPSET, dynamic=False)

    os.makedirs(ONNX_DIR, exist_ok=True)
    target = os.path.join(ONNX_DIR, f"{name}_{imgsz}.onnx")
    shutil.move(exported, target)
    print(f"[+] gespeichert: {target}")
    return target


def check(path):
    """Kurzer Test: Modell mit onnxruntime laden und Ein-/Ausgabeform ausgeben."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    result = sess.run(None, {inp.name: np.zeros(inp.shape, np.float32)})[0]
    print(f"    Eingabe {inp.name} {inp.shape} -> Ausgabe {out.name} {list(result.shape)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    ap.add_argument("--imgsz", nargs="+", type=int, default=IMG_SIZES)
    args = ap.parse_args()

    for name in args.models:
        for imgsz in args.imgsz:
            check(export(name, imgsz))


if __name__ == "__main__":
    main()
