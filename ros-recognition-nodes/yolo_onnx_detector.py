"""YOLO-Detektor (YOLOv8/11, Ausgabe [1, 4+nc, N]) über onnxruntime – ohne Ultralytics/PyTorch.

Benötigt nur: pip install onnxruntime opencv-python numpy

Modell exportieren (auf dem PC, mit Ultralytics):
    from ultralytics import YOLO
    YOLO("bone_recognition_yolov8.pt").export(format="onnx", imgsz=320, simplify=True, opset=17)
"""
import cv2
import numpy as np
import onnxruntime as ort


class YoloOnnxDetector:

    def __init__(self, model_path, num_threads=2):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_h, self.input_w = inp.shape[2], inp.shape[3]

    def _letterbox(self, frame):
        """Seitenverhältnis erhalten und mit grauem Rand auf Modellgrösse auffüllen (wie im Training)."""
        h, w = frame.shape[:2]
        scale = min(self.input_w / w, self.input_h / h)
        new_w, new_h = round(w * scale), round(h * scale)
        pad_x, pad_y = (self.input_w - new_w) / 2, (self.input_h - new_h) / 2
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top, left = round(pad_y - 0.1), round(pad_x - 0.1)
        canvas = cv2.copyMakeBorder(
            resized, top, self.input_h - new_h - top, left, self.input_w - new_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        return canvas, scale, left, top

    def detect(self, frame, conf_thres=0.5, iou_thres=0.45):
        """Gibt Liste von (x1, y1, x2, y2, conf, cls_id) in Original-Bildkoordinaten zurück."""
        img, scale, pad_x, pad_y = self._letterbox(frame)
        blob = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR->RGB, HWC->NCHW
        blob = np.ascontiguousarray(blob)

        preds = self.session.run(None, {self.input_name: blob})[0][0].T  # (N, 4+nc)
        class_scores = preds[:, 4:]
        cls_ids = class_scores.argmax(axis=1)
        confs = class_scores[np.arange(len(cls_ids)), cls_ids]

        mask = confs >= conf_thres
        if not mask.any():
            return []
        preds, confs, cls_ids = preds[mask], confs[mask], cls_ids[mask]

        # cx, cy, w, h (Modellkoordinaten) -> x, y, w, h (Originalbild)
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x = (cx - bw / 2 - pad_x) / scale
        y = (cy - bh / 2 - pad_y) / scale
        boxes = np.stack([x, y, bw / scale, bh / scale], axis=1)

        # klassenweise NMS (wie Ultralytics): Boxen pro Klasse verschieben
        nms_boxes = boxes.copy()
        nms_boxes[:, :2] += cls_ids[:, None] * 10000.0
        keep = cv2.dnn.NMSBoxes(nms_boxes.tolist(), confs.tolist(), conf_thres, iou_thres)

        h, w = frame.shape[:2]
        detections = []
        for i in np.array(keep).flatten():
            bx, by, bw_, bh_ = boxes[i]
            x1, y1 = max(0.0, bx), max(0.0, by)
            x2, y2 = min(float(w), bx + bw_), min(float(h), by + bh_)
            detections.append((x1, y1, x2, y2, float(confs[i]), int(cls_ids[i])))
        detections.sort(key=lambda d: d[4], reverse=True)
        return detections
