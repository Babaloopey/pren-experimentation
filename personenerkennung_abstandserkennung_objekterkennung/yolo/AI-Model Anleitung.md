Hier ist deine vollständige, praxiserprobte Schritt-für-Schritt-Anleitung für das nächste Mal. Sie führt dich vom rohen Bildmaterial über die Annotation und das YOLO-Training bis hin zum finalen Export als `.tflite`-Modell für die Zielhardware.

---

## 1. Datenannotation (MakeSense.ai)

1. **Bilder hochladen:** Lade deine ca. 160 Objektbilder (Knochen, Pylonen) sowie die ca. 20 reinen Hintergrundbilder in [MakeSense.ai](https://www.makesense.ai/) hoch.
2. **Klassen definieren (`labels.txt`):** Erstelle die Klassen in exakt dieser Reihenfolge:
```text
knochen
pylon

```


3. **Bounding Boxes zeichnen:** Annotiere alle Knochen und Pylonen.
4. **Wichtig für Hintergrundbilder:** Für die reinen Hintergrundbilder (Negative Samples) musst du ebenfalls eine Label-Datei exportieren, damit sie leer bleiben (keine Boxen enthalten). Das verhindert mAP-Berechnungsfehler beim Training.
5. **Export-Format:** Exportiere die Labels im **YOLO-Format** (Zip-Archiv mit `.txt`-Dateien pro Bild).

---

## 2. Dataset-Struktur in Google Colab

Organisiere deine Daten in Google Colab im folgenden Verzeichnisbaum:

```text
/content/dataset/
├── images/
│   ├── train/  (z.B. 80 Knochen, 80 Pylonen)
│   └── val/    (z.B. Hintergrundbilder oder Validierungs-Frames)
└── labels/
    ├── train/  (entsprechende .txt Dateien)
    └── val/    (leere .txt Dateien für Hintergrundbilder)

```

Erstelle im Hauptverzeichnis (`/content/`) eine Konfigurationsdatei namens **`data.yaml`**:

```yaml
path: /dataset  # Wurzelverzeichnis relativ zum Projekt
train: images/train
val: images/val

names:
  0: knochen
  1: pylon

```

---

## 3. Modelltraining in Google Colab (YOLOv8)

Dank Ultralytics YOLO umgehen wir veraltete Framework-Inkompatibilitäten und trainieren ein extrem stabiles, performantes Modell in wenigen Minuten.

Führe diesen Code in einer Colab-Zelle aus:

```python
from ultralytics import YOLO

# 1. YOLOv8 Nano Basis-Modell laden (lädt sich automatisch herunter)
model = YOLO("yolov8n.pt")

# 2. Training starten
results = model.train(
    data="data.yaml",       # Pfad zu deiner YAML-Konfig
    epochs=30,              # Anzahl der Epochen
    imgsz=640,              # Bildgröße
    batch=16,               # Batch-Größe (bei wenig RAM auf 8 reduzieren)
    device="cpu",           # Auf "cpu" belassen oder "0" für GPU (T4) setzen
    project="bone_pylon_project",
    name="yolo_train_run"
)

print("Training erfolgreich abgeschlossen!")

```

* **Dein trainiertes PyTorch-Modell liegt danach unter:**
`/content/bone_pylon_project/yolo_train_run/weights/best.pt`

---

## 4. Direkter Export als `.tflite` für die Zielhardware

Da der klassische MediaPipe Model Maker unter modernen Python-Umgebungen (wie Python 3.13) veraltete C++-Abhängigkeiten verlangt, ist der sauberste und stabilste Weg der **Direktexport aus YOLO**. Ultralytics konvertiert das Modell direkt in das für Edge-Geräte optimierte TensorFlow-Lite-Format.

Führe diesen Befehl aus, um das `.tflite`-Modell zu generieren:

```python
from ultralytics import YOLO

# 1. Dein trainiertes Best-Modell laden
model = YOLO("/content/bone_pylon_project/yolo_train_run/weights/best.pt")

# 2. Ins TensorFlow Lite Format (.tflite) exportieren
model.export(format="tflite")

print("Modell erfolgreich als .tflite exportiert!")

```

* **Das fertige `.tflite`-Modell findest du hier:**
`/content/bone_pylon_project/yolo_train_run/weights/best_saved_model/best_float32.tflite` (oder im selben Verzeichnis als `.tflite`-Datei).

Mit diesem Workflow hast du sowohl ein performantes YOLO-Modell (`.pt` / `.onnx`) als auch das leichtgewichtige `.tflite`-Modell parat, um beide Varianten direkt auf der Zielhardware gegeneinander zu testen und zu vergleichen!