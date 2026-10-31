import os
import threading
import time
import numpy as np
import pyaudio
from openwakeword.model import Model

# Absoluten Pfad zum Modell ermitteln
current_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = "Sitz_20261014_145414.onnx"
model_path = os.path.join(current_dir, model_filename)

print(f"Lade Modell von: {model_path}")

# Modell laden mit aktivierter VAD (Voice Activity Detection) gemäss Doku-Empfehlung
try:
  model = Model(
      wakeword_models=[model_path],
      inference_framework="onnx",
      vad_threshold=0.3,  # Filtert Hintergrundgeräusche & Rauschen heraus!
  )
except Exception as e:
  print(
      "[HINWEIS] Falls melspectrogram.onnx fehlt, ausführen:"
      " python -c 'import openwakeword; openwakeword.utils.download_models()'"
  )
  raise e

# Audio-Stream Konfiguration (80ms Chunks = 1280 Samples bei 16kHz)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
)

model_key = os.path.splitext(model_filename)[0]
print(
    f"[*] Höre zu mit VAD-Filter... Sage: '{model_key}' (STRG+C zum Beenden)"
)


# Funktion für Aktionen nach dem Erkennen (läuft im Hintergrund, blockiert Audio nicht)
def handle_wake_word_action(score):
  print(
      f"\n[!] AKTION AUSGEFÜHRT! '{model_key}' erkannt! Konfidenz-Score:"
      f" {score:.2f}"
  )
  # Hier kannst du z.B. Lichter schalten, Aufnahmen starten oder Befehle verarbeiten
  # time.sleep(1) # Simuliere eine kurze Aufgabe


try:
  last_trigger_time = 0
  cooldown_seconds = 1.5  # Sperrzeit gegen mehrfaches Auslösen
  consecutive_hits = (
      0  # Zählt aufeinanderfolgende Frames über dem Schwellenwert
  )
  required_consecutive = 2  # Muss in X Frames nacheinander da sein

  while True:
    data = stream.read(CHUNK, exception_on_overflow=False)
    audio_data = np.frombuffer(data, dtype=np.int16)

    # Vorhersage berechnen
    prediction = model.predict(audio_data)
    score = prediction.get(model_key, 0)

    current_time = time.time()

    # Schwellenwert-Prüfung (0.1)
    if score > 0.1:
      consecutive_hits += 1
    else:
      consecutive_hits = 0  # Zurücksetzen, wenn der Pegel abfällt

    # Prüfen, ob die Bedingung erfüllt ist UND der Cooldown abgelaufen ist
    if (
        consecutive_hits >= required_consecutive
        and (current_time - last_trigger_time) > cooldown_seconds
    ):
      last_trigger_time = current_time
      consecutive_hits = 0  # Reset nach erfolgreichem Auslösen

      # Aktion in einem eigenen Thread starten, damit der Stream flüssig weiterläuft
      threading.Thread(
          target=handle_wake_word_action, args=(score,), daemon=True
      ).start()

except KeyboardInterrupt:
  print("\n[*] Skript wird beendet...")
finally:
  stream.stop_stream()
  stream.close()
  p.terminate()