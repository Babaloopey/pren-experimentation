import os

import numpy as np
import pyaudio
from openwakeword.model import Model

# Absoluten Pfad zum Modell ermitteln
current_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = "hey_watson_20261013_031428.onnx"
model_path = os.path.join(current_dir, model_filename)

print(f"Lade Modell von: {model_path}")

# Modell laden mit aktivierter VAD (Voice Activity Detection) gemäss Doku-Empfehlung
try:
  model = Model(
      wakeword_models=[model_path],
      inference_framework="onnx",
      vad_threshold=0.5,  # Filtert Hintergrundgeräusche & Rauschen heraus!
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

try:
  while True:
    data = stream.read(CHUNK, exception_on_overflow=False)
    audio_data = np.frombuffer(data, dtype=np.int16)

    # Vorhersage berechnen
    prediction = model.predict(audio_data)

    # Gezielter Blick auf den Score unseres Modells
    score = prediction.get(model_key, 0)

    # Schwellenwert (kann zwischen 0.3 und 0.6 justiert werden)
    if score > 0.4:
      print(
          f"[!] Treffer! '{model_key}' erkannt! Konfidenz-Score: {score:.2f}"
      )

except KeyboardInterrupt:
  print("\n[*] Skript wird beendet...")
finally:
  stream.stop_stream()
  stream.close()
  p.terminate()