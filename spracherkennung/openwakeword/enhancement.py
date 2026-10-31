import os
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
import scipy.io.wavfile
import torch
from torch import nn
import openwakeword
import openwakeword.data
import openwakeword.utils

# 1. Feature Extractor initialisieren
F = openwakeword.utils.AudioFeatures()

# 2. Bestehendes Modell laden
existing_model_path = "Sitz_20261014_145414.onnx"
print(f"[*] Analysiere bestehendes Modell: {existing_model_path}")

if not os.path.exists(existing_model_path):
  raise FileNotFoundError(
      f"Modelldatei '{existing_model_path}' wurde nicht gefunden!"
  )

onnx_model = onnx.load(existing_model_path)
initializer_map = {
    init.name: numpy_helper.to_array(init)
    for init in onnx_model.graph.initializer
}

# Architektur-Parameter auslesen
first_linear_weight = None
for name, arr in initializer_map.items():
  if arr.ndim == 2 and arr.shape[1] > 1:
    first_linear_weight = arr
    break

if first_linear_weight is None:
  first_linear_weight = list(initializer_map.values())[0]

expected_input_features = first_linear_weight.shape[1]
layer_dim = first_linear_weight.shape[0]
n_timesteps = expected_input_features // 96

print(f"[*] Erwartete Eingabegrösse: {expected_input_features} Features")
print(f"[*] Erkannte Layer-Dimension (layer_dim): {layer_dim}")
print(f"[*] Entspricht exakt {n_timesteps} Timesteps.")

# 3. Deine 18 echten Aufnahmen laden
training_dir = Path("Training_Data")
positive_clips = [str(p) for p in training_dir.glob("*.wav")]

print(f"[*] {len(positive_clips)} echte Aufnahmen erfolgreich gefunden.")
if len(positive_clips) == 0:
  raise ValueError("Keine WAV-Dateien in 'Training_Data' gefunden!")

# Audio-Wunschlänge für die Rohdaten berechnen
total_length_seconds = n_timesteps * 0.08
sr = 16000
total_length = int(sr * total_length_seconds)

wav_data_list = []
for path in positive_clips:
  sr_file, dat = scipy.io.wavfile.read(path)
  padded = np.zeros(total_length, dtype=np.int16)
  start_idx = max(0, total_length - len(dat))
  padded[start_idx:] = dat[:total_length]
  wav_data_list.append(padded)

stacked_wavs = np.array(wav_data_list, dtype=np.int16)
positive_features = F.embed_clips(stacked_wavs, batch_size=16, ncpu=4)

# FIX: Garantiere, dass JEDER Clip exakt n_timesteps (16) hat (Padding/Trunkierung auf Feature-Ebene)
fixed_features = []
for feat in positive_features:
  if feat.shape[0] < n_timesteps:
    # Mit Nullen auffüllen, wenn zu kurz
    padded_feat = np.zeros((n_timesteps, feat.shape[1]), dtype=np.float32)
    padded_feat[-feat.shape[0] :] = feat
    fixed_features.append(padded_feat)
  elif feat.shape[0] > n_timesteps:
    # Abschneiden, wenn zu lang
    fixed_features.append(feat[:n_timesteps])
  else:
    fixed_features.append(feat)

positive_features = np.array(fixed_features, dtype=np.float32)

# 4. Trainingsdaten vorbereiten
X = positive_features
y = np.ones((len(positive_features), 1), dtype=np.float32)

training_data = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
    batch_size=8,
    shuffle=True,
)

# 5. PyTorch-Modell aufbauen
fcn = nn.Sequential(
    nn.Flatten(),
    nn.Linear(expected_input_features, layer_dim),
    nn.LayerNorm(layer_dim),
    nn.ReLU(),
    nn.Linear(layer_dim, layer_dim),
    nn.LayerNorm(layer_dim),
    nn.ReLU(),
    nn.Linear(layer_dim, 1),
    nn.Sigmoid(),
)

# Gewichte des alten Modells übertragen
with torch.no_grad():
  onnx_arrays = list(initializer_map.values())
  for name, param in fcn.named_parameters():
    param_shape = list(param.shape)
    for arr in onnx_arrays:
      if list(arr.shape) == param_shape:
        param.copy_(torch.from_numpy(arr.copy()))
        onnx_arrays.remove(arr)
        break

loss_function = torch.nn.functional.binary_cross_entropy
optimizer = torch.optim.Adam(
    fcn.parameters(), lr=0.00005
)  # Noch etwas sanfter fürs Fine-Tuning

# 6. Fine-Tuning ausführen
print("[*] Starte Fine-Tuning auf deine 18 echten Aufnahmen...")
n_epochs = 25

for epoch in range(n_epochs):
  epoch_loss = 0
  for batch in training_data:
    x_b, y_b = batch[0], batch[1]

    optimizer.zero_grad()
    predictions = fcn(x_b)
    loss = loss_function(predictions, y_b)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()

  if (epoch + 1) % 5 == 0:
    print(
        f"Epoche {epoch+1}/{n_epochs} abgeschlossen. Loss:"
        f" {epoch_loss / len(training_data):.4f}"
    )

# 7. Als neues ONNX-Modell exportieren
output_onnx_path = "Sitz_fine_tuned_2m.onnx"
torch.onnx.export(
    fcn, args=torch.zeros((1, n_timesteps, 96)), f=output_onnx_path
)
print(
    "[*] Fertig! Dein feingetuntes Modell wurde erfolgreich gespeichert unter:"
    f" {output_onnx_path}"
)