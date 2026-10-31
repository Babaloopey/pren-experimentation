#!/usr/bin/env python3
"""
Fine-Tuning eines bestehenden openWakeWord-Modells (.onnx) mit eigenen Aufnahmen.

Was das Skript macht:
  1. Liest die Architektur + Gewichte des bestehenden ONNX-Modells aus und baut
     daraus ein PyTorch-Modell (mit Verifikation gegen onnxruntime).
  2. Bereitet deine Aufnahmen auf: Stille trimmen, am Fensterende ausrichten
     (so wie openWakeWord trainiert), augmentieren (Tempo, Lautstärke,
     Hintergrundgeräusche mit zufälligem SNR).
  3. Erzeugt Negativbeispiele (Sprache/Rauschen/Stille). OHNE Negative würde das
     Modell lernen "immer ja" zu sagen -> deshalb zwingend nötig.
  4. Trainiert mit kleiner Lernrate, Anker-Regularisierung zu den alten Gewichten
     und Early Stopping auf zurückgehaltenen Aufnahmen.
  5. Exportiert nach ONNX (ohne onnxscript, via klassischem Exporter) und prüft
     das Ergebnis gegen PyTorch.

Beispiel:
  python enhance_wakeword.py --model Sitz_20261014_145414.onnx \
      --positives Training_Data --negatives Negative_Data \
      --output Sitz_fine_tuned.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import scipy.io.wavfile
import scipy.signal
import torch
from onnx import numpy_helper
from torch import nn
import openwakeword.utils

SR = 16000
HOP = 1280  # 80 ms pro Embedding-Frame
FEAT_DIM = 96


# --------------------------------------------------------------------------- #
# Audio-Helfer
# --------------------------------------------------------------------------- #
def load_wav(path):
    """Lädt eine WAV als mono float32 in [-1, 1] bei 16 kHz."""
    sr, dat = scipy.io.wavfile.read(str(path))
    if dat.ndim > 1:
        dat = dat.mean(axis=1)
    if np.issubdtype(dat.dtype, np.integer):
        dat = dat.astype(np.float32) / float(np.iinfo(dat.dtype).max)
    else:
        dat = dat.astype(np.float32)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        dat = scipy.signal.resample_poly(dat, SR // g, int(sr) // g).astype(np.float32)
    return dat


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def trim_silence(x, thresh_db=-35.0, margin_s=0.05):
    """Schneidet Stille vorne/hinten weg, damit das Wort sauber am Ende sitzt."""
    win = max(1, int(0.01 * SR))
    env = np.convolve(np.abs(x), np.ones(win) / win, mode="same")
    peak = env.max()
    if peak <= 0:
        return x
    idx = np.where(env > peak * 10 ** (thresh_db / 20))[0]
    if len(idx) == 0:
        return x
    m = int(margin_s * SR)
    return x[max(0, idx[0] - m): idx[-1] + m]


def random_crop(x, n, rng):
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / len(x))) + 1)
    s = int(rng.integers(0, len(x) - n + 1))
    return x[s: s + n]


def find_window_samples(F, n_timesteps):
    """Findet die Fensterlänge (Samples), die exakt n_timesteps Embedding-Frames ergibt."""
    n = (n_timesteps + 9) * HOP
    for _ in range(10):
        emb = F.embed_clips(np.zeros((1, n), dtype=np.int16), batch_size=1, ncpu=1)
        t = emb.shape[1]
        if t == n_timesteps:
            return n
        n += (n_timesteps - t) * HOP
    raise RuntimeError("Konnte keine passende Fensterlänge bestimmen.")


def embed(F, windows, n_timesteps, chunk=128):
    """float-Fenster -> openWakeWord-Features (N, n_timesteps, 96)."""
    out = []
    for i in range(0, len(windows), chunk):
        arr = np.stack(windows[i: i + chunk])
        arr = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
        feats = F.embed_clips(arr, batch_size=64, ncpu=1).astype(np.float32)
        if feats.shape[1] != n_timesteps:
            raise RuntimeError(f"Unerwartete Frame-Anzahl: {feats.shape[1]} statt {n_timesteps}")
        out.append(feats)
    return np.concatenate(out, axis=0) if out else np.zeros((0, n_timesteps, FEAT_DIM), np.float32)


# --------------------------------------------------------------------------- #
# Positive: Augmentierung
# --------------------------------------------------------------------------- #
def make_positive_window(clip, window, bg_pool, rng, snr_low, snr_high, clean=False):
    x = clip.copy()
    if not clean:  # leichte Tempo-/Tonhöhenvariation
        factor = rng.uniform(0.9, 1.1)
        x = scipy.signal.resample(x, max(1, int(len(x) / factor))).astype(np.float32)
    if len(x) > window:
        x = x[-window:]

    # Wort endet 0-200 ms vor Fensterende (wie im openWakeWord-Training)
    jitter = 0 if clean else int(rng.uniform(0, 0.2) * SR)
    end = window - jitter
    start = end - len(x)
    if start < 0:
        x = x[-start:]
        start = 0
    fg = np.zeros(window, np.float32)
    fg[start: start + len(x)] = x

    snr = 30.0 if clean else float(rng.uniform(snr_low, snr_high))
    if bg_pool and (clean or rng.random() < 0.9):
        bg = random_crop(bg_pool[int(rng.integers(len(bg_pool)))], window, rng)
        bg = bg * (rms(x) / (rms(bg) * 10 ** (snr / 20)))
    else:
        bg = rng.normal(0, 1, window).astype(np.float32)
        bg = bg * (rms(x) * 10 ** (-snr / 20))

    mix = fg + bg
    mix = mix / (np.abs(mix).max() + 1e-8)
    gain = 0.9 if clean else float(rng.uniform(0.1, 0.9))
    return (mix * gain).astype(np.float32)


# --------------------------------------------------------------------------- #
# Negative
# --------------------------------------------------------------------------- #
def windows_from_audio(x, window, max_per_file):
    if len(x) == 0:
        return []
    if len(x) < window:
        x = np.tile(x, int(np.ceil(window / len(x))) + 1)
    starts = list(range(0, len(x) - window + 1, window // 2))
    if len(starts) > max_per_file:
        starts = list(np.linspace(0, len(starts) - 1, max_per_file).astype(int))
        starts = [list(range(0, len(x) - window + 1, window // 2))[i] for i in starts]
    return [x[s: s + window] for s in starts]


def build_negatives(F, files, window, n_timesteps, rng, max_total, max_per_file=300):
    """Gibt (train_feats, val_feats, bg_pool) zurück."""
    files = list(files)
    rng.shuffle(files)
    train_f, val_f, bg_pool = [], [], []
    total = 0
    for f in files:
        if total >= max_total:
            break
        try:
            x = load_wav(f)
        except Exception as e:  # defekte Datei überspringen
            print(f"    [!] Überspringe {f}: {e}")
            continue
        if len(bg_pool) < 50:
            bg_pool.append(x)
        wins = windows_from_audio(x, window, max_per_file)
        n = len(wins)
        if n >= 5:  # zusammenhängender Block am Ende = Validierung (kein Overlap-Leak)
            nv = max(1, int(0.2 * n))
            tr, va = wins[: n - nv - 1], wins[n - nv:]
        else:
            tr, va = wins, []
        gains = lambda k: rng.uniform(0.5, 1.5, size=k)
        tr = [w * g for w, g in zip(tr, gains(len(tr)))]
        va = [w * g for w, g in zip(va, gains(len(va)))]
        if tr:
            train_f.append(embed(F, tr, n_timesteps))
        if va:
            val_f.append(embed(F, va, n_timesteps))
        total += n

    # Synthetische Negative: Stille / leises Rauschen
    noise = [(rng.normal(0, 1, window) * rng.uniform(0.0005, 0.05)).astype(np.float32) for _ in range(120)]
    train_f.append(embed(F, noise[:100], n_timesteps))
    val_f.append(embed(F, noise[100:], n_timesteps))

    cat = lambda lst: np.concatenate(lst, axis=0) if lst else np.zeros((0, n_timesteps, FEAT_DIM), np.float32)
    return cat(train_f), cat(val_f), bg_pool


# --------------------------------------------------------------------------- #
# ONNX -> PyTorch
# --------------------------------------------------------------------------- #
def onnx_to_torch(onnx_model):
    """
    Liest Linear-/LayerNorm-Schichten in Graph-Reihenfolge aus dem ONNX-Modell
    (statt Gewichte über die Shape zuzuordnen - das ist mehrdeutig, weil
    Linear-Bias und LayerNorm-Gewichte gleich geformt sind).
    """
    g = onnx_model.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    inits[n.output[0]] = numpy_helper.to_array(a.t)

    nodes = list(g.node)
    ops = {n.op_type for n in nodes}
    layers = []
    for i, node in enumerate(nodes):
        if node.op_type == "Gemm":
            attrs = {a.name: a for a in node.attribute}
            W = inits[node.input[1]]
            if not (("transB" in attrs) and attrs["transB"].i == 1):
                W = W.T
            b = inits[node.input[2]] if len(node.input) > 2 else np.zeros(W.shape[0], np.float32)
            layers.append(["linear", W.copy(), b.copy()])
        elif node.op_type == "MatMul":
            w_names = [n for n in node.input if n in inits and inits[n].ndim == 2]
            if not w_names:
                continue
            W = inits[w_names[0]].T  # (in, out) -> (out, in)
            b = np.zeros(W.shape[0], np.float32)
            for nxt in nodes[i + 1: i + 3]:
                if nxt.op_type == "Add" and node.output[0] in nxt.input:
                    other = [n for n in nxt.input if n != node.output[0]][0]
                    if other in inits:
                        b = inits[other].copy()
                    break
            layers.append(["linear", W.copy(), b])
        elif node.op_type == "LayerNormalization":
            attrs = {a.name: a for a in node.attribute}
            eps = attrs["epsilon"].f if "epsilon" in attrs else 1e-5
            scale = inits[node.input[1]]
            bias = inits[node.input[2]] if len(node.input) > 2 else np.zeros_like(scale)
            layers.append(["ln", scale.copy(), bias.copy(), float(eps)])
        elif node.op_type == "Sqrt":  # zerlegtes LayerNorm (ältere Opsets)
            eps = 1e-5
            if i > 0 and nodes[i - 1].op_type == "Add":
                for n in nodes[i - 1].input:
                    if n in inits and inits[n].size == 1:
                        eps = float(inits[n].reshape(-1)[0])
            scale = bias = None
            for nxt in nodes[i + 1: i + 5]:
                vec = [n for n in nxt.input if n in inits and inits[n].ndim == 1]
                if nxt.op_type == "Mul" and vec:
                    scale = inits[vec[0]].copy()
                if nxt.op_type == "Add" and vec:
                    bias = inits[vec[0]].copy()
            if scale is not None and bias is not None:
                layers.append(["ln", scale, bias, eps])

    n_linear = sum(1 for l in layers if l[0] == "linear")
    if n_linear < 2:
        raise RuntimeError("Unbekannte Modellarchitektur (weniger als 2 Linear-Schichten gefunden).")
    if "Relu" not in ops:
        raise RuntimeError("Modell nutzt kein ReLU - Architektur wird nicht unterstützt.")

    mods = [nn.Flatten()]
    seen = 0
    for k, l in enumerate(layers):
        if l[0] == "linear":
            seen += 1
            lin = nn.Linear(l[1].shape[1], l[1].shape[0])
            with torch.no_grad():
                lin.weight.copy_(torch.from_numpy(l[1]))
                lin.bias.copy_(torch.from_numpy(l[2]))
            mods.append(lin)
            nxt_is_ln = k + 1 < len(layers) and layers[k + 1][0] == "ln"
            if seen < n_linear and not nxt_is_ln:
                mods.append(nn.ReLU())
        else:
            ln = nn.LayerNorm(l[1].shape[0], eps=l[3])
            with torch.no_grad():
                ln.weight.copy_(torch.from_numpy(l[1]))
                ln.bias.copy_(torch.from_numpy(l[2]))
            mods.append(ln)
            mods.append(nn.ReLU())
    if "Sigmoid" in ops:
        mods.append(nn.Sigmoid())
    return nn.Sequential(*mods)


def verify_against_onnx(model, onnx_path, n_timesteps, tag):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    model.eval()
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(5):
        x = rng.normal(0, 1, (1, n_timesteps, FEAT_DIM)).astype(np.float32)
        ref = sess.run(None, {name: x})[0].reshape(-1)
        with torch.no_grad():
            out = model(torch.from_numpy(x)).numpy().reshape(-1)
        worst = max(worst, float(np.abs(ref - out).max()))
    print(f"    {tag}: max. Abweichung ONNX vs. PyTorch = {worst:.2e}")
    if worst > 1e-3:
        raise RuntimeError(f"{tag}: PyTorch-Nachbau stimmt nicht mit dem ONNX-Modell überein!")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).squeeze(1).numpy()


def metrics(model, Xp, Xn, thr=0.5):
    out = {}
    if len(Xp):
        out["recall"] = float((predict(model, Xp) >= thr).mean())
    if len(Xn):
        out["fpr"] = float((predict(model, Xn) >= thr).mean())
    return out


def fmt(m):
    return ", ".join(f"{k}={v:.3f}" for k, v in m.items()) or "-"


def weighted_loss(pred, y, neg_weight):
    w = torch.where(y > 0.5, torch.ones_like(y), torch.full_like(y, neg_weight))
    return torch.nn.functional.binary_cross_entropy(pred, y, weight=w)


def export_onnx(model, n_timesteps, path, in_name, out_name):
    model.eval()
    dummy = torch.zeros(1, n_timesteps, FEAT_DIM)
    # Rein numerische Namen (z.B. '39') lehnt der Exporter ab -> gültige Namen verwenden
    if not in_name.isidentifier():
        in_name = "input"
    if not out_name.isidentifier():
        out_name = "output"
    kwargs = dict(
        input_names=[in_name],
        output_names=[out_name],
        opset_version=17,
        dynamic_axes={in_name: {0: "batch"}, out_name: {0: "batch"}},
    )
    try:
        # dynamo=False -> klassischer Exporter, braucht KEIN onnxscript
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # ältere Torch-Version kennt 'dynamo' nicht
        torch.onnx.export(model, dummy, str(path), **kwargs)
    except ModuleNotFoundError as e:
        sys.exit(f"[!] {e}\n    Lösung: pip install onnxscript")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Sitz_20261014_145414.onnx", help="bestehendes openWakeWord-Modell")
    ap.add_argument("--positives", default="Training_Data", help="Ordner mit deinen Aufnahmen des Wakewords")
    ap.add_argument(
        "--negatives", nargs="+",
        default=["openwakeword/notebooks/training_tutorial_data/negative",
                 "notebooks/training_tutorial_data/negative", "Negative_Data"],
        help="Ordner mit Negativ-Audio (anderes Sprechen, Geräusche, Musik ...)",
    )
    ap.add_argument("--output", default="Sitz_fine_tuned.onnx")
    ap.add_argument("--aug-per-clip", type=int, default=60, help="Augmentierungen pro Aufnahme")
    ap.add_argument("--max-neg-windows", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10, help="Early-Stopping-Geduld (Epochen)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--anchor", type=float, default=0.1, help="Stärke der Bindung an die alten Gewichte (0 = aus)")
    ap.add_argument("--neg-weight", type=float, default=2.0, help="Gewicht der Negativbeispiele im Loss")
    ap.add_argument("--val-fraction", type=float, default=0.2, help="Anteil der Aufnahmen für Validierung")
    ap.add_argument("--snr-low", type=float, default=5.0)
    ap.add_argument("--snr-high", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # 1. Bestehendes Modell laden --------------------------------------------
    print(f"[1/6] Lade bestehendes Modell: {args.model}")
    if not Path(args.model).exists():
        sys.exit(f"[!] Modelldatei '{args.model}' nicht gefunden.")
    onnx_model = onnx.load(args.model)
    init_names = {i.name for i in onnx_model.graph.initializer}
    in_name = next(i.name for i in onnx_model.graph.input if i.name not in init_names)
    out_name = onnx_model.graph.output[0].name

    model = onnx_to_torch(onnx_model)
    first_lin = next(m for m in model if isinstance(m, nn.Linear))
    n_timesteps = first_lin.in_features // FEAT_DIM
    print(f"    Timesteps={n_timesteps}, Layer-Dim={first_lin.out_features}")
    verify_against_onnx(model, args.model, n_timesteps, "Original")
    original_params = [p.detach().clone() for p in model.parameters()]

    try:
        F = openwakeword.utils.AudioFeatures()
    except Exception:
        openwakeword.utils.download_models()
        F = openwakeword.utils.AudioFeatures()
    window = find_window_samples(F, n_timesteps)
    print(f"    Fensterlänge: {window} Samples ({window / SR:.2f} s)")

    # 2. Aufnahmen laden ------------------------------------------------------
    print(f"[2/6] Lade Aufnahmen aus '{args.positives}'")
    pos_files = sorted(Path(args.positives).glob("*.wav"))
    if not pos_files:
        sys.exit(f"[!] Keine WAV-Dateien in '{args.positives}'.")
    clips = [trim_silence(load_wav(p)) for p in pos_files]
    print(f"    {len(clips)} Aufnahmen, Länge {min(map(len, clips)) / SR:.2f}-{max(map(len, clips)) / SR:.2f} s (nach Trimmen)")
    if max(map(len, clips)) > window:
        print("    [!] Mind. eine Aufnahme ist länger als das Modellfenster - vorne wird abgeschnitten.")

    order = rng.permutation(len(clips))
    n_val = max(1, int(round(args.val_fraction * len(clips)))) if len(clips) >= 5 else 0
    val_clips = [clips[i] for i in order[:n_val]]
    train_clips = [clips[i] for i in order[n_val:]]
    print(f"    Training: {len(train_clips)} Aufnahmen, Validierung: {len(val_clips)} Aufnahmen")

    # 3. Negative -------------------------------------------------------------
    print("[3/6] Erzeuge Negativbeispiele")
    neg_files = []
    for d in args.negatives:
        if Path(d).is_dir():
            neg_files += sorted(Path(d).rglob("*.wav"))
    if not neg_files:
        sys.exit("[!] Keine Negativ-WAVs gefunden. Gib --negatives <Ordner> an. "
                 "Ohne Negative würde das Modell auf 'immer ja' trainiert.")
    Xn_train, Xn_val, bg_pool = build_negatives(F, neg_files, window, n_timesteps, rng, args.max_neg_windows)
    print(f"    {len(neg_files)} Negativ-Dateien -> {len(Xn_train)} Train- / {len(Xn_val)} Val-Fenster")

    # 4. Positive -------------------------------------------------------------
    print("[4/6] Augmentiere und berechne Features der Aufnahmen")

    def positives(cl, n_aug):
        wins = [make_positive_window(c, window, bg_pool, rng, args.snr_low, args.snr_high, clean=(k == 0))
                for c in cl for k in range(n_aug)]
        return embed(F, wins, n_timesteps)

    Xp_train = positives(train_clips, args.aug_per_clip)
    Xp_val = positives(val_clips, max(10, args.aug_per_clip // 3)) if val_clips else np.zeros((0, n_timesteps, FEAT_DIM), np.float32)
    print(f"    {len(Xp_train)} Train- / {len(Xp_val)} Val-Positive")

    X = np.concatenate([Xp_train, Xn_train])
    y = np.concatenate([np.ones(len(Xp_train)), np.zeros(len(Xn_train))]).astype(np.float32)[:, None]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=64, shuffle=True,
    )
    have_val = len(Xp_val) > 0 and len(Xn_val) > 0
    if have_val:
        Xv = np.concatenate([Xp_val, Xn_val])
        yv = torch.from_numpy(np.concatenate([np.ones(len(Xp_val)), np.zeros(len(Xn_val))]).astype(np.float32))

    # 5. Training -------------------------------------------------------------
    print("[5/6] Fine-Tuning")
    if have_val:
        print(f"    Vorher (Original) auf Validierung: {fmt(metrics(model, Xp_val, Xn_val))}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss, best_state, bad = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = weighted_loss(model(xb), yb, args.neg_weight)
            if args.anchor > 0:
                loss = loss + args.anchor * sum(((p - r) ** 2).sum() for p, r in zip(model.parameters(), original_params))
            loss.backward()
            opt.step()
            tot += loss.item()
        msg = f"    Epoche {epoch:3d}/{args.epochs}  train-loss={tot / len(loader):.4f}"
        if have_val:
            pv = torch.from_numpy(predict(model, Xv))
            vl = weighted_loss(pv, yv, args.neg_weight).item()
            msg += f"  val-loss={vl:.4f}  [{fmt(metrics(model, Xp_val, Xn_val))}]"
            if vl < best_loss - 1e-4:
                best_loss, bad = vl, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        print(msg)
        if have_val and bad >= args.patience:
            print("    Early Stopping.")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        print("    Beste Epoche (val-loss) wiederhergestellt.")
        print(f"    Nachher auf Validierung: {fmt(metrics(model, Xp_val, Xn_val))}")

    # 6. Export ---------------------------------------------------------------
    print(f"[6/6] Exportiere nach {args.output}")
    export_onnx(model, n_timesteps, args.output, in_name, out_name)
    verify_against_onnx(model, args.output, n_timesteps, "Export")
    print(f"[*] Fertig! Neues Modell: {args.output}")
    print("    Teste es z.B. mit openwakeword/examples/detect_from_microphone.py "
          "und passe den Threshold in der echten Umgebung an.")


if __name__ == "__main__":  # Guard nötig unter Windows (Multiprocessing)
    main()