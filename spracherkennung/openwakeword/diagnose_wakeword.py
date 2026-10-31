#!/usr/bin/env python3
"""
Fine-Tuning v2: verbessert ein bestehendes openWakeWord-Modell mit eigenen Aufnahmen,
OHNE es zu verschlechtern.

Unterschiede zu v1:
  * Features kommen aus durchlaufenden Sequenzen (wie im Live-Betrieb), nicht aus
    einzelnen zugeschnittenen Fenstern.
  * Das Label "Positiv" sitzt dort, wo das ALTE Modell sein Maximum hat (kalibriert).
  * Viel mehr und vielfältigere Negative (Audio-Dateien, optional openWakeWord-
    Feature-Datei, synthetische Stille/Rauschen/Brummen/Klicks, Kontext um das Wort).
  * Validierung wie im Live-Betrieb: Maximum über den Stream, False-Positives pro Stunde,
    Stille/Rauschen-Test.
  * Schutz: Es wird nur exportiert, wenn eine Epoche das Original auf der Validierung
    schlägt UND keine zusätzlichen False Positives erzeugt. Sonst bleibt das Original.

Aufruf (Beispiel):
  python enhance_wakeword_v2.py --model Sitz_20261014_145414.onnx --positives Training_Data ^
      --negatives openwakeword/notebooks/training_tutorial_data/negative Negative_Data ^
      --neg-features validation_set_features.npy --output Sitz_fine_tuned_v2.onnx

validation_set_features.npy (~11 h Negativ-Features, klein) gibt es hier:
  https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import scipy.signal
import torch
from torch import nn
import openwakeword.utils

from enhance_wakeword import (  # Hilfsfunktionen aus dem ersten Skript wiederverwenden
    FEAT_DIM, HOP, SR, export_onnx, find_window_samples, load_wav, onnx_to_torch,
    rms, trim_silence, verify_against_onnx, random_crop,
)


# --------------------------------------------------------------------------- #
# Features / Scoring
# --------------------------------------------------------------------------- #
def to_pcm(x):
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)


def embed_seq(F, audio):
    """Ganze Audio-Sequenz -> Embeddings (T, 96), wie im Streaming."""
    return np.asarray(F._get_embeddings(to_pcm(audio)), dtype=np.float32)


def sliding(E, n):
    if len(E) < n:
        return np.zeros((0, n, FEAT_DIM), np.float32)
    idx = np.arange(len(E) - n + 1)[:, None] + np.arange(n)[None, :]
    return E[idx]


def score_windows(model, W, bs=8192):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(W), bs):
            out.append(model(torch.from_numpy(np.ascontiguousarray(W[i: i + bs]))).reshape(-1).numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def count_events(scores, thr, refractory=12):
    """Zählt Auslösungen (aufeinanderfolgende Überschreitungen = 1 Ereignis)."""
    ev, last = 0, -10 ** 9
    for i in np.where(scores >= thr)[0]:
        if i - last > refractory:
            ev += 1
        last = i
    return ev


# --------------------------------------------------------------------------- #
# Sequenzen mit Wort
# --------------------------------------------------------------------------- #
def make_sequence(clip, bg_pool, rng, snr_low, snr_high, pre_s, post_s, clean=False):
    """Stille/Hintergrund + Wort + Hintergrund. Gibt (audio, Wortende in Samples) zurück."""
    x = clip.copy()
    if not clean:
        factor = rng.uniform(0.9, 1.1)
        x = scipy.signal.resample(x, max(1, int(len(x) / factor))).astype(np.float32)
    pre, post = int(pre_s * SR), int(post_s * SR)
    total = pre + len(x) + post
    fg = np.zeros(total, np.float32)
    fg[pre: pre + len(x)] = x

    snr = 30.0 if clean else float(rng.uniform(snr_low, snr_high))
    if bg_pool and (clean or rng.random() < 0.9):
        bg = random_crop(bg_pool[int(rng.integers(len(bg_pool)))], total, rng)
        bg = bg * (rms(x) / (rms(bg) * 10 ** (snr / 20)))
    else:
        bg = (rng.normal(0, 1, total) * rms(x) * 10 ** (-snr / 20)).astype(np.float32)
    mix = fg + bg
    mix = mix / (np.abs(mix).max() + 1e-8)
    gain = 0.9 if clean else float(rng.uniform(0.15, 0.9))
    return (mix * gain).astype(np.float32), pre + len(x)


def s_star(e_end, window):
    """Fensterstart (in Frames), bei dem das Fenster genau am Wortende endet."""
    return (e_end - (window - 400)) / HOP


def calibrate_offset(F, model, clips, names, window, T, rng, thr):
    """Wo (relativ zum Wortende) hat das alte Modell sein Maximum?"""
    diffs = []
    print("    Kalibrierung (altes Modell, Stream-Maximum je Aufnahme):")
    for c, nm in zip(clips, names):
        seq, e_end = make_sequence(c, [], rng, 30, 30, 2.1, 2.1, clean=True)
        sc = score_windows(model, sliding(embed_seq(F, seq), T))
        peak = int(np.argmax(sc))
        ok = sc.max() >= thr
        if ok:
            diffs.append(peak - s_star(e_end, window))
        print(f"      {nm:14s} max={sc.max():.2f}  Peak-Offset={peak - s_star(e_end, window):+.1f} Frames{'' if ok else '  (unter Schwelle)'}")
    if len(diffs) >= 3:
        d = int(round(float(np.median(diffs))))
        print(f"    -> Positiv-Zone liegt {d:+d} Frames relativ zum Wortende (Median aus {len(diffs)} Aufnahmen)")
        return d
    print("    [!] Zu wenige erkannte Aufnahmen für Kalibrierung, verwende Offset 0.")
    return 0


def collect_positives(F, clips, n_aug, d, bg_pool, rng, args, window, T, width=2):
    pos, ctx_neg = [], []
    for c in clips:
        for k in range(n_aug):
            pre, post = rng.uniform(2.1, 2.6), rng.uniform(2.1, 2.6)
            seq, e_end = make_sequence(c, bg_pool, rng, args.snr_low, args.snr_high, pre, post, clean=(k == 0))
            E = embed_seq(F, seq)
            s_t = int(round(s_star(e_end, window) + d))
            for s in range(s_t - width, s_t + width + 1):
                if 0 <= s <= len(E) - T:
                    pos.append(E[s: s + T])
            # Kontext ohne Wort: reiner Vorlauf bzw. Nachlauf
            for s in range(0, int((pre * SR - window) / HOP) + 1):
                if s <= len(E) - T:
                    ctx_neg.append(E[s: s + T])
            first_post = int(np.ceil(e_end / HOP)) + 1
            for s in range(first_post, len(E) - T + 1):
                ctx_neg.append(E[s: s + T])
    return np.stack(pos), (np.stack(ctx_neg) if ctx_neg else np.zeros((0, T, FEAT_DIM), np.float32))


def recording_sets(F, clips, n, bg_pool, rng, args, T):
    """Validierungssets: pro Aufnahme eine Sequenz (sauber + verrauschte Varianten) als Fensterliste."""
    sets = []
    for c in clips:
        for k in range(n):
            seq, _ = make_sequence(c, bg_pool, rng, 10, 20, 2.1, 2.1, clean=(k == 0))
            sets.append(sliding(embed_seq(F, seq), T))
    return sets


# --------------------------------------------------------------------------- #
# Negative
# --------------------------------------------------------------------------- #
def synth_audio(rng, dur=8.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    items = [("Stille", np.zeros(n, np.float32))]
    for a in (0.001, 0.005, 0.02, 0.1, 0.3):
        items.append((f"weisses Rauschen {a}", (rng.normal(0, 1, n) * a)))
    items.append(("50-Hz-Brummen", 0.05 * np.sin(2 * np.pi * 50 * t)))
    items.append(("Sinus 440 Hz", 0.2 * np.sin(2 * np.pi * 440 * t)))
    clicks = np.zeros(n)
    clicks[rng.integers(0, n, 20)] = rng.uniform(0.3, 1.0, 20)
    items.append(("Klicks", clicks))
    return [(nm, np.clip(x, -1, 1).astype(np.float32)) for nm, x in items]


def synth_windows(F, rng, T):
    return [(nm, sliding(embed_seq(F, a), T)) for nm, a in synth_audio(rng)]


def collect_wav_negatives(F, files, T, rng, stride=3):
    train, val_streams, pool = [], [], []
    files = list(files)
    rng.shuffle(files)
    for f in files:
        try:
            x = load_wav(f)
        except Exception as e:
            print(f"    [!] Überspringe {f}: {e}")
            continue
        if len(x) == 0:
            continue
        if len(pool) < 50:
            pool.append(x)
        if len(x) < 4 * SR:
            x = np.tile(x, int(np.ceil(4 * SR / len(x))))
        W = sliding(embed_seq(F, x[: 10 * 60 * SR]), T)
        n = len(W)
        if n >= 20:
            nv = max(5, int(0.2 * n))
            train.append(W[: max(0, n - nv - T)][::stride])
            val_streams.append(W[n - nv:])
        else:
            train.append(W[::stride])
    cat = np.concatenate(train) if train else np.zeros((0, T, FEAT_DIM), np.float32)
    return cat, val_streams, pool


def collect_feature_negatives(path, T, rng, max_train):
    """openWakeWord-Feature-Datei: 2D (Frames,96) -> Train + Val-Stream; 3D (N,T,96) -> nur Train."""
    A = np.load(path, mmap_mode="r")
    if A.ndim == 2:
        n_tr = int(len(A) * 0.8)
        block = np.asarray(A[:n_tr], dtype=np.float32)
        starts = rng.integers(0, len(block) - T, size=min(max_train, len(block) - T))
        train = block[starts[:, None] + np.arange(T)[None, :]]
        val = np.asarray(A[n_tr: n_tr + 50000], dtype=np.float32)  # max. ~67 min, spart Speicher
        return train, [sliding(val, T)]
    if A.ndim == 3:
        idx = np.sort(rng.choice(len(A), size=min(max_train, len(A)), replace=False))
        return np.asarray(A[idx], dtype=np.float32), []
    raise ValueError(f"Unerwartete Form der Feature-Datei: {A.shape}")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Sitz_20261014_145414.onnx")
    ap.add_argument("--positives", default="Training_Data")
    ap.add_argument("--negatives", nargs="+", default=["openwakeword/notebooks/training_tutorial_data/negative"])
    ap.add_argument("--neg-features", default=None, help="optional: openWakeWord-Feature-.npy (z.B. validation_set_features.npy)")
    ap.add_argument("--output", default="Sitz_fine_tuned_v2.onnx")
    ap.add_argument("--aug-per-clip", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--anchor", type=float, default=0.5, help="Bindung an die alten Gewichte (0 = aus)")
    ap.add_argument("--pos-weight", type=float, default=3.0)
    ap.add_argument("--thr", type=float, default=0.5, help="Erkennungsschwelle für die Validierung")
    ap.add_argument("--max-fph", type=float, default=0.5, help="erlaubte False Positives/Stunde auf Val-Negativen")
    ap.add_argument("--freeze-hidden", action="store_true", help="nur die letzte Linear-Schicht trainieren")
    ap.add_argument("--max-neg", type=int, default=60000)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--snr-low", type=float, default=5.0)
    ap.add_argument("--snr-high", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # 1. Modell ---------------------------------------------------------------
    print(f"[1/6] Lade Modell: {args.model}")
    if not Path(args.model).exists():
        sys.exit(f"[!] '{args.model}' nicht gefunden.")
    onnx_model = onnx.load(args.model)
    init_names = {i.name for i in onnx_model.graph.initializer}
    in_name = next(i.name for i in onnx_model.graph.input if i.name not in init_names)
    out_name = onnx_model.graph.output[0].name
    model = onnx_to_torch(onnx_model)
    T = next(m for m in model if isinstance(m, nn.Linear)).in_features // FEAT_DIM
    verify_against_onnx(model, args.model, T, "Original")
    original_params = [p.detach().clone() for p in model.parameters()]
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    try:
        F = openwakeword.utils.AudioFeatures()
    except Exception:
        openwakeword.utils.download_models()
        F = openwakeword.utils.AudioFeatures()
    window = find_window_samples(F, T)

    # 2. Aufnahmen + Negative laden -----------------------------------------
    print("[2/6] Lade Aufnahmen und Negative")
    pos_files = sorted(Path(args.positives).glob("*.wav"))
    if not pos_files:
        sys.exit(f"[!] Keine WAVs in '{args.positives}'.")
    clips = [trim_silence(load_wav(p)) for p in pos_files]
    names = [p.name for p in pos_files]

    neg_files = []
    for d_ in args.negatives:
        if Path(d_).is_dir():
            neg_files += sorted(Path(d_).rglob("*.wav"))
    Xn_wav, val_streams, bg_pool = collect_wav_negatives(F, neg_files, T, rng) if neg_files else (np.zeros((0, T, FEAT_DIM), np.float32), [], [])
    print(f"    {len(neg_files)} Negativ-WAVs -> {len(Xn_wav)} Train-Fenster")
    Xn_feat = np.zeros((0, T, FEAT_DIM), np.float32)
    if args.neg_features:
        Xn_feat, vs = collect_feature_negatives(args.neg_features, T, rng, args.max_neg)
        val_streams += vs
        print(f"    Feature-Datei -> {len(Xn_feat)} Train-Fenster")
    if len(Xn_wav) + len(Xn_feat) < 2000:
        print("    [!] WARNUNG: Sehr wenige Negative. Das Risiko für False Positives ist hoch. "
              "Nutze --neg-features und/oder mehr eigene Negativ-Aufnahmen.")

    order = rng.permutation(len(clips))
    n_val = max(1, int(round(args.val_fraction * len(clips)))) if len(clips) >= 5 else 0
    val_idx, train_idx = list(order[:n_val]), list(order[n_val:])
    train_clips = [clips[i] for i in train_idx]
    val_clips = [clips[i] for i in val_idx]
    print(f"    Aufnahmen: {len(train_clips)} Training, {len(val_clips)} Validierung")

    # 3. Kalibrierung -----------------------------------------------------------
    print("[3/6] Kalibriere Positiv-Zone am alten Modell")
    d = calibrate_offset(F, model, clips, names, window, T, rng, args.thr)

    # 4. Trainingsdaten ---------------------------------------------------------
    print("[4/6] Erzeuge Trainings- und Validierungsdaten")
    Xp, Xn_ctx = collect_positives(F, train_clips, args.aug_per_clip, d, bg_pool, rng, args, window, T)
    synth_train = np.concatenate([w for _, w in synth_windows(F, rng, T)])
    synth_eval = synth_windows(F, rng, T)  # andere Zufallsziehung
    Xn = np.concatenate([Xn_wav, Xn_feat, Xn_ctx, synth_train])
    print(f"    Positive: {len(Xp)}, Negative: {len(Xn)} (davon Kontext {len(Xn_ctx)}, synthetisch {len(synth_train)})")

    eval_clips = val_clips if val_clips else train_clips
    val_pos_sets = recording_sets(F, eval_clips, 5, bg_pool, rng, args, T)
    if not val_clips:
        print("    [!] Zu wenige Aufnahmen für eine eigene Validierung: Recall wird auf Trainingsaufnahmen gemessen.")
    n_val_frames = sum(len(s) for s in val_streams)
    print(f"    Val-Negativ-Stream: {n_val_frames * 0.08 / 60:.1f} Minuten")

    X = np.concatenate([Xp, Xn])
    y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))]).astype(np.float32)[:, None]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=256, shuffle=True,
    )

    def evaluate(m):
        rec = float(np.mean([score_windows(m, W).max() >= args.thr for W in val_pos_sets]))
        ev = sum(count_events(score_windows(m, W), args.thr) for W in val_streams)
        hours = n_val_frames * 0.08 / 3600
        fph = ev / hours if hours > 0 else 0.0
        synth = max(float(score_windows(m, W).max()) for _, W in synth_eval)
        return rec, fph, synth

    # 5. Training ---------------------------------------------------------------
    print("[5/6] Fine-Tuning (mit Schutz vor Verschlechterung)")
    if args.freeze_hidden:
        last_lin = [m for m in model if isinstance(m, nn.Linear)][-1]
        for p in model.parameters():
            p.requires_grad = False
        for p in last_lin.parameters():
            p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)

    base_rec, base_fph, base_synth = evaluate(model)
    fph_limit = max(base_fph, args.max_fph)
    print(f"    Epoche  0 (Original)  recall={base_rec:.2f}  FP/h={base_fph:.2f}  synth-max={base_synth:.2f}")
    best_key, best_epoch, best_state = (base_rec, -base_fph), 0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            w = torch.where(yb > 0.5, torch.full_like(yb, args.pos_weight), torch.ones_like(yb))
            loss = torch.nn.functional.binary_cross_entropy(pred, yb, weight=w)
            if args.anchor > 0:
                loss = loss + args.anchor * sum(((p - r) ** 2).sum() for p, r in zip(model.parameters(), original_params))
            loss.backward()
            opt.step()
            tot += loss.item()
        rec, fph, synth = evaluate(model)
        ok = synth < args.thr and fph <= fph_limit
        key = (rec, -fph)
        mark = ""
        if ok and key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            mark = "  <- bisher beste"
        print(f"    Epoche {epoch:2d}  loss={tot / len(loader):.4f}  recall={rec:.2f}  FP/h={fph:.2f}  "
              f"synth-max={synth:.2f}  {'ok' if ok else 'VERWORFEN'}{mark}")

    if best_state is None:
        print("\n[*] Keine Epoche war besser als das Original (bei gleichzeitig keinen zusätzlichen False Positives).")
        print("    Es wird NICHTS exportiert - dein Original bleibt die bessere Wahl.")
        print("    Ideen: mehr/andere Negative (--neg-features), --freeze-hidden, kleinere --lr, mehr Aufnahmen.")
        return
    model.load_state_dict(best_state)
    print(f"    Beste Epoche: {best_epoch}")

    # 6. Export + Vergleich -----------------------------------------------------
    print(f"[6/6] Exportiere nach {args.output}")
    export_onnx(model, T, args.output, in_name, out_name)
    verify_against_onnx(model, args.output, T, "Export")

    old = onnx_to_torch(onnx.load(args.model))
    print("\n    Vergleich (max. Score im Stream)        alt    neu")
    for nm, W in synth_eval:
        print(f"    {nm:38s} {score_windows(old, W).max():5.2f}  {score_windows(model, W).max():5.2f}")
    for nm, c in list(zip(names, clips)):
        seq, _ = make_sequence(c, [], rng, 30, 30, 2.1, 2.1, clean=True)
        W = sliding(embed_seq(F, seq), T)
        print(f"    Aufnahme {nm:29s} {score_windows(old, W).max():5.2f}  {score_windows(model, W).max():5.2f}")
    print("\n[*] Fertig. Prüfe mit diagnose_wakeword.py und live mit detect_from_microphone.py,")
    print("    bevor du das neue Modell einsetzt.")


if __name__ == "__main__":
    main()