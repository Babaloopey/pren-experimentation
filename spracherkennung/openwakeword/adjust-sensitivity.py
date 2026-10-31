#!/usr/bin/env python3
"""
Macht ein bestehendes openWakeWord-Modell empfindlicher - ohne Training, ohne Risiko
für den Rest des Modells.

Idee: Das letzte Layer ist  Score = sigmoid(z).  Zieht man vom Logit z eine Konstante ab,
dann entspricht "0.5" im neuen Modell genau der Schwelle t im alten Modell. Nichts anderes
am Netz ändert sich (kein Vergessen, kein Sättigen auf 1.00). Das Skript
  1. misst mit der echten Streaming-Pipeline (wie live), bei welcher Schwelle wie viele
     deiner Aufnahmen erkannt werden und wie viele Fehlauslösungen auf Negativ-Audio und
     Rauschen entstehen,
  2. schlägt eine Schwelle vor,
  3. schreibt auf Wunsch (--threshold) ein neues .onnx, in dem diese Schwelle eingebacken ist.

Beispiele:
  # nur messen
  python adjust_sensitivity.py --model Sitz_20261014_145414.onnx --positives Training_Data ^
      --negatives openwakeword/notebooks/training_tutorial_data/negative Negative_Data

  # Schwelle 0.25 einbacken
  python adjust_sensitivity.py --model Sitz_20261014_145414.onnx --positives Training_Data ^
      --negatives openwakeword/notebooks/training_tutorial_data/negative --threshold 0.25 ^
      --output Sitz_sensitiv.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import scipy.io.wavfile
import scipy.signal
from onnx import numpy_helper

SR = 16000
FRAME_S = 0.08
GRID = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]


# --------------------------------------------------------------------------- #
# Audio + Streaming-Scoring (wie live)
# --------------------------------------------------------------------------- #
def load_wav(path):
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


def make_oww_model(path):
    import openwakeword
    try:
        return openwakeword.Model(wakeword_models=[str(path)], inference_framework="onnx")
    except TypeError:
        return openwakeword.Model(wakeword_model_paths=[str(path)])


def stream_scores(m, audio):
    """Score pro 80-ms-Frame, genau wie beim Live-Betrieb."""
    try:
        m.reset()
    except Exception:
        pass
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    res = m.predict_clip(pcm)
    key = list(res[0].keys())[0]
    return np.array([r[key] for r in res], dtype=np.float32)


def count_events(scores, thr, refractory=12):
    ev, last = 0, -10 ** 9
    for i in np.where(scores >= thr)[0]:
        if i - last > refractory:
            ev += 1
        last = i
    return ev


def synthetic_negatives(rng, dur=10.0):
    n = int(dur * SR)
    items = [("Stille", np.zeros(n, np.float32))]
    for a in (0.001, 0.005, 0.02, 0.1, 0.3):
        items.append((f"weisses Rauschen {a}", np.clip(rng.normal(0, 1, n) * a, -1, 1).astype(np.float32)))
    return items


# --------------------------------------------------------------------------- #
# ONNX: Schwelle einbacken (Bias der letzten Schicht verschieben)
# --------------------------------------------------------------------------- #
def logit(p):
    p = float(np.clip(p, 1e-7, 1 - 1e-7))
    return float(np.log(p / (1 - p)))


def shift_final_bias(model, shift):
    """Addiert 'shift' auf den Bias der Schicht direkt vor dem letzten Sigmoid."""
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    sigmoids = [n for n in g.node if n.op_type == "Sigmoid"]
    if not sigmoids:
        raise RuntimeError("Kein Sigmoid im Modell gefunden - Architektur nicht unterstützt.")
    producer = {out: n for n in g.node for out in n.output}
    node = producer.get(sigmoids[-1].input[0])
    if node is None:
        raise RuntimeError("Konnte die Schicht vor dem Sigmoid nicht finden.")

    bias_name = None
    if node.op_type == "Gemm" and len(node.input) > 2:
        bias_name = node.input[2]
    elif node.op_type == "Add":
        cand = [x for x in node.input if x in inits]
        if len(cand) == 1:
            bias_name = cand[0]
    if bias_name is None or bias_name not in inits:
        raise RuntimeError(
            f"Die Schicht vor dem Sigmoid ({node.op_type}) hat keinen änderbaren Bias. "
            "Alternativ die Schwelle einfach live in deinem Code senken."
        )
    users = sum(1 for n in g.node for x in n.input if x == bias_name)
    if users != 1:
        raise RuntimeError("Der Bias wird mehrfach verwendet - Änderung wäre nicht sicher.")

    init = inits[bias_name]
    arr = numpy_helper.to_array(init)
    if arr.size != 1:
        raise RuntimeError(f"Bias hat {arr.size} Einträge, erwartet 1 (ein Ausgabewert).")
    new = (arr + shift).astype(arr.dtype)
    init.CopyFrom(numpy_helper.from_array(new, name=init.name))
    return float(arr.reshape(-1)[0]), float(new.reshape(-1)[0])


def verify_shift(old_path, new_path, shift):
    so = ort.InferenceSession(str(old_path), providers=["CPUExecutionProvider"])
    sn = ort.InferenceSession(str(new_path), providers=["CPUExecutionProvider"])
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in so.get_inputs()[0].shape]
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(10):
        x = rng.normal(0, 1, shape).astype(np.float32)
        po = float(so.run(None, {so.get_inputs()[0].name: x})[0].reshape(-1)[0])
        pn = float(sn.run(None, {sn.get_inputs()[0].name: x})[0].reshape(-1)[0])
        expect = 1 / (1 + np.exp(-(logit(po) + shift)))
        worst = max(worst, abs(pn - expect))
    print(f"    Kontrolle ONNX alt vs. neu: max. Abweichung zur erwarteten Verschiebung = {worst:.2e}")
    if worst > 1e-3:
        raise RuntimeError("Das neue Modell verhält sich nicht wie erwartet!")


def transform(scores, shift):
    z = np.log(np.clip(scores, 1e-7, 1 - 1e-7) / (1 - np.clip(scores, 1e-7, 1 - 1e-7)))
    return 1 / (1 + np.exp(-(z + shift)))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Sitz_20261014_145414.onnx")
    ap.add_argument("--positives", default="Training_Data")
    ap.add_argument("--negatives", nargs="*", default=[], help="Ordner mit Negativ-Audio (je mehr/echter, desto besser)")
    ap.add_argument("--threshold", type=float, default=None, help="Schwelle des alten Modells, die im neuen Modell zu 0.5 wird")
    ap.add_argument("--output", default="Sitz_sensitiv.onnx")
    ap.add_argument("--max-neg-minutes", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    if not Path(args.model).exists():
        sys.exit(f"[!] '{args.model}' nicht gefunden.")
    pos_files = sorted(Path(args.positives).glob("*.wav"))
    if not pos_files:
        sys.exit(f"[!] Keine WAVs in '{args.positives}'.")

    print(f"[1/3] Messe Original-Modell im Stream: {args.model}")
    m = make_oww_model(args.model)

    pos_max = {}
    for p in pos_files:
        pos_max[p.name] = float(stream_scores(m, load_wav(p)).max())

    neg_streams, neg_frames = [], 0
    neg_files = []
    for d in args.negatives:
        if Path(d).is_dir():
            neg_files += sorted(Path(d).rglob("*.wav"))
    budget = int(args.max_neg_minutes * 60 / FRAME_S)
    for f in neg_files:
        if neg_frames >= budget:
            break
        x = load_wav(f)[: 5 * 60 * SR]
        if len(x) < SR:
            continue
        sc = stream_scores(m, x)
        neg_streams.append((f.name, sc))
        neg_frames += len(sc)
    synth = [(nm, float(stream_scores(m, a).max())) for nm, a in synthetic_negatives(rng)]

    neg_hours = neg_frames * FRAME_S / 3600
    neg_max = max([float(s.max()) for _, s in neg_streams] + [v for _, v in synth])
    print(f"    {len(pos_files)} Aufnahmen, Negativ-Audio: {neg_frames * FRAME_S / 60:.1f} Minuten ({len(neg_streams)} Dateien)")
    if neg_frames * FRAME_S < 600:
        print("    [!] Wenig Negativ-Audio (<10 min). Die Aussage über Fehlauslösungen ist entsprechend unsicher.")

    print("\n[2/3] Ergebnis je Schwelle (alte Modell-Scores)")
    print(f"    {'Schwelle':>8s} {'erkannt':>10s} {'Fehlausl. Negativ-Audio':>26s} {'Rauschen/Stille':>16s}")
    for t in GRID:
        rec = sum(v >= t for v in pos_max.values())
        ev = sum(count_events(s, t) for _, s in neg_streams)
        per_h = f"{ev / neg_hours:.1f}/h" if neg_hours > 0 else "-"
        sy = sum(v >= t for _, v in synth)
        print(f"    {t:8.2f} {rec:4d}/{len(pos_max):<5d} {ev:8d} ({per_h:>7s}) {sy:10d}/{len(synth)}")

    print("\n    Höchster Score auf allen Negativen/Rauschen: " f"{neg_max:.2f}")
    rec_sorted = sorted(pos_max.items(), key=lambda kv: kv[1])
    print("    Schwächste Aufnahmen: " + ", ".join(f"{k}={v:.2f}" for k, v in rec_sorted[:5]))
    safe = [t for t in GRID if t >= neg_max + 0.05]
    if safe:
        rec_t = min(safe)
        print(f"    Vorschlag: Schwelle {rec_t:.2f} (kleinste Rasterschwelle mit Abstand zum höchsten Negativ-Score)")
    else:
        rec_t = None
        print("    Kein Rasterwert liegt sicher über den Negativ-Scores - bleib bei 0.5 oder teste mit mehr Negativ-Audio.")

    if args.threshold is None:
        print("\n[3/3] Kein --threshold angegeben: es wurde nichts geschrieben.")
        if rec_t is not None and rec_t < 0.5:
            print(f"    Zum Einbacken z. B.: --threshold {rec_t:.2f} --output Sitz_sensitiv.onnx")
        return

    t = args.threshold
    if not 0.0 < t < 1.0:
        sys.exit("[!] --threshold muss zwischen 0 und 1 liegen.")
    shift = -logit(t)
    print(f"\n[3/3] Backe Schwelle {t:.2f} ein (Logit-Verschiebung {shift:+.3f}) -> {args.output}")
    model = onnx.load(args.model)
    old_b, new_b = shift_final_bias(model, shift)
    print(f"    Bias der letzten Schicht: {old_b:+.4f} -> {new_b:+.4f}")
    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    verify_shift(args.model, args.output, shift)

    m2 = make_oww_model(args.output)
    first = pos_files[0]
    chk = float(stream_scores(m2, load_wav(first)).max())
    expect = float(transform(np.array([pos_max[first.name]]), shift)[0])
    print(f"    Stream-Kontrolle {first.name}: neu={chk:.3f}, erwartet={expect:.3f}")

    new_rec = sum(float(transform(np.array([v]), shift)[0]) >= 0.5 for v in pos_max.values())
    new_ev = sum(count_events(transform(s, shift), 0.5) for _, s in neg_streams)
    print(f"    Mit Standardschwelle 0.5 erkennt das neue Modell {new_rec}/{len(pos_max)} Aufnahmen, "
          f"{new_ev} Fehlauslösungen auf dem Negativ-Audio.")
    print(f"[*] Fertig: {args.output}  (verwende es wie das alte Modell, Standardschwelle 0.5)")


if __name__ == "__main__":
    main()