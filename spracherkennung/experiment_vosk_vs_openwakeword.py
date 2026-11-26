#!/usr/bin/env python3
"""
Experiment: Befehlserkennung Vosk vs. openWakeWord auf aufgenommenen Befehlen.

Datensatz
---------
    <data-dir>/Brav/*.wav            Trainings-Sprecher (für openWakeWord mittrainiert)
    <data-dir>/Brav/Validation/*.wav Holdout-Sprecher (nie trainiert)  -> fairer Vergleich
    ... gleich für Hier, Sitz, Such

Der Ordnername ist das erwartete Wort. Vosk ist nicht auf diese Daten trainiert, openWakeWord
schon (ausser Validation). Darum werden alle Kennzahlen getrennt für "train", "validation" und
"alle" ausgegeben. Massgebend für den Vergleich ist "validation".

Ablauf
------
1. Jede Aufnahme wird geladen (Mono, 16 kHz), vorne und hinten mit Stille aufgefüllt und
   optional mit weissem Rauschen bei einem bestimmten SNR versehen (--snr).
2. Vosk: Erkennung wie in vosk/test_vosk.py (Grammatik = Befehle + Ähnlichwörter). Pro Clip
   wird der beste gültige Befehl und dessen Konfidenz gespeichert (ohne Schwelle).
3. openWakeWord: Alle vier Modelle (*_de.onnx) gleichzeitig, wie im Live-Betrieb in 80-ms-Blöcken
   gefüttert. Pro Clip werden die Score-Verläufe aller Modelle gespeichert.
4. Danach werden die Entscheidungsregeln nachträglich angewendet:
     Vosk:  Befehl gilt, wenn Konfidenz >= min_conf
     oWW:   Modell löst aus, wenn Score > Schwelle in >= N aufeinanderfolgenden Frames
            (wie test_command_wakewords.py). Lösen mehrere aus, gewinnt der höchste Score.
   So lassen sich Schwellen-Sweeps ohne Neuberechnung erstellen.

Kennzahlen (pro System, Split und SNR)
---------------------------------------
* Accuracy       richtig erkannt / alle Clips
* Fehlerkennung  anderer Befehl erkannt (gefährlicher als "nichts erkannt")
* Verpasst       kein Befehl erkannt
* Recall je Wort, Konfusionsmatrix (inkl. Spalte "-" = nichts)
* Cross-Trigger  (nur oWW) Anteil Clips, bei denen ein *falsches* Modell feuert - auch wenn
                 das richtige höher liegt. Entspricht Fehlalarmen eines einzelnen Modells.
* Rechenzeit     ms pro Clip und Real-Time-Factor (RTF, < 1 = schneller als Echtzeit)

Nutzung
-------
    python experiment_vosk_vs_openwakeword.py
    python experiment_vosk_vs_openwakeword.py --snr clean 20 10 5
    python experiment_vosk_vs_openwakeword.py --oww-threshold 0.5 --vosk-min-conf 0.8
    python experiment_vosk_vs_openwakeword.py --no-vad --limit 5     # schneller Probelauf

Ausgabe in experiment_results/<Zeitstempel>/:
    per_clip.csv        ein Eintrag pro Clip, System und SNR (inkl. Rohwerten)
    summary.csv         Kennzahlen bei den gewählten Arbeitspunkten
    sweep.csv           Kennzahlen über alle Schwellen
    confusion_*.png     Konfusionsmatrizen
    sweep_*.png         Accuracy / Fehlerkennung über der Schwelle
    report.md           Zusammenfassung
"""
import argparse
import csv
import importlib.util
import json
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
SR = 16000
COMMANDS = ["brav", "hier", "sitz", "such"]
LABELS = COMMANDS + ["-"]                      # "-" = nichts erkannt
DEFAULT_DATA = Path(r"C:\Users\flori\OneDrive\05_Studium\Semester 05\PREN\Training_Data\Sprache")
OWW_MODELS = {c: HERE / "openwakeword" / "models" / f"{c}_de.onnx" for c in COMMANDS}
OWW_CHUNK = 1280                               # 80 ms, wie im Live-Skript
VOSK_BLOCK = 4000                              # 250 ms, wie in test_vosk.py
PAD_BEFORE_S, PAD_AFTER_S = 1.5, 1.0

# Arbeitspunkte der bestehenden Skripte
OWW_THRESHOLD = 0.1          # test_command_wakewords.py: score > 0.1
OWW_CONSECUTIVE = 2          # ... in 2 Frames nacheinander
OWW_VAD = 0.3
VOSK_MIN_CONF = 0.80         # test_vosk.py: MIN_CONF


# ------------------------------------------------------------------ Hilfsfunktionen
def load_vosk_module():
    """test_vosk.py als Modul laden, damit Vokabular und Bewertung identisch sind."""
    spec = importlib.util.spec_from_file_location("test_vosk", HERE / "vosk" / "test_vosk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError(f"{path}: nur 16-Bit-PCM unterstützt")
    x = np.frombuffer(raw, np.int16).astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return x


def find_clips(data_dir, limit=None):
    clips = []
    for cmd_dir in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()):
        label = cmd_dir.name.lower()
        if label not in COMMANDS:
            continue
        per_label = []
        for f in sorted(cmd_dir.rglob("*.wav")):
            split = "validation" if "validation" in (p.lower() for p in f.relative_to(cmd_dir).parts[:-1]) else "train"
            speaker = f.stem.split(".")[0].split("_")[0] if ("." in f.stem or "_" in f.stem) else "unbekannt"
            per_label.append(dict(path=f, label=label, split=split, speaker=speaker,
                                  name=str(f.relative_to(data_dir))))
        if limit:   # je Split begrenzen, damit auch Validation im Probelauf vorkommt
            per_label = [c for s in ("train", "validation") for c in [c for c in per_label if c["split"] == s][:limit]]
        clips += per_label
    return clips


def prepare(x, snr, rng, floor=0.0):
    """Stille anfügen und optional Rauschen bei gegebenem SNR (dB, bezogen auf die Sprachleistung).
    floor: Standardabweichung eines konstanten Grundrauschens (int16-Einheiten), damit die Stille
    nicht aus exakten Nullen besteht (ein echtes Mikrofon liefert nie digitale Null)."""
    y = np.concatenate([np.zeros(int(PAD_BEFORE_S * SR)), x, np.zeros(int(PAD_AFTER_S * SR))])
    if floor > 0:
        y = y + rng.normal(0, floor, len(y))
    if snr != "clean":
        p_sig = np.mean(x ** 2) + 1e-9
        p_noise = p_sig / (10 ** (float(snr) / 10))
        y = y + rng.normal(0, np.sqrt(p_noise), len(y))
    return np.clip(y, -32768, 32767).astype(np.int16)


# ------------------------------------------------------------------ Vosk
class VoskRunner:
    def __init__(self, model_path=None):
        self.tv = load_vosk_module()
        from vosk import KaldiRecognizer, Model, SetLogLevel
        SetLogLevel(-1)
        self.model = Model(model_path=str(model_path)) if model_path else Model(lang="de")
        self.KR = KaldiRecognizer
        self.grammar = json.dumps(self.tv.build_vocabulary(), ensure_ascii=False)
        self.rec = self.KR(self.model, SR, self.grammar)   # einmal erstellen, pro Clip Reset()
        self.rec.SetWords(True)

    def run(self, audio):
        rec = self.rec
        rec.Reset()
        results = []
        t0 = time.perf_counter()
        for i in range(0, len(audio), VOSK_BLOCK):
            if rec.AcceptWaveform(audio[i:i + VOSK_BLOCK].tobytes()):
                results.append(json.loads(rec.Result()))
        results.append(json.loads(rec.FinalResult()))
        dt = time.perf_counter() - t0
        best_cmd, best_conf, texts = None, 0.0, []
        for r in results:
            if r.get("text"):
                texts.append(r["text"])
            cmd, conf, _ = self.tv.evaluate_result(r, COMMANDS, min_conf=0.0)
            if cmd is not None and conf > best_conf:
                best_cmd, best_conf = cmd, conf
        return dict(raw_cmd=best_cmd, score=best_conf, text=" | ".join(texts), seconds=dt)


def vosk_decide(row, min_conf):
    return row["raw_cmd"] if row["raw_cmd"] and row["score"] >= min_conf else "-"


# ------------------------------------------------------------------ openWakeWord
class OwwRunner:
    def __init__(self, vad_threshold=OWW_VAD):
        from openwakeword.model import Model
        self.model = Model(wakeword_models=[str(p) for p in OWW_MODELS.values()],
                           inference_framework="onnx", vad_threshold=vad_threshold)
        self.keys = {c: OWW_MODELS[c].stem for c in COMMANDS}

    def _reset(self):
        self.model.reset()
        vad = getattr(self.model, "vad", None)
        if vad is not None:
            if hasattr(vad, "reset_states"):
                vad.reset_states()
            if hasattr(vad, "prediction_buffer"):
                vad.prediction_buffer.clear()

    def run(self, audio):
        self._reset()
        frames = {c: [] for c in COMMANDS}
        t0 = time.perf_counter()
        for i in range(0, len(audio) - OWW_CHUNK + 1, OWW_CHUNK):
            pred = self.model.predict(audio[i:i + OWW_CHUNK])
            for c in COMMANDS:
                frames[c].append(float(pred.get(self.keys[c], 0.0)))
        dt = time.perf_counter() - t0
        frames = {c: np.array(v) for c, v in frames.items()}
        best = max(COMMANDS, key=lambda c: frames[c].max())
        return dict(raw_cmd=best, score=float(frames[best].max()), frames=frames,
                    max_scores={c: float(frames[c].max()) for c in COMMANDS}, seconds=dt)


def fired_peak(scores, threshold, consecutive):
    """Höchster Score innerhalb einer Folge von >= consecutive Frames über der Schwelle, sonst None."""
    run, peak, best = 0, 0.0, None
    for s in scores:
        if s > threshold:
            run += 1
            peak = max(peak, s)
            if run >= consecutive:
                best = peak if best is None else max(best, peak)
        else:
            run, peak = 0, 0.0
    return best


def oww_decide(row, threshold, consecutive):
    """Gibt (Entscheidung, Liste der ausgelösten Modelle) zurück."""
    fired = {c: p for c in COMMANDS if (p := fired_peak(row["frames"][c], threshold, consecutive)) is not None}
    if not fired:
        return "-", []
    return max(fired, key=fired.get), sorted(fired)


# ------------------------------------------------------------------ Auswertung
def metrics(rows, preds, fired_lists=None):
    n = len(rows)
    if n == 0:
        return {}
    labels = [r["label"] for r in rows]
    correct = sum(p == l for p, l in zip(preds, labels))
    wrong = sum(p not in ("-", l) for p, l in zip(preds, labels))
    m = dict(n=n, accuracy=correct / n, wrong_cmd=wrong / n, missed=preds.count("-") / n)
    for c in COMMANDS:
        idx = [i for i, l in enumerate(labels) if l == c]
        m[f"recall_{c}"] = (sum(preds[i] == c for i in idx) / len(idx)) if idx else float("nan")
    if fired_lists is not None:
        m["cross_trigger"] = sum(any(f != l for f in fl) for fl, l in zip(fired_lists, labels)) / n
    m["ms_per_clip"] = 1000 * np.mean([r["seconds"] for r in rows])
    m["rtf"] = sum(r["seconds"] for r in rows) / sum(r["audio_s"] for r in rows)
    return m


def confusion(rows, preds):
    mat = np.zeros((len(COMMANDS), len(LABELS)), int)
    for r, p in zip(rows, preds):
        mat[COMMANDS.index(r["label"]), LABELS.index(p)] += 1
    return mat


def plot_confusion(mat, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    norm = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(COMMANDS)), COMMANDS)
    ax.set_xlabel("erkannt")
    ax.set_ylabel("gesagt")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_sweep(sweep_rows, split, snr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, system, xlabel in [(axes[0], "openwakeword", "oWW-Schwelle"), (axes[1], "vosk", "Vosk min_conf")]:
        rs = [r for r in sweep_rows if r["system"] == system and r["split"] == split and r["snr"] == snr]
        x = [r["threshold"] for r in rs]
        ax.plot(x, [r["accuracy"] for r in rs], label="Accuracy", color="#2a7ab8")
        ax.plot(x, [r["wrong_cmd"] for r in rs], label="falscher Befehl", color="#d1495b")
        ax.plot(x, [r["missed"] for r in rs], label="verpasst", color="#999999", ls="--")
        if system == "openwakeword":
            ax.plot(x, [r["cross_trigger"] for r in rs], label="Cross-Trigger", color="#e8a33d", ls=":")
        ax.set_xlabel(xlabel)
        ax.set_title(f"{system} ({split}, SNR {snr})", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Anteil der Clips")
    axes[0].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fmt(v):
    return f"{v:.1%}" if isinstance(v, float) and not np.isnan(v) else str(v)


# ------------------------------------------------------------------ Hauptprogramm
def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=HERE / "experiment_results")
    ap.add_argument("--vosk-model-path", help="Vosk-Modellordner (Standard: vosk-model-small-de aus dem Cache)")
    ap.add_argument("--snr", nargs="+", default=["clean"], help="'clean' und/oder SNR-Werte in dB, z. B. clean 20 10 5")
    ap.add_argument("--oww-threshold", type=float, default=OWW_THRESHOLD)
    ap.add_argument("--oww-consecutive", type=int, default=OWW_CONSECUTIVE)
    ap.add_argument("--vosk-min-conf", type=float, default=VOSK_MIN_CONF)
    ap.add_argument("--no-vad", action="store_true", help="openWakeWord ohne Silero-VAD")
    ap.add_argument("--systems", nargs="+", default=["vosk", "openwakeword"], choices=["vosk", "openwakeword"])
    ap.add_argument("--limit", type=int, help="max. Clips pro Wort und Split (Probelauf)")
    ap.add_argument("--noise-floor", type=float, default=10.0,
                    help="Grundrauschen in int16-Einheiten (Std.), z. B. 10 = ca. -70 dBFS; 0 = digitale Stille")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    clips = find_clips(args.data_dir, args.limit)
    if not clips:
        raise SystemExit(f"Keine WAV-Dateien in {args.data_dir}")
    out = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(clips)} Clips ({sum(c['split'] == 'validation' for c in clips)} Validation) -> {out}")

    audio = {c["name"]: read_wav(c["path"]) for c in clips}
    runners = {}
    if "vosk" in args.systems:
        runners["vosk"] = VoskRunner(args.vosk_model_path)
    if "openwakeword" in args.systems:
        runners["openwakeword"] = OwwRunner(0 if args.no_vad else OWW_VAD)

    # ---- 1. Rohdaten berechnen
    results = {}   # (system, snr) -> Liste von Zeilen
    for snr in args.snr:
        rng = np.random.default_rng(args.seed)
        prepared = {c["name"]: prepare(audio[c["name"]], snr, rng, args.noise_floor) for c in clips}
        for system, runner in runners.items():
            rows = []
            t0 = time.time()
            for k, c in enumerate(clips, 1):
                r = runner.run(prepared[c["name"]])
                r.update(c, audio_s=len(prepared[c["name"]]) / SR)
                rows.append(r)
                print(f"\r  {system:13s} SNR {snr:>5}: {k}/{len(clips)}", end="", flush=True)
            print(f"  ({time.time() - t0:.1f} s)")
            results[(system, snr)] = rows

    # ---- 2. Arbeitspunkte, Konfusionsmatrizen, per-Clip-CSV
    def decide(system, rows, thr):
        if system == "vosk":
            return [vosk_decide(r, thr) for r in rows], None
        d = [oww_decide(r, thr, args.oww_consecutive) for r in rows]
        return [p for p, _ in d], [f for _, f in d]

    op = {"vosk": args.vosk_min_conf, "openwakeword": args.oww_threshold}
    splits = {"validation": lambda r: r["split"] == "validation", "train": lambda r: r["split"] == "train",
              "alle": lambda r: True}
    summary = []
    per_clip_path = out / "per_clip.csv"
    with open(per_clip_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["system", "snr", "split", "speaker", "file", "label", "prediction", "correct",
                    "raw_cmd", "score", "fired_models", "vosk_text",
                    *[f"oww_max_{c}" for c in COMMANDS], "ms"])
        for (system, snr), rows in results.items():
            preds, fired = decide(system, rows, op[system])
            for i, r in enumerate(rows):
                w.writerow([system, snr, r["split"], r["speaker"], r["name"], r["label"], preds[i],
                            int(preds[i] == r["label"]), r["raw_cmd"] or "-", f"{r['score']:.3f}",
                            ",".join(fired[i]) if fired else "", r.get("text", ""),
                            *[f"{r['max_scores'][c]:.3f}" if "max_scores" in r else "" for c in COMMANDS],
                            f"{1000 * r['seconds']:.1f}"])
            for split, f in splits.items():
                idx = [i for i, r in enumerate(rows) if f(r)]
                sub = [rows[i] for i in idx]
                if not sub:
                    continue
                m = metrics(sub, [preds[i] for i in idx], [fired[i] for i in idx] if fired else None)
                summary.append(dict(system=system, snr=snr, split=split, threshold=op[system], **m))
                if split in ("validation", "alle"):
                    plot_confusion(confusion(sub, [preds[i] for i in idx]),
                                   f"{system} | {split} | SNR {snr} | Schwelle {op[system]}",
                                   out / f"confusion_{system}_{split}_{snr}.png")

    # ---- 3. Schwellen-Sweep
    sweep = []
    grids = {"vosk": np.round(np.arange(0.0, 1.001, 0.05), 2),
             "openwakeword": np.round(np.concatenate([[0.01, 0.02, 0.05], np.arange(0.1, 0.951, 0.05)]), 2)}
    for (system, snr), rows in results.items():
        for thr in grids[system]:
            preds, fired = decide(system, rows, float(thr))
            for split, f in splits.items():
                idx = [i for i, r in enumerate(rows) if f(r)]
                if idx:
                    m = metrics([rows[i] for i in idx], [preds[i] for i in idx],
                                [fired[i] for i in idx] if fired else None)
                    sweep.append(dict(system=system, snr=snr, split=split, threshold=float(thr), **m))
    if len(runners) == 2:
        for snr in args.snr:
            for split in ("validation", "alle"):
                if any(r["split"] == split for r in sweep):
                    plot_sweep(sweep, split, snr, out / f"sweep_{split}_{snr}.png")

    # Bester Arbeitspunkt je System (max. Accuracy - 2 * Fehlerkennung, auf Validation)
    best = {}
    for (system, snr) in results:
        cand = [r for r in sweep if r["system"] == system and r["snr"] == snr and r["split"] == "validation"]
        if cand:
            best[(system, snr)] = max(cand, key=lambda r: r["accuracy"] - 2 * r["wrong_cmd"])

    keys = list(dict.fromkeys(k for r in summary + sweep for k in r))
    for name, data in (("summary.csv", summary), ("sweep.csv", sweep)):
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, delimiter=";")
            w.writeheader()
            w.writerows(data)

    # ---- 4. Bericht
    cols = ["n", "accuracy", "wrong_cmd", "missed", *[f"recall_{c}" for c in COMMANDS], "cross_trigger", "ms_per_clip", "rtf"]
    lines = [f"# Vosk vs. openWakeWord ({datetime.now():%Y-%m-%d %H:%M})", "",
             f"Daten: `{args.data_dir}` ({len(clips)} Clips)  ",
             f"Arbeitspunkte: Vosk min_conf={args.vosk_min_conf}, oWW Schwelle={args.oww_threshold} "
             f"x {args.oww_consecutive} Frames, VAD={'aus' if args.no_vad else OWW_VAD}, "
             f"Grundrauschen={args.noise_floor}", "",
             "> openWakeWord wurde auf den Trainings-Sprechern trainiert. Massgebend ist **validation**.", "",
             "## Arbeitspunkte", "", "| System | SNR | Split | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 3)]
    for r in summary:
        vals = [fmt(r.get(c, "")) if c not in ("n", "ms_per_clip", "rtf") else
                (f"{r[c]:.1f}" if c == "ms_per_clip" else f"{r[c]:.3f}" if c == "rtf" else str(r[c])) for c in cols]
        lines.append(f"| {r['system']} | {r['snr']} | {r['split']} | " + " | ".join(vals) + " |")
    lines += ["", "## Bester Arbeitspunkt auf Validation (max. Accuracy - 2*Fehlerkennung)", "",
              "| System | SNR | Schwelle | Accuracy | falscher Befehl | verpasst |", "|---|---|---|---|---|---|"]
    for (system, snr), r in best.items():
        lines.append(f"| {system} | {snr} | {r['threshold']} | {fmt(r['accuracy'])} | {fmt(r['wrong_cmd'])} | {fmt(r['missed'])} |")
    lines += ["", "## Fehler (Validation, Arbeitspunkt)", ""]
    for (system, snr), rows in results.items():
        preds, _ = decide(system, rows, op[system])
        errs = [(r, p) for r, p in zip(rows, preds) if r["split"] == "validation" and p != r["label"]]
        lines.append(f"**{system}, SNR {snr}:** " + (", ".join(
            f"`{Path(r['name']).name}` → {p}" + (f" ('{r['text']}')" if r.get("text") else "") for r, p in errs) or "keine"))
        lines.append("")
    report = "\n".join(lines)
    (out / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nErgebnisse: {out}")


if __name__ == "__main__":
    main()
