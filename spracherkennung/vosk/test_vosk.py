#!/usr/bin/env python3
"""
Sprachbefehle mit Vosk (Ersatz für die openWakeWord-Modelle Brav / Hier / Sitz / Such).

Funktionsweise
--------------
* Vosk (deutsches Modell) erkennt nur ein kleines Vokabular: die Befehlswörter plus ähnlich
  klingende Wörter (Witz, Bitz, Blitz, ...) als "Fallen" und "[unk]" für alles andere.
  Dadurch landet z. B. "Witz" nicht fälschlich bei "Sitz".
* Ein Befehl gilt nur dann als erkannt, wenn die ganze Äusserung nur aus (demselben) Befehlswort
  besteht und die Vosk-Konfidenz über der Schwelle liegt. Ähnliche Wörter, "[unk]" und Mischungen
  werden verworfen.
* Nach einem Befehl gilt eine kurze Sperrzeit (Cooldown), damit er nicht doppelt auslöst.

Installation (Windows / Linux / Raspberry Pi):
    pip install vosk sounddevice numpy

Nutzung:
    python vosk_befehle.py                        # Mikrofon, Modell wird beim ersten Start geladen
    python vosk_befehle.py --list-devices         # Mikrofone auflisten
    python vosk_befehle.py --device 2             # bestimmtes Mikrofon
    python vosk_befehle.py --model-path pfad/zum/vosk-model-small-de-0.15
    python vosk_befehle.py --wav aufnahmen/Sitz   # WAV-Dateien oder Ordner testen (kein Mikrofon)

Hinweis: Das GROSSE deutsche Modell (vosk-model-de-0.21) unterstützt keine Grammatik zur Laufzeit.
Verwende das kleine Modell (vosk-model-small-de-0.15), das ist auch der Standard.

Einbindung in ein eigenes Skript:
    listener = VoskBefehle(on_command=lambda wort, conf: print(wort, conf))
    listener.start()      # läuft in einem eigenen Thread
    ...
    listener.stop()
"""
import argparse
import json
import queue
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000

# ------------------------------------------------------------------ Einstellungen
COMMANDS = ["brav", "hier", "sitz", "such"]

# Ähnlich klingende Wörter und andere Kommandos. Sie stehen im Vokabular, damit Vosk sie als
# solche erkennt, und werden danach verworfen. Beliebig erweiterbar (Kleinschreibung).
# Wörter, die das Modell nicht kennt, ignoriert Vosk (Warnung im Log). Sie fallen dann unter [unk].
DISTRACTORS = [
    # ähnlich zu "sitz"
    "witz", "bitz", "blitz", "spitz", "ritz", "fritz", "schmitz", "sitzt", "sitzen", "sitzung",
    "setz", "satz", "schatz", "sitze", "platz", "ritze", "hitze", "spritz", "glitz",
    # ähnlich zu "such"
    "suche", "suchen", "sucht", "buch", "fluch", "tuch", "zug", "besuch", "bruch", "zucht",
    "zuck", "suff", "sucre",
    # ähnlich zu "hier"
    "bier", "tier", "dir", "wir", "mir", "ihr", "hierher", "hiermit", "papier", "vier", "für",
    # ähnlich zu "brav"
    "bravo", "graf", "schaf", "draht", "brauch", "braun", "brave", "raf", "grav",
    # andere Kommandos / häufige Alltagswörter
    "komm", "bleib", "aus", "nein", "gut", "stopp", "warte", "los", "pfui", "hallo", "danke",
    "nimm", "auf", "ja", "ok", "okay",
]

MIN_CONF = 0.80       # Mindest-Konfidenz von Vosk (0..1) für jedes erkannte Wort
COOLDOWN_S = 1.5      # Sperrzeit nach einem Befehl
BLOCK_SIZE = 4000     # 0.25 s pro Audioblock


def build_vocabulary(commands=COMMANDS, distractors=DISTRACTORS):
    words = list(dict.fromkeys([*commands, *distractors]))   # eindeutig, Reihenfolge erhalten
    return words + ["[unk]"]


def evaluate_result(result, commands=COMMANDS, min_conf=MIN_CONF):
    """Bewertet ein Vosk-Endergebnis (dict mit "text" und "result").
    Gibt (befehl, konfidenz, grund) zurück. befehl ist None, wenn die Äusserung verworfen wird."""
    text = (result.get("text") or "").strip()
    if not text:
        return None, 0.0, "leer"
    words = result.get("result")
    if words:
        tokens = [w["word"] for w in words]
        confs = [float(w.get("conf", 1.0)) for w in words]
    else:                                   # Modell ohne Wortdetails
        tokens, confs = text.split(), [1.0] * len(text.split())
    if any(t not in commands for t in tokens):
        return None, min(confs), f"verworfen (enthält Nicht-Befehl): {text}"
    if len(set(tokens)) != 1:
        return None, min(confs), f"verworfen (verschiedene Befehle): {text}"
    conf = min(confs)
    if conf < min_conf:
        return None, conf, f"verworfen (Konfidenz {conf:.2f} < {min_conf}): {text}"
    return tokens[0], conf, "ok"


class VoskBefehle:
    def __init__(self, on_command, model_path=None, device=None, commands=COMMANDS,
                 distractors=DISTRACTORS, min_conf=MIN_CONF, cooldown=COOLDOWN_S, verbose=True):
        from vosk import KaldiRecognizer, Model, SetLogLevel
        SetLogLevel(-1)
        self.on_command = on_command
        self.commands, self.min_conf, self.cooldown = list(commands), min_conf, cooldown
        self.device, self.verbose = device, verbose
        self.vocab = build_vocabulary(commands, distractors)
        self.model = Model(model_path=str(model_path)) if model_path else Model(lang="de")
        self._Recognizer = KaldiRecognizer
        self.rec = self._new_recognizer()
        self._last = 0.0
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._thread = None

    def _new_recognizer(self):
        rec = self._Recognizer(self.model, SAMPLE_RATE, json.dumps(self.vocab, ensure_ascii=False))
        rec.SetWords(True)
        return rec

    # ---- Verarbeitung eines Endergebnisses
    def _handle(self, result):
        cmd, conf, why = evaluate_result(result, self.commands, self.min_conf)
        if self.verbose and why != "leer":
            print(f"  [{why}]" if cmd is None else f"  [erkannt: {cmd} ({conf:.2f})]")
        if cmd is None:
            return None
        now = time.monotonic()
        if now - self._last < self.cooldown:
            if self.verbose:
                print("  [Cooldown, ignoriert]")
            return None
        self._last = now
        self.on_command(cmd, conf)
        return cmd

    def feed(self, pcm_bytes):
        """Audio (16 kHz, mono, int16, Bytes) einspeisen. Gibt den ausgelösten Befehl oder None zurück."""
        if self.rec.AcceptWaveform(pcm_bytes):
            return self._handle(json.loads(self.rec.Result()))
        return None

    def flush(self):
        return self._handle(json.loads(self.rec.FinalResult()))

    # ---- Mikrofon
    def _audio_cb(self, indata, frames, t, status):
        if status and self.verbose:
            print("Audio-Status:", status, file=sys.stderr)
        self._q.put(bytes(indata))

    def _run(self):
        import sounddevice as sd
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, dtype="int16",
                               channels=1, device=self.device, callback=self._audio_cb):
            if self.verbose:
                print("Höre zu. Befehle:", ", ".join(self.commands), "(Strg+C zum Beenden)")
            while not self._stop.is_set():
                try:
                    data = self._q.get(timeout=0.2)
                except queue.Empty:
                    continue
                self.feed(data)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


# ------------------------------------------------------------------ WAV-Test ohne Mikrofon
def read_wav_16k(path):
    with wave.open(str(path), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError(f"{path}: nur 16-Bit-PCM unterstützt")
    x = np.frombuffer(raw, np.int16).astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SAMPLE_RATE:
        n = int(round(len(x) * SAMPLE_RATE / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return np.clip(x, -32768, 32767).astype(np.int16)


def test_wavs(listener, paths):
    files = []
    for p in paths:
        p = Path(p)
        files += sorted(p.rglob("*.wav")) if p.is_dir() else [p]
    stats = {}
    for f in files:
        listener.rec = listener._new_recognizer()
        listener._last = 0.0
        x = read_wav_16k(f)
        x = np.concatenate([np.zeros(SAMPLE_RATE // 2, np.int16), x, np.zeros(SAMPLE_RATE, np.int16)])  # Stille für das Satzende
        hits = []
        listener.on_command = lambda c, conf: hits.append(c)
        for i in range(0, len(x), BLOCK_SIZE):
            listener.feed(x[i:i + BLOCK_SIZE].tobytes())
        listener.flush()
        got = hits[0] if hits else "-"
        print(f"{f.parent.name}/{f.name}: {got}")
        stats.setdefault(f.parent.name.lower(), []).append(got)
    print("\nZusammenfassung je Ordner (Ordnername = erwartetes Wort):")
    for folder, res in stats.items():
        ok = sum(1 for r in res if r == folder)
        print(f"  {folder}: {ok}/{len(res)} richtig, falsch: {sum(1 for r in res if r not in ('-', folder))}, nichts: {res.count('-')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", help="Ordner mit dem Vosk-Modell (sonst automatischer Download)")
    ap.add_argument("--device", help="Mikrofon (Nummer oder Name), siehe --list-devices")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--min-conf", type=float, default=MIN_CONF)
    ap.add_argument("--cooldown", type=float, default=COOLDOWN_S)
    ap.add_argument("--wav", nargs="+", help="WAV-Dateien oder Ordner testen statt Mikrofon")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    device = int(args.device) if args.device and args.device.isdigit() else args.device

    def on_command(cmd, conf):
        print(f">>> BEFEHL: {cmd.upper()}  (Konfidenz {conf:.2f})")
        # Hier den Roboter / das PREN-Skript ansteuern.

    listener = VoskBefehle(on_command, model_path=args.model_path, device=device,
                           min_conf=args.min_conf, cooldown=args.cooldown)
    if args.wav:
        test_wavs(listener, args.wav)
        return
    listener.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()


if __name__ == "__main__":
    main()