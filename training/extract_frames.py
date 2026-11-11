#!/usr/bin/env python3
"""
Extrahiert Frames aus einem Video und speichert sie als Bilder (z.B. für YOLO-Training).

Installation:
    pip install opencv-python

Beispiele:
    python extract_frames.py knochen.mp4 frames/
    python extract_frames.py knochen.mp4 frames/ --fps 3
    python extract_frames.py knochen.mp4 frames/ --every 10 --blur-threshold 80
"""

import argparse
from pathlib import Path

import cv2


def blur_score(frame) -> float:
    """Varianz des Laplace-Operators: je kleiner, desto unschärfer."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    parser = argparse.ArgumentParser(description="Video in einzelne Frames zerlegen.")
    parser.add_argument("video", type=Path, help="Pfad zum Eingabevideo")
    parser.add_argument("output", type=Path, help="Ausgabeordner für die Bilder")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--every", type=int, default=None,
                       help="Jeden N-ten Frame speichern")
    group.add_argument("--fps", type=float, default=None,
                       help="Ziel-Frames pro Sekunde (z.B. 3 = 3 Bilder/Sekunde)")
    parser.add_argument("--blur-threshold", type=float, default=0.0,
                        help="Frames mit Schärfe unter diesem Wert überspringen (0 = aus, ~50-100 sinnvoll)")
    parser.add_argument("--max-size", type=int, default=0,
                        help="Längste Kante auf diesen Wert skalieren (0 = Originalgröße)")
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--quality", type=int, default=95, help="JPEG-Qualität (1-100)")
    parser.add_argument("--prefix", default=None,
                        help="Dateinamen-Präfix (Standard: Videoname)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Video konnte nicht geöffnet werden: {args.video}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Schrittweite bestimmen
    if args.every:
        step = max(1, args.every)
    elif args.fps:
        step = max(1, round(video_fps / args.fps))
    else:
        step = 1

    args.output.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.video.stem
    params = [cv2.IMWRITE_JPEG_QUALITY, args.quality] if args.format == "jpg" else []

    print(f"Video: {args.video.name} | {video_fps:.1f} FPS | {total} Frames | Schrittweite: {step}")

    idx = saved = skipped_blur = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % step == 0:
            if args.blur_threshold and blur_score(frame) < args.blur_threshold:
                skipped_blur += 1
            else:
                if args.max_size:
                    h, w = frame.shape[:2]
                    scale = args.max_size / max(h, w)
                    if scale < 1:
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                           interpolation=cv2.INTER_AREA)
                name = f"{prefix}_{idx:06d}.{args.format}"
                cv2.imwrite(str(args.output / name), frame, params)
                saved += 1
        idx += 1

    cap.release()
    print(f"Fertig: {saved} Bilder gespeichert, {skipped_blur} wegen Unschärfe übersprungen.")


if __name__ == "__main__":
    main()