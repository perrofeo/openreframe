"""Video I/O helpers: probing, cut detection, reading one shot at a time, encoding."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
                          "-show_entries", "format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    s = info["streams"][0]
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    has_audio = bool(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                                     "stream=index", "-of", "csv=p=0", str(path)],
                                    capture_output=True, text=True).stdout.strip())
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": num / den,
            "frames": int(s["nb_read_frames"]), "audio": has_audio}


def detect_cuts(path: Path, fps: float, frames: int, threshold: float = 8.0, min_len: int = 3) -> list[tuple[int, int]]:
    """Hard cuts with ffmpeg's scdet. Returns (first_frame, length) per shot."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
                          f"scdet=threshold={threshold},metadata=print:key=lavfi.scd.time:file=-",
                          "-f", "null", "-"], capture_output=True, text=True)
    times = [float(line.split("=")[1]) for line in out.stdout.splitlines() if "lavfi.scd.time=" in line]
    edges = [0] + sorted({round(t * fps) for t in times}) + [frames]
    shots = []
    for a, b in zip(edges, edges[1:]):
        if b - a < min_len and shots:            # a flash, not a shot: glue it to the previous one
            shots[-1] = (shots[-1][0], shots[-1][1] + b - a)
        elif b > a:
            shots.append((a, b - a))
    return shots


def read_shots(path: Path, shots: list[tuple[int, int]]):
    """Yield (index, frames) for each shot, keeping only one shot in memory."""
    cap = cv2.VideoCapture(str(path))
    for k, (start, n) in enumerate(shots):
        frames = []
        for _ in range(n):
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        yield k, frames
    cap.release()


def write_mp4(frames: list[np.ndarray], path: Path, fps: float, crf: int = 12) -> None:
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                          "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
                          str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError(f"ffmpeg failed writing {path}")


def read_mp4(path: Path, limit: int | None = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out = []
    while limit is None or len(out) < limit:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def proxy(path: Path, dest: Path, max_width: int = 960) -> float:
    """Downscaled copy for detection uploads (Cloud single uploads top out around 100 MB).
    Returns the factor to scale proxy coordinates back to the source."""
    w = probe(path)["width"]
    if w <= max_width:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-an", "-c:v", "libx264", "-crf", "18",
                        "-pix_fmt", "yuv420p", str(dest)], check=True)
        return 1.0
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-an", "-vf", f"scale={max_width}:-2",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(dest)], check=True)
    return w / max_width
