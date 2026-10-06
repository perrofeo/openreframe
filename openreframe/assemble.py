"""Put the shots back together: paste the real pixels over the AI result, join, add the original
audio, and derive the 4:5 and 1:1 versions from the 9:16 one."""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from .video import read_mp4

FEATHER = 16   # px of blend at each seam when pasting the original back


def paste_back(ai_frames, real_frames, top: int, bottom: int) -> list[np.ndarray]:
    crop_h = real_frames[0].shape[0]
    alpha = np.ones(crop_h, np.float32)
    ramp = np.linspace(0, 1, FEATHER, dtype=np.float32)
    if top:
        alpha[:FEATHER] = ramp
    if bottom:
        alpha[-FEATHER:] = ramp[::-1]
    alpha = alpha[:, None, None]
    out = []
    for ai, real in zip(ai_frames, real_frames):
        f = ai.copy()
        band = f[top:top + crop_h].astype(np.float32)
        f[top:top + crop_h] = np.clip(alpha * real.astype(np.float32) + (1 - alpha) * band, 0, 255).astype(np.uint8)
        out.append(f)
    return out


def window(out_h: int, win_h: int, top: int, crop_h: int, face_y: float | None) -> int:
    """Top row of a shorter (4:5, 1:1) window cut from the 9:16 frame: centred on the real pixels,
    moved if needed so the face stays inside with some headroom."""
    y = top + crop_h / 2 - win_h / 2
    if face_y is not None:
        fy = top + face_y
        y = min(max(y, fy - 0.75 * win_h), fy - 0.25 * win_h)
    return int(round(min(max(y, 0), out_h - win_h)))


class Writer:
    def __init__(self, path: Path, w: int, h: int, fps: float):
        self.path = path
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", "14",
                                   "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait():
            raise RuntimeError(f"ffmpeg failed writing {self.path}")


def add_audio(silent: Path, source: Path, dest: Path, has_audio: bool) -> None:
    if not has_audio:
        silent.replace(dest)
        return
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", str(source), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", str(dest)], check=True)
    silent.unlink()


def side_by_side(source: Path, vertical: Path, dest: Path, out_h: int) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-i", str(vertical), "-filter_complex",
                    f"[0:v]scale=-2:{out_h}[o];[o][1:v]hstack=2,format=yuv420p[v]", "-map", "[v]", "-map", "1:a?",
                    "-c:v", "libx264", "-crf", "18", "-c:a", "copy", "-shortest", str(dest)], check=True)


def ai_frames_of(path: Path, n: int):
    frames = read_mp4(path, n)
    if len(frames) < n:
        raise RuntimeError(f"{path} has {len(frames)} frames, expected {n}")
    return frames


def debug_view(source: Path, vertical: Path, dest: Path, shots: list, boxes: list, out_w: int, out_h: int,
               fps: float, has_audio: bool, subjects: set) -> None:
    """The edit with what the tool decided (cuts, boxes, face, crop window, how much is invented) next to
    the vertical result with its seams dashed and the invented bands labelled."""
    src = cv2.VideoCapture(str(source))
    ver = cv2.VideoCapture(str(vertical))
    font = cv2.FONT_HERSHEY_DUPLEX
    silent = dest.with_name(dest.stem + "_silent.mp4")
    writer = None
    for k, s in enumerate(shots):
        cw, ch, top, bottom = s["crop_w"], s["crop_h"], s["top"], s["bottom"]
        for i in range(s["n"]):
            ok, f = src.read()
            ok2, v = ver.read()
            if not (ok and ok2):
                raise RuntimeError("debug view: ran out of frames")
            H, W = f.shape[:2]
            x0 = int(round(s["x0"][i]))
            f[:, :x0] = (f[:, :x0] * 0.35).astype(np.uint8)
            f[:, x0 + cw:] = (f[:, x0 + cw:] * 0.35).astype(np.uint8)
            for b in boxes[s["start"] + i]:
                if b["label"] in subjects and b["score"] >= 0.4 and b["width"] * b["height"] <= 0.5 * W * H:
                    p0 = (int(b["x"]), int(b["y"]))
                    p1 = (int(b["x"] + b["width"]), int(b["y"] + b["height"]))
                    cv2.rectangle(f, p0, p1, (80, 220, 80), 2)
                    cv2.putText(f, b["label"], (p0[0] + 4, p0[1] + 22), font, 0.7, (80, 220, 80), 2)
            face = s["faces"][i] if s["has_face"] else None
            if face:
                cv2.rectangle(f, (face[0], face[1]), (face[2], face[3]), (0, 210, 255), 2)
            cv2.rectangle(f, (x0, 1), (x0 + cw - 1, H - 2), (255, 255, 255), 3)
            if i < 8:
                cv2.putText(f, f"CUT {k + 1}/{len(shots)}", (24, 56), font, 1.5, (0, 0, 255), 3)
            cv2.putText(f, f"crop {cw / H:.2f}:1  |  AI fills {1 - ch / out_h:.0%}  (top {top}px, bottom {bottom}px)",
                        (24, H - 24), font, 0.8, (255, 255, 255), 2)
            left = cv2.resize(f, (int(round(W * out_h / H / 2)) * 2, out_h), interpolation=cv2.INTER_AREA)
            for y in ([top] if top else []) + ([top + ch] if bottom else []):
                for xx in range(0, out_w, 16):
                    cv2.line(v, (xx, y), (min(xx + 8, out_w - 1), y), (255, 255, 255), 1)
            if top > 40:
                cv2.putText(v, "AI", (12, 32), font, 0.8, (255, 255, 255), 2)
            if bottom > 40:
                cv2.putText(v, "AI", (12, out_h - 14), font, 0.8, (255, 255, 255), 2)
            frame = np.hstack([left, v])
            if writer is None:
                writer = Writer(silent, frame.shape[1], frame.shape[0], fps)
            writer.write(frame)
    writer.close()
    add_audio(silent, source, dest, has_audio)
