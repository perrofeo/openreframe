"""Where to put the vertical window in each shot, and how wide it should be.

Interest per column comes from the detector boxes (people, cars...) and faces; motion only breaks
ties. Motion alone fails on tracking shots: when the camera follows a car, what moves on screen is
the background, and a motion-driven crop drifts onto the buildings.
"""
from __future__ import annotations

import numpy as np
import cv2

SUBJECTS = {"person", "car", "truck", "bus", "motorcycle", "bicycle", "dog", "horse", "cat", "bird"}
CROP_RATIOS = (3 / 4, 1.0, 4 / 3)   # candidate crop widths, as width / source height
COVER = 0.70                        # share of the interest the crop must hold in 75 % of the frames
FACE_MIN_SCORE = 0.7                # MediaPipe: false positives sat at 0.50-0.68, real faces at 0.76-0.96
FACE_SHOT = 0.30                    # a shot "has a face" if one is found in this share of its frames
FACE_MARGIN = 0.15                  # keep the face this far (share of crop width) from the crop edges


def _even(x: float) -> int:
    return int(round(x / 2)) * 2


def motion_profile(frames: list[np.ndarray]) -> np.ndarray:
    g = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 3) for f in frames]
    n, w = len(g), g[0].shape[1]
    mov = np.zeros((n, w), np.float32)
    for i in range(1, n):
        mov[i] = np.abs(g[i] - g[i - 1]).mean(axis=0)
    if n > 1:
        mov[0] = mov[1]
    return np.maximum(mov - np.median(mov, axis=1, keepdims=True), 0)   # drop uniform (camera) motion


def plan_shot(frames, boxes, faces, out_w: int, out_h: int, crop_only: bool = False, keep_most: bool = False) -> dict:
    """boxes/faces: per frame, lists of {x, y, width, height, label, score} in source pixels.
    crop_only: the director's call for a shot the model cannot extend well: a plain crop at the target
    ratio, still following who matters, with nothing invented.
    keep_most: when widening (vertical -> horizontal) the narrowest crop turns the shot into a zoomed-in
    close-up and loses the hands and the table; there the widest candidate wins and the sides are invented."""
    n = len(frames)
    H, W = frames[0].shape[:2]
    mov = motion_profile(frames)

    face = []
    for i in range(n):
        c = [b for b in faces[i] if b.get("score", 1) >= FACE_MIN_SCORE]
        if c:
            b = max(c, key=lambda b: b["width"])
            face.append(np.array([b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]], np.float32))
        else:
            face.append(None)
    has_face = np.mean([f is not None for f in face]) >= FACE_SHOT
    if has_face:
        seen = [i for i, f in enumerate(face) if f is not None]
        face = [face[min(seen, key=lambda j: abs(j - i))] for i in range(n)]

    interest = mov / (mov.max(axis=1, keepdims=True) + 1e-6) * 0.2
    with_boxes = 0
    for i in range(n):
        prof = np.zeros(W, np.float32)
        for b in boxes[i]:
            if b["label"] not in SUBJECTS or b["score"] < 0.4 or b["width"] * b["height"] > 0.5 * W * H:
                continue                    # giant boxes out: a car interior is detected as one "car"
            x0, x1 = int(max(0, b["x"])), int(min(W, b["x"] + b["width"]))
            if x1 <= x0:
                continue
            m = float(mov[i, x0:x1].mean() / (mov[i].max() + 1e-6))
            prof[x0:x1] += b["score"] * np.sqrt(b["width"] * b["height"]) * (1 + 3 * m) / (x1 - x0)
        if prof.any():
            with_boxes += 1
            interest[i] = prof / prof.sum() + interest[i] / (interest[i].sum() + 1e-6) * 0.1
    interest = np.stack([interest[max(0, i - 6):i + 7].mean(axis=0) for i in range(n)])   # boxes flicker

    crop_w = None
    for r in (reversed(CROP_RATIOS) if keep_most else CROP_RATIOS):
        w = min(W, _even(H * r))
        held = []
        for i in range(n):
            ac = np.concatenate([[0], np.cumsum(interest[i])])
            held.append((ac[w:] - ac[:-w]).max() / (ac[-1] + 1e-6) if w < W else 1.0)
        if np.percentile(held, 25) >= COVER:
            crop_w = w
            break
    crop_w = crop_w or min(W, _even(H * CROP_RATIOS[-1]))
    if has_face:
        widest = max(f[2] - f[0] for f in face)
        crop_w = min(min(W, _even(H * CROP_RATIOS[-1])), max(crop_w, min(W, _even(widest / 0.5))))
    if crop_only:
        crop_w = min(W, _even(H * out_w / out_h))
    crop_h = _even(H * out_w / crop_w)          # height of the real pixels once scaled to out_w
    if out_h - crop_h < 16:                     # a few pixels short is rounding: stretch, do not invent them
        crop_h = out_h
    crop_h = min(crop_h, out_h)

    margin = FACE_MARGIN * crop_w
    x0s = np.arange(W - crop_w + 1)
    xs = []
    for i in range(n):
        pf = interest[max(0, i - 12):i + 13].mean(axis=0)
        ac = np.concatenate([[0], np.cumsum(pf)])
        inside = ac[crop_w:] - ac[:-crop_w]
        center = 1 - 0.3 * np.abs((x0s + crop_w / 2) - W / 2) / (W / 2)
        score = inside / (inside.max() + 1e-6) * 0.7 + center * 0.3
        if has_face:
            fx0, fx1 = face[i][0], face[i][2]
            ok = (x0s <= fx0 - margin) & (x0s + crop_w >= fx1 + margin)
            score = np.where(ok, score, -1) if ok.any() else -np.abs((x0s + crop_w / 2) - (fx0 + fx1) / 2)
        xs.append(float(x0s[int(np.argmax(score))]))
    xs = np.array(xs)
    v = max(3, min(25, (n // 2) * 2 - 1))       # smoothing scaled to the shot length
    if n >= 3:
        xs = np.convolve(np.pad(xs, v // 2, mode="edge"), np.ones(v) / v, mode="valid")

    face_y = [float((face[i][1] + face[i][3]) / 2) * crop_h / H for i in range(n)] if has_face else []
    return {"crop_w": int(crop_w), "crop_h": int(crop_h), "x0": [round(float(x), 1) for x in xs],
            "has_face": bool(has_face), "frames_with_boxes": with_boxes,
            "face_y_max": round(max(face_y), 1) if face_y else None,
            "faces": [[round(float(v)) for v in f] if f is not None else None for f in face]}


def crop_frames(frames, x0: list[float], crop_w: int, out_w: int, crop_h: int) -> list[np.ndarray]:
    return [cv2.resize(f[:, int(round(x)):int(round(x)) + crop_w], (out_w, crop_h), interpolation=cv2.INTER_AREA)
            for f, x in zip(frames, x0)]
