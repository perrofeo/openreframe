"""Where the AI fills: always towards the background, never into what is close to the camera.

Filling below a table edge makes the model invent what is under the table (legs at odd angles,
the wrong shoes); filling the sky above it does not. Depth decides which border is far.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

FACE_MAX = 0.45     # a face's centre never ends up below this share of the frame height
CENTER_MAX = 0.60   # nor does the centre of the real pixels: platforms put captions and buttons low
MIN_BAND = 48       # fill bands thinner than this are not worth it

# Describe only what is visible. An earlier instruction asked for "what is most likely just above and
# just below", and the captioner obliged: an umbrella floating over a street, lightning in a car's
# windscreen. The fill model paints whatever the prompt names; the pixels already give it the context.
CAPTION_INSTRUCTION = (
    "This is one frame from a shot of a film. In one paragraph of plain English, at most 50 words, describe "
    "only what is visible: the place, the time of day, the light, the weather, the surfaces and materials. "
    "If a person is visible, name their clothing in a few words. Do not guess what lies outside the picture, "
    "do not add anything that is not there, and do not mention the camera, the frame or the image.")
PROMPT_TAIL = " Cinematic film, natural light, fine grain."
# Filling above an open sky, the model sometimes builds a roof or a ceiling (a rooftop at night got a flat
# white ceiling). Depth alone cannot tell sky from far walls; Depth Anything 3 mono has a sky mask.
SKY_PHRASE = "Open sky continues above; nothing overhead."
SKY_SHARE = 0.30


def sky_above(sky_png: str) -> bool:
    m = cv2.imread(sky_png, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return False
    band = max(4, int(0.08 * m.shape[0]))
    return float(m[:band].astype(np.float32).mean() / 255) >= SKY_SHARE


def split(depth, total: int, out_h: int, crop_h: int, face_y_max: float | None, safe_zone: bool = True) -> dict:
    """Pixels to fill before and after the real image along the fill axis (above/below for a vertical
    target). `depth` is the Depth Anything render (bright = near) already in that orientation.
    safe_zone: the platform rules (face high, real image not too low) only apply to vertical targets."""
    d = depth.astype(np.float32) / 255 if depth.dtype != np.float32 else depth
    band = max(4, int(0.08 * d.shape[0]))
    near_top, near_bottom = float(d[:band].mean()), float(d[-band:].mean())
    top = int(round(total * float(np.clip(0.5 + 2.0 * (near_bottom - near_top), 0, 1))))
    face_cap = total
    if safe_zone:
        if face_y_max is not None:
            face_cap = max(0, int(FACE_MAX * out_h - face_y_max))
        top = min(top, face_cap, max(0, int(CENTER_MAX * out_h - crop_h / 2)))
    if 0 < total - top < MIN_BAND and face_cap >= total:
        top = total                 # a thin strip under the foreground only invents what is under it
    if 0 < top < MIN_BAND and not safe_zone:
        top = 0
    top -= top % 2
    return {"near_top": round(near_top, 3), "near_bottom": round(near_bottom, 3), "top": top, "bottom": total - top}


def coherent_time_of_day(captions: list[str]) -> list[str]:
    """One edit, one time of day: the captioner sometimes calls a night shot 'late afternoon'."""
    night = sum(bool(re.search(r"\bnight", c or "", re.I)) for c in captions)
    if night <= len(captions) / 2:
        return captions
    day = r"\b(late afternoon sun|late afternoon|cloudy afternoon|afternoon|daytime|daylight|sunlight|sunset|dusk)\b"
    return [re.sub(day, "night", c or "", flags=re.I) for c in captions]
