"""openreframe-web: a small Gradio app.

- Examples: finished results, free to watch.
- Try it: upload an edit and paste YOUR Comfy API key. The key lives only in memory for that run; it
  is never logged, stored or written to the run folder. The GPU time is billed to the key's owner.

    pip install "openreframe[web]"
    OPENREFRAME_EXAMPLES=./examples openreframe-web --port 7860
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

from .pipeline import Run
from .video import probe

MAX_SECONDS = 60
MAX_MB = 200


def _examples(folder: Path) -> list[tuple[str, str]]:
    """Every *_compare.mp4 in the folder, titled from its file name."""
    if not folder.is_dir():
        return []
    return [(str(p), p.stem.replace("_compare", "").replace("_", " ")) for p in sorted(folder.glob("*_compare.mp4"))]


TARGETS = {"To vertical (9:16)": ("9:16", ["9:16", "4:5", "1:1"]),
           "To horizontal (16:9)": ("16:9", ["16:9", "4:3", "1:1"])}


def _reframe(video_path, api_key, direction, ratios, captioner):
    import gradio as gr
    # gr.Error shows the message to the visitor; a plain exception only shows "Error" on every box
    if not video_path:
        raise gr.Error("Upload a video first.")
    if not api_key or not api_key.strip():
        raise gr.Error("Paste your Comfy API key (https://platform.comfy.org/profile/api-keys). "
                       "It is used for this run only.")
    target, allowed = TARGETS[direction]
    ratios = [r for r in allowed if r in (ratios or [])]
    if target not in ratios:
        ratios = [target] + ratios
    src = Path(video_path)
    if src.stat().st_size > MAX_MB * 1024 * 1024:
        raise gr.Error(f"Keep it under {MAX_MB} MB.")
    info = probe(src)
    if info["frames"] / info["fps"] > MAX_SECONDS:
        raise gr.Error(f"This demo takes edits up to {MAX_SECONDS} s; the command line has no limit.")
    work = Path(tempfile.mkdtemp(prefix="openreframe_"))
    local = work / ("input" + src.suffix.lower())
    shutil.copy(src, local)
    lines: list[str] = []
    result: dict = {}

    def job():
        try:
            run = Run(local, work / "out", target=target, ratios=tuple(ratios), api_key=api_key.strip(),
                      logger=lines.append, captioner=captioner)
            run.run()
            result["ok"] = True
        except Exception as exc:              # the message, never the key
            result["error"] = f"{type(exc).__name__}: {exc}"

    t = threading.Thread(target=job, daemon=True)
    t.start()
    while t.is_alive():
        yield "\n".join(lines[-40:]), None, None, None, None
        time.sleep(2)
    if "error" in result:
        lines.append("ERROR " + result["error"])
        yield "\n".join(lines[-40:]), None, None, None, None
        return
    out = work / "out"
    vids = [gr.update(value=str(out / f"input_{r.replace(':', 'x')}.mp4"), label=r) for r in ratios]
    vids += [gr.update(value=None, label="")] * (3 - len(vids))
    yield ("\n".join(lines[-40:]), str(out / "input_compare.mp4"), *vids[:3])


def build_app(examples: Path):
    import gradio as gr

    with gr.Blocks(title="openreframe") as app:
        gr.Markdown("# openreframe\nReframe a finished edit to another ratio, **shot by shot**, with open models "
                    "on ComfyUI: horizontal to vertical, or vertical to horizontal. "
                    "[Code](https://github.com/perrofeo/openreframe)")
        with gr.Tab("Examples"):
            ex = _examples(examples)
            if not ex:
                gr.Markdown("No examples here yet.")
            for path, title in ex:
                gr.Video(value=path, label=title, interactive=False)
        with gr.Tab("Try it"):
            gr.Markdown(f"Up to {MAX_SECONDS} s and {MAX_MB} MB. Runs on **your** Comfy Cloud credits "
                        "(API access needs a paid plan): about a minute of GPU per 20 s of edit. "
                        "Your key is used for this run only and is never stored.")
            with gr.Row():
                with gr.Column():
                    video = gr.Video(label="Your edit", sources=["upload"])
                    key = gr.Textbox(label="Comfy API key", type="password", placeholder="comfyui-...")
                    direction = gr.Radio(list(TARGETS), value="To vertical (9:16)", label="Direction")
                    ratios = gr.CheckboxGroup(TARGETS["To vertical (9:16)"][1], value=TARGETS["To vertical (9:16)"][1],
                                              label="Ratios from the same render")
                    captioner = gr.Dropdown(["Qwen3-VL-4B-Instruct", "Qwen3-VL-8B-Instruct", "Qwen3-VL-2B-Instruct"],
                                            value="Qwen3-VL-4B-Instruct", label="Shot captioner")
                    go = gr.Button("Reframe", variant="primary")
                with gr.Column():
                    log = gr.Textbox(label="Progress", lines=14, max_lines=14, autoscroll=True)
                    compare = gr.Video(label="Original | result")
            with gr.Row():
                v1 = gr.Video(label="9:16")
                v2 = gr.Video(label="4:5")
                v3 = gr.Video(label="1:1")
            direction.change(lambda d: gr.update(choices=TARGETS[d][1], value=TARGETS[d][1]), direction, ratios)
            go.click(_reframe, [video, key, direction, ratios, captioner], [log, compare, v1, v2, v3])
    return app


def main(argv=None):
    ap = argparse.ArgumentParser(prog="openreframe-web")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)
    examples = Path(os.environ.get("OPENREFRAME_EXAMPLES", "examples"))
    build_app(examples).queue(default_concurrency_limit=2).launch(server_name=a.host, server_port=a.port,
                                                                    allowed_paths=[str(examples.resolve())])


if __name__ == "__main__":
    main()
