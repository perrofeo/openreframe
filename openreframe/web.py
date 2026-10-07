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


def _reframe(video_path, api_key, ratios, captioner):
    import gradio as gr
    # gr.Error shows the message to the visitor; a plain exception only shows "Error" on every box
    if not video_path:
        raise gr.Error("Upload a video first.")
    if not api_key or not api_key.strip():
        raise gr.Error("Paste your Comfy API key (https://platform.comfy.org/profile/api-keys). "
                       "It is used for this run only.")
    if not ratios:
        raise gr.Error("Pick at least one ratio.")
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
            run = Run(local, work / "out", ratios=tuple(ratios), api_key=api_key.strip(),
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
    pick = lambda name: str(out / name) if (out / name).exists() else None  # noqa: E731
    yield ("\n".join(lines[-40:]), pick("input_compare.mp4"), pick("input_9x16.mp4"), pick("input_4x5.mp4"),
           pick("input_1x1.mp4"))


def build_app(examples: Path):
    import gradio as gr

    with gr.Blocks(title="openreframe") as app:
        gr.Markdown("# openreframe\nTurn a finished 16:9 edit into vertical, **shot by shot**, with open models "
                    "on ComfyUI. [Code](https://github.com/perrofeo/openreframe)")
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
                    video = gr.Video(label="Your horizontal edit", sources=["upload"])
                    key = gr.Textbox(label="Comfy API key", type="password", placeholder="comfyui-...")
                    ratios = gr.CheckboxGroup(["9:16", "4:5", "1:1"], value=["9:16", "4:5", "1:1"], label="Ratios")
                    captioner = gr.Dropdown(["Qwen3-VL-4B-Instruct", "Qwen3-VL-8B-Instruct", "Qwen3-VL-2B-Instruct"],
                                            value="Qwen3-VL-4B-Instruct", label="Shot captioner")
                    go = gr.Button("Reframe", variant="primary")
                with gr.Column():
                    log = gr.Textbox(label="Progress", lines=14, max_lines=14, autoscroll=True)
                    compare = gr.Video(label="Original | 9:16")
            with gr.Row():
                v916 = gr.Video(label="9:16")
                v45 = gr.Video(label="4:5")
                v11 = gr.Video(label="1:1")
            go.click(_reframe, [video, key, ratios, captioner], [log, compare, v916, v45, v11])
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
