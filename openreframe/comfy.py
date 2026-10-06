"""Thin layer over the official Comfy SDK: load a workflow template, fill its inputs, run it,
download its outputs. Works against Comfy Cloud (default), a local ComfyUI behind
`comfy-api-proxy`, or a Comfy API deployment: only COMFY_BASE_URL changes."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from comfy_sdk import Comfy

from . import __version__

WORKFLOWS = Path(__file__).parent / "workflows"


def client(api_key: str | None = None) -> Comfy:
    """COMFY_BASE_URL picks the target. The key comes from the caller (the web app passes each visitor's
    own) or from COMFY_API_KEY; it is never written anywhere by openreframe."""
    if api_key:
        return Comfy(api_key=api_key, client_info=f"openreframe/{__version__}")
    return Comfy(client_info=f"openreframe/{__version__}")


def template(name: str) -> dict:
    return json.loads((WORKFLOWS / name).read_text(encoding="utf-8"))


def build(c: Comfy, graph: dict, inputs: dict[tuple[str, str], object]):
    """`inputs` maps (node_id, input_name) -> value. Paths (Path) are uploaded as assets."""
    wf = c.workflows.from_json(copy.deepcopy(graph))
    for (node, name), value in inputs.items():
        if isinstance(value, Path):
            value = c.assets.from_file(str(value))
        wf.set_input(node, name, value)
    return wf


def download(job, node_id: str, dest: Path) -> list[Path]:
    """Save every output of `node_id` into `dest` (a file path for a single output)."""
    saved = []
    outs = job.get_outputs(node_id)
    for k, out in enumerate(outs):
        target = dest if len(outs) == 1 else dest.with_name(f"{dest.stem}_{k}{dest.suffix}")
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_file(str(target))
        saved.append(target)
    return saved
