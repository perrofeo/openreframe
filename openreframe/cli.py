"""openreframe INPUT.mp4 [-o OUT] [--to 9:16|16:9|...] — reframe a finished edit to another ratio, shot by shot."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import EXTRA, STEPS, Run


def main(argv=None):
    ap = argparse.ArgumentParser(prog="openreframe", description=__doc__)
    ap.add_argument("input", type=Path, help="horizontal video (a finished edit, with cuts and sound)")
    ap.add_argument("-o", "--out", type=Path, help="output folder (default: <input name>_openreframe)")
    ap.add_argument("--to", default="9:16", help="target ratio: 9:16 (default), 16:9, 4:5, 1:1, 4:3...")
    ap.add_argument("--ratios", help="extra cuts from the same render (default: 4:5,1:1 for 9:16; 4:3,1:1 for 16:9)")
    ap.add_argument("--size", help="render size WxH, multiples of 16 (default: long side 960)")
    ap.add_argument("--seed", type=int, default=6061003)
    ap.add_argument("--steps", default=",".join(STEPS), help=f"run only some steps: {','.join(STEPS)}")
    ap.add_argument("--redo", default="", help="run this step again, and every step after it")
    ap.add_argument("--debug-view", action="store_true", help="also render the decisions over the edit")
    ap.add_argument("--reroll", action="append", default=[], metavar="N",
                    help="render shot N again with a new seed (only that shot is re-rendered)")
    ap.add_argument("--crop-only", action="append", default=[], metavar="N",
                    help="shot N gets a plain crop, nothing invented (for shots the model cannot extend well)")
    ap.add_argument("--recaption", action="store_true", help="describe every shot again (otherwise captions are kept)")
    ap.add_argument("--captioner", help="Qwen-VL model for the shot descriptions (default Qwen3-VL-4B-Instruct; "
                                        "Qwen3-VL-2B-Instruct for GPUs with 8 GB)")
    ap.add_argument("--note", action="append", default=[], metavar='N="TEXT"',
                    help='add something the model cannot know to shot N\'s prompt, e.g. 2="His car is black."')
    a = ap.parse_args(argv)
    if not a.input.exists():
        sys.exit(f"{a.input}: not found")
    size = None
    if a.size:
        size = tuple(int(x) for x in a.size.lower().split("x"))
        if size[0] % 16 or size[1] % 16:
            sys.exit("--size must be multiples of 16")
    ratios = None
    if a.ratios:
        ratios = (a.to,) + tuple(r.strip() for r in a.ratios.split(",") if r.strip() != a.to)
    run = Run(a.input, a.out or a.input.with_name(f"{a.input.stem}_openreframe"), target=a.to, size=size,
              seed=a.seed, ratios=ratios, captioner=a.captioner)
    if a.crop_only:
        co = run.state.setdefault("crop_only", [])
        co += [str(int(n)) for n in a.crop_only if str(int(n)) not in co]
        if "analyze" in run.state["done"]:
            run.reset("analyze")                  # renders of the other shots are reused
    if a.recaption:
        run.recaption()
    if a.reroll:
        rr = run.state.setdefault("rerolls", {})
        for n in a.reroll:
            rr[str(int(n))] = rr.get(str(int(n)), 0) + 1
        if "fill" in run.state["done"]:
            run.reset("fill")                     # only the rerolled shots render again
    for step in filter(None, a.redo.split(",")):
        run.reset(step)
    if a.note:
        notes = run.state.setdefault("notes", {})
        for item in a.note:
            n, _, text = item.partition("=")
            notes[n.strip()] = text.strip().strip('"')
        if "split" in run.state["done"]:
            run.reset("split")                    # only the shots whose prompt changed render again
    steps = [s for s in a.steps.split(",") if s in STEPS + EXTRA]
    if a.debug_view and "debug" not in steps:
        steps.append("debug")
        if "debug" in run.state["done"]:
            run.state["done"].remove("debug")
    run.run(steps)


if __name__ == "__main__":
    main()
