"""The whole run, step by step, with its state in <work>/plan.json so an interrupted run resumes
without paying twice for a shot that already rendered."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
from pathlib import Path

import cv2

from . import assemble, comfy, fill, framing, video

STEPS = ("detect", "analyze", "look", "split", "fill", "assemble")
EXTRA = ("debug",)                                   # only when asked for
DEFAULT_RATIOS = {"9:16": ("9:16", "4:5", "1:1"), "16:9": ("16:9", "4:3", "1:1")}


def parse_ratio(r: str) -> float:
    w, h = (float(x) for x in r.split(":"))
    return w / h


def base_size(target: str) -> tuple[int, int]:
    """Render size for the target ratio: long side 960, both sides multiples of 16."""
    a = parse_ratio(target)
    if abs(a - 1) < 1e-6:
        return 720, 720
    long_, short = 960, int(round(960 * min(a, 1 / a) / 16)) * 16
    return (short, long_) if a < 1 else (long_, short)


def canon(img, widen: bool):
    """The tool always works as if it cropped across x and invented along y (horizontal -> vertical).
    Going vertical -> horizontal is the same problem turned 90 degrees."""
    return img.transpose(1, 0, *range(2, img.ndim)).copy() if widen else img


def canon_boxes(frames: list, widen: bool) -> list:
    if not widen:
        return frames
    return [[dict(b, x=b["y"], y=b["x"], width=b["height"], height=b["width"]) for b in f] for f in frames]


def log(msg: str) -> None:
    print(msg, flush=True)


class Run:
    def __init__(self, source: Path, out_dir: Path, target: str = "9:16", size=None, seed=6061003, ratios=None,
                 api_key: str | None = None, logger=None, captioner: str | None = None):
        self.source = source.resolve()
        self.out = out_dir.resolve()
        self.work = self.out / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.state_path = self.work / "plan.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.state.setdefault("source", str(self.source))
        self.state.setdefault("target", target)
        w, h = size or base_size(self.state["target"])
        self.state.setdefault("out", {"w": w, "h": h})
        self.state.setdefault("seed", seed)
        self.state.setdefault("done", [])
        self.ratios = tuple(ratios) if ratios else DEFAULT_RATIOS.get(self.state["target"], (self.state["target"],))
        self._client = None
        self._api_key = api_key
        self._log = logger or log
        if captioner:
            self.state["captioner"] = captioner

    # -- bookkeeping ---------------------------------------------------------------------------
    @property
    def c(self):
        if self._client is None:
            self._client = comfy.client(self._api_key)
        return self._client

    def save(self):
        self.state_path.write_text(json.dumps(self.state, indent=1, ensure_ascii=False))

    @property
    def widen(self) -> bool:
        return bool(self.state.get("widen", False))

    def canon_out(self) -> tuple[int, int]:
        """Output size in the working orientation (crop across x, invent along y)."""
        w, h = self.state["out"]["w"], self.state["out"]["h"]
        return (h, w) if self.widen else (w, h)

    def shot_seed(self, k: int) -> int:
        """Each shot has its own seed; --reroll N moves shot N to a new one."""
        return self.state["seed"] + k + 1000 * self.state.get("rerolls", {}).get(str(k + 1), 0)

    def shot_dir(self, k: int) -> Path:
        d = self.work / "shots" / f"{k:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def recaption(self):
        for k, s in enumerate(self.state.get("shots", [])):
            s.pop("caption", None)
            (self.work / "shots" / f"{k:03d}" / "caption.txt").unlink(missing_ok=True)
        self.reset("look")

    def reset(self, step: str):
        """Run a step again, and the steps after it. Renders are reused when their inputs did not change."""
        order = STEPS + EXTRA
        for st in order[order.index(step):]:
            if st in self.state["done"]:
                self.state["done"].remove(st)
        if step == "look":                        # captions are kept; --recaption asks for new ones
            for k in range(len(self.state.get("shots", []))):
                for fname in ("depth.png", "sky.png"):
                    (self.work / "shots" / f"{k:03d}" / fname).unlink(missing_ok=True)
        self.save()

    def run(self, steps=STEPS):
        for step in steps:
            if step in self.state["done"]:
                self._log(f"[{step}] done before, skipping")
                continue
            self._log(f"[{step}]")
            getattr(self, step)()
            self.state["done"].append(step)
            self.save()

    # -- 1. cuts + detection (Comfy) -------------------------------------------------------------
    def detect(self):
        info = video.probe(self.source)
        self.state["probe"] = info
        self.state["widen"] = info["width"] / info["height"] < parse_ratio(self.state["target"])
        self._log(f"  {'widening' if self.widen else 'narrowing'} to {self.state['target']} "
                  f"({self.state['out']['w']}x{self.state['out']['h']})")
        shots = video.detect_cuts(self.source, info["fps"], info["frames"])
        self.state["shots"] = [{"start": a, "n": n} for a, n in shots]
        self._log(f"  {info['width']}x{info['height']} @ {info['fps']:.3f} fps, {info['frames']} frames, {len(shots)} shots")
        proxy = self.work / "proxy.mp4"
        scale = video.proxy(self.source, proxy)
        wf = comfy.build(self.c, comfy.template("detect.json"), {("load_video", "file"): proxy})
        job = self.c.run(wf)
        for node, name in (("save_boxes", "boxes"), ("save_faces", "faces")):
            raw = self.work / f"{name}_proxy.json"
            comfy.download(job, node, raw)
            per_frame = json.loads(raw.read_text())
            if len(per_frame) != info["frames"]:
                raise RuntimeError(f"{name}: {len(per_frame)} frames from the detector, {info['frames']} in the video")
            for frame in per_frame:
                for b in frame:
                    for key in ("x", "y", "width", "height"):
                        b[key] *= scale
            (self.work / f"{name}.json").write_text(json.dumps(per_frame))
        self._log(f"  boxes and faces for {info['frames']} frames")

    # -- 2. framing (local) ---------------------------------------------------------------------
    def analyze(self):
        boxes = canon_boxes(json.loads((self.work / "boxes.json").read_text()), self.widen)
        faces = canon_boxes(json.loads((self.work / "faces.json").read_text()), self.widen)
        cw, ch = self.canon_out()
        fps = self.state["probe"]["fps"]
        shots = self.state["shots"]
        for k, frames in video.read_shots(self.source, [(s["start"], s["n"]) for s in shots]):
            s = shots[k]
            a, n = s["start"], len(frames)
            s["n"] = n
            frames = [canon(f, self.widen) for f in frames]
            only = str(k + 1) in self.state.get("crop_only", [])
            s.update(framing.plan_shot(frames, boxes[a:a + n], faces[a:a + n], cw, ch, crop_only=only,
                                       keep_most=self.widen))
            crops = [canon(c, self.widen) for c in framing.crop_frames(frames, s["x0"], s["crop_w"], cw, s["crop_h"])]
            s["length"] = n + (-(n - 1)) % 4                  # Wan wants 4k+1 frames: repeat the last one
            crops += [crops[-1]] * (s["length"] - n)
            d = self.shot_dir(k)
            video.write_mp4(crops, d / "crop.mp4", fps, crf=10)
            cv2.imwrite(str(d / "middle.png"), crops[n // 2])
            self._log(f"  shot {k + 1:3d}: {n:4d} fr  crop {s['crop_w']}px ({s['crop_w'] / frames[0].shape[0]:.2f}:1)"
                      f"  AI fills {1 - s['crop_h'] / ch:.0%}  {'face' if s['has_face'] else 'subjects/motion'}"
                      f"{'  (crop only)' if only else ''}")

    # -- 3. depth + caption per shot (Comfy, one job) ---------------------------------------------
    LOOK_PARTS = {"depth": (("depth_run", "depth", "save_depth"), "save_depth", "depth.png"),
                  "sky": (("sky_run", "sky", "save_sky"), "save_sky", "sky.png"),
                  "caption": (("caption", "save_caption"), "save_caption", "caption.txt")}
    LOOK_SHARED = {"depth_model", "sky_model"}

    def look(self):
        """One Comfy job for every shot that still lacks its depth map, sky mask or caption."""
        base = comfy.template("look.json")
        graph, inputs, todo = {}, {}, {}
        for k in range(len(self.state["shots"])):
            d = self.shot_dir(k)
            parts = [part for part, (_, _, fname) in self.LOOK_PARTS.items() if not (d / fname).exists()]
            if not parts:
                continue
            todo[k] = parts
            p = f"s{k:03d}_"
            wanted = {"image"} | {n for part in parts for n in self.LOOK_PARTS[part][0]}
            for nid in wanted:
                node = copy.deepcopy(base[nid])
                for name, v in node["inputs"].items():
                    if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                        if v[0] in self.LOOK_SHARED:
                            graph[v[0]] = copy.deepcopy(base[v[0]])
                        else:
                            node["inputs"][name] = [p + v[0], v[1]]
                graph[p + nid] = node
            inputs[(p + "image", "image")] = d / "middle.png"
            if "caption" in parts:
                inputs[(p + "caption", "custom_prompt")] = fill.CAPTION_INSTRUCTION
                if self.state.get("captioner"):
                    inputs[(p + "caption", "model_name")] = self.state["captioner"]
        if not todo:
            return
        job = self.c.run(comfy.build(self.c, graph, inputs))
        for k, parts in todo.items():
            p, d = f"s{k:03d}_", self.shot_dir(k)
            for part in parts:
                _, save, fname = self.LOOK_PARTS[part]
                comfy.download(job, p + save, d / fname)
            self.state["shots"][k]["caption"] = (d / "caption.txt").read_text(encoding="utf-8").strip()
        self._log(f"  depth, sky and caption for {len(todo)} shots")

    # -- 4. where to fill (local) ---------------------------------------------------------------
    def split(self):
        cw, ch = self.canon_out()
        shots = self.state["shots"]
        captions = fill.coherent_time_of_day([s.get("caption", "") for s in shots])
        side = ("left", "right") if self.widen else ("above", "below")
        for k, (s, cap) in enumerate(zip(shots, captions)):
            depth = canon(cv2.imread(str(self.shot_dir(k) / "depth.png"), cv2.IMREAD_GRAYSCALE), self.widen)
            s.update(fill.split(depth, ch - s["crop_h"], ch, s["crop_h"], s["face_y_max"], safe_zone=not self.widen))
            note = self.state.get("notes", {}).get(str(k + 1), "")
            sky = (not self.widen and s["top"] > 0 and fill.sky_above(str(self.shot_dir(k) / "sky.png")))
            s["sky_above"] = sky
            s["prompt"] = (cap + (" " + fill.SKY_PHRASE if sky else "") + (" " + note if note else "")
                           + fill.PROMPT_TAIL)
            self._log(f"  shot {k + 1:3d}: fill {s['top']}px {side[0]}, {s['bottom']}px {side[1]}")

    # -- 5. outpaint every shot (Comfy, one job per shot) ----------------------------------------
    def fill(self):
        out_w, out_h = self.state["out"]["w"], self.state["out"]["h"]
        base = comfy.template("fill_wan_vace.json")
        digest = hashlib.sha1(f"{self.source}:{self.source.stat().st_size}".encode()).hexdigest()[:12]
        pending = []
        for k, s in enumerate(self.state["shots"]):
            # the key covers everything that defines a shot's render, so a result is reused only when
            # nothing changed, a changed note re-renders just that shot, and nothing is paid twice
            spec = json.dumps([s["prompt"], s["top"], s["bottom"], s["length"], self.shot_seed(k),
                               s["crop_w"], s["x0"], out_w, out_h, base] + (["widen"] if self.widen else []),
                              sort_keys=True)
            key = f"openreframe-{digest}-{k:03d}-{hashlib.sha1(spec.encode()).hexdigest()[:16]}"
            if s.get("fill_key") == key and (self.shot_dir(k) / "fill.mp4").exists():
                continue
            if s["top"] + s["bottom"] == 0:           # nothing to invent (e.g. already vertical): no GPU, no cost
                shutil.copy(self.shot_dir(k) / "crop.mp4", self.shot_dir(k) / "fill.mp4")
                s["fill_key"] = key
                continue
            if s.get("job_key") != key or not s.get("job_id"):
                wf = comfy.build(self.c, base, {
                    ("load_video", "file"): self.shot_dir(k) / "crop.mp4",
                    **({("pad", "left"): s["top"], ("pad", "right"): s["bottom"]} if self.widen
                       else {("pad", "top"): s["top"], ("pad", "bottom"): s["bottom"]}),
                    ("mask_rep", "amount"): s["length"],
                    ("vace", "width"): out_w, ("vace", "height"): out_h, ("vace", "length"): s["length"],
                    ("positive", "text"): s["prompt"],
                    ("sampler", "seed"): self.shot_seed(k),
                    ("save", "filename_prefix"): f"openreframe/{digest}_{k:03d}"})
                s["job_id"], s["job_key"] = self.c.submit(wf, idempotency_key=key).id, key
                self.save()
            pending.append(k)
        self._log(f"  {len(pending)} shots to render, {len(self.state['shots']) - len(pending)} reused")
        # poll every job and report the queue, so a run waiting for a free GPU never looks frozen;
        # download each shot as soon as it is done, in whatever order they finish
        jobs = {k: self.c.jobs.get(self.state["shots"][k]["job_id"]) for k in pending}
        last = None
        while jobs:
            counts = {"waiting": 0, "running": 0}
            for k, job in list(jobs.items()):
                job.refresh()
                st = str(job.status).lower()
                if any(w in st for w in ("succeed", "complete", "success")):
                    s = self.state["shots"][k]
                    comfy.download(job, "save", self.shot_dir(k) / "fill.mp4")
                    s["fill_key"] = s["job_key"]
                    self.save()
                    del jobs[k]
                    self._log(f"  shot {k + 1:3d} ready")
                elif any(w in st for w in ("fail", "error", "cancel")):
                    job.result()                          # raises with the per-node detail
                elif "run" in st or "progress" in st:
                    counts["running"] += 1
                else:
                    counts["waiting"] += 1
            if jobs and counts != last:
                self._log(f"  {counts['running']} rendering, {counts['waiting']} waiting in the Comfy queue")
                last = counts
            if jobs:
                time.sleep(10)

    # -- 6. paste back, join, audio, other ratios (local) ---------------------------------------------
    def assemble(self):
        cw, ch = self.canon_out()
        fps = self.state["probe"]["fps"]
        name = self.source.stem
        target = self.state["target"]
        writers = {}
        for r in self.ratios:
            a = parse_ratio(r)
            a_c = 1 / a if self.widen else a             # width/height in the working orientation
            win = ch if r == target else min(ch, int(round(cw / a_c / 2)) * 2)
            size = (win, cw) if self.widen else (cw, win)
            writers[r] = (assemble.Writer(self.work / f"{name}_{r.replace(':', 'x')}_silent.mp4", *size, fps), win)
        shots = self.state["shots"]
        for k, frames in video.read_shots(self.source, [(s["start"], s["n"]) for s in shots]):
            s = shots[k]
            real = framing.crop_frames([canon(f, self.widen) for f in frames], s["x0"], s["crop_w"], cw, s["crop_h"])
            ai = [canon(f, self.widen) for f in assemble.ai_frames_of(self.shot_dir(k) / "fill.mp4", s["n"])]
            done = assemble.paste_back(ai, real, s["top"], s["bottom"])
            for r, (w, win) in writers.items():
                y = 0 if win == ch else assemble.window(ch, win, s["top"], s["crop_h"],
                                                        None if self.widen else s["face_y_max"])
                for f in done:
                    w.write(canon(f[y:y + win], self.widen))
        for r, (w, _) in writers.items():
            w.close()
            final = self.out / f"{name}_{r.replace(':', 'x')}.mp4"
            assemble.add_audio(w.path, self.source, final, self.state["probe"]["audio"])
            self._log(f"  {final}")
        cmp_ = self.out / f"{name}_compare.mp4"
        assemble.side_by_side(self.source, self.out / f"{name}_{target.replace(':', 'x')}.mp4", cmp_,
                              self.state["out"]["h"])
        self._log(f"  {cmp_}")

    # -- optional: the decisions drawn over the edit ----------------------------------------------
    def debug(self):
        name = self.source.stem
        dest = self.out / f"{name}_debug.mp4"
        assemble.debug_view(self.source, self.out / f"{name}_{self.state['target'].replace(':', 'x')}.mp4", dest,
                            self.state["shots"], json.loads((self.work / "boxes.json").read_text()),
                            self.state["out"]["w"], self.state["out"]["h"], self.state["probe"]["fps"],
                            self.state["probe"]["audio"], framing.SUBJECTS, widen=self.widen)
        self._log(f"  {dest}")
