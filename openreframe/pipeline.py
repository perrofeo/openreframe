"""The whole run, step by step, with its state in <work>/plan.json so an interrupted run resumes
without paying twice for a shot that already rendered."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import cv2

from . import assemble, comfy, fill, framing, video

STEPS = ("detect", "analyze", "look", "split", "fill", "assemble")
EXTRA = ("debug",)                                   # only when asked for
RATIOS = {"9:16": None, "4:5": 5 / 4, "1:1": 1.0}   # height / width of the shorter cuts


def log(msg: str) -> None:
    print(msg, flush=True)


class Run:
    def __init__(self, source: Path, out_dir: Path, size=(544, 960), seed=6061003, ratios=("9:16", "4:5", "1:1")):
        self.source = source.resolve()
        self.out = out_dir.resolve()
        self.work = self.out / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.state_path = self.work / "plan.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.state.setdefault("source", str(self.source))
        self.state.setdefault("out", {"w": size[0], "h": size[1]})
        self.state.setdefault("seed", seed)
        self.state.setdefault("done", [])
        self.ratios = ratios
        self._client = None

    # -- bookkeeping ---------------------------------------------------------------------------
    @property
    def c(self):
        if self._client is None:
            self._client = comfy.client()
        return self._client

    def save(self):
        self.state_path.write_text(json.dumps(self.state, indent=1, ensure_ascii=False))

    def shot_dir(self, k: int) -> Path:
        d = self.work / "shots" / f"{k:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def reset(self, step: str):
        """Run a step again, and the steps after it. Renders are reused when their inputs did not change."""
        order = STEPS + EXTRA
        for st in order[order.index(step):]:
            if st in self.state["done"]:
                self.state["done"].remove(st)
        if step == "look":
            for k, s in enumerate(self.state.get("shots", [])):
                s.pop("caption", None)
                (self.work / "shots" / f"{k:03d}" / "caption.txt").unlink(missing_ok=True)
        self.save()

    def run(self, steps=STEPS):
        for step in steps:
            if step in self.state["done"]:
                log(f"[{step}] done before, skipping")
                continue
            log(f"[{step}]")
            getattr(self, step)()
            self.state["done"].append(step)
            self.save()

    # -- 1. cuts + detection (Comfy) -------------------------------------------------------------
    def detect(self):
        info = video.probe(self.source)
        self.state["probe"] = info
        shots = video.detect_cuts(self.source, info["fps"], info["frames"])
        self.state["shots"] = [{"start": a, "n": n} for a, n in shots]
        log(f"  {info['width']}x{info['height']} @ {info['fps']:.3f} fps, {info['frames']} frames, {len(shots)} shots")
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
        log(f"  boxes and faces for {info['frames']} frames")

    # -- 2. framing (local) ---------------------------------------------------------------------
    def analyze(self):
        boxes = json.loads((self.work / "boxes.json").read_text())
        faces = json.loads((self.work / "faces.json").read_text())
        out_w, out_h, fps = self.state["out"]["w"], self.state["out"]["h"], self.state["probe"]["fps"]
        shots = self.state["shots"]
        for k, frames in video.read_shots(self.source, [(s["start"], s["n"]) for s in shots]):
            s = shots[k]
            a, n = s["start"], len(frames)
            s["n"] = n
            s.update(framing.plan_shot(frames, boxes[a:a + n], faces[a:a + n], out_w, out_h))
            crops = framing.crop_frames(frames, s["x0"], s["crop_w"], out_w, s["crop_h"])
            s["length"] = n + (-(n - 1)) % 4                  # Wan wants 4k+1 frames: repeat the last one
            crops += [crops[-1]] * (s["length"] - n)
            d = self.shot_dir(k)
            video.write_mp4(crops, d / "crop.mp4", fps, crf=10)
            cv2.imwrite(str(d / "middle.png"), crops[n // 2])
            log(f"  shot {k + 1:3d}: {n:4d} fr  crop {s['crop_w']}px ({s['crop_w'] / frames[0].shape[0]:.2f}:1)"
                f"  AI fills {1 - s['crop_h'] / out_h:.0%}  {'face' if s['has_face'] else 'subjects/motion'}")

    # -- 3. depth + caption per shot (Comfy, one job) ---------------------------------------------
    def look(self):
        base = comfy.template("look.json")
        shared = {"depth_model"}
        graph, todo = {}, []
        for k, s in enumerate(self.state["shots"]):
            if s.get("caption") and (self.shot_dir(k) / "depth.png").exists():
                continue
            todo.append(k)
            p = f"s{k:03d}_"
            for nid, node in base.items():
                if nid in shared:
                    graph[nid] = copy.deepcopy(node)
                    continue
                node = copy.deepcopy(node)
                for name, v in node["inputs"].items():
                    if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                        node["inputs"][name] = [v[0] if v[0] in shared else p + v[0], v[1]]
                graph[p + nid] = node
        if not todo:
            return
        inputs = {}
        for k in todo:
            p = f"s{k:03d}_"
            inputs[(p + "image", "image")] = self.shot_dir(k) / "middle.png"
            inputs[(p + "caption", "custom_prompt")] = fill.CAPTION_INSTRUCTION
        job = self.c.run(comfy.build(self.c, graph, inputs))
        for k in todo:
            p, d = f"s{k:03d}_", self.shot_dir(k)
            comfy.download(job, p + "save_depth", d / "depth.png")
            comfy.download(job, p + "save_caption", d / "caption.txt")
            self.state["shots"][k]["caption"] = (d / "caption.txt").read_text(encoding="utf-8").strip()
        log(f"  depth and caption for {len(todo)} shots")

    # -- 4. where to fill (local) ---------------------------------------------------------------
    def split(self):
        out_h = self.state["out"]["h"]
        shots = self.state["shots"]
        captions = fill.coherent_time_of_day([s.get("caption", "") for s in shots])
        for k, (s, cap) in enumerate(zip(shots, captions)):
            s.update(fill.split(str(self.shot_dir(k) / "depth.png"), out_h - s["crop_h"], out_h, s["crop_h"],
                                s["face_y_max"]))
            note = self.state.get("notes", {}).get(str(k + 1), "")
            s["prompt"] = cap + (" " + note if note else "") + fill.PROMPT_TAIL
            log(f"  shot {k + 1:3d}: fill {s['top']}px above, {s['bottom']}px below")

    # -- 5. outpaint every shot (Comfy, one job per shot) ----------------------------------------
    def fill(self):
        out_w, out_h = self.state["out"]["w"], self.state["out"]["h"]
        base = comfy.template("fill_wan_vace.json")
        digest = hashlib.sha1(f"{self.source}:{self.source.stat().st_size}".encode()).hexdigest()[:12]
        pending = []
        for k, s in enumerate(self.state["shots"]):
            # the key covers everything that defines a shot's render, so a result is reused only when
            # nothing changed, a changed note re-renders just that shot, and nothing is paid twice
            spec = json.dumps([s["prompt"], s["top"], s["bottom"], s["length"], self.state["seed"] + k,
                               s["crop_w"], s["x0"], out_w, out_h, base], sort_keys=True)
            key = f"openreframe-{digest}-{k:03d}-{hashlib.sha1(spec.encode()).hexdigest()[:16]}"
            if s.get("fill_key") == key and (self.shot_dir(k) / "fill.mp4").exists():
                continue
            if s.get("job_key") != key or not s.get("job_id"):
                wf = comfy.build(self.c, base, {
                    ("load_video", "file"): self.shot_dir(k) / "crop.mp4",
                    ("pad", "top"): s["top"], ("pad", "bottom"): s["bottom"],
                    ("mask_rep", "amount"): s["length"],
                    ("vace", "width"): out_w, ("vace", "height"): out_h, ("vace", "length"): s["length"],
                    ("positive", "text"): s["prompt"],
                    ("sampler", "seed"): self.state["seed"] + k,
                    ("save", "filename_prefix"): f"openreframe/{digest}_{k:03d}"})
                s["job_id"], s["job_key"] = self.c.submit(wf, idempotency_key=key).id, key
                self.save()
            pending.append(k)
        log(f"  {len(pending)} shots to render, {len(self.state['shots']) - len(pending)} reused")
        for k in pending:
            s = self.state["shots"][k]
            job = self.c.jobs.get(s["job_id"])
            job.wait()
            job.result()
            comfy.download(job, "save", self.shot_dir(k) / "fill.mp4")
            s["fill_key"] = s["job_key"]
            self.save()
            log(f"  shot {k + 1:3d} ready")

    # -- 6. paste back, join, audio, other ratios (local) ---------------------------------------------
    def assemble(self):
        out_w, out_h, fps = self.state["out"]["w"], self.state["out"]["h"], self.state["probe"]["fps"]
        name = self.source.stem
        writers = {}
        for r in self.ratios:
            h = out_h if RATIOS[r] is None else int(round(out_w * RATIOS[r] / 2)) * 2
            writers[r] = (assemble.Writer(self.work / f"{name}_{r.replace(':', 'x')}_silent.mp4", out_w, h, fps), h)
        shots = self.state["shots"]
        for k, frames in video.read_shots(self.source, [(s["start"], s["n"]) for s in shots]):
            s = shots[k]
            real = framing.crop_frames(frames, s["x0"], s["crop_w"], out_w, s["crop_h"])
            ai = assemble.ai_frames_of(self.shot_dir(k) / "fill.mp4", s["n"])
            done = assemble.paste_back(ai, real, s["top"], s["bottom"])
            for r, (w, h) in writers.items():
                if RATIOS[r] is None:
                    for f in done:
                        w.write(f)
                    continue
                fy = s["face_y_max"]
                y = assemble.window(out_h, h, s["top"], s["crop_h"], fy)
                for f in done:
                    w.write(f[y:y + h])
        for r, (w, _) in writers.items():
            w.close()
            final = self.out / f"{name}_{r.replace(':', 'x')}.mp4"
            assemble.add_audio(w.path, self.source, final, self.state["probe"]["audio"])
            log(f"  {final}")
        if "9:16" in self.ratios:
            cmp_ = self.out / f"{name}_compare.mp4"
            assemble.side_by_side(self.source, self.out / f"{name}_9x16.mp4", cmp_, out_h)
            log(f"  {cmp_}")

    # -- optional: the decisions drawn over the edit ----------------------------------------------
    def debug(self):
        name = self.source.stem
        dest = self.out / f"{name}_debug.mp4"
        assemble.debug_view(self.source, self.out / f"{name}_9x16.mp4", dest, self.state["shots"],
                            json.loads((self.work / "boxes.json").read_text()), self.state["out"]["w"],
                            self.state["out"]["h"], self.state["probe"]["fps"], self.state["probe"]["audio"],
                            framing.SUBJECTS)
        log(f"  {dest}")
