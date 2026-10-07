"""Publish the Space: this folder + the openreframe package + the example videos.

    python space/deploy.py EXAMPLES_DIR [--repo perrofeo/openreframe]

Example videos are not in git; pass the folder that holds the *_compare.mp4 files.
Needs `hf auth login` with a write token."""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("examples", type=Path)
ap.add_argument("--repo", default="perrofeo/openreframe")
a = ap.parse_args()
here = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as tmp:
    t = Path(tmp)
    for f in ("app.py", "requirements.txt", "packages.txt", "README.md"):
        shutil.copy(here / f, t / f)
    shutil.copytree(here.parent / "openreframe", t / "openreframe", ignore=shutil.ignore_patterns("__pycache__"))
    (t / "examples").mkdir()
    for v in sorted(a.examples.glob("*_compare.mp4")):
        shutil.copy(v, t / "examples" / v.name)
    api = HfApi()
    api.upload_folder(folder_path=str(t), repo_id=a.repo, repo_type="space", commit_message="deploy",
                      delete_patterns=["examples/*", "openreframe/**"])
    print("deployed", a.repo, [p.name for p in (t / "examples").iterdir()])
