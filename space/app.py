"""Hugging Face Space for openreframe: https://github.com/perrofeo/openreframe"""
from pathlib import Path

from openreframe.web import build_app

examples = Path(__file__).parent / "examples"
build_app(examples).queue(default_concurrency_limit=2).launch(allowed_paths=[str(examples)])
