import os
import sys
from pathlib import Path


def pick_model() -> str:
    here = Path(__file__).resolve().parent
    os.chdir(here)
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    models_dir = here.parent / "models"
    if not models_dir.exists():
        sys.exit("No models/ folder found. Download a HuggingFace model into models/ first.")
    for folder in sorted(models_dir.iterdir()):
        if folder.is_dir() and (folder / "config.json").exists():
            return f"../models/{folder.name}"
    sys.exit("No usable model found under models/. Download a HuggingFace model into models/ first.")
