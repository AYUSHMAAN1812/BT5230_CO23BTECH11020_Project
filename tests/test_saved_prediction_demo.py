from __future__ import annotations

import importlib.util
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/run_saved_prediction_demo.py"


def load_demo_module():
    spec = importlib.util.spec_from_file_location("run_saved_prediction_demo", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_saved_prediction_replay_creates_panel(tmp_path: Path) -> None:
    demo = load_demo_module()
    record = demo.load_record("augmented", demo.DEFAULT_IMAGE_ID)
    output = tmp_path / "demo.png"
    summary = demo.create_panel(
        record,
        demo.IMAGE_DIR / f"{demo.DEFAULT_IMAGE_ID}.png",
        output,
        "augmented",
    )

    assert output.is_file()
    assert summary["mode"] == "saved_prediction_replay_no_inference"
    assert summary["image_id"] == demo.DEFAULT_IMAGE_ID
    assert summary["valid"] is True
    with Image.open(output) as image:
        assert image.size == (1440, 620)
