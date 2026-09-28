#!/usr/bin/env python
"""Fetch and verify the pinned official MobileNetV3-Small ImageNet weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

URL = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
FILENAME = "mobilenet_v3_small-047dcff4.pth"
SHA256 = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pretrained"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.load_state_dict_from_url(
        URL, model_dir=str(args.output_dir), check_hash=True, progress=True
    )
    path = args.output_dir / FILENAME
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != SHA256:
        raise ValueError(f"Downloaded pretrained weights failed SHA-256 verification: {actual}")
    print(json.dumps({"path": str(path), "sha256": actual, "source": URL}, indent=2))


if __name__ == "__main__":
    main()
