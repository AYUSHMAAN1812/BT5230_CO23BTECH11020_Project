from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "demo" / "molscribe_gradio_app.py"


def load_demo_module():
    spec = importlib.util.spec_from_file_location("molscribe_gradio_app", APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_hash_verification(tmp_path: Path) -> None:
    app = load_demo_module()
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"test-checkpoint")
    expected = app.sha256_file(checkpoint)

    assert app.verify_checkpoint(checkpoint, expected) == expected
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        app.verify_checkpoint(checkpoint, "0" * 64)


def test_model_args_force_current_vocab_path(tmp_path: Path) -> None:
    app = load_demo_module()
    vocab = tmp_path / "molscribe" / "vocab" / "vocab_chars.json"
    vocab.parent.mkdir(parents=True)
    vocab.write_text("{}", encoding="utf-8")
    args = app.ChartokMolScribePredictor._model_args(
        {
            "formats": ["chartok_coords"],
            "vocab_file": "/content/old-session/vocab_chars.json",
        },
        tmp_path,
    )

    assert args.formats == ["chartok_coords"]
    assert args.vocab_file == str(tmp_path / "molscribe" / "vocab" / "vocab_chars.json")


def test_model_args_reject_incompatible_format(tmp_path: Path) -> None:
    app = load_demo_module()
    vocab = tmp_path / "molscribe" / "vocab" / "vocab_chars.json"
    vocab.parent.mkdir(parents=True)
    vocab.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="chartok_coords"):
        app.ChartokMolScribePredictor._model_args(
            {"formats": ["chartok_coords", "edges"]}, tmp_path
        )


def test_molscribe_source_validation(tmp_path: Path) -> None:
    app = load_demo_module()
    with pytest.raises(FileNotFoundError, match="MolScribe source"):
        app.verify_molscribe_source(tmp_path)

    vocab = tmp_path / "molscribe" / "vocab" / "vocab_chars.json"
    vocab.parent.mkdir(parents=True)
    vocab.write_text("{}", encoding="utf-8")
    assert app.verify_molscribe_source(tmp_path) == vocab
