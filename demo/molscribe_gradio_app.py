#!/usr/bin/env python
"""Single-image Gradio interface for the project-trained augmented checkpoint.

Run this file with the pinned MolScribe environment prepared by the Colab demo notebook.
The project checkpoint contains the ``chartok_coords`` decoder but no edge
decoder, so this loader intentionally does not use MolScribe's stock interface,
which expects an ``edges`` prediction.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

AUGMENTED_CHECKPOINT_SHA256 = (
    "20bba8c832365f1b708cfe7738b5dfd5f90b025bf6a4619288240e0c6f313741"
)
MOLSCRIBE_COMMIT = "7296a30413eb55436702011efdff78131f66d162"


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a 1.1 GB file at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(path: Path, expected_sha256: str) -> str:
    """Fail closed if the selected checkpoint is missing or has changed."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            "Checkpoint SHA-256 mismatch. "
            f"Expected {expected_sha256}, received {actual}."
        )
    return actual


def verify_molscribe_source(path: Path) -> Path:
    """Validate the checked-out MolScribe source needed by the custom loader."""
    vocab = path / "molscribe" / "vocab" / "vocab_chars.json"
    if not vocab.is_file():
        raise FileNotFoundError(
            "MolScribe source is incomplete or the --molscribe-root path is incorrect: "
            f"missing {vocab}"
        )
    return vocab


class ChartokMolScribePredictor:
    """Inference wrapper for the chartok-only transfer-learning checkpoint."""

    def __init__(self, checkpoint: Path, molscribe_root: Path) -> None:
        # Imports are intentionally lazy: verification tests do not need the
        # large Colab-only MolScribe/PyTorch environment.
        import torch
        from molscribe.dataset import get_transforms
        from molscribe.interface import safe_load
        from molscribe.model import Decoder, Encoder
        from molscribe.tokenizer import get_tokenizer

        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        states = torch.load(str(checkpoint), map_location=torch.device("cpu"))
        args = self._model_args(states.get("args", {}), molscribe_root)

        tokenizer = get_tokenizer(args)
        encoder = Encoder(args, pretrained=False)
        args.encoder_dim = encoder.n_features
        decoder = Decoder(args, tokenizer)
        safe_load(encoder, states["encoder"])
        safe_load(decoder, states["decoder"])
        encoder.to(self.device).eval()
        decoder.to(self.device).eval()

        self.encoder = encoder
        self.decoder = decoder
        self.transform = get_transforms(args.input_size, augment=False)

    @staticmethod
    def _model_args(saved: dict[str, Any], molscribe_root: Path) -> argparse.Namespace:
        """Reconstruct MolScribe defaults, then apply checkpoint metadata."""
        defaults: dict[str, Any] = {
            "encoder": "swin_base",
            "decoder": "transformer",
            "trunc_encoder": False,
            "no_pretrained": True,
            "use_checkpoint": True,
            "dropout": 0.5,
            "embed_dim": 256,
            "enc_pos_emb": False,
            "dec_num_layers": 6,
            "dec_hidden_size": 256,
            "dec_attn_heads": 8,
            "dec_num_queries": 128,
            "hidden_dropout": 0.1,
            "attn_dropout": 0.1,
            "max_relative_positions": 0,
            "continuous_coords": False,
            "compute_confidence": False,
            "input_size": 384,
            "coord_bins": 64,
            "sep_xy": True,
            "formats": ["chartok_coords"],
        }
        defaults.update(saved)
        # The stored Colab path is session-specific, so always use the current clone.
        defaults["vocab_file"] = str(verify_molscribe_source(molscribe_root))
        if defaults.get("formats") != ["chartok_coords"]:
            raise ValueError(
                "This interface supports the project-trained chartok_coords checkpoint."
            )
        return argparse.Namespace(**defaults)

    def predict(self, image_path: str | None):
        """Return a rendered molecule, processed SMILES, raw SMILES, and status."""
        if not image_path:
            return None, "", "", "Upload a PNG or JPEG chemical-structure image."

        import cv2
        from molscribe.chemistry import _postprocess_smiles
        from rdkit import Chem
        from rdkit.Chem import Draw

        image = cv2.imread(str(image_path))
        if image is None:
            return None, "", "", "The uploaded file could not be read as an image."
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=image, keypoints=[])["image"].unsqueeze(0)
        tensor = tensor.to(self.device)

        with self._torch.no_grad():
            features, hiddens = self.encoder(tensor)
            prediction = self.decoder.decode(features, hiddens)[0]
        raw_smiles = prediction["chartok_coords"]["smiles"]
        processed_smiles, _, success = _postprocess_smiles(raw_smiles)

        mol = Chem.MolFromSmiles(processed_smiles) if success else None
        if mol is None:
            return (
                None,
                processed_smiles,
                raw_smiles,
                "Invalid prediction: RDKit could not construct a molecule.",
            )
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        rendered = Draw.MolToImage(mol, size=(600, 400))
        device_name = "GPU" if self.device.type == "cuda" else "CPU"
        return rendered, canonical, raw_smiles, f"Valid RDKit molecule - inference on {device_name}"


def build_demo(predictor: ChartokMolScribePredictor):
    """Create the presentation interface after the model has loaded."""
    import gradio as gr

    description = (
        "Upload one chemical-structure drawing. The project-trained augmented MolScribe "
        "checkpoint returns a SMILES prediction and an RDKit rendering. Research "
        "prototype only: always verify the structure before scientific use."
    )
    return gr.Interface(
        fn=predictor.predict,
        inputs=gr.Image(type="filepath", label="Chemical structure image"),
        outputs=[
            gr.Image(type="pil", label="Predicted molecule"),
            gr.Textbox(label="Canonical predicted SMILES"),
            gr.Textbox(label="Raw model output"),
            gr.Textbox(label="Validation status"),
        ],
        title="Hand-Drawn Chemical Structure Recognition",
        description=description,
        allow_flagging="never",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--molscribe-root", type=Path, required=True)
    parser.add_argument(
        "--expected-sha256", default=AUGMENTED_CHECKPOINT_SHA256
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_molscribe_source(args.molscribe_root)
    actual = verify_checkpoint(args.checkpoint, args.expected_sha256)
    print(f"Verified checkpoint SHA-256: {actual}")
    print("Loading model; this can take a minute...")
    predictor = ChartokMolScribePredictor(args.checkpoint, args.molscribe_root)
    print(f"Model ready on {predictor.device}.")
    demo = build_demo(predictor)
    demo.queue().launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=args.server_port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
