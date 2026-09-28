#!/usr/bin/env python
"""Decode a synthetic split from a saved project checkpoint.

This tool never loads the real DECIMER hand-drawn benchmark. It creates a prediction
CSV for the separate chemistry-aware evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocsr_project.chemistry import canonicalize_smiles
from ocsr_project.dataset import ImageSmilesDataset, make_collate_fn
from ocsr_project.modeling import build_model
from ocsr_project.tokenizer import SmilesTokenizer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-decode-length", type=int)
    parser.add_argument("--decoding-method", choices=("greedy", "beam"), default="greedy")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--valid-rerank", action="store_true")
    parser.add_argument("--nbest-output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.beam_width < 1 or (args.valid_rerank and args.decoding_method != "beam"):
        raise ValueError("Validity reranking requires beam decoding with positive beam width")
    if args.nbest_output is not None and args.decoding_method != "beam":
        raise ValueError("N-best output is available only for beam decoding")

    evaluation_path = args.evaluation_manifest or args.manifest
    if any(
        marker in str(path).lower()
        for path in (args.manifest, evaluation_path)
        for marker in ("benchmark", "decimer")
    ):
        raise ValueError("This command is restricted to synthetic manifests")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.manifest)
    if set(frame["condition"]) != {config["condition"]}:
        raise ValueError("Manifest condition differs from training configuration")
    evaluation_frame = pd.read_csv(evaluation_path)
    columns = ["compound_id", "canonical_smiles", "split"]
    left = frame[columns].sort_values("compound_id").reset_index(drop=True)
    right = evaluation_frame[columns].sort_values("compound_id").reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("Evaluation manifest must have identical molecules and split assignments")
    selected = evaluation_frame.loc[evaluation_frame["split"] == args.split].copy()
    if args.max_examples is not None:
        selected = selected.head(args.max_examples)
    if selected.empty:
        raise ValueError(f"No examples found in {args.split} split")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    tokenizer = SmilesTokenizer.load(args.run_dir / "tokenizer.json")
    dataset = ImageSmilesDataset(
        selected,
        tokenizer,
        image_size=int(config["image_size"]),
        max_sequence_length=int(config["max_sequence_length"]),
        image_normalization=config.get("image_normalization", "symmetric"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer),
    )
    model_config = config["model"]
    model = build_model(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        embedding_dim=int(model_config["embedding_dim"]),
        decoder_layers=int(model_config["decoder_layers"]),
        attention_heads=int(model_config["attention_heads"]),
        feedforward_dim=int(model_config["feedforward_dim"]),
        dropout=float(model_config["dropout"]),
        max_sequence_length=int(config["max_sequence_length"]),
        encoder_position=model_config.get("encoder_position", "none"),
        encoder_norm=model_config.get("encoder_norm", "batch"),
        encoder_kind=model_config["encoder"],
        pretrained_weights_path=model_config.get("pretrained_weights_path"),
        pretrained_weights_sha256=model_config.get("pretrained_weights_sha256"),
    ).to(device)
    checkpoint_path = args.run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["config_hash"] != sha256(args.config):
        raise ValueError("Config hash differs from the selected checkpoint")
    if checkpoint["manifest_hash"] != sha256(args.manifest):
        raise ValueError("Manifest hash differs from the selected checkpoint")
    if (
        args.split == "validation"
        and checkpoint["validation_manifest_hash"] != sha256(evaluation_path)
    ):
        raise ValueError("Validation manifest hash differs from the selected checkpoint")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    decode_length = args.max_decode_length or int(config["max_sequence_length"])
    if decode_length < 2 or decode_length > int(config["max_sequence_length"]):
        raise ValueError("Decode length must be between 2 and max_sequence_length")
    rows: list[dict[str, str]] = []
    nbest_rows: list[dict[str, str | int | float | bool]] = []
    raw_top1_valid_count = 0
    reranked_count = 0
    for batch in loader:
        if args.decoding_method == "greedy":
            generated = model.greedy_decode(
                batch["images"].to(device),
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                max_length=decode_length,
            )
            predictions = [tokenizer.decode(ids.tolist()) for ids in generated.cpu()]
        else:
            generated, scores = model.beam_decode(
                batch["images"].to(device),
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                max_length=decode_length,
                beam_width=args.beam_width,
            )
            predictions = []
            for image_id, beams, beam_scores in zip(
                batch["compound_ids"], generated.cpu(), scores.cpu(), strict=True
            ):
                candidates = [tokenizer.decode(ids.tolist()) for ids in beams]
                valid = [canonicalize_smiles(candidate) is not None for candidate in candidates]
                raw_top1_valid_count += int(valid[0])
                chosen = next((index for index, okay in enumerate(valid) if okay), 0) if args.valid_rerank else 0
                reranked_count += int(chosen > 0)
                predictions.append(candidates[chosen])
                for rank, (candidate, score, okay) in enumerate(
                    zip(candidates, beam_scores.tolist(), valid, strict=True), start=1
                ):
                    nbest_rows.append(
                        {
                            "image_id": str(image_id),
                            "rank": rank,
                            "log_probability": float(score),
                            "predicted_smiles": candidate,
                            "valid": okay,
                        }
                    )
        for image_id, reference, prediction in zip(
            batch["compound_ids"], batch["smiles"], predictions, strict=True
        ):
            rows.append(
                {
                    "image_id": str(image_id),
                    "reference_smiles": reference,
                    "predicted_smiles": prediction,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_id", "reference_smiles", "predicted_smiles"]
        )
        writer.writeheader()
        writer.writerows(rows)
    if args.nbest_output is not None:
        args.nbest_output.parent.mkdir(parents=True, exist_ok=True)
        with args.nbest_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["image_id", "rank", "log_probability", "predicted_smiles", "valid"],
            )
            writer.writeheader()
            writer.writerows(nbest_rows)
    print(
        json.dumps(
            {
                "status": "complete",
                "prediction_count": len(rows),
                "split": args.split,
                "output": str(args.output),
                "checkpoint_epoch": checkpoint["epoch"],
                "decoding_method": args.decoding_method,
                "beam_width": args.beam_width if args.decoding_method == "beam" else None,
                "raw_top1_valid_count": (
                    raw_top1_valid_count if args.decoding_method == "beam" else None
                ),
                "reranked_count": reranked_count if args.valid_rerank else None,
                "note": "Synthetic evaluation only; real benchmark remains held out",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
