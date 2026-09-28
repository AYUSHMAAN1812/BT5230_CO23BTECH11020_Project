#!/usr/bin/env python
"""Test whether the spatial OCSR model can memorize a tiny training-only set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocsr_project.dataset import ImageSmilesDataset, make_collate_fn
from ocsr_project.modeling import build_model
from ocsr_project.tokenizer import SmilesTokenizer, tokenize_smiles
from ocsr_project.training import seed_everything, train_epoch


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--check-every", type=int, default=20)
    args = parser.parse_args()

    if "benchmark" in str(args.manifest).lower() or "decimer" in str(args.manifest).lower():
        raise ValueError("The held-out real benchmark is prohibited in this probe")
    if args.examples < 2 or args.steps < 1 or args.check_every < 1:
        raise ValueError("Probe counts must be positive and at least two examples are required")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.manifest)
    if set(frame["condition"]) != {"clean"}:
        raise ValueError("This memorization probe requires clean synthetic images")
    training = frame.loc[frame["split"] == "train"].copy()
    training["smiles_tokens"] = training["canonical_smiles"].map(
        lambda value: len(tokenize_smiles(str(value)))
    )
    selected = training.sort_values(["smiles_tokens", "compound_id"]).head(args.examples)
    if len(selected) != args.examples:
        raise ValueError("Not enough training images for the requested probe")

    seed_everything(int(config["seed"]))
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    tokenizer = SmilesTokenizer.build(training["canonical_smiles"].tolist())
    dataset = ImageSmilesDataset(
        selected,
        tokenizer,
        image_size=int(config["image_size"]),
        max_sequence_length=int(config["max_sequence_length"]),
        image_normalization=config.get("image_normalization", "symmetric"),
    )
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer),
    )
    batch = next(iter(loader))
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
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    max_decode_length = int(selected["smiles_tokens"].max()) + 6
    trace = []
    for step in range(1, args.steps + 1):
        loss = train_epoch(model, loader, optimizer, pad_id=tokenizer.pad_id, device="cpu")
        if step % args.check_every != 0 and step != args.steps:
            continue
        with torch.no_grad():
            generated = model.greedy_decode(
                batch["images"],
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                max_length=max_decode_length,
            )
        predictions = [tokenizer.decode(row.tolist()) for row in generated]
        exact_count = sum(
            prediction == reference
            for prediction, reference in zip(predictions, batch["smiles"], strict=True)
        )
        record = {"step": step, "loss": loss, "exact_count": exact_count}
        trace.append(record)
        print(json.dumps(record), flush=True)
        if exact_count == len(dataset):
            break

    results = {
        "purpose": "training-only memorization diagnostic, not validation performance",
        "compound_ids": selected["compound_id"].tolist(),
        "reference_smiles": batch["smiles"],
        "predicted_smiles": predictions,
        "vocabulary_size": tokenizer.vocab_size,
        "max_decode_length": max_decode_length,
        "trace": trace,
        "memorized_all": exact_count == len(dataset),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"memorized_all": results["memorized_all"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
