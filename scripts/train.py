#!/usr/bin/env python
"""Train or resume a controlled OCSR experiment using synthetic data only."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocsr_project.curriculum import select_short_molecules, training_phase
from ocsr_project.dataset import ImageSmilesDataset, make_collate_fn
from ocsr_project.modeling import build_model
from ocsr_project.tokenizer import SmilesTokenizer
from ocsr_project.training import evaluate_loss, seed_everything, train_epoch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def require_synthetic_manifest(path: Path, frame: pd.DataFrame, condition: str) -> None:
    if "benchmark" in str(path).lower() or "decimer" in str(path).lower():
        raise ValueError("The held-out real benchmark must never be used for training")
    if "condition" not in frame.columns or set(frame["condition"]) != {condition}:
        raise ValueError("Manifest image condition does not match the experiment config")
    if not {"train", "validation"}.issubset(set(frame["split"])):
        raise ValueError("Training requires nonempty train and validation partitions")
    if frame["compound_id"].duplicated().any() or frame["canonical_smiles"].duplicated().any():
        raise ValueError("Training manifest contains duplicate molecules or IDs")


def require_matched_validation(training: pd.DataFrame, validation: pd.DataFrame) -> None:
    columns = ["compound_id", "canonical_smiles", "split"]
    left = training[columns].sort_values("compound_id").reset_index(drop=True)
    right = validation[columns].sort_values("compound_id").reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("Validation manifest must have identical molecules and split assignments")
    if set(validation["condition"]) != {"clean"}:
        raise ValueError("Shared validation manifest must contain clean images")


def save_checkpoint(path: Path, payload: dict) -> None:
    import torch

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def make_checkpoint(
    *,
    model,
    optimizer,
    epoch: int,
    best_validation_loss: float,
    stale_epochs: int,
    config_hash: str,
    manifest_hash: str,
    validation_manifest_hash: str,
    history: list[dict[str, float | int]],
) -> dict:
    import torch

    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_validation_loss": best_validation_loss,
        "stale_epochs": stale_epochs,
        "config_hash": config_hash,
        "manifest_hash": manifest_hash,
        "validation_manifest_hash": validation_manifest_hash,
        "history": history,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.manifest)
    require_synthetic_manifest(args.manifest, frame, config["condition"])
    validation_path = args.validation_manifest or args.manifest
    validation_frame = pd.read_csv(validation_path)
    if args.validation_manifest is not None:
        if "benchmark" in str(validation_path).lower() or "decimer" in str(validation_path).lower():
            raise ValueError("The held-out real benchmark must never be used for model selection")
        require_matched_validation(frame, validation_frame)
    seed = int(config["seed"])
    seed_everything(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    train_frame = frame.loc[frame["split"] == "train"]
    tokenizer_path = args.run_dir / "tokenizer.json"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        tokenizer = SmilesTokenizer.load(tokenizer_path)
    else:
        tokenizer = SmilesTokenizer.build(train_frame["canonical_smiles"].tolist())
        tokenizer.save(tokenizer_path)

    image_size = int(config["image_size"])
    image_normalization = config.get("image_normalization", "symmetric")
    max_length = int(config["max_sequence_length"])
    train_dataset = ImageSmilesDataset(
        frame,
        tokenizer,
        split="train",
        image_size=image_size,
        max_sequence_length=max_length,
        image_normalization=image_normalization,
    )
    training_config = config["training"]
    eos_loss_weight = float(training_config.get("eos_loss_weight", 1.0))
    if eos_loss_weight <= 0:
        raise ValueError("eos_loss_weight must be positive")
    epochs = int(training_config["epochs"])
    curriculum_config = training_config.get("curriculum")
    warmup_epochs = 0
    short_dataset = None
    if curriculum_config is not None:
        warmup_epochs = int(curriculum_config["epochs"])
        if warmup_epochs < 1 or warmup_epochs >= epochs:
            raise ValueError("Curriculum epochs must be at least one and fewer than total epochs")
        short_frame = select_short_molecules(
            train_frame, max_smiles_tokens=int(curriculum_config["max_smiles_tokens"])
        )
        short_dataset = ImageSmilesDataset(
            short_frame,
            tokenizer,
            image_size=image_size,
            max_sequence_length=max_length,
            image_normalization=image_normalization,
        )
    validation_dataset = ImageSmilesDataset(
        validation_frame,
        tokenizer,
        split="validation",
        image_size=image_size,
        max_sequence_length=max_length,
        image_normalization=image_normalization,
    )
    batch_size = int(training_config["batch_size"])
    collate = make_collate_fn(tokenizer)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
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
        max_sequence_length=max_length,
        encoder_position=model_config.get("encoder_position", "none"),
        encoder_norm=model_config.get("encoder_norm", "batch"),
        encoder_kind=model_config["encoder"],
        pretrained_weights_path=model_config.get("pretrained_weights_path"),
        pretrained_weights_sha256=model_config.get("pretrained_weights_sha256"),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    config_hash = sha256(args.config)
    manifest_hash = sha256(args.manifest)
    validation_manifest_hash = sha256(validation_path)
    history: list[dict[str, float | int]] = []
    start_epoch = 1
    best_validation_loss = float("inf")
    stale_epochs = 0
    last_path = args.run_dir / "last.pt"
    best_path = args.run_dir / "best.pt"
    if args.resume:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if (
            checkpoint["config_hash"] != config_hash
            or checkpoint["manifest_hash"] != manifest_hash
            or checkpoint["validation_manifest_hash"] != validation_manifest_hash
        ):
            raise ValueError("Cannot resume: config or manifest has changed")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        if torch.cuda.is_available() and checkpoint["torch_cuda_random_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_random_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(checkpoint["best_validation_loss"])
        stale_epochs = int(checkpoint["stale_epochs"])
        history = checkpoint["history"]
    elif last_path.exists():
        raise FileExistsError(f"Run directory already contains a checkpoint: {last_path}")

    metadata = {
        "experiment_id": config["experiment_id"],
        "condition": config["condition"],
        "started_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(args.config),
        "config_sha256": config_hash,
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest_hash,
        "validation_manifest_path": str(validation_path),
        "validation_manifest_sha256": validation_manifest_hash,
        "seed": seed,
        "device": device,
        "torch_version": torch.__version__,
        "train_count": len(train_dataset),
        "curriculum_train_count": len(short_dataset) if short_dataset is not None else None,
        "curriculum_epochs": warmup_epochs,
        "eos_loss_weight": eos_loss_weight,
        "validation_count": len(validation_dataset),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "image_normalization": image_normalization,
    }
    metadata_path = args.run_dir / "metadata.json"
    if not metadata_path.exists():
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    patience = int(config["training"]["early_stopping_patience"])
    for epoch in range(start_epoch, epochs + 1):
        phase = training_phase(epoch, warmup_epochs)
        epoch_dataset = short_dataset if phase == "short" else train_dataset
        generator = torch.Generator().manual_seed(seed + epoch)
        train_loader = DataLoader(
            epoch_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate,
        )
        started = time.perf_counter()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            pad_id=tokenizer.pad_id,
            device=device,
            clip_norm=float(config["training"]["gradient_clip_norm"]),
            eos_id=tokenizer.eos_id,
            eos_loss_weight=eos_loss_weight,
        )
        validation_loss = evaluate_loss(
            model,
            validation_loader,
            pad_id=tokenizer.pad_id,
            device=device,
            eos_id=tokenizer.eos_id,
            eos_loss_weight=eos_loss_weight,
        )
        elapsed = time.perf_counter() - started
        record = {
            "epoch": epoch,
            "phase": phase,
            "train_count": len(epoch_dataset),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "seconds": elapsed,
        }
        history.append(record)
        improved = phase == "full" and validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            stale_epochs = 0
        elif phase == "full":
            stale_epochs += 1
        payload = make_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_validation_loss=best_validation_loss,
            stale_epochs=stale_epochs,
            config_hash=config_hash,
            manifest_hash=manifest_hash,
            validation_manifest_hash=validation_manifest_hash,
            history=history,
        )
        save_checkpoint(last_path, payload)
        if improved:
            save_checkpoint(best_path, payload)
        (args.run_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(json.dumps(record), flush=True)
        if phase == "full" and stale_epochs >= patience:
            print(f"Early stopping after epoch {epoch}", flush=True)
            break

    print(
        json.dumps(
            {
                "status": "complete",
                "epochs_completed": len(history),
                "best_validation_loss": best_validation_loss,
                "run_dir": str(args.run_dir),
                "note": "Synthetic validation only; held-out real benchmark was not accessed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
