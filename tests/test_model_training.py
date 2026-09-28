import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from ocsr_project.dataset import ImageSmilesDataset, make_collate_fn
from ocsr_project.images import generate_depiction
from ocsr_project.modeling import build_model
from ocsr_project.tokenizer import SmilesTokenizer
from ocsr_project.training import evaluate_loss, seed_everything, token_cross_entropy, train_epoch


def test_model_forward_decode_and_train_step(tmp_path):
    seed_everything(17)
    values = ["CCO", "CCN"]
    tokenizer = SmilesTokenizer.build(values)
    rows = []
    for index, smiles in enumerate(values):
        image_path = tmp_path / f"{index}.png"
        generate_depiction(smiles, image_path, width=64, height=64)
        rows.append(
            {
                "compound_id": str(index),
                "canonical_smiles": smiles,
                "split": "train",
                "image_path": image_path,
            }
        )
    dataset = ImageSmilesDataset(
        pd.DataFrame(rows), tokenizer, split="train", image_size=64, max_sequence_length=16
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=make_collate_fn(tokenizer))
    model = build_model(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        embedding_dim=32,
        decoder_layers=1,
        attention_heads=4,
        feedforward_dim=64,
        max_sequence_length=16,
    )
    batch = next(iter(loader))
    logits = model(batch["images"], batch["sequences"][:, :-1])
    assert logits.shape[:2] == batch["sequences"][:, :-1].shape
    assert logits.shape[-1] == tokenizer.vocab_size

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    training_loss = train_epoch(model, loader, optimizer, pad_id=tokenizer.pad_id, device="cpu")
    validation_loss = evaluate_loss(model, loader, pad_id=tokenizer.pad_id, device="cpu")
    assert training_loss > 0
    assert validation_loss > 0

    decoded = model.greedy_decode(
        batch["images"], bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id, max_length=8
    )
    assert decoded.shape[0] == 2
    assert decoded.shape[1] <= 8
    single_beam, single_scores = model.beam_decode(
        batch["images"],
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        max_length=8,
        beam_width=1,
    )
    assert torch.equal(single_beam[:, 0], decoded)
    assert single_scores.shape == (2, 1)
    beams, scores = model.beam_decode(
        batch["images"],
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        max_length=8,
        beam_width=3,
    )
    assert beams.shape[:2] == (2, 3)
    assert torch.all(scores[:, 0] >= scores[:, 1])


def test_2d_image_encoding_is_spatial_and_checkpointed():
    model = build_model(
        vocab_size=8,
        pad_id=0,
        embedding_dim=32,
        decoder_layers=1,
        attention_heads=4,
        feedforward_dim=64,
        max_sequence_length=16,
        encoder_position="sincos_2d",
    )
    positions = model.image_position_encoding
    assert positions.shape == (1, 49, 32)
    assert not torch.equal(positions[:, 0], positions[:, 1])
    assert not torch.equal(positions[:, 0], positions[:, 7])
    assert "image_position_encoding" in model.state_dict()


def test_invalid_encoder_position_is_rejected():
    with pytest.raises(ValueError, match="Unsupported encoder position"):
        build_model(vocab_size=8, pad_id=0, encoder_position="unknown")


def test_group_normalization_has_no_running_statistics():
    model = build_model(
        vocab_size=8,
        pad_id=0,
        embedding_dim=32,
        decoder_layers=1,
        attention_heads=4,
        feedforward_dim=64,
        max_sequence_length=16,
        encoder_norm="group",
    )
    assert any(isinstance(layer, torch.nn.GroupNorm) for layer in model.encoder)
    assert not any(isinstance(layer, torch.nn.BatchNorm2d) for layer in model.encoder)
    assert not any("running_mean" in key for key in model.state_dict())


def test_eos_loss_weight_emphasizes_termination_error():
    logits = torch.tensor([[[0.0, 4.0, 0.0], [0.0, 4.0, 0.0]]])
    target = torch.tensor([[1, 2]])
    plain = token_cross_entropy(logits, target, pad_id=0, eos_id=2)
    weighted = token_cross_entropy(logits, target, pad_id=0, eos_id=2, eos_loss_weight=5.0)
    assert weighted > plain
    with pytest.raises(ValueError, match="non-padding eos_id"):
        token_cross_entropy(logits, target, pad_id=0, eos_loss_weight=5.0)
