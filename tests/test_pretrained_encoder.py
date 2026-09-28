import hashlib

import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision.models import mobilenet_v3_small

from ocsr_project.dataset import ImageSmilesDataset
from ocsr_project.modeling import build_model
from ocsr_project.tokenizer import SmilesTokenizer


def test_imagenet_normalization_and_fixed_encoder(tmp_path):
    torch.set_num_threads(2)
    image_path = tmp_path / "white.png"
    Image.new("RGB", (224, 224), "white").save(image_path)
    frame = pd.DataFrame(
        [
            {
                "compound_id": "one",
                "canonical_smiles": "CCO",
                "split": "train",
                "image_path": image_path,
            }
        ]
    )
    tokenizer = SmilesTokenizer.build(["CCO"])
    image = ImageSmilesDataset(
        frame, tokenizer, image_size=224, image_normalization="imagenet"
    )[0]["image"]
    expected = (1 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor(
        [0.229, 0.224, 0.225]
    )
    assert torch.allclose(image[:, 0, 0], expected)

    backbone = mobilenet_v3_small(weights=None)
    weights_path = tmp_path / "backbone.pth"
    torch.save(backbone.state_dict(), weights_path)
    weights_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    model = build_model(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        embedding_dim=32,
        decoder_layers=1,
        attention_heads=4,
        feedforward_dim=64,
        max_sequence_length=16,
        encoder_position="sincos_2d",
        encoder_kind="mobilenet_v3_small_frozen",
        pretrained_weights_path=weights_path,
        pretrained_weights_sha256=weights_hash,
    )
    model.train()
    assert not model.encoder.training
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.projection.parameters())
    logits = model(image.unsqueeze(0), torch.tensor([[tokenizer.bos_id]]))
    assert logits.shape == (1, 1, tokenizer.vocab_size)
    with pytest.raises(ValueError, match="checksum mismatch"):
        build_model(
            vocab_size=tokenizer.vocab_size,
            pad_id=tokenizer.pad_id,
            encoder_kind="mobilenet_v3_small_frozen",
            pretrained_weights_path=weights_path,
            pretrained_weights_sha256="0" * 64,
        )
