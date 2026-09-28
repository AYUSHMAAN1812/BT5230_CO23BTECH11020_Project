"""Compact CNN-Transformer image-to-SMILES model for the controlled student experiments."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path


def _imports():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for model construction") from exc
    return torch, nn


def build_model(
    *,
    vocab_size: int,
    pad_id: int,
    embedding_dim: int = 192,
    decoder_layers: int = 3,
    attention_heads: int = 6,
    feedforward_dim: int = 512,
    dropout: float = 0.1,
    max_sequence_length: int = 128,
    encoder_position: str = "none",
    encoder_norm: str = "batch",
    encoder_kind: str = "tiny_cnn",
    pretrained_weights_path: str | Path | None = None,
    pretrained_weights_sha256: str | None = None,
):
    torch, nn = _imports()
    if encoder_kind not in {"tiny_cnn", "mobilenet_v3_small_frozen"}:
        raise ValueError(f"Unsupported image encoder: {encoder_kind}")
    if encoder_norm not in {"batch", "group"}:
        raise ValueError(f"Unsupported encoder normalization: {encoder_norm}")
    if encoder_position not in {"none", "sincos_2d"}:
        raise ValueError(f"Unsupported encoder position encoding: {encoder_position}")
    if encoder_position == "sincos_2d" and embedding_dim % 4:
        raise ValueError("2D sinusoidal encoding requires embedding_dim divisible by four")

    def make_norm(channels: int):
        if encoder_norm == "group":
            return nn.GroupNorm(math.gcd(8, channels), channels)
        return nn.BatchNorm2d(channels)

    def make_2d_encoding(height: int, width: int):
        quarter = embedding_dim // 4
        frequencies = torch.exp(
            -torch.log(torch.tensor(10000.0))
            * torch.arange(quarter, dtype=torch.float32)
            / max(quarter - 1, 1)
        )
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        y_angles = y.reshape(-1, 1) * frequencies
        x_angles = x.reshape(-1, 1) * frequencies
        return torch.cat(
            (y_angles.sin(), y_angles.cos(), x_angles.sin(), x_angles.cos()), dim=1
        ).unsqueeze(0)

    class TinyOCSRModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pad_id = pad_id
            self.max_sequence_length = max_sequence_length
            self.encoder_position = encoder_position
            self.freeze_encoder = encoder_kind == "mobilenet_v3_small_frozen"
            if self.freeze_encoder:
                if pretrained_weights_path is None or pretrained_weights_sha256 is None:
                    raise ValueError("Frozen MobileNet requires a local weight path and SHA-256")
                weights_path = Path(pretrained_weights_path)
                actual_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
                if actual_hash.lower() != pretrained_weights_sha256.lower():
                    raise ValueError("Pretrained MobileNet weight checksum mismatch")
                from torchvision.models import mobilenet_v3_small

                backbone = mobilenet_v3_small(weights=None)
                backbone.load_state_dict(
                    torch.load(weights_path, map_location="cpu", weights_only=True), strict=True
                )
                self.encoder = backbone.features
                self.encoder.requires_grad_(False)
                self.projection = nn.Conv2d(576, embedding_dim, kernel_size=1)
            else:
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
                    make_norm(32),
                    nn.GELU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    make_norm(64),
                    nn.GELU(),
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    make_norm(128),
                    nn.GELU(),
                    nn.Conv2d(128, embedding_dim, kernel_size=3, stride=2, padding=1),
                    make_norm(embedding_dim),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d((7, 7)),
                )
                self.projection = nn.Identity()
            self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
            self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim)
            layer = nn.TransformerDecoderLayer(
                d_model=embedding_dim,
                nhead=attention_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(layer, num_layers=decoder_layers)
            self.output = nn.Linear(embedding_dim, vocab_size)
            if encoder_position == "sincos_2d":
                self.register_buffer("image_position_encoding", make_2d_encoding(7, 7))

        def encode_images(self, images):
            if self.freeze_encoder:
                with torch.no_grad():
                    features = self.encoder(images)
            else:
                features = self.encoder(images)
            features = self.projection(features)
            features = nn.functional.adaptive_avg_pool2d(features, (7, 7))
            memory = features.flatten(2).transpose(1, 2)
            if self.encoder_position == "sincos_2d":
                memory = memory + self.image_position_encoding.to(dtype=memory.dtype)
            return memory

        def train(self, mode: bool = True):
            super().train(mode)
            if self.freeze_encoder:
                self.encoder.eval()
            return self

        def decode_tokens(self, memory, decoder_input_ids):
            length = decoder_input_ids.shape[1]
            if length > self.max_sequence_length:
                raise ValueError("Decoder input exceeds configured maximum sequence length")
            positions = torch.arange(length, device=decoder_input_ids.device).unsqueeze(0)
            target = self.token_embedding(decoder_input_ids) + self.position_embedding(positions)
            causal_mask = torch.triu(
                torch.ones(length, length, device=decoder_input_ids.device, dtype=torch.bool),
                diagonal=1,
            )
            padding_mask = decoder_input_ids.eq(self.pad_id)
            decoded = self.decoder(
                target,
                memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=padding_mask,
            )
            return self.output(decoded)

        def forward(self, images, decoder_input_ids):
            return self.decode_tokens(self.encode_images(images), decoder_input_ids)

        @torch.no_grad()
        def greedy_decode(self, images, *, bos_id: int, eos_id: int, max_length: int):
            self.eval()
            memory = self.encode_images(images)
            generated = torch.full(
                (images.shape[0], 1), bos_id, dtype=torch.long, device=images.device
            )
            finished = torch.zeros(images.shape[0], dtype=torch.bool, device=images.device)
            for _ in range(max_length - 1):
                next_token = self.decode_tokens(memory, generated)[:, -1].argmax(dim=-1)
                next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
                generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
                finished |= next_token.eq(eos_id)
                if bool(finished.all()):
                    break
            return generated

        @torch.no_grad()
        def beam_decode(
            self,
            images,
            *,
            bos_id: int,
            eos_id: int,
            max_length: int,
            beam_width: int,
        ):
            """Return raw score-ranked token beams without chemistry-based filtering."""
            if beam_width < 1 or max_length < 2 or max_length > self.max_sequence_length:
                raise ValueError("Invalid beam width or decode length")
            self.eval()
            memory = self.encode_images(images)
            batch_size = images.shape[0]
            generated = torch.full(
                (batch_size, beam_width, 1), bos_id, dtype=torch.long, device=images.device
            )
            scores = torch.full(
                (batch_size, beam_width), float("-inf"), device=images.device
            )
            scores[:, 0] = 0.0
            finished = torch.zeros(batch_size, beam_width, dtype=torch.bool, device=images.device)
            repeated_memory = memory.repeat_interleave(beam_width, dim=0)
            for _ in range(max_length - 1):
                current_length = generated.shape[-1]
                logits = self.decode_tokens(
                    repeated_memory, generated.reshape(batch_size * beam_width, current_length)
                )[:, -1]
                vocab_size = logits.shape[-1]
                log_probabilities = torch.log_softmax(logits, dim=-1).reshape(
                    batch_size, beam_width, vocab_size
                )
                log_probabilities = log_probabilities.masked_fill(
                    finished.unsqueeze(-1), float("-inf")
                )
                log_probabilities[:, :, eos_id] = torch.where(
                    finished,
                    torch.zeros_like(scores),
                    log_probabilities[:, :, eos_id],
                )
                candidates = scores.unsqueeze(-1) + log_probabilities
                scores, flat_indices = candidates.reshape(batch_size, -1).topk(beam_width, dim=-1)
                parent_indices = flat_indices // vocab_size
                next_tokens = flat_indices % vocab_size
                generated = torch.gather(
                    generated,
                    1,
                    parent_indices.unsqueeze(-1).expand(-1, -1, current_length),
                )
                generated = torch.cat((generated, next_tokens.unsqueeze(-1)), dim=-1)
                finished = torch.gather(finished, 1, parent_indices) | next_tokens.eq(eos_id)
                if bool(finished.all()):
                    break
            return generated, scores

    return TinyOCSRModel()
