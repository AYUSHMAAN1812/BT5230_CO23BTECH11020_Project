"""A deterministic, chemistry-aware tokenizer for SMILES sequences."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|Se|Na|Li|Ca|Mg|Al|Fe|Zn|Cu|Mn|Hg|Ag|Au|Sn|As|"
    r"se|as|@@?|%[0-9]{2}|[0-9]|[BCNOPSFIKVYWHbcnops]|\(|\)|\.|=|#|-|\+|\\|/|:|~|\?|>|\*|\$)"
)

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)


def tokenize_smiles(smiles: str) -> list[str]:
    """Tokenize SMILES and fail if any input character is unaccounted for."""
    tokens = TOKEN_PATTERN.findall(smiles)
    if "".join(tokens) != smiles:
        raise ValueError(f"SMILES tokenizer could not losslessly tokenize: {smiles!r}")
    return tokens


@dataclass
class SmilesTokenizer:
    token_to_id: dict[str, int]

    def __post_init__(self) -> None:
        missing = [token for token in SPECIAL_TOKENS if token not in self.token_to_id]
        if missing:
            raise ValueError(f"Tokenizer vocabulary is missing special tokens: {missing}")
        if sorted(self.token_to_id.values()) != list(range(len(self.token_to_id))):
            raise ValueError("Tokenizer ids must be contiguous and start at zero")
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}

    @classmethod
    def build(cls, smiles_values: Iterable[str]) -> SmilesTokenizer:
        vocabulary: set[str] = set()
        for smiles in smiles_values:
            vocabulary.update(tokenize_smiles(smiles))
        ordered = list(SPECIAL_TOKENS) + sorted(vocabulary.difference(SPECIAL_TOKENS))
        return cls({token: index for index, token in enumerate(ordered)})

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(
        self, smiles: str, *, add_special_tokens: bool = True, max_length: int | None = None
    ) -> list[int]:
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokenize_smiles(smiles)]
        if add_special_tokens:
            ids = [self.bos_id, *ids, self.eos_id]
        if max_length is not None and len(ids) > max_length:
            raise ValueError(
                f"Encoded sequence length {len(ids)} exceeds configured maximum {max_length}"
            )
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        tokens: list[str] = []
        for index in ids:
            token = self.id_to_token.get(int(index), UNK_TOKEN)
            if token == EOS_TOKEN:
                break
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return "".join(tokens)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"token_to_id": self.token_to_id}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> SmilesTokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({str(k): int(v) for k, v in payload["token_to_id"].items()})

