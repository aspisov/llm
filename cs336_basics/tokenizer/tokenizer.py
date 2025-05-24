import base64
import json
from collections.abc import Iterable, Iterator

import numpy as np
import regex as re
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.id_to_byte = vocab
        self.byte_to_id = {v: k for k, v in self.id_to_byte.items()}
        self.merges = merges
        self.merge_priorities = {pair: i for i, pair in enumerate(self.merges)}
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_json = json.load(f)
        vocab = {
            int(token_id): base64.b64decode(token_str.encode("utf-8")) for token_id, token_str in vocab_json.items()
        }

        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n\r")  # Only remove newlines, not spaces/tabs
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) == 2:
                    merges.append(
                        (base64.b64decode(parts[0].encode("utf-8")), base64.b64decode(parts[1].encode("utf-8")))
                    )

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode_pretoken(self, pretoken: str) -> list[int]:
        tokens = [bytes([b]) for b in pretoken.encode("utf-8")]

        while len(tokens) > 1:
            merge_pairs = []
            for pair in zip(tokens, tokens[1:]):
                if pair in self.merge_priorities:
                    merge_pairs.append((self.merge_priorities[pair], pair))

            if not merge_pairs:
                return [self.byte_to_id[b] for b in tokens]

            _, merge_pair = min(merge_pairs)

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == merge_pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return [self.byte_to_id[b] for b in tokens]

    def encode(self, text: str) -> list[int]:
        ids = []
        if not self.special_tokens:
            for pretoken in tqdm(re.finditer(PAT, text)):
                ids.extend(self.encode_pretoken(pretoken.group()))
            return ids

        delimiters = "|".join(re.escape(token) for token in self.special_tokens)
        for chunk in tqdm(re.split(f"({delimiters})", text)):
            if not chunk:
                continue

            if chunk in self.special_tokens:
                ids.append(self.byte_to_id[chunk.encode("utf-8")])
            else:
                for pretoken in re.finditer(PAT, chunk):
                    ids.extend(self.encode_pretoken(pretoken.group()))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        byte_sequence = b"".join(self.id_to_byte[id] for id in ids)
        text = byte_sequence.decode("utf-8", errors="replace")
        return text

    def serialize(self, path: str, ids: list[int]):
        ids_np = np.array(ids, dtype=np.uint16)
        np.savez_compressed(path, ids_np)
