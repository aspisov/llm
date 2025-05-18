import multiprocessing
import os
from collections import Counter
from typing import BinaryIO

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def find_most_common_pair(
    word_counts: dict[tuple[int, ...], int], vocab: dict[int, bytes]
):
    """Most common pair search"""
    pair_counts = Counter()

    for word, count in word_counts.items():
        for token1, token2 in zip(word, word[1:]):
            pair_counts[(token1, token2)] += count

    max_freq = max(pair_counts.values())

    max_freq_pairs = [
        pair for pair, freq in pair_counts.items() if freq == max_freq
    ]

    return max(max_freq_pairs, key=lambda x: (vocab[x[0]], vocab[x[1]]))


def update_word_counts(
    word_counts: dict[tuple[int, ...], int],
    pair: tuple[int, int],
    next_token: int,
) -> dict[tuple[int, ...], int]:
    """Updates counts of words."""
    new_counts = Counter()
    for word, count in word_counts.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(next_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_counts[tuple(new_word)] = count
    return new_counts


def pre_tokenize(
    docs: str, special_tokens: list[str]
) -> dict[tuple[int, ...], int]:
    """Pretokenization"""
    delimeters = "|".join(re.escape(token) for token in special_tokens)
    word_counts = Counter()
    for doc in re.split(delimeters, docs):
        pre_tokens = [
            tuple(list(pre_token.encode("utf-8")))
            for pre_token in re.findall(PAT, doc)
        ]
        word_counts.update(pre_tokens)
    return word_counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train BPE tokenizer"""

    vocab = {idx: bytes([idx]) for idx in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    with open(input_path, "r") as f:
        docs = f.read()

    word_counts = pre_tokenize(docs=docs, special_tokens=special_tokens)

    # ----------------------------- MERGES -------------------------------------
    next_token = 256
    for _ in range(vocab_size - next_token - len(special_tokens)):
        pair = find_most_common_pair(word_counts, vocab)
        merges.append((vocab[pair[0]], vocab[pair[1]]))
        vocab[next_token] = vocab[pair[0]] + vocab[pair[1]]

        word_counts = update_word_counts(
            word_counts=word_counts, pair=pair, next_token=next_token
        )
        next_token += 1

    for special_token in special_tokens:
        vocab[next_token] = special_token.encode("utf-8")
        next_token += 1

    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe(
        "data/TinyStoriesV2-GPT4-valid.txt", 300, ["<|endoftext|>"]
    )
    print(vocab, merges, sep="\n-----------------------------------------\n")
