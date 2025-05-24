import base64
import json
import multiprocessing
import os
import time
from collections import Counter, defaultdict
from typing import BinaryIO

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

NUM_PROCESSES = 11


def count_pretokens_in_chunk(args) -> dict[str, int]:
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    delimeters = "|".join(re.escape(token) for token in special_tokens)
    pretoken_counts = Counter()
    for doc in re.split(delimeters, chunk):
        pre_tokens = [tuple(list(pre_token.encode("utf-8"))) for pre_token in re.findall(PAT, doc)]
        pretoken_counts.update(pre_tokens)
    return pretoken_counts


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

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


def find_most_common_pair(pair_counts: dict[tuple[int, int], int], vocab: dict[int, bytes]) -> tuple[int, int]:
    """Most common pair search"""
    max_cnt = max(pair_counts.values())

    return max(
        (pair for pair, cnt in pair_counts.items() if cnt == max_cnt),
        key=lambda x: (vocab[x[0]], vocab[x[1]]),
    )


def update_pairs(
    pretokens: list[tuple[tuple[int, ...], int]],
    pair_counts: dict[tuple[int, int], int],
    pair_ids: dict[tuple[int, int], set[int]],
    pair: tuple[int, int],
    token: int,
) -> tuple[
    list[tuple[tuple[int, ...], int]],
    dict[tuple[int, int], int],
    dict[tuple[int, int], set[int]],
]:
    """Update pair counts."""
    indices_to_process = list(pair_ids.get(pair, set()))
    for idx in indices_to_process:
        pretoken, count = pretokens[idx]

        # subtract old
        for token1, token2 in zip(pretoken, pretoken[1:]):
            pair_counts[(token1, token2)] -= count
            pair_ids[(token1, token2)].discard(idx)
            if pair_counts[(token1, token2)] == 0:
                del pair_counts[(token1, token2)]

        # create new pretoken
        new_pretoken = []
        i = 0
        while i < len(pretoken):
            if i + 1 < len(pretoken) and (pretoken[i], pretoken[i + 1]) == pair:
                new_pretoken.append(token)
                i += 2
            else:
                new_pretoken.append(pretoken[i])
                i += 1

        # add new
        for token1, token2 in zip(new_pretoken, new_pretoken[1:]):
            pair_counts[(token1, token2)] += count
            pair_ids[(token1, token2)].add(idx)

        pretokens[idx] = tuple(new_pretoken), count

    del pair_counts[pair]
    del pair_ids[pair]

    return pretokens, pair_counts, pair_ids


def create_pairs(
    pretokens: list[tuple[tuple[int, ...], int]],
) -> tuple[
    dict[tuple[int, int], int],
    dict[tuple[int, int], set[int]],
]:
    pair_counts = Counter()
    pair_ids: dict[tuple[int, int], set[int]] = defaultdict(set)
    for i, (pretoken, count) in enumerate(pretokens):
        for token1, token2 in zip(pretoken, pretoken[1:]):
            pair_counts[(token1, token2)] += count
            pair_ids[(token1, token2)].add(i)

    return pair_counts, pair_ids


def pretokenize(input_path: str | os.PathLike, special_tokens: list[str]) -> list[tuple[tuple[int, ...], int]]:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, special_tokens[0].encode("utf-8"))

    jobs = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with multiprocessing.Pool(NUM_PROCESSES) as pool:
        results = pool.map(count_pretokens_in_chunk, jobs)

    pretoken_counts = Counter()
    for partial in results:
        pretoken_counts.update(partial)

    return [(pretoken, count) for pretoken, count in pretoken_counts.items()]


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train BPE tokenizer"""

    vocab = {idx: bytes([idx]) for idx in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    # --------------------------- PRETOKENS ------------------------------------
    pretokens = pretokenize(input_path=input_path, special_tokens=special_tokens)

    # ----------------------------- PAIRS --------------------------------------
    pair_counts, pair_ids = create_pairs(pretokens=pretokens)

    # ----------------------------- MERGES -------------------------------------
    next_token = 256
    for _ in range(vocab_size - next_token - len(special_tokens)):
        if not pair_counts:
            break

        pair = find_most_common_pair(pair_counts=pair_counts, vocab=vocab)

        merges.append((vocab[pair[0]], vocab[pair[1]]))
        vocab[next_token] = vocab[pair[0]] + vocab[pair[1]]

        pretokens, pair_counts, pair_ids = update_pairs(
            pretokens=pretokens,
            pair_counts=pair_counts,
            pair_ids=pair_ids,
            pair=pair,
            token=next_token,
        )

        next_token += 1

    for special_token in special_tokens:
        vocab[next_token] = special_token.encode("utf-8")
        next_token += 1

    return vocab, merges


def train_and_save_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str], vocab_save_path: str, merges_save_path: str
):
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    vocab_json = {k: base64.b64encode(v).decode("utf-8") for k, v in vocab.items()}

    with open(vocab_save_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    with open(merges_save_path, "w", encoding="utf-8") as f:
        for pair in merges:
            f.write(f"{base64.b64encode(pair[0]).decode('utf-8')}\t{base64.b64encode(pair[1]).decode('utf-8')}\n")


if __name__ == "__main__":
    prefix = "tiny_stories"

    vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 32000, ["<|endoftext|>"])

    vocab_path = "outputs/tokenizers/new_vocab.json"
    merges_path = "outputs/tokenizers/new_merges.txt"
