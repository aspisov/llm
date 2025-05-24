import time

from cs336_basics.tokenizer.tokenizer import Tokenizer


def load_docs(path: str, n: int, special_token: str) -> str:
    docs = []
    with open(path, encoding="utf-8") as f:
        cur_doc = ""
        while len(docs) < n:
            line = f.readline()
            if special_token in line:
                cur_doc += line[: line.index(special_token) + len(special_token)]
                docs.append(cur_doc)
                cur_doc = line[line.index(special_token) + len(special_token) :]
            else:
                cur_doc += line
    return "\n".join(docs)


def save_encoded(input_path: str, output_path: str, tokenizer: Tokenizer):
    # throughput
    with open(input_path, encoding="utf-8") as f:
        corpus = f.read()

        start = time.time()
        total_bytes = len(corpus.encode("utf-8"))
        ids = tokenizer.encode(corpus)
        elapse_time = time.time() - start
        print(f"Throughput: {total_bytes / elapse_time} bytes/second")
        tokenizer.serialize(output_path, ids)
        print(f"Saved data to {output_path}")


def main():
    tinystories_path = "data/TinyStoriesV2-GPT4-valid.txt"

    docs = load_docs(tinystories_path, 10, "<|endoftext|>")

    tinystories_tokenizer = Tokenizer.from_files(
        "outputs/tokenizers/tinystories_bpe_vocab.json",
        "outputs/tokenizers/tinystories_bpe_merges.txt",
        ["<|endoftext|>"],
    )

    bytes = docs.encode("utf-8")
    tokens = tinystories_tokenizer.encode(docs)

    print("Compression ration for TinyStories:", len(bytes) / len(tokens))

    owt_path = "data/owt_valid.txt"

    docs = load_docs(owt_path, 10, "<|endoftext|>")

    owt_tokenizer = Tokenizer.from_files(
        "outputs/tokenizers/owt_bpe_vocab.json",
        "outputs/tokenizers/owt_bpe_merges.txt",
        ["<|endoftext|>"],
    )

    bytes = docs.encode("utf-8")
    tokens = owt_tokenizer.encode(docs)

    print("Compression ration for OWT:", len(bytes) / len(tokens))

    tokens = tinystories_tokenizer.encode(docs)
    print("Compression ration for OWT with TinyStories tokenizer:", len(bytes) / len(tokens))

    save_encoded(tinystories_path, "data/TinyStories-valid", tinystories_tokenizer)
    save_encoded(owt_path, "data/owt-valid", owt_tokenizer)


if __name__ == "__main__":
    main()
