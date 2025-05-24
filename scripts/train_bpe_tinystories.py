from cs336_basics.tokenizer.bpe_trainer import train_and_save_bpe

if __name__ == "__main__":
    train_and_save_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        vocab_save_path="outputs/tokenizers/tinystories_bpe_vocab.json",
        merges_save_path="outputs/tokenizers/tinystories_bpe_merges.txt",
    )
