from cs336_basics.tokenizer.bpe_trainer import train_and_save_bpe

if __name__ == "__main__":
    train_and_save_bpe(
        input_path="data/owt_valid.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
        vocab_save_path="outputs/tokenizers/owt_bpe_vocab.json",
        merges_save_path="outputs/tokenizers/owt_bpe_merges.txt",
    )
