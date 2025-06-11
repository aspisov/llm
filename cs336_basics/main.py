import torch

from cs336_basics.model import Transformer
from cs336_basics.tokenizer import Tokenizer

tokenizer = Tokenizer.from_files(
    "outputs/tokenizers/tinystories_bpe_vocab.json",
    "outputs/tokenizers/tinystories_bpe_merges.txt",
    ["<|endoftext|>"],
)
llm = Transformer(10000, 1024, 12, 768, 12, 768 * 4, 10, torch.device("mps"), torch.float32)


response = llm.generate_text(prompt="I'm a language model", tokenizer=tokenizer)

print(response)
