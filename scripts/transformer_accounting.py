def count_parameters(vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int):
    embeddings = vocab_size * d_model
    mha = d_model**2 * 4 + d_model * 2
    ffn = d_model * d_ff * 3
    final_ln = d_model
    lm_head = d_model * vocab_size

    total = embeddings + (mha + ffn) * num_layers + final_ln + lm_head
    print(f"Total: {total * 4 / 10**9:.02f} GB")


def count_flops(vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int):
    embeddings = 0
    mha = (
        2 * d_model * d_model * context_length * 3
        + 2 * context_length**2 * d_model * 2
        + 2 * context_length * d_model * d_model
    )
    ffn = 2 * context_length * d_ff * d_model * 3
    lm_head = 2 * d_model * context_length * vocab_size

    total = embeddings + (mha + ffn) * num_layers + lm_head
    print(
        f"MHA: {mha * num_layers / total * 100:.02f}%, FFN: {ffn * num_layers / total * 100:.02f}%, LM HEAD: {lm_head / total * 100:.02f}%"
    )
    print(f"Total: {total / 10**12:.2f} TFLOPS")


if __name__ == "__main__":
    count_parameters(vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=6400)
    print("-" * 100, "\n", "GPT-2 small", sep="")
    count_flops(vocab_size=50257, context_length=1024, num_layers=12, d_model=768, num_heads=12, d_ff=6400)
    print("-" * 100, "\n", "GPT-2 medium", sep="")
    count_flops(vocab_size=50257, context_length=1024, num_layers=24, d_model=1024, num_heads=16, d_ff=6400)
    print("-" * 100, "\n", "GPT-2 large", sep="")
    count_flops(vocab_size=50257, context_length=1024, num_layers=36, d_model=1280, num_heads=20, d_ff=6400)
    print("-" * 100, "\n", "GPT-2 XL", sep="")
    count_flops(vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=6400)
    print("-" * 100, "\n", "GPT-2 XL with 16384 context length", sep="")
    count_flops(vocab_size=50257, context_length=16384, num_layers=48, d_model=1600, num_heads=25, d_ff=6400)
    