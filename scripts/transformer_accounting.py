def count_parameters(
    vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, batch_size: int = 1
) -> int:
    embeddings = vocab_size * d_model
    mha = d_model**2 * 4 + d_model * 2
    ffn = d_model * d_ff * 3
    final_ln = d_model
    lm_head = d_model * vocab_size

    total = embeddings + (mha + ffn) * num_layers + final_ln + lm_head
    return total


def count_gradients(
    vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, batch_size: int = 1
) -> int:
    embeddings = vocab_size * d_model
    mha = d_model**2 * 4 + d_model * 2
    ffn = d_model * d_ff * 3
    final_ln = d_model
    lm_head = d_model * vocab_size

    total = embeddings + (mha + ffn) * num_layers + final_ln + lm_head
    return total


def count_optim(
    vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, batch_size: int = 1
) -> int:
    embeddings = vocab_size * d_model
    mha = d_model**2 * 4 + d_model * 2
    ffn = d_model * d_ff * 3
    final_ln = d_model
    lm_head = d_model * vocab_size

    total = embeddings + (mha + ffn) * num_layers + final_ln + lm_head
    return total * 2


def count_activations(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    batch_size: int = 1,
) -> int:
    # Transformer block (per layer)
    rms_norms = batch_size * context_length * d_model * 2
    residuals = batch_size * context_length * d_model * 2

    # MHA (properly accounting for heads)
    qkv_projections = batch_size * context_length * d_model * 3
    scores = batch_size * num_heads * context_length * context_length
    attn_weights = batch_size * num_heads * context_length * context_length
    output_projection = batch_size * context_length * d_model

    # FFN
    w_1_multiply = batch_size * context_length * d_ff
    w_3_multiply = batch_size * context_length * d_ff
    silu = batch_size * context_length * d_ff
    w_2_multiply = batch_size * context_length * d_model
    w_1_w_3_elementwise = batch_size * context_length * d_ff  # SiLU(W1) * W3

    # Final components
    final_rms_norm = batch_size * context_length * d_model
    output_embeddings = batch_size * context_length * vocab_size
    cross_entropy = batch_size * context_length

    mha = qkv_projections + scores + attn_weights + output_projection
    ffn = w_1_multiply + w_2_multiply + w_3_multiply + silu + w_1_w_3_elementwise
    transformer_block = rms_norms + residuals + mha + ffn
    total = transformer_block * num_layers + final_rms_norm + output_embeddings + cross_entropy

    return total


def calculate_memory(
    vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, batch_size: int = 1
) -> None:
    parameters = count_parameters(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, batch_size)
    gradients = count_gradients(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, batch_size)
    optimizer_states = count_optim(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, batch_size)
    activations = count_activations(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, batch_size)
    print(f"Parameters: {parameters * 4 / 10**9:.02f} GB")
    print(f"Gradients: {gradients * 4 / 10**9:.02f} GB")
    print(f"Optimizer states: {optimizer_states * 4 / 10**9:.02f} GB")
    print(f"Activations: {activations * 4 / 10**9:.02f} GB")
    print(f"Total: {(parameters + gradients + optimizer_states + activations) * 4 / 10**9:.02f} GB")


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
    calculate_memory(
        vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=6400, batch_size=2
    )
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
