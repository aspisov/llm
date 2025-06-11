import torch
import torch.nn as nn

from cs336_basics.model import Embedding, Linear, MultiHeadSelfAttention, RMSNorm, SwiGLU
from cs336_basics.model.inference import top_p_sampling
from cs336_basics.tokenizer import Tokenizer


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, rope_theta, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device)

        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.context_length = context_length

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    rope_theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

    def generate_text(
        self, prompt: str, tokenizer: Tokenizer, temperature: float = 1, top_p: float = 1, max_tokens: int = 5
    ):
        special_token_ids = {}
        if tokenizer.special_tokens:
            special_token_ids = set(tokenizer.encode("".join(tokenizer.special_tokens)))

        device = next(self.parameters()).device
        token_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

        for _ in range(max_tokens):
            if token_ids.shape[1] >= self.context_length:
                break

            logits = self.forward(token_ids)

            preds = top_p_sampling(logits=logits[:, -1, :], p=top_p, temperature=temperature)

            if preds[-1].item() in special_token_ids:
                break

            token_ids = torch.cat([token_ids, preds], dim=-1)

        return tokenizer.decode(token_ids.squeeze(0).cpu().tolist())
