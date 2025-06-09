## Problem (unicode1): Understanding Unicode (1 point)

(a) What Unicode character does `chr(0)` return?
* Deliverable: Nothing, it's a unicode null character.

(b) How does this character’s string representation (`__repr__()`) differ from its printed representation?
* Deliverable: '\x00'

(c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:
```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```
* Deliverable: It is invisible, however it is still there.

## Problem (unicode2): Unicode Encodings (3 points)

(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

* Deliverable: UTF-16 and UTF-32 are sparse, which will result into longer sequences (more bytes).

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")) 
'hello'
```

* Deliverable: `"привет".encode("utf-8")`, it works on bytes individually, however UTF-8 is variable-width, thus some of the bytes will end up being invalid.

(c) Give a two byte sequence that does not decode to any Unicode character(s).

* Deliverable: `b'\xf0\x9f'`, UTF-8 has some preserved sequences of 2 bytes that mean that there is going to be 3rd byte after these two.

## Problem (train_bpe): BPE Tokenizer Training (15 points)

- Deliverable: Write a function that, given a path to an input text file, trains a (byte-level) BPE tokenizer. Your BPE training function should handle (at least) the following input parameters:

`input_path: str` Path to a text file with BPE tokenizer training data.

`vocab_size: int` A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).

`special_tokens: list[str]` A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training.

Your BPE training function should return the resulting vocabulary and merges:

`vocab: dict[int, bytes]` The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).

`merges: list[tuple[bytes, bytes]]` A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.

To test your BPE training function against our provided tests, you will first need to implement the test adapter at `[adapters.run_train_bpe]`. Then, run `uv run pytest tests/test_train_bpe.py`. Your implementation should be able to pass all tests. Optionally (this could be a large time-investment), you can implement the key parts of your training method using some systems language, for instance C++ (consider cppyy for this) or Rust (using PyO3). If you do this, be aware of which operations require copying vs reading directly from Python memory, and make sure to leave build instructions, or make sure it builds using only pyproject.toml. Also note that the GPT-2 regex is not well-supported in most regex engines and will be too slow in most that do. We have verified that Oniguruma is reasonably fast and supports negative lookahead, but the regex package in Python is, if anything, even faster.

[code](cs336_basics/tokenizer/bpe.py)

## Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

(a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories `<|endoftext|>` special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How many hours and memory did training take? What is the longest token in the vocabulary? Does it make sense?

Resource requirements: ≤ 30 minutes (no GPUs), ≤ 30GB RAM Hint You should be able to get under 2 minutes for BPE training using multiprocessing during pretokenization and the following two facts:

(a) The <|endoftext|> token delimits documents in the data files.

(b) The <|endoftext|> token is handled as a special case before the BPE merges are applied.

- Deliverable: It took 1 minute 10 seconds to train, about 1GB of RAM. The longest token is `b' accomplishment'`.

(b) Profile your code. What part of the tokenizer training process takes the most time?

- Deliverable: Loading and counting pretokens takes the most time.

## Problem (tokenizer_experiments): Experiments with tokenizers (4 points)

(a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?

- Deliverable: 
    - TinyStories: 4.15 bytes/token
    - OpenWebText: 4.51 bytes/token


(b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.

- Deliverable: 3.41, it drops significantly.

(c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?

- Deliverable: Around 1 million bytes/second, it would take 572 days to tokenize Pile dataset.

(d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. We’ll use this later to train our language model. We recommend serializing the token IDs as a NumPy array of datatype uint16. Why is uint16 an appropriate choice?

- Deliverable: because `uint16` has supports values from 0 to 65535, which is just enough for all our ids.



## Problem (transformer_accounting): Transformer LM resource accounting (5 points)

(a) Consider GPT-2 XL, which has the following configuration:


- `vocab_size` : 50,257
- `context_length` : 1,024
- `num_layers` : 48 
- `d_model` : 1,600
- `num_heads` : 25 
- `d_ff` : 6,400

Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

- Deliverable: `vocab_size` * `d_model` + (`d_model`^2 * 4 + `d_model` * 2 + `d_model` * `d_ff` * 3) * `num_layers` + `d_model` + `d_model` * `vocab_size` = $50257 \cdot 1600 + (1600^2 \cdot 4 + 1600 \cdot 2 + 1600 \cdot 6400 \cdot 3) \cdot 48 + 1600 + 1600 \cdot 50257 = 2 127 057 600$ $\implies$ **8.51 GB**

(b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.

Deliverable: 
- Matrix multiplies: 
  - Embeddings: 0
  - Transformer Block:
    - MHA: 
      - q_proj @ x: 2 * d_model * d_model * seq_len
      - k_proj @ x: 2 * d_model * d_model * seq_len
      - v_proj @ x: 2 * d_model * d_model * seq_len
      - Q @ K^T: 2 * seq_len * d_model * seq_len
      - attn @ V: 2 * seq_len * seq_len * d_model
      - output_proj @ attn_output: 2 * seq_len * d_model * d_model
    - FFN: 
      - w1 @ x: 2 * seq_len * d_ff * d_model
      - w2 @ x: 2 * seq_len * d_ff * d_model
      - w3 @ h: 2 * d_model * d_ff * seq_len
  - LM Head: 2 * seq_len * d_model * vocab_size
- Total: (2 * 1024 * 1600 * 1600 * 3 + 2 * 1024^2 * 1600 + 2 * 1024^2 * 1600 + 2 * 1024 * 1600 * 1600 + 2 * 1024 * 6400 * 1600 * 3) * 48 + 2 * 1024 * 1600 * 50257 = 4 513 336 524 800 FPOPS = 4.51 TFLOPS
  - MHA: 1 328 755 507 200
  - FFN: 3 019 898 880 000
  - LM Head: 164 682 137 600


(c) Based on your analysis above, which parts of the model require the most FLOPs?

- Deliverable: FFN

(d) Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?

Deliverable: For each model, provide a breakdown of model components and its associated FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide a one-to-two sentence description of how varying the model size changes the proportional FLOPs of each component.
- GPT-2 small: 0.54 TFLOPS
  - MHA: 17.96% 
  - FFN: 67.35% 
  - LM HEAD: 14.69%
- GPT-2 medium: 1.38 TFLOPS
  - MHA: 22.39% 
  - FFN: 69.98% 
  - LM HEAD: 7.63%
- GPT-2 large: 2.62 TFLOPS
  - MHA: 25.82%
  - FFN: 69.15%
  - LM HEAD: 5.03%
- GPT-2 XL: 4.51 TFLOPS
  - MHA: 29.44% 
  - FFN: 66.91% 
  - LM HEAD: 3.65%

As model size increases LM head becomes less and less percentage of total FLOPS.

(e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How do the relative contribution of FLOPs of the model components change?

- Deliverable: 
GPT-2 XL with 16384 context length: 149.52 TFLOPS
  - MHA: 65.92% 
  - FFN: 32.32% 
  - LM HEAD: 1.76%

With large context length we get a much greater impact of MHA on total FLOPS count because it has context length squared term.


## Problem (adamwAccounting): Resource accounting for training with AdamW (2 points)

Let us compute how much memory and compute running AdamW requires. Assume we are using float32 for every tensor.

(a) How much peak memory does running AdamW require? Decompose your answer based on the memory usage of the parameters, activations, gradients, and optimizer state. Express your answer in terms of the batch_size and the model hyperparameters (vocab_size, context_length, num_layers, d_model, num_heads). Assume d_ff = 4 × d_model.

For simplicity, when calculating memory usage of activations, consider only the following components:

- Transformer block
  - RMSNorm(s)
  - Multi-head self-attention sublayer: $QKV$ projections, $Q^T K$ matrix multiply, softmax, weighted sum of values, output projection.
  - Position-wise feed-forward: $W_1$ matrix multiply, $\mathrm{SiLU}$, $W_2$ matrix multiply
- final RMSNorm
- output embedding
- cross-entropy on logits

- Deliverable: [script](scripts/transformer_accounting.py)



(b) Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on the batch_size. What is the maximum batch size you can use and still fit within 80GB memory?

- Deliverable: [script](scripts/transformer_accounting.py), 2

(c) How many FLOPs does running one step of AdamW take?

Deliverable: An algebraic expression, with a brief justification.

(d) Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second) relative to the hardware’s theoretical peak FLOP throughput [Chowdhery et al., 2022]. An NVIDIA A100 GPU has a theoretical peak of 19.5 teraFLOP/s for float32 operations. Assuming you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a batch size of 1024 on a single A100? Following Kaplan et al. [2020] and Hoffmann et al. [2022], assume that the backward pass has twice the FLOPs of the forward pass.

Deliverable: The number of days training would take, with a brief justification.