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

**Deliverable**: Write a function that, given a path to an input text file, trains a (byte-level) BPE tokenizer. Your BPE training function should handle (at least) the following input parameters:

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

* Deliverable: It took 1 minute 10 seconds to train, about 1GB of RAM. The longest token is `b' accomplishment'`.

(b) Profile your code. What part of the tokenizer training process takes the most time?

Deliverable: Loading and counting pretokens takes the most time.

