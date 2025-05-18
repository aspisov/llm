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