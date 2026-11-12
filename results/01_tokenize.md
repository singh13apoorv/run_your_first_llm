# Output (Same on any machine)

## Apple Silicon M5 Pro (MPS)
```
$ uv run python 01_tokenize.py
Token ids:      [128000, 791, 6864, 315, 9822, 374]
Tokens:         ['<|begin_of_text|>', 'The', 'Ġcapital', 'Ġof', 'ĠFrance', 'Ġis']
Decoded:        <|begin_of_text|>The capital of France is
BOS:            <|begin_of_text|> (128000)
EOS:            <|end_of_text|> (128001)
Vocab size:     128256
```

> Six tokens: BOS (128000) plus five words. `Ġ` marks a leading space, so each word after the first is a single token including its space.

## NVIDIA RTX 4090 (CUDA)
torch 2.15.0+cu132 · CUDA 13.2 · compute capability 8.9
```
Token ids:      [128000, 791, 6864, 315, 9822, 374]
Tokens:         ['<|begin_of_text|>', 'The', 'Ġcapital', 'Ġof', 'ĠFrance', 'Ġis']
Decoded:        <|begin_of_text|>The capital of France is
BOS:            <|begin_of_text|> (128000)
EOS:            <|end_of_text|> (128001)
Vocab size:     128256
```