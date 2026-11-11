# Output (Same on any machine)

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