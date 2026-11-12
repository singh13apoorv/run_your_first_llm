# Output
**Greedy, so identical on every run; matches 03_generate.py's greedy output, extended to 64 tokens.**

The text appears a few characters at a time rather than all at once: each piece is one full pass
through the model's 16 GB of weights. On the M5 Pro the rhythm is visibly slower than on the RTX 4090,
and 05_throughput.py puts numbers on it.

## Apple Silicon M5 Pro (MPS)
```
$ uv run python 04_stream.py
Loading weights: 100%|███████████████████████| 291/291 [00:03<00:00, 95.90it/s]
The capital of France is a city of many faces. It is a city of history, a city of culture, a city of art, a city of fashion, a city of food, a city of romance, a city of architecture, a city of music, a city of literature, a city of science, a city of education, a
```

## NVIDIA RTX 4090 (CUDA)
torch 2.15.0+cu132 · CUDA 13.2 · compute capability 8.9
```
Loading weights: 100%|██████████| 291/291 [00:04<00:00, 72.63it/s]
The capital of France is a city of many faces. It is a city of history, a city of culture, a city of art, a city of fashion, a city of food, a city of romance, a city of architecture, a city of music, a city of literature, a city of science, a city of education, a
```