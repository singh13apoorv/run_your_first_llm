# 04_stream.py output

Greedy, so identical on every run; it matches `03_generate.py`'s greedy output, extended to 64 tokens. The text is identical on all three machines.

The text appears a few characters at a time rather than all at once: each piece is one full pass through the model's 16 GB of weights. The rhythm is visibly faster on each machine in turn (M5 Pro, then RTX 4090, then H100), and `05_throughput.py` puts numbers on it.

## Apple M5 Pro, 48 GB (MPS)

```
$ uv run python 04_stream.py
The capital of France is a city of many faces. It is a city of history, a city of culture, a city of art, a city of fashion, a city of food, a city of romance, a city of architecture, a city of music, a city of literature, a city of science, a city of education, a
```

## NVIDIA RTX 4090, 24 GB (CUDA)

torch 2.15.0+cu132 · CUDA 13.2 · compute capability 8.9

```
The capital of France is a city of many faces. It is a city of history, a city of culture, a city of art, a city of fashion, a city of food, a city of romance, a city of architecture, a city of music, a city of literature, a city of science, a city of education, a
```

## NVIDIA H100 SXM, 80 GB HBM3 (CUDA)

torch 2.15.0+cu132 · CUDA 13.2 · compute capability 9.0

```
The capital of France is a city of many faces. It is a city of history, a city of culture, a city of art, a city of fashion, a city of food, a city of romance, a city of architecture, a city of music, a city of literature, a city of science, a city of education, a
```