# Throughput from different GPUs

## Apple Silicon M5 Pro
```
$ uv run python 05_throughput.py
Loading weights: 100%|██████████████████████| 291/291 [00:02<00:00, 108.85it/s]
run 1: 256 tokens in 14.88 s -> 17.2 tok/s
run 2: 256 tokens in 14.87 s -> 17.2 tok/s
run 3: 256 tokens in 14.86 s -> 17.2 tok/s

mps: median 17.2 tok/s (end-to-end, includes prefill)
effective bandwidth ≈ 276 GB/s (weights read once per token)
```

## NVIDIA RTX 4090 (upcoming)
.....

## NVIDIA H100 (upcoming)