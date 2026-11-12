# Throughput from different GPUs

## Apple Silicon M5 Pro (MPS)
```
$ uv run python 05_throughput.py
Loading weights: 100%|██████████████████████| 291/291 [00:02<00:00, 108.85it/s]
run 1: 256 tokens in 14.88 s -> 17.2 tok/s
run 2: 256 tokens in 14.87 s -> 17.2 tok/s
run 3: 256 tokens in 14.86 s -> 17.2 tok/s

mps: median 17.2 tok/s (end-to-end, includes prefill)
effective bandwidth ≈ 276 GB/s (weights read once per token)
```

## NVIDIA RTX 4090 (CUDA)
torch 2.15.0+cu132 · CUDA 13.2 · compute capability 8.9
```
Loading weights: 100%|██████████| 291/291 [00:04<00:00, 70.00it/s]
run 1: 256 tokens in 6.68 s -> 38.3 tok/s
run 2: 256 tokens in 6.55 s -> 39.1 tok/s
run 3: 256 tokens in 6.61 s -> 38.7 tok/s

cuda: median 38.7 tok/s (end-to-end, includes prefill)
effective bandwidth ≈ 622 GB/s (weights read once per token)
```

## NVIDIA H100 (upcoming)