# 05_throughput.py output

256 new tokens, greedy, three timed runs after a warm-up; the result is the median.

| Machine | Median | Effective bandwidth | Spec bandwidth | % of peak |
|---|---|---|---|---|
| Apple M5 Pro (MPS) | 17.2 tok/s | 276 GB/s | 307 GB/s | 90% |
| NVIDIA RTX 4090 | 38.7 tok/s | 622 GB/s | 1,008 GB/s | 62% |
| NVIDIA H100 SXM | 83.0 tok/s | 1,333 GB/s | 3,350 GB/s | 40% |

## Apple M5 Pro, 48 GB (MPS)

```
$ uv run python 05_throughput.py
run 1: 256 tokens in 14.88 s -> 17.2 tok/s
run 2: 256 tokens in 14.87 s -> 17.2 tok/s
run 3: 256 tokens in 14.86 s -> 17.2 tok/s

mps: median 17.2 tok/s (end-to-end, includes prefill)
effective bandwidth ≈ 276 GB/s (weights read once per token)
```

## NVIDIA RTX 4090, 24 GB (CUDA)

torch 2.15.0+cu132 · CUDA 13.2 · compute capability 8.9

```
run 1: 256 tokens in 6.68 s -> 38.3 tok/s
run 2: 256 tokens in 6.55 s -> 39.1 tok/s
run 3: 256 tokens in 6.61 s -> 38.7 tok/s

cuda: median 38.7 tok/s (end-to-end, includes prefill)
effective bandwidth ≈ 622 GB/s (weights read once per token)
```

## NVIDIA H100 SXM, 80 GB HBM3 (CUDA)

torch 2.15.0+cu132 · CUDA 13.2 · compute capability 9.0 · host CPU: Intel Xeon Platinum 8568Y+

```
run 1: 256 tokens in 9.11 s -> 28.1 tok/s
run 2: 256 tokens in 3.09 s -> 83.0 tok/s
run 3: 256 tokens in 3.02 s -> 84.6 tok/s

cuda: median 83.0 tok/s (end-to-end, includes prefill)
effective bandwidth ≈ 1333 GB/s (weights read once per token)
```

> Run 1 took three times as long as runs 2 and 3 even after the warm-up, so the H100 needed more than one generation to settle. Runs 2 and 3 agree within 2%, and the median uses them. On the Mac and the 4090 all three runs agree.

**Why % of peak falls on faster hardware:** at batch size 1, each token runs about a thousand small GPU kernels launched one by one from Python. On the Mac, those launches hide behind slow memory reads (90% of peak). On faster GPUs the memory reads finish sooner, so launch overhead takes a bigger share: 62% on the 4090 and 40% on the H100, where reading the weights takes only about 4.8 ms per token but each token takes about 12 ms.