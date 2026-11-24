# 02_inspect.py output

Parameters, tensors, dtype and memory are the same on every machine. Only the memory-reporting call differs (see the MPS note).

## Apple M5 Pro, 48 GB (MPS)

```
$ uv run python 02_inspect.py
Device:      mps
Parameters:  8,030,261,248
Tensors:     291
Dtype:       torch.bfloat16
Memory:      16.07 GB allocated
Weights on:  mps:0
```

> On MPS, `torch.mps.current_allocated_memory()` reports 0 GB after loading, because the weights bypass PyTorch's allocator on unified memory; `driver_allocated_memory()` shows the real 16.07 GB.

## NVIDIA RTX 4090, 24 GB (CUDA)

torch 2.15.0+cu132 · CUDA 13.2 · compute capability 8.9

```
Device:      cuda
Parameters:  8,030,261,248
Tensors:     291
Dtype:       torch.bfloat16
Memory:      16.06 GB allocated
Weights on:  cuda:0
```

## NVIDIA H100 SXM, 80 GB HBM3 (CUDA)

torch 2.15.0+cu132 · CUDA 13.2 · compute capability 9.0

```
Device:      cuda
Parameters:  8,030,261,248
Tensors:     291
Dtype:       torch.bfloat16
Memory:      16.06 GB allocated
Weights on:  cuda:0
```