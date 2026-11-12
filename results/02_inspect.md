# Output (Parameters, Tensors, Dtype and Memory should be same across devices)

## Apple Silicon M5 Pro

```
$ uv run python 02_inspect.py
Loading weights: 100%|██████████████████████| 291/291 [00:02<00:00, 136.10it/s]
Device:      mps
Parameters:  8,030,261,248
Tensors:     291
Dtype:       torch.bfloat16
Memory:      16.07 GB allocated
Weights on:  mps:0
```

> On MPS, `torch.mps.current_allocated_memory()` reports 0 GB after loading, because the weights bypass PyTorch's allocator on unified memory; `driver_allocated_memory()` shows the real 16.07 GB.

## NVIDIA RTX 4090

_To be filled after the GPU run._