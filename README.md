# Run Your First LLM

Run Llama 3.1 8B from scratch with PyTorch and Hugging Face Transformers, serve it over HTTP, and measure how fast it really is on different hardware: an Apple Silicon laptop and rented NVIDIA GPUs.

Everything here is pinned (Python, every library, and the exact model snapshot), so anyone with this repository and a suitable machine gets the same results.

Built while working through Module 01 ("Run Your First LLM") of the [Fanout Inference Engineering](https://fanout.sh) course.

## Results

| Device | Decode speed | Effective bandwidth | Spec bandwidth | % of peak |
|---|---|---|---|---|
| Apple M5 Pro (MPS) | 17.2 tok/s | 276 GB/s | 307 GB/s | 90% |
| NVIDIA RTX 4090 (CUDA) | _pending_ | _pending_ | 1,008 GB/s | _pending_ |
| NVIDIA H100 (CUDA) | _pending_ | _pending_ | 3,350 GB/s | _pending_ |

Llama 3.1 8B in BF16, greedy decoding, 256 new tokens, median of 3 runs after a warm-up. See [`examples/05_throughput.py`](examples/05_throughput.py).

**Why bandwidth?** Generating one token reads all ~16 GB of weights from memory once. The math itself is quick, so decoding is limited by memory bandwidth: the speed ceiling is roughly `bandwidth ÷ 16 GB` tokens per second. "Effective bandwidth" is `tok/s × 16.06 GB`, and the last column shows how close each machine gets to its hardware limit.

## What's inside

```
run_your_first_llm/
├── server.py            # FastAPI server: /generate and /stream
├── run.sh               # one command: sync the environment, start the server
├── examples/
│   ├── common.py        # device selection, pinned model, loading helpers
│   ├── 01_tokenize.py   # text → token IDs and back
│   ├── 02_inspect.py    # parameters, tensors, dtype, memory
│   ├── 03_generate.py   # greedy vs sampled vs seeded generation
│   ├── 04_stream.py     # stream tokens as they're generated
│   └── 05_throughput.py # tokens/s and effective bandwidth
├── pyproject.toml
└── uv.lock              # every dependency pinned
```

Each example has a matching `*_out.md` file with real output from my runs.

## Requirements

- **Hardware**, one of:
  - An NVIDIA GPU with at least 24 GB of VRAM and compute capability 8.0 or higher (needed for BF16): RTX 3090/4090, A10, L4, A100, H100, and so on. The driver must support **CUDA 13.2 or newer** (check the top-right of `nvidia-smi`).
  - An Apple Silicon Mac with at least 32 GB of unified memory.
- [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and every pinned dependency.
- About 20 GB of free disk space for the model.

The lock file installs the right PyTorch build automatically: the CUDA 13.2 build on Linux, and the standard build with MPS (Apple GPU) support on macOS.

## Setup

```
git clone <repo-url> && cd run_your_first_llm
uv sync
uv run hf download NousResearch/Meta-Llama-3.1-8B \
    --revision 1f47e50cdbe801ad8a5174156ec3a0655108fb9f --exclude "original/*"
```

- **No login is needed.** `NousResearch/Meta-Llama-3.1-8B` is an ungated copy of Meta's base model.
- **The revision is pinned** so everyone downloads exactly the same files. The code loads this same revision.
- **`--exclude "original/*"`** skips a second copy of the weights in Meta's own format, which transformers doesn't use. That saves 16 GB.

## Examples

Run them from the `examples/` folder:

```
cd examples
uv run python 01_tokenize.py
```

| Script | What it shows | GPU? |
|---|---|---|
| `01_tokenize.py` | How text becomes token IDs: 6 tokens for "The capital of France is", including the start-of-text token | No |
| `02_inspect.py` | 8,030,261,248 parameters in 291 tensors, stored in BF16, using about 16 GB | Yes |
| `03_generate.py` | Greedy decoding repeats exactly; sampling varies unless you set a seed | Yes |
| `04_stream.py` | Text appearing token by token, one full pass through the weights for each | Yes |
| `05_throughput.py` | Tokens per second, and how close that comes to the hardware's bandwidth limit | Yes |

## Serving the model

### On your own machine

Terminal 1:
```
bash run.sh
```

Terminal 2:
```
# Full response at once
curl -s localhost:8000/generate -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_new_tokens": 64}'

# Streamed, token by token
curl -N localhost:8000/stream -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_new_tokens": 64}'
```

### On a remote GPU box

1. On the box, inside `tmux`, run the setup steps above, then `bash run.sh`.
2. On your laptop, open an SSH tunnel and keep it open:
   ```
   ssh -L 8000:localhost:8000 <your-ssh-alias>
   ```
3. In another terminal on your laptop, run the same curl commands as above.

The server listens only on `127.0.0.1`, so it's reachable through your SSH tunnel and not from the open internet.

### Server design

- **One request at a time.** A lock makes requests take turns on the GPU. That's simple and predictable, but it means a second user waits for the first to finish. Serving many users efficiently (batching) is what inference engines like vLLM are for.
- **Greedy decoding** (`do_sample=False`), so the same prompt always returns the same text.

Expected start of the response:
```
The capital of France is a city of many faces. It is a city of history, ...
```

This is a **base** model: it continues text rather than answering like an assistant, so "Paris" isn't the guaranteed next word. For chat-style answers, use `Meta-Llama-3.1-8B-Instruct`.

## Notes and findings

- **Greedy output matches across machines here:** the M5 Pro produced the same continuation the course recorded on an H100. That's typical but not guaranteed: tiny floating-point differences between hardware can change one token, and with it everything after.
- **Seeded sampling repeats only on the same machine.** Apple GPUs and NVIDIA GPUs use different random number generators, so the same seed gives different text on each.
- **Measuring memory on Apple Silicon:** after loading, `torch.mps.current_allocated_memory()` reports 0 GB, because the weights reach the GPU without going through PyTorch's allocator on unified memory. `torch.mps.driver_allocated_memory()` shows the real 16.07 GB. CUDA's `torch.cuda.memory_allocated()` counts the weights normally.
- **Timing GPU work:** GPU operations run asynchronously, so the throughput script waits for the GPU to finish (`synchronize`) before stopping the clock. Without that, timings come out too fast.

## License

The code in this repository is MIT licensed. The model weights are **not** included: they're downloaded from Hugging Face and covered by the [Llama 3.1 Community License](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/LICENSE).