# Run Your First LLM

Run Llama 3.1 8B from scratch with PyTorch and Hugging Face Transformers, serve it over HTTP, and measure how fast it really is on different hardware: an Apple Silicon laptop and rented NVIDIA GPUs.

Everything here is pinned (Python, every library, and the exact model snapshot), so anyone with this repository and a suitable machine gets the same results.

Built while working through Module 01 ("Run Your First LLM") of the [Fanout Inference Engineering](https://fanout.sh) course.

## Results

| Device | Decode speed | Effective bandwidth | Spec bandwidth | % of peak |
|---|---|---|---|---|
| Apple M5 Pro, 48 GB unified memory (MPS) | 17.2 tok/s | 276 GB/s | 307 GB/s | 90% |
| NVIDIA RTX 4090, 24 GB (CUDA) | 38.7 tok/s | 622 GB/s | 1,008 GB/s | 62% |
| NVIDIA H100 SXM, 80 GB HBM3 (CUDA) | 83.0 tok/s | 1,333 GB/s | 3,350 GB/s | 40% |

Llama 3.1 8B in BF16, greedy decoding, 256 new tokens, median of 3 runs after a warm-up. The prompt is 6 tokens, so prefill is a negligible part of each run. See [`examples/05_throughput.py`](examples/05_throughput.py) and the raw output in [`results/05_throughput.md`](results/05_throughput.md).

Test machines: torch 2.15.0, with the CUDA 13.2 build on both GPUs (compute capability 8.9 and 9.0). The GPUs were rented on Vast.ai; the H100's host CPU was an Intel Xeon Platinum 8568Y+.

**Why bandwidth?** Generating one token reads all ~16 GB of weights from memory once. The math itself is quick, so decoding is limited by memory bandwidth: the speed ceiling is roughly `bandwidth ÷ 16 GB` tokens per second. "Effective bandwidth" is `tok/s × 16.06 GB`, and the last column shows how close each machine gets to its hardware limit.

**Why faster GPUs get further from their limit.** At batch size 1, each token runs about a thousand small GPU kernels, launched one by one from Python. On the Mac, those launches hide behind slow memory reads, so it reaches 90% of peak. On faster GPUs the memory reads finish sooner and launch overhead takes a bigger share: 62% on the 4090 and 40% on the H100. On the H100, reading the weights takes about 4.8 ms per token, but each token takes about 12 ms. The H100 is still the fastest here, at 2.1x the 4090, but well short of the 3.3x its bandwidth would allow. Removing that overhead (CUDA graphs, `torch.compile`, or an engine like vLLM) is the next step after this module.

## What's inside

```
run_your_first_llm/
├── src/run_your_first_llm/
│   ├── server.py          # FastAPI server: /generate and /stream
│   ├── run.sh             # one command: sync the environment, start the server
│   └── __init__.py
├── examples/
│   ├── common.py          # device selection, pinned model, loading helpers
│   ├── 01_tokenize.py     # text → token IDs and back
│   ├── 02_inspect.py      # parameters, tensors, dtype, memory
│   ├── 03_generate.py     # greedy vs sampled vs seeded generation
│   ├── 04_stream.py       # stream tokens as they're generated
│   └── 05_throughput.py   # tokens/s and effective bandwidth
├── results/
│   ├── 01_tokenize.md     # real output from each example, per machine
│   ├── 02_inspect.md
│   ├── 03_generate.md
│   ├── 04_stream.md
│   └── 05_throughput.md
├── run_all.sh             # detects the machine, runs examples 01-05, saves logs
├── serve.sh               # starts the server and saves its log
├── curl_server.sh         # sends both requests and saves the responses
├── server_out.md          # server logs and responses, per machine
├── pyproject.toml
└── uv.lock                # every dependency pinned
```

Each example's real output, for every machine, is in `results/`. `logs/` holds raw output from each run and isn't committed.

## Requirements

- **Hardware**, one of:
  - An NVIDIA GPU with at least 24 GB of VRAM and compute capability 8.0 or higher (needed for BF16): RTX 3090/4090, A10, L4, A100, H100, and so on. The driver must support **CUDA 13.2 or newer** (check the top-right of `nvidia-smi`).
  - An Apple Silicon Mac with at least 32 GB of unified memory.
- [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and every pinned dependency.
- About 20 GB of free disk space for the model.

The lock file installs the right PyTorch build automatically: the CUDA 13.2 build on Linux, and the standard build with MPS (Apple GPU) support on macOS.

## Setup

```
git clone https://github.com/singh13apoorv/run_your_first_llm.git
cd run_your_first_llm
uv sync
uv run hf download NousResearch/Meta-Llama-3.1-8B \
    --revision 1f47e50cdbe801ad8a5174156ec3a0655108fb9f --exclude "original/*"
```

Optional: log in to Hugging Face. It isn't needed for this ungated model, but it is for gated repos such as Meta's official `meta-llama/...` models (create a **Read** token under Settings → Access Tokens on huggingface.co):

```
uv run hf auth login
```

- **No login is needed.** `NousResearch/Meta-Llama-3.1-8B` is an ungated copy of Meta's base model.
- **The revision is pinned** so everyone downloads exactly the same files. The code loads this same revision.
- **`--exclude "original/*"`** skips a second copy of the weights in Meta's own format, which transformers doesn't use. That saves 16 GB.

## Examples

Run everything on the current machine with one command:

```
bash run_all.sh
```

It names the machine from its hardware (`m5pro`, `rtx4090`, `h100-sxm`, and so on), records the hardware and versions, and runs examples 01-05, each in its own process, saving the output to `logs/<machine>/`. Pass a name to override it: `bash run_all.sh mybox`.

Or run a single example from the `examples/` folder:

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
bash src/run_your_first_llm/run.sh
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

1. On the box, inside `tmux`, run the setup steps above, then `bash serve.sh`. It starts the server with `run.sh` and saves its log to `logs/<machine>/server.txt`.
2. On your laptop, open an SSH tunnel and keep it open:
   ```
   ssh -L 8000:localhost:8000 <your-ssh-alias>
   ```
3. In another terminal on your laptop, run `bash curl_server.sh <machine>`, or the same curl commands as above. The script saves both responses to `logs/<machine>/`.

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

- **Greedy output matches across machines here:** the M5 Pro, the RTX 4090 and the H100 produced the same continuation, which also matches the course's H100 reference. That's typical but not guaranteed: tiny floating-point differences between hardware can change one token, and with it everything after.
- **Seeded sampling repeats only on the same machine.** With seed 42, each machine repeats its own text exactly, but all three produce different text, including the two NVIDIA GPUs. The random number generator and the order of floating-point operations both differ between devices.
- **The H100 needed more than one warm-up.** Its first timed run took 9.11 s, three times as long as the next two (3.09 s and 3.02 s). The median ignores it, but a single timed run would have understated the H100 by two thirds.
- **Measuring memory on Apple Silicon:** after loading, `torch.mps.current_allocated_memory()` reports 0 GB, because the weights reach the GPU without going through PyTorch's allocator on unified memory. `torch.mps.driver_allocated_memory()` shows the real 16.07 GB. CUDA's `torch.cuda.memory_allocated()` counts the weights normally (16.06 GB on both GPUs).
- **Timing GPU work:** GPU operations run asynchronously, so the throughput script waits for the GPU to finish (`synchronize`) before stopping the clock. Without that, timings come out too fast.

## License

The code in this repository is MIT licensed. The model weights are **not** included: they're downloaded from Hugging Face and covered by the [Llama 3.1 Community License](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/LICENSE).