import time
import statistics
import torch
from common import load_model, sync, DEVICE

PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 256
RUNS = 3


def generate(model, inputs, n):
    with torch.inference_mode():
        return model.generate(**inputs, max_new_tokens=n, do_sample=False)


def main():
    tokenizer, model = load_model()
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    generate(model, inputs, 16)  # warm-up

    rates = []
    for i in range(RUNS):
        sync(); t0 = time.perf_counter()
        out = generate(model, inputs, MAX_NEW_TOKENS)
        sync(); elapsed = time.perf_counter() - t0
        n = out.shape[1] - inputs.input_ids.shape[1]
        rates.append(n / elapsed)
        print(f"run {i + 1}: {n} tokens in {elapsed:.2f} s -> {n / elapsed:.1f} tok/s")

    tps = statistics.median(rates)
    print(f"\n{DEVICE}: median {tps:.1f} tok/s (end-to-end, includes prefill)")
    print(f"effective bandwidth ≈ {tps * weight_bytes / 1e9:.0f} GB/s "
          f"(weights read once per token)")


if __name__ == "__main__":
    main()