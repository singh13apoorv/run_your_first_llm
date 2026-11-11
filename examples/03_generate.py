import torch
from common import load_model

PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 40


def run(model, tokenizer, inputs, title, **kwargs):
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, **kwargs)
    print(f"------ {title} ------")
    print(tokenizer.decode(out[0], skip_special_tokens=True), "\n")


def main():
    tokenizer, model = load_model()
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

    run(model, tokenizer, inputs, "Greedy (identical every run)", do_sample=False)

    # do_sample uses generation_config defaults: temperature=0.6, top_p=0.9
    run(model, tokenizer, inputs, "Sampled, no seed (changes every run)", do_sample=True)

    for i in (1, 2):
        torch.manual_seed(42)
        run(model, tokenizer, inputs, f"Sampled, seed 42, run {i} (identical)", do_sample=True)


if __name__ == "__main__":
    main()