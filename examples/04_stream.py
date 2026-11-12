import torch
from common import load_model
from transformers import TextStreamer

PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 64

def main():
    tokenizer, model = load_model()
    input = tokenizer(PROMPT, return_tensors="pt").to(model.device)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    print(PROMPT, end="", flush=True)
    with torch.inference_mode():
        model.generate(**input, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, streamer=streamer)

if __name__ == "__main__":
    main()