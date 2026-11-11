import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "NousResearch/Meta-Llama-3.1-8B"
REVISION = "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()

def memory_gb():
    if DEVICE == "cuda":
        return torch.cuda.memory_allocated() / 1e9
    if DEVICE == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    return 0.0

def load_model(with_model=True):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION, clean_up_tokenization_spaces=False)
    if not with_model:
        return tokenizer, None

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=REVISION,
        device_map=DEVICE,
        dtype='auto'
    ).eval()

    return tokenizer, model