import threading
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_NAME = "NousResearch/Meta-Llama-3.1-8B"
REVISION = "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION, clean_up_tokenization_spaces=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=REVISION, dtype="auto", device_map=DEVICE).eval()

app = FastAPI()
lock = threading.Lock()  # one GPU: one generation at a time, on purpose

class Req(BaseModel):
    prompt: str
    max_new_tokens: int = 64

@app.post("/generate")
def generate(req: Req):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    with lock, torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=req.max_new_tokens, do_sample=False)
    return {"text": tokenizer.decode(out[0], skip_special_tokens=True)}

@app.post("/stream")
def stream(req: Req):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def work():
        with lock, torch.inference_mode():
            model.generate(**inputs, max_new_tokens=req.max_new_tokens, do_sample=False, streamer=streamer)

    threading.Thread(target=work).start()
    return StreamingResponse(streamer, media_type="text/plain")