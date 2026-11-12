# Server output

## Apple Silicon M5 Pro (MPS)

```
$ bash src/run_your_first_llm/run.sh
Resolved 66 packages in 3ms
Checked 45 packages in 1ms
Loading weights: 100%|██████████████████████| 291/291 [00:02<00:00, 129.62it/s]
INFO:     Started server process [32590]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:52751 - "POST /generate HTTP/1.1" 200 OK
INFO:     127.0.0.1:52777 - "POST /stream HTTP/1.1" 200 OK
```

```
$ curl -s localhost:8000/generate -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_new_tokens": 64}'
{"text":"The capital of France is a city of many faces. It is a city of history, a city of culture, a city of art, a city of fashion, a city of food, a city of romance, a city of architecture, a city of music, a city of literature, a city of science, a city of education, a"}
```

```
$ curl -N localhost:8000/stream -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_new_tokens": 64}'
 a city of many faces. It is a city of history, a city of culture, a city of art, a city of fashion, a city of food, a city of romance, a city of architecture, a city of music, a city of literature, a city of science, a city of education, a
```

Both endpoints return the same greedy continuation as `examples/03_generate.py` and `04_stream.py`.

## NVIDIA RTX 4090 (CUDA)
torch 2.15.0+cu132 · CUDA 13.2 · compute capability 8.9

**Server outputs**
```
Loading weights: 100%|██████████| 291/291 [00:04<00:00, 70.92it/s]
INFO:     Started server process [4831]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:36846 - "POST /generate HTTP/1.1" 200 OK
INFO:     127.0.0.1:38792 - "POST /stream HTTP/1.1" 200 OK
```

**Generate text**
```
{"text":"The capital of France is a city of many faces. It is a city of history, a city of culture, a city of art, a city of fashion, a city of food, a city of romance, a city of architecture, a city of music, a city of literature, a city of science, a city of education, a"}
```

**Stream text**
```
 a city of many faces. It is a city of history, a city of culture, a city of art, a city of fashion, a city of food, a city of romance, a city of architecture, a city of music, a city of literature, a city of science, a city of education, a
```

_To be filled after the GPU run._