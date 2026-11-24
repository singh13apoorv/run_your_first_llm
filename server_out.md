# Server output

On every machine, both endpoints return the same greedy continuation as `examples/03_generate.py` and `04_stream.py`.

Request used on each machine:

```
curl -s localhost:8000/generate -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_new_tokens": 64}'

curl -N localhost:8000/stream -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_new_tokens": 64}'
```

On the rented GPUs, the server ran on the box and the requests came from a laptop through `ssh -L 8000:localhost:8000`.

## Apple M5 Pro, 48 GB (MPS)

**Server log**
```
$ bash src/run_your_first_llm/run.sh
Resolved 66 packages in 3ms
Checked 45 packages in 1ms
INFO:     Started server process [32590]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:52751 - "POST /generate HTTP/1.1" 200 OK
INFO:     127.0.0.1:52777 - "POST /stream HTTP/1.1" 200 OK
```

**/generate**
```
{"text":"The capital of France is a city of many faces. It is a city of history, a city of culture, a city of art, a city of fashion, a city of food, a city of romance, a city of architecture, a city of music, a city of literature, a city of science, a city of education, a"}
```

**/stream**
```
 a city of many faces. It is a city of history, a city of culture, a city of art, a city of fashion, a city of food, a city of romance, a city of architecture, a city of music, a city of literature, a city of science, a city of education, a
```

## NVIDIA RTX 4090, 24 GB (CUDA)

torch 2.15.0+cu132 · CUDA 13.2 · compute capability 8.9

**Server log**
```
INFO:     Started server process [4831]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:36846 - "POST /generate HTTP/1.1" 200 OK
INFO:     127.0.0.1:38792 - "POST /stream HTTP/1.1" 200 OK
```

**/generate**
```
{"text":"The capital of France is a city of many faces. It is a city of history, a city of culture, a city of art, a city of fashion, a city of food, a city of romance, a city of architecture, a city of music, a city of literature, a city of science, a city of education, a"}
```

**/stream**
```
 a city of many faces. It is a city of history, a city of culture, a city of art, a city of fashion, a city of food, a city of romance, a city of architecture, a city of music, a city of literature, a city of science, a city of education, a
```

## NVIDIA H100 SXM, 80 GB HBM3 (CUDA)

torch 2.15.0+cu132 · CUDA 13.2 · compute capability 9.0

**Server log**
```
$ bash serve.sh
Resolved 66 packages in 1ms
Checked 64 packages in 0.95ms
INFO:     Started server process [3292]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:47792 - "POST /generate HTTP/1.1" 200 OK
INFO:     127.0.0.1:47808 - "POST /stream HTTP/1.1" 200 OK
```

**/generate**
```
{"text":"The capital of France is a city of many faces. It is a city of history, a city of culture, a city of art, a city of fashion, a city of food, a city of romance, a city of architecture, a city of music, a city of literature, a city of science, a city of education, a"}
```

**/stream**
```
 a city of many faces. It is a city of history, a city of culture, a city of art, a city of fashion, a city of food, a city of romance, a city of architecture, a city of music, a city of literature, a city of science, a city of education, a
```