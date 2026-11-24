# 03_generate.py output

- **Greedy:** identical on every run, and identical on all three machines here. It also matches the course's H100 reference. Across hardware this is typical but not guaranteed.
- **Seeded sampling:** identical on every run on the same machine, but different on every machine, including between the two NVIDIA GPUs.
- **Unseeded sampling:** different on every run, by design.

## Apple M5 Pro, 48 GB (MPS)

```
$ uv run python 03_generate.py
------ Greedy (identical every run) ------
The capital of France is a city of many faces. It is a city of history, a city of culture, a city of art, a city of fashion, a city of food, a city of romance, a city

------ Sampled, no seed (changes every run) ------
The capital of France is the city of Paris. It is the most populated city in France with a population of about 2,150,000. It is also the most popular tourist destination in the world with over 30

------ Sampled, seed 42, run 1 (identical) ------
The capital of France is home to some of the most famous landmarks in the world. The city is also home to some of the most iconic restaurants in the world. This is a city that is always in the spotlight, and

------ Sampled, seed 42, run 2 (identical) ------
The capital of France is home to some of the most famous landmarks in the world. The city is also home to some of the most iconic restaurants in the world. This is a city that is always in the spotlight, and
```

## NVIDIA RTX 4090, 24 GB (CUDA)

torch 2.15.0+cu132 · CUDA 13.2 · compute capability 8.9

```
------ Greedy (identical every run) ------
The capital of France is a city of many faces. It is a city of history, a city of culture, a city of art, a city of fashion, a city of food, a city of romance, a city

------ Sampled, no seed (changes every run) ------
The capital of France is a city of endless possibilities. The city has a rich history, amazing architecture, and a thriving culture. It is also home to some of the best restaurants in the world. If you are looking for

------ Sampled, seed 42, run 1 (identical) ------
The capital of France is Paris, the most visited city in the world. The city has a population of about 2.2 million people and is located in the north central part of the country. The city is divided into

------ Sampled, seed 42, run 2 (identical) ------
The capital of France is Paris, the most visited city in the world. The city has a population of about 2.2 million people and is located in the north central part of the country. The city is divided into
```

## NVIDIA H100 SXM, 80 GB HBM3 (CUDA)

torch 2.15.0+cu132 · CUDA 13.2 · compute capability 9.0

```
------ Greedy (identical every run) ------
The capital of France is a city of many faces. It is a city of history, a city of culture, a city of art, a city of fashion, a city of food, a city of romance, a city

------ Sampled, no seed (changes every run) ------
The capital of France is the most popular city in Europe. The city is famous for its history, culture, and architecture. It is also a very popular tourist destination. There are many things to see and do in Paris,

------ Sampled, seed 42, run 1 (identical) ------
The capital of France is Paris, the country is divided into 22 regions. The climate of France is oceanic, but in the mountainous regions, it is more continental. France is a country in Western Europe, bordered

------ Sampled, seed 42, run 2 (identical) ------
The capital of France is Paris, the country is divided into 22 regions. The climate of France is oceanic, but in the mountainous regions, it is more continental. France is a country in Western Europe, bordered
```