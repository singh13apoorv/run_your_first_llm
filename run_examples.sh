cd examples
for s in 01_tokenize 02_inspect 03_generate 04_stream 05_throughput; do
  uv run python $s.py 2>&1 | tee ../logs/rtx4090/$s.txt
done
cd ..