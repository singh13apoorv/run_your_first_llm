#!/usr/bin/env bash
# Record the machine and run examples 01-05, one process each, into logs/<label>/.
#
#   bash run_all.sh              # label detected automatically
#   bash run_all.sh h100-sxm     # or force one
#
# Labels: Apple M5 Pro -> m5pro, RTX 4090 -> rtx4090,
#         H100 -> h100-sxm / h100-pcie / h100-nvl, anything else -> slug of the GPU name.
set -euo pipefail
cd "$(dirname "$0")"            # repo root (this script lives there)

detect_label() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local name
    name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    case "$name" in
      *H100*NVL*)  echo "h100-nvl" ;;
      *H100*PCIe*) echo "h100-pcie" ;;
      *H100*HBM3*) echo "h100-sxm" ;;
      *) echo "$name" | tr '[:upper:]' '[:lower:]' \
           | sed -e 's/nvidia//g' -e 's/geforce//g' -e 's/[^a-z0-9]//g' ;;
    esac
  elif [[ "$(uname)" == "Darwin" ]]; then
    sysctl -n machdep.cpu.brand_string | tr '[:upper:]' '[:lower:]' \
      | sed -e 's/apple//g' -e 's/[^a-z0-9]//g'
  else
    echo "cpu"
  fi
}

LABEL=${1:-$(detect_label)}
OUT="logs/$LABEL"
mkdir -p "$OUT"
echo "=== machine label: $LABEL"

uv sync

# --- What this machine is ---------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$OUT/nvidia-smi.txt"
  lscpu | head -20 > "$OUT/cpu.txt"     # host CPU matters when decode is overhead-bound
else
  { sysctl -n machdep.cpu.brand_string
    echo "memory: $(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 )) GB"; } > "$OUT/machine.txt"
fi

uv run python - <<'EOF' 2>&1 | tee "$OUT/env.txt"
import torch
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
elif torch.backends.mps.is_available():
    print("Apple MPS")
EOF

# --- Examples 01-05, each in its own process ---------------------------------
cd examples
for s in 01_tokenize 02_inspect 03_generate 04_stream 05_throughput; do
  echo "=== $s"
  uv run python "$s.py" 2>&1 | tee "../$OUT/$s.txt"
done
cd ..

echo "=== done -> $OUT"
ls -1 "$OUT"
echo "Next: bash serve.sh   (then run curl_server.sh from your laptop)"