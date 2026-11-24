#!/usr/bin/env bash
# Start the server and save its log to logs/<label>/server.txt. Stop with Ctrl+C.
#
#   bash serve.sh              # same auto label as run_all.sh
#   bash serve.sh h100-sxm
set -euo pipefail
cd "$(dirname "$0")"

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
mkdir -p "logs/$LABEL"
echo "=== machine label: $LABEL   (log -> logs/$LABEL/server.txt)"

bash src/run_your_first_llm/run.sh 2>&1 | tee "logs/$LABEL/server.txt"