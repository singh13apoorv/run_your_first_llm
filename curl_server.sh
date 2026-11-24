#!/usr/bin/env bash
# Hit /generate and /stream and save the responses to logs/<label>/.
# Run where the server is reachable at localhost:8000:
#   - on the same machine, or
#   - on your laptop with the tunnel open:  ssh -L 8000:localhost:8000 vastgpubox
#
#   bash curl_server.sh h100-sxm      # label = the machine the SERVER runs on
set -euo pipefail
cd "$(dirname "$0")"

LABEL=${1:?usage: curl_server.sh <label of the server machine>}
OUT="logs/$LABEL"
mkdir -p "$OUT"
BODY='{"prompt": "The capital of France is", "max_new_tokens": 64}'

echo "=== /generate"
curl -s localhost:8000/generate -H "Content-Type: application/json" -d "$BODY" \
  | tee "$OUT/generate.txt"
echo; echo

echo "=== /stream"
curl -sN localhost:8000/stream -H "Content-Type: application/json" -d "$BODY" \
  | tee "$OUT/stream.txt"
echo