#!/usr/bin/env bash
# Launch DataView. Finds a Python that has huggingface_hub + datasets installed.
set -e
cd "$(dirname "$0")"

CANDIDATES=(
  "/opt/homebrew/opt/python@3.11/bin/python3.11"
  "$(command -v python3 || true)"
  "$(command -v python || true)"
)

PY=""
for c in "${CANDIDATES[@]}"; do
  [ -z "$c" ] && continue
  if "$c" -c "import huggingface_hub, fastapi, uvicorn, pandas, pyarrow" 2>/dev/null; then
    PY="$c"; break
  fi
done

if [ -z "$PY" ]; then
  echo "No suitable Python found. Need: huggingface_hub, fastapi, uvicorn, pandas, pyarrow."
  echo "Install with:  pip install 'huggingface_hub[hf_transfer]' fastapi uvicorn pandas pyarrow"
  exit 1
fi

echo "Using: $PY"
exec "$PY" server.py
