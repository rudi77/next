#!/usr/bin/env bash
set -e
VENV=/home/rudi/src/next/.venv
"$VENV/bin/pip" install --force-reinstall --no-cache-dir \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
