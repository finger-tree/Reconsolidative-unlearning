#!/usr/bin/env bash
set -euo pipefail

# pyenv install 3.10.14   # if not already installed
# pyenv local 3.10.14     # in the project directory
# python -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt
# python launch.py

if command -v pyenv >/dev/null 2>&1; then
  pyenv local 3.10.14 || true
fi

if [ ! -d .venv ]; then
  python -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Running the evaluation locally..."
python -m main \
  --data_dir ./data \
  --checkpoint_dir ./checkpoints \
  --output_dir ./outputs


#test training with 1 epoch
# cd /Users/ansen/development/Reconsolidative-unlearning/code/23comp_metric && . .venv/bin/activate && python -m main --num_models 1 --data_dir ./data --checkpoint_dir ./checkpoints --output_dir ./outputs