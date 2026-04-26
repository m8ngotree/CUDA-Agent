#!/usr/bin/env bash
# Four-stage training driver matching paper §3.3.
#
# Usage: bash scripts/run_training.sh configs/cuda_pipeline.yaml
#
# Stages, in order:
#   1. single-turn PPO warmup        (initialize actor + critic)
#   2. RFT                           (collect agent traces, SFT actor)
#   3. value pretraining             (initialize critic on RFT traces)
#   4. agentic PPO                   (full multi-turn RL)

set -euo pipefail

CONFIG="${1:-configs/cuda_pipeline.yaml}"
NPROC=${NPROC:-8}

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

echo "[1/4] Single-turn PPO warmup"
torchrun --nproc_per_node="$NPROC" -m train.single_turn_warmup --config "$CONFIG"

echo "[2/4] Rejection Fine-Tuning"
torchrun --nproc_per_node="$NPROC" -m train.rft --config "$CONFIG"

echo "[3/4] Value pretraining"
torchrun --nproc_per_node="$NPROC" -m train.value_pretrain --config "$CONFIG"

echo "[4/4] Agentic PPO"
torchrun --nproc_per_node="$NPROC" -m train.agentic_rl --config "$CONFIG"

echo "Training complete. Checkpoints under $(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['train']['output_dir'])")"
