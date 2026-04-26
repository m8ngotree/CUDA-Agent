#!/usr/bin/env bash
# Evaluation driver. Runs every benchmark listed under eval.benchmarks
# in the config.
#
# Usage: bash scripts/run_eval.sh configs/cuda_pipeline.yaml [checkpoint_path]

set -euo pipefail

CONFIG="${1:-configs/cuda_pipeline.yaml}"
CKPT="${2:-}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

BENCHMARKS=$(python -c "
import yaml
c = yaml.safe_load(open('$CONFIG'))
print(' '.join(c['eval']['benchmarks']))
")

for b in $BENCHMARKS; do
    case "$b" in
        kernelbench_*)
            python -m eval.eval_kernelbench --config "$CONFIG" --benchmark "$b" \
                ${CKPT:+--checkpoint "$CKPT"}
            ;;
        tritonbench)
            python -m eval.eval_tritonbench --config "$CONFIG" \
                ${CKPT:+--checkpoint "$CKPT"}
            ;;
        *)
            echo "Unknown benchmark: $b" >&2
            exit 1
            ;;
    esac
done

echo "Eval done."
