#!/usr/bin/env bash
# End-to-end data pipeline driver.
#
# Usage: bash scripts/run_data_pipeline.sh configs/cuda_pipeline.yaml
#
# Each stage is independently runnable and resumable. If you want to re-run
# only one stage, comment out the others.

set -euo pipefail

CONFIG="${1:-configs/cuda_pipeline.yaml}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

echo "[1/5] Crawl torch operators"
python -m data.crawl.crawl_torch_ops --config "$CONFIG"

echo "[2/5] Crawl transformers operators"
python -m data.crawl.crawl_transformers_ops --config "$CONFIG"

echo "[3/5] Synthesize fused problems via LLM"
python -m data.synthesis.synthesize_fused_ops --config "$CONFIG"

echo "[4/5] Execution-based filter"
python -m data.filter.filter_pipeline --config "$CONFIG"

echo "[5/5] Build HuggingFace dataset"
python -m data.dataset.build_dataset --config "$CONFIG"

echo "Done. Yield funnel: $(python -c "import yaml,json,sys; c=yaml.safe_load(open('$CONFIG'));print(c['filter']['output_dir']+'/filter_stats.json')")"
