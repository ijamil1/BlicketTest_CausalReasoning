#!/bin/bash

models=(
  "anthropic/claude-3.5-haiku"
  "qwen/qwen3-30b-a3b-instruct-2507"
  "Qwen/Qwen3-30B-A3B-Instruct-2507:n3vjii616j5c0qh91k1yw1zn"
  "Qwen/Qwen3-30B-A3B-Instruct-2507:ghbtibxgz1mrnwiknbd5y4dl"
  "Qwen/Qwen3-30B-A3B-Instruct-2507:s4fuawjjlm76pjyudakekox4"
  "Qwen/Qwen3-30B-A3B-Instruct-2507:v81qpxqqp9ahrzpuovvdrq9o"
)

EVALS_DIR="./environments/BlicketTest_CausalReasoning/outputs/evals"

for model in "${models[@]}"; do
  short_name="${model#*/}"
  if ls "$EVALS_DIR" 2>/dev/null | grep -q "$short_name"; then
    echo "Skipping $model (already has eval results for $short_name)"
  else
    echo "Running eval with model: $model"
    prime eval run irfanjamil/BlicketTest_CausalReasoning@0.1.4 -n 60 -r 3 -m "$model"
  fi
  echo ""
done