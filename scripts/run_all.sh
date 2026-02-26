#!/usr/bin/env bash
set -euo pipefail

SCRIPTS=(
  "characterise_epilink.py"
  "generate_tree.py"
  "generate_datasets.py"
  "pairwise_discrimination.py"
  "sparsify_effects.py"
  "run_clustering.py"
  "evaluate_clustering.py"
  "cluster_stability.py"
  "boston_clustering.py"
)

echo "Starting pipeline..."

for script in "${SCRIPTS[@]}"; do
  echo "Running $script"
  python "$script"
done

echo "All scripts completed successfully."