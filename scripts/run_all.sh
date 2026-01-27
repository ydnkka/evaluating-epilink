#!/usr/bin/env bash
set -euo pipefail

SCRIPTS=(
  "01_characterise_mechanism.py"
  "02.1_generate_tree.py"
  "02.2_simulate_datasets.py"
  "03_evaluate_edges.py"
  "04.1_sparsify_effects.py"
  "04.2_run_clustering.py"
  "05_evaluate_clustering.py"
)

echo "Starting pipeline..."

for script in "${SCRIPTS[@]}"; do
  echo "Running $script"
  python "$script"
done

echo "All scripts completed successfully."