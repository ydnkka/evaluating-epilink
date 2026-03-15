# evaluating-epilink

Reproducibility workflows and manuscript analyses for the `epilink` paper.

This repository contains the paper-facing analysis pipeline: synthetic data generation, pairwise benchmarking, clustering, temporal stability analyses, the Boston application, and manuscript figure rendering.

## Setup

Create the environment:

```bash
conda env create -f environment.yml
conda activate evaluating-epilink
```

Install `epilink` in one of two ways:

From PyPI:

```bash
pip install epilink
```

From a local clone:

```bash
cd ../epilink
pip install -e . --no-deps
```

Install this repository:

```bash
cd ../evaluating-epilink
pip install -e .
```

## Run

Run the full workflow:

```bash
bash scripts/run_all.sh
```

Or use Snakemake:

```bash
snakemake -s workflow/Snakefile --cores 1
```

To render figures only:

```bash
python scripts/render_figures.py
```

## Layout

```text
configs/                 workflow and study configuration
data/                    raw inputs and processed intermediates
results/                 generated tables, figures, and manifests
scripts/                 command-line entrypoints
src/evaluating_epilink/  analysis code and epilink adapter
workflow/                Snakemake workflow
tests/                   smoke tests
```

## Notes

- Boston uses `included_intermediate_counts: [0]`.
- Figures are rendered from `src/evaluating_epilink/plotting/manuscript.py`.
- Stage manifests and the pipeline report are written to `results/manifests/`.
