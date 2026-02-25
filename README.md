# evaluating-epilink
Reproducibility code and analyses for *Threshold-Free Probabilistic Inference of Superspreading Molecular Transmission Clusters*.

---

This repository contains the code and analysis workflows used in the paper:

***Threshold-Free Probabilistic Inference of Superspreading Molecular Transmission Clusters***

The paper introduces a probabilistic method for inferring recent transmission and identifying superspreading molecular 
transmission clusters (SMTCs) from pathogen genomic data and sampling dates, without relying on fixed genetic or temporal thresholds.

The methodological framework itself is implemented in the companion tool **[epilink](https://github.com/ydnkka/epilink)**. This repository is dedicated
to the reproducibility of the results presented in the paper and includes scripts for data generation, analysis,
benchmarking, and figure reproduction.

The repository supports:

* Simulation of epidemiological and genomic data along known transmission trees
* Application of *epilink*
* Network construction and community detection analyses
* Evaluation of inferred clusters using [BCubed](https://github.com/hhromic/python-bcubed) metrics
* Scenario-based sensitivity analyses
* Benchmarking against logistic regression baselines
* Reproduction of all figures and tables in the main text and supplementary material

This repository is intended for transparency and reproducibility and is not a standalone implementation of the framework.

All analyses are fully configuration-driven. No parameters affecting results are hard-coded.

The full implementation of the framework is available in the **[epilink](https://github.com/ydnkka/epilink)** repository.

## Repository layout

```
config/     configuration files for simulation and analyses
data/       raw and processed inputs (see data/README.md)
figures/    manuscript figures
tables/     manuscript tables
notebooks/  exploratory notebooks
scripts/    reproducible pipelines used for the manuscript
```

## Reproducing the paper

1. Install dependencies (see `environment.yml`)
2. Install [epilink](https://github.com/ydnkka/epilink)
3. Run `scripts/run_all.sh`

### Boston empirical analysis

The Boston pipeline is implemented in `scripts/06_boston_pipeline.py` and configured via
`config/boston.yaml`. Inputs are expected in `data/processed/boston/`, and outputs are
written to `figures/` and `tables/`.
