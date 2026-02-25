# Data layout

This repository separates raw, processed, and derived artifacts to keep the manuscript
pipeline reproducible and lightweight.

```
data/
  raw/                 # external or large inputs (not fully versioned)
  processed/           # intermediate outputs used by scripts and figures
```

## Boston dataset

- `data/raw/boston/` is not versioned; add raw inputs there if needed.
- `data/processed/boston/` contains the curated Boston metadata and pairwise distances
  used by `scripts/06_boston_pipeline.py`.

If you are distributing the repository without raw data, keep `data/processed/boston/`
and document any missing sources in the manuscript data availability statement.
