# Source-aware stabilized soil ML dataset and analysis

This repository accompanies the manuscript:

**Source-aware machine learning for stabilized expansive soils: transfer validation, decision screening and carbon-efficient mixture ranking**

## Contents

- `data/`: cleaned real UCS and CBR datasets, combined dataset, source inventory and dataset summary.
- `results/`: classification, ranking and Pareto-screening output tables.
- `scripts/`: Python scripts used to regenerate the main analyses and manuscript-supporting outputs.

## Reproducibility

The analysis uses Python, pandas, NumPy, scikit-learn and Matplotlib. Main random seeds are fixed at 42 or explicitly stated in the scripts. Synthetic auxiliary rows are not part of the real experimental dataset and are generated only inside training folds.

## Data-use note

The datasets are literature-derived and harmonised for research use. Users should verify original source papers before using values for design.
