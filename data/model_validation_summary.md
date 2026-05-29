# Model validation summary

The manuscript uses a cleaned literature-derived dataset for stabilised expansive soil mixture screening. The validation design separates ordinary internal validation from grouped-source validation so that performance is tested against independent source groups rather than only random record splits.

## Data role

- The master modelling table is `modelling_dataset.csv`.
- The dataset contains laboratory observations for unconfined compressive strength and California bearing ratio screening.
- Regional soil-health descriptors are retained only as negative-control context variables.
- Synthetic records discussed in the manuscript are training-only auxiliary records and are not treated as experimental observations.

## Validation role

- `model_comparison_summary.csv` reports model-level comparison metrics.
- `foldwise_validation_metrics.csv` reports fold-level grouped-source validation behaviour.
- `data_audit.csv` records dataset assembly checks.

The practical interpretation is intentionally conservative: the models are used to shortlist candidate mixtures and support emissions-aware ranking before local laboratory confirmation.
