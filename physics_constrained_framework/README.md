# Physics-Constrained Stabilized Soil ML Pipeline

## Input

Place the cleaned or raw CSV file at:

`stabilized_soil_dataset.csv`

The script standardizes common column-name variants, checks missing variables,
flags physically impossible records, engineers geotechnical features, performs
grouped-source validation, tests physical monotonicity, estimates cradle-to-gate
embodied carbon using editable placeholder emission factors, and generates
explainability outputs.

## Selected target

`UCS_kPa`

## Model features

- `Clay_Fraction`
- `Plasticity_Index`
- `OMC`
- `MDD`
- `Water_Content`
- `Curing_Days`
- `Cement_Content`
- `Lime_Content`
- `Fly_Ash_Content`
- `GGBS_Content`
- `A_clay`
- `Delta_w`
- `Binder_Total`
- `Binder_Clay_Ratio`
- `log_curing`

## Reproduction

```bash
pip install -r outputs/requirements.txt
python physics_constrained_stabilized_soil_ml.py
```

## Important LCA note

Emission factors are placeholders and must be replaced with region-specific EPD
or accepted database values before journal submission.
