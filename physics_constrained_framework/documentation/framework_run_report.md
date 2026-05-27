# Framework Run Report

## Data quality summary

```csv
item,value
rows_loaded,2213
columns_loaded,50
selected_target,UCS_kPa
physically_flagged_rows,0
column_mappings_applied,"{""source_dataset"": ""Source"", ""ucs_kpa"": ""UCS_kPa"", ""cbr_pct"": ""CBR"", ""clay_pct"": ""Clay_Fraction"", ""pi_pct"": ""Plasticity_Index"", ""omc_pct"": ""OMC"", ""mdd_gcm3"": ""MDD"", ""water_content_pct"": ""Water_Content"", ""curing_days"": ""Curing_Days"", ""cement_pct"": ""Cement_Content"", ""lime_pct"": ""Lime_Content"", ""fly_ash_pct"": ""Fly_Ash_Content"", ""ggbs_pct"": ""GGBS_Content""}"
missing_required_features,"Binder_Content, SiO2, Al2O3, CaO"
```

## Model metric summary

```csv
model,R2_mean,R2_std,RMSE_mean,RMSE_std,MAE_mean,MAE_std,MAPE_mean,MAPE_std
unconstrained,-1.2533005778202726,3.0063565437221804,2019.3753787885507,1310.7985864058282,1434.4748800096984,913.0841280785392,293.5817830674667,368.6989116010568
physics_constrained,-0.763123348745843,1.8158260578814516,1931.737503670152,1214.0249280206754,1388.311938999908,865.2215980355088,289.8115113752931,436.0452626740234
```

## Physical consistency summary

```csv
feature,violation_count,total_checks,violation_rate_percent,note,model
Curing_Days,28,300,9.333333333333334,expected monotonic increase,unconstrained
MDD,137,300,45.66666666666666,expected monotonic increase,unconstrained
Cement_Content,99,300,33.0,expected monotonic increase,unconstrained
Lime_Content,0,300,0.0,expected monotonic increase,unconstrained
Delta_w,0,300,0.0,expected monotonic decrease,unconstrained
Curing_Days,0,300,0.0,expected monotonic increase,physics_constrained
MDD,0,300,0.0,expected monotonic increase,physics_constrained
Cement_Content,0,300,0.0,expected monotonic increase,physics_constrained
Lime_Content,0,300,0.0,expected monotonic increase,physics_constrained
Delta_w,0,300,0.0,expected monotonic decrease,physics_constrained
```

## Interpretation

The present compiled dataset remains difficult for held-out-source prediction. The grouped-source validation metrics should therefore be interpreted as evidence of source transfer difficulty, not as final design-grade prediction accuracy. The physics-constrained model reduced the risk of non-physical prediction by eliminating the tested monotonicity violations in the controlled perturbation checks. For a manuscript, this result is best framed as a transparent source-transferability diagnosis and physics-consistency framework, not as a universal strength predictor.
