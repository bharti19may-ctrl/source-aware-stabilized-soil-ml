"""
Physics-constrained machine learning framework for chemically stabilized soils.

Default input:
    DATA_PATH = "stabilized_soil_dataset.csv"

Outputs are written to:
    outputs/

Important:
    Emission factors in the LCA module are editable placeholders. For journal
    submission, replace them with region-specific EPD, ecoinvent, ICE, or
    nationally accepted database values.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


DATA_PATH = "stabilized_soil_dataset.csv"
OUTPUT_DIR = Path("outputs")
RANDOM_SEED = 42


STANDARD_COLUMNS = {
    "Source": ["source", "study", "study_id", "paper_id", "reference", "laboratory_id", "lab_id", "dataset", "source_id", "source_dataset", "data_source"],
    "Study_ID": ["studyid", "study_id", "paper", "paper_id", "publication", "reference_id"],
    "UCS_kPa": ["ucs", "ucs_kpa", "qu", "unconfined_compressive_strength", "compressive_strength", "ucs_strength"],
    "CBR": ["cbr", "cbr_percent", "cbr_pct", "california_bearing_ratio"],
    "Resilient_Modulus": ["resilient_modulus", "mr", "modulus_resilient", "resilient_modulus_mpa"],
    "Clay_Fraction": ["clay", "clay_fraction", "clay_pct", "clay_percent"],
    "Plasticity_Index": ["pi", "plasticity_index", "pi_pct", "plastic_index"],
    "OMC": ["omc", "optimum_moisture_content", "omc_pct", "omc_percent"],
    "MDD": ["mdd", "maximum_dry_density", "mdd_gcm3", "dry_density", "max_dry_density"],
    "Water_Content": ["water_content", "moisture_content", "water_content_pct", "w_pct", "w"],
    "Curing_Days": ["curing_days", "curing", "age", "age_days", "curing_time_days"],
    "Cement_Content": ["cement", "cement_content", "cement_pct", "cement_percent"],
    "Lime_Content": ["lime", "lime_content", "lime_pct", "lime_percent"],
    "Binder_Content": ["binder", "binder_content", "binder_pct", "stabilizer_content", "additive_content"],
    "Fly_Ash_Content": ["fly_ash", "flyash", "fly_ash_pct", "fly_ash_content"],
    "GGBS_Content": ["ggbs", "ggbs_pct", "slag", "slag_content"],
    "SiO2": ["sio2", "sio2_pct", "silica", "silicon_dioxide"],
    "Al2O3": ["al2o3", "al2o3_pct", "alumina", "aluminium_oxide", "aluminum_oxide"],
    "CaO": ["cao", "cao_pct", "calcium_oxide"],
}

TARGET_PRIORITY = ["UCS_kPa", "CBR", "Resilient_Modulus"]
GROUP_CANDIDATES = ["Source", "Study_ID"]
BASE_REQUIRED_FEATURES = [
    "Clay_Fraction",
    "Plasticity_Index",
    "OMC",
    "MDD",
    "Water_Content",
    "Curing_Days",
    "Cement_Content",
    "Lime_Content",
    "Binder_Content",
    "SiO2",
    "Al2O3",
    "CaO",
]


@dataclass
class ValidationResult:
    fold_table: pd.DataFrame
    summary_table: pd.DataFrame
    predictions: pd.DataFrame


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def clean_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[%()/.\\-]+", "_", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"__+", "_", name)
    return name.strip("_")


def normalise_for_match(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def standardize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    renamed = {c: clean_name(c) for c in df.columns}
    df = df.rename(columns=renamed)
    lookup = {normalise_for_match(c): c for c in df.columns}
    mapping: Dict[str, str] = {}

    for standard, aliases in STANDARD_COLUMNS.items():
        candidates = [standard] + aliases
        for candidate in candidates:
            key = normalise_for_match(candidate)
            if key in lookup and standard not in df.columns:
                mapping[lookup[key]] = standard
                break

    df = df.rename(columns=mapping)
    return df, mapping


def choose_target(df: pd.DataFrame) -> str:
    available = [t for t in TARGET_PRIORITY if t in df.columns and df[t].notna().sum() >= 10]
    if not available:
        raise ValueError(
            "No usable target found. At least one of UCS_kPa, CBR, or Resilient_Modulus "
            "must exist with at least 10 non-missing values."
        )
    return available[0]


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def detect_physical_quality(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    flags = pd.DataFrame(index=df.index)
    flags["negative_target"] = df[target] < 0
    flags["negative_curing"] = df.get("Curing_Days", pd.Series(False, index=df.index)) < 0

    for c in ["Cement_Content", "Lime_Content", "Binder_Content", "Fly_Ash_Content", "GGBS_Content"]:
        if c in df.columns:
            flags[f"negative_{c}"] = df[c] < 0

    if "Plasticity_Index" in df.columns:
        flags["invalid_plasticity_index"] = (df["Plasticity_Index"] < 0) | (df["Plasticity_Index"] > 150)
    if "OMC" in df.columns:
        flags["unrealistic_omc"] = (df["OMC"] < 0) | (df["OMC"] > 80)
    if "MDD" in df.columns:
        flags["unrealistic_mdd"] = (df["MDD"] < 0.8) | (df["MDD"] > 2.6)
    if "Clay_Fraction" in df.columns:
        flags["invalid_clay_fraction"] = (df["Clay_Fraction"] <= 0) | (df["Clay_Fraction"] > 100)
    if "Water_Content" in df.columns:
        flags["unrealistic_water_content"] = (df["Water_Content"] < 0) | (df["Water_Content"] > 100)

    flags = flags.fillna(False)
    df["physical_quality_flag"] = flags.any(axis=1)
    summary = pd.DataFrame(
        {
            "check": flags.columns,
            "flagged_records": flags.sum(axis=0).astype(int).values,
        }
    )
    return df, summary


def make_data_quality_summary(df: pd.DataFrame, target: str, mapping: Dict[str, str], missing_required: List[str]) -> pd.DataFrame:
    rows = [
        {"item": "rows_loaded", "value": len(df)},
        {"item": "columns_loaded", "value": df.shape[1]},
        {"item": "selected_target", "value": target},
        {"item": "physically_flagged_rows", "value": int(df["physical_quality_flag"].sum()) if "physical_quality_flag" in df else 0},
        {"item": "column_mappings_applied", "value": json.dumps(mapping)},
        {"item": "missing_required_features", "value": ", ".join(missing_required) if missing_required else "None"},
    ]
    return pd.DataFrame(rows)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in BASE_REQUIRED_FEATURES + ["Fly_Ash_Content", "GGBS_Content"]:
        if c not in df.columns:
            df[c] = np.nan

    clay = df["Clay_Fraction"].replace(0, np.nan)
    cao = df["CaO"].replace(0, np.nan)

    df["A_clay"] = df["Plasticity_Index"] / clay
    df["phi_ox"] = (df["SiO2"] + df["Al2O3"]) / cao
    df["Delta_w"] = df["Water_Content"] - df["OMC"]
    binder_components = ["Cement_Content", "Lime_Content", "Binder_Content", "Fly_Ash_Content", "GGBS_Content"]
    df["Binder_Total"] = df[binder_components].fillna(0).sum(axis=1)
    df["Binder_Clay_Ratio"] = df["Binder_Total"] / clay
    df["log_curing"] = np.log1p(df["Curing_Days"].clip(lower=0))
    return df


def usable_feature_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "Clay_Fraction",
        "Plasticity_Index",
        "OMC",
        "MDD",
        "Water_Content",
        "Curing_Days",
        "Cement_Content",
        "Lime_Content",
        "Binder_Content",
        "Fly_Ash_Content",
        "GGBS_Content",
        "SiO2",
        "Al2O3",
        "CaO",
        "A_clay",
        "phi_ox",
        "Delta_w",
        "Binder_Total",
        "Binder_Clay_Ratio",
        "log_curing",
    ]
    return [c for c in candidates if c in df.columns and df[c].notna().sum() > 0]


def minimum_data_check(df: pd.DataFrame, target: str, features: List[str]) -> None:
    missing = [c for c in BASE_REQUIRED_FEATURES if c not in df.columns or df[c].notna().sum() == 0]
    available_core = [c for c in BASE_REQUIRED_FEATURES if c not in missing]
    if len(available_core) < 5:
        raise ValueError(
            "Insufficient predictors. At least five standard geotechnical/chemical "
            f"features are required. Available: {available_core}; missing: {missing}"
        )
    if df[target].notna().sum() < 30:
        raise ValueError("Insufficient target records after cleaning. At least 30 target values are recommended.")
    if len(features) < 5:
        raise ValueError(f"Too few usable model features after feature engineering: {features}")


def monotonic_constraints_for_features(features: List[str]) -> List[int]:
    positive = {"Curing_Days", "log_curing", "MDD", "Cement_Content", "Lime_Content", "Binder_Total", "Binder_Clay_Ratio"}
    negative = {"Delta_w", "Plasticity_Index", "A_clay"}
    constraints = []
    for f in features:
        if f in positive:
            constraints.append(1)
        elif f in negative:
            constraints.append(-1)
        else:
            constraints.append(0)
    return constraints


def build_model(features: List[str], constrained: bool) -> Pipeline:
    constraints = monotonic_constraints_for_features(features) if constrained else None
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=600,
        l2_regularization=0.05,
        min_samples_leaf=8,
        random_state=RANDOM_SEED,
        monotonic_cst=constraints,
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                features,
            )
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", regressor)])


def select_cv(df: pd.DataFrame) -> Tuple[object, Optional[str], np.ndarray]:
    group_col = next((c for c in GROUP_CANDIDATES if c in df.columns and df[c].nunique(dropna=True) >= 2), None)
    if group_col is None:
        warnings.warn("No source/study grouping column found. Falling back to KFold validation.")
        return KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED), None, np.arange(len(df))

    groups = df[group_col].astype(str).fillna("unknown").values
    n_groups = len(np.unique(groups))
    if n_groups <= 10:
        return LeaveOneGroupOut(), group_col, groups
    return GroupKFold(n_splits=5), group_col, groups


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan
    valid_mape = np.abs(y_true) > 1e-9
    mape = np.mean(np.abs((y_true[valid_mape] - y_pred[valid_mape]) / y_true[valid_mape])) * 100 if valid_mape.any() else np.nan
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape}


def grouped_validation(df: pd.DataFrame, target: str, features: List[str], constrained: bool) -> ValidationResult:
    model_name = "physics_constrained" if constrained else "unconstrained"
    model = build_model(features, constrained=constrained)
    cv, group_col, groups = select_cv(df)

    fold_rows = []
    pred_rows = []
    x = df[features]
    y = df[target].values.astype(float)

    split_iter = cv.split(x, y, groups) if group_col else cv.split(x, y)
    for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
        fold_model = clone(model)
        fold_model.fit(x.iloc[train_idx], y[train_idx])
        pred = fold_model.predict(x.iloc[test_idx])
        metrics = calculate_metrics(y[test_idx], pred)
        test_group = ", ".join(sorted(set(groups[test_idx]))) if group_col else f"random_fold_{fold}"
        fold_rows.append(
            {
                "model": model_name,
                "fold": fold,
                "test_group": test_group,
                "n_test": len(test_idx),
                **metrics,
            }
        )
        for idx, yt, yp in zip(test_idx, y[test_idx], pred):
            pred_rows.append(
                {
                    "model": model_name,
                    "row_index": int(df.index[idx]),
                    "fold": fold,
                    "test_group": test_group,
                    "measured": yt,
                    "predicted": yp,
                    "residual": yt - yp,
                }
            )

    fold_table = pd.DataFrame(fold_rows)
    metric_cols = ["R2", "RMSE", "MAE", "MAPE"]
    summary = (
        fold_table.groupby("model")[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join([x for x in col if x]).rstrip("_") for col in summary.columns.to_flat_index()]
    return ValidationResult(fold_table, summary, pd.DataFrame(pred_rows))


def physical_consistency_test(model: Pipeline, base_df: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    checks = {
        "Curing_Days": (0.75, 1.50, "increase"),
        "MDD": (0.98, 1.02, "increase"),
        "Cement_Content": (0.75, 1.50, "increase"),
        "Lime_Content": (0.75, 1.50, "increase"),
        "Delta_w": (0.00, 5.00, "decrease"),
    }
    rows = []
    sample = base_df[features].copy().dropna(how="all")
    if len(sample) > 300:
        sample = sample.sample(300, random_state=RANDOM_SEED)

    for feature, (low_factor, high_factor, direction) in checks.items():
        if feature not in features:
            rows.append({"feature": feature, "violation_count": np.nan, "total_checks": 0, "violation_rate_percent": np.nan, "note": "feature unavailable"})
            continue
        low = sample.copy()
        high = sample.copy()
        if feature == "Delta_w":
            low[feature] = 0.0
            high[feature] = sample[feature].fillna(0) + high_factor
        else:
            low[feature] = sample[feature].fillna(sample[feature].median()) * low_factor
            high[feature] = sample[feature].fillna(sample[feature].median()) * high_factor

        y_low = model.predict(low)
        y_high = model.predict(high)
        if direction == "increase":
            violations = np.sum(y_high < y_low - 1e-9)
        else:
            violations = np.sum(y_high > y_low + 1e-9)
        rows.append(
            {
                "feature": feature,
                "violation_count": int(violations),
                "total_checks": int(len(sample)),
                "violation_rate_percent": 100 * violations / max(len(sample), 1),
                "note": "expected monotonic " + direction,
            }
        )
    return pd.DataFrame(rows)


def embodied_carbon_module(df: pd.DataFrame) -> pd.DataFrame:
    ef = {
        "Natural_soil": 0.005,
        "Cement_Content": 0.92,
        "Lime_Content": 1.20,
        "Fly_Ash_Content": 0.02,
        "GGBS_Content": 0.14,
    }
    density = df["MDD"].copy() if "MDD" in df.columns else pd.Series(np.nan, index=df.index)
    rho_d = np.where(density < 10, density * 1000, density)  # g/cm3 to kg/m3 when applicable
    rho_d = pd.Series(rho_d, index=df.index).fillna(pd.Series(rho_d, index=df.index).median())
    rho_d = rho_d.fillna(1700.0)

    rows = []
    for factor in [0.8, 1.0, 1.2]:
        ec = rho_d * ef["Natural_soil"] * factor
        for mat in ["Cement_Content", "Lime_Content", "Fly_Ash_Content", "GGBS_Content"]:
            if mat in df.columns:
                mass_fraction = df[mat].fillna(0).clip(lower=0) / 100.0
                ec += rho_d * mass_fraction * ef[mat] * factor
        out = pd.DataFrame(
            {
                "scenario_factor": factor,
                "EC_mix_kgCO2e_m3": ec,
                "rho_d_kg_m3": rho_d,
            }
        )
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def save_figures(predictions: pd.DataFrame, fold_table: pd.DataFrame, violation: pd.DataFrame, carbon: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 5))
    for model, grp in predictions.groupby("model"):
        plt.scatter(grp["measured"], grp["predicted"], s=18, alpha=0.65, label=model)
    lims = [predictions[["measured", "predicted"]].min().min(), predictions[["measured", "predicted"]].max().max()]
    plt.plot(lims, lims, "k--", lw=1)
    plt.xlabel("Measured")
    plt.ylabel("Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predicted_vs_measured.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    for model, grp in predictions.groupby("model"):
        plt.scatter(grp["predicted"], grp["residual"], s=18, alpha=0.65, label=model)
    plt.axhline(0, color="k", linestyle="--", lw=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_plot.png", dpi=300)
    plt.close()

    for metric in ["R2", "RMSE"]:
        plt.figure(figsize=(7, 4))
        for model, grp in fold_table.groupby("model"):
            plt.plot(grp["fold"], grp[metric], marker="o", label=model)
        plt.xlabel("Fold")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"foldwise_{metric}.png", dpi=300)
        plt.close()

    plt.figure(figsize=(7, 4))
    pivot = violation.pivot_table(index="feature", columns="model", values="violation_rate_percent", aggfunc="mean")
    pivot.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Violation rate (%)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "physical_violation_rate_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    carbon.boxplot(column="EC_mix_kgCO2e_m3", by="scenario_factor", grid=False)
    plt.suptitle("")
    plt.title("Embodied carbon sensitivity")
    plt.xlabel("Emission factor multiplier")
    plt.ylabel("EC_mix (kg CO2e/m3)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "embodied_carbon_sensitivity.png", dpi=300)
    plt.close()


def shap_or_permutation(model: Pipeline, df: pd.DataFrame, target: str, features: List[str]) -> pd.DataFrame:
    x = df[features]
    y = df[target]
    try:
        import shap

        transformed = model.named_steps["preprocess"].transform(x)
        feature_names = features
        explainer = shap.Explainer(model.named_steps["model"])
        values = explainer(transformed)

        shap.summary_plot(values, transformed, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "shap_summary_plot.png", dpi=300)
        plt.close()

        shap.plots.bar(values, show=False)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "shap_bar_plot.png", dpi=300)
        plt.close()

        for feature in ["Curing_Days", "Binder_Total", "Cement_Content", "Delta_w", "MDD"]:
            if feature in feature_names:
                shap.dependence_plot(feature, values.values, transformed, feature_names=feature_names, show=False)
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / f"shap_dependence_{feature}.png", dpi=300)
                plt.close()

        importance = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": np.abs(values.values).mean(axis=0),
                "method": "SHAP",
            }
        ).sort_values("mean_abs_shap", ascending=False)
    except Exception as exc:
        warnings.warn(f"SHAP failed; using permutation importance fallback. Reason: {exc}")
        perm = permutation_importance(model, x, y, n_repeats=20, random_state=RANDOM_SEED, scoring="r2")
        importance = pd.DataFrame(
            {
                "feature": features,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
                "method": "permutation_importance",
            }
        ).sort_values("importance_mean", ascending=False)
        plt.figure(figsize=(7, 5))
        top = importance.head(15).iloc[::-1]
        plt.barh(top["feature"], top["importance_mean"])
        plt.xlabel("Permutation importance")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "permutation_importance_fallback.png", dpi=300)
        plt.close()
    importance.to_csv(OUTPUT_DIR / "explainability_importance.csv", index=False)
    return importance


def write_requirements() -> None:
    requirements = [
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "joblib",
        "shap",
        "# optional: xgboost",
    ]
    (OUTPUT_DIR / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")


def write_readme(target: str, features: List[str]) -> None:
    text = f"""# Physics-Constrained Stabilized Soil ML Pipeline

## Input

Place the cleaned or raw CSV file at:

`{DATA_PATH}`

The script standardizes common column-name variants, checks missing variables,
flags physically impossible records, engineers geotechnical features, performs
grouped-source validation, tests physical monotonicity, estimates cradle-to-gate
embodied carbon using editable placeholder emission factors, and generates
explainability outputs.

## Selected target

`{target}`

## Model features

{chr(10).join(f"- `{f}`" for f in features)}

## Reproduction

```bash
pip install -r outputs/requirements.txt
python physics_constrained_stabilized_soil_ml.py
```

## Important LCA note

Emission factors are placeholders and must be replaced with region-specific EPD
or accepted database values before journal submission.
"""
    (OUTPUT_DIR / "README_reproducibility.md").write_text(text, encoding="utf-8")


def main(data_path: str = DATA_PATH) -> None:
    seed_everything()
    OUTPUT_DIR.mkdir(exist_ok=True)

    raw = pd.read_csv(data_path)
    df, mapping = standardize_columns(raw)
    numeric_candidates = sorted(set(TARGET_PRIORITY + BASE_REQUIRED_FEATURES + ["Fly_Ash_Content", "GGBS_Content"]))
    df = coerce_numeric(df, numeric_candidates)
    target = choose_target(df)
    df, physical_flags = detect_physical_quality(df, target)
    df_clean = df.loc[~df["physical_quality_flag"]].copy()
    df_clean = df_clean.loc[df_clean[target].notna()].copy()
    df_clean = feature_engineering(df_clean)
    features = usable_feature_columns(df_clean)
    missing_required = [c for c in BASE_REQUIRED_FEATURES if c not in df_clean.columns or df_clean[c].notna().sum() == 0]
    minimum_data_check(df_clean, target, features)

    quality_summary = make_data_quality_summary(df, target, mapping, missing_required)
    quality_summary.to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False)
    physical_flags.to_csv(OUTPUT_DIR / "physical_quality_flags.csv", index=False)
    df_clean.to_csv(OUTPUT_DIR / "processed_dataset.csv", index=False)

    uncon = grouped_validation(df_clean, target, features, constrained=False)
    cons = grouped_validation(df_clean, target, features, constrained=True)
    fold_table = pd.concat([uncon.fold_table, cons.fold_table], ignore_index=True)
    summary_table = pd.concat([uncon.summary_table, cons.summary_table], ignore_index=True)
    predictions = pd.concat([uncon.predictions, cons.predictions], ignore_index=True)

    fold_table.to_csv(OUTPUT_DIR / "model_fold_metrics.csv", index=False)
    summary_table.to_csv(OUTPUT_DIR / "model_metric_summary.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "cross_validated_predictions.csv", index=False)

    final_uncon = build_model(features, constrained=False).fit(df_clean[features], df_clean[target])
    final_cons = build_model(features, constrained=True).fit(df_clean[features], df_clean[target])

    v_uncon = physical_consistency_test(final_uncon, df_clean, features, target)
    v_uncon["model"] = "unconstrained"
    v_cons = physical_consistency_test(final_cons, df_clean, features, target)
    v_cons["model"] = "physics_constrained"
    violation_table = pd.concat([v_uncon, v_cons], ignore_index=True)
    violation_table.to_csv(OUTPUT_DIR / "physical_violation_table.csv", index=False)

    carbon = embodied_carbon_module(df_clean)
    carbon.to_csv(OUTPUT_DIR / "lca_carbon_summary.csv", index=False)

    best_summary = summary_table.copy()
    best_model_name = best_summary.sort_values("R2_mean", ascending=False).iloc[0]["model"]
    final_model = final_cons if best_model_name == "physics_constrained" else final_uncon
    joblib.dump(final_model, OUTPUT_DIR / "final_trained_model.joblib")

    save_figures(predictions, fold_table, violation_table, carbon)
    shap_or_permutation(final_model, df_clean, target, features)
    write_requirements()
    write_readme(target, features)

    run_meta = {
        "random_seed": RANDOM_SEED,
        "selected_target": target,
        "features": features,
        "missing_required_features": missing_required,
        "best_model_by_mean_R2": best_model_name,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
