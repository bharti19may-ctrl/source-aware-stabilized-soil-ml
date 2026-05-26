from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

PROJECT = Path(r"D:\kec folder\my files\student project\Elsevier_Q2_ExternalOnly_StabilizedSoil_ML")
DECISION_DIR = PROJECT / "Decision_Support_Framework_v1_2026-05-25"
SYN_DIR = PROJECT / "Synthetic_Augmentation_PrePaper_Audit_v1_2026-05-26"
DATA_DIR = PROJECT / "Final_Reliable_Dataset_For_Elsevier_v1_2026-05-25"
OUT = PROJECT / "Review_Report_Required_Analyses_v1_2026-05-26"
FIG = OUT / "figures"
TAB = OUT / "tables"
for p in [OUT, FIG, TAB]:
    p.mkdir(parents=True, exist_ok=True)


def fold_uncertainty():
    pred = pd.read_csv(DECISION_DIR / "decision_support_classification_predictions_v1.csv")
    rows = []
    for keys, g in pred.groupby(["dataset", "label", "validation", "model"]):
        vals = []
        for fold, f in g.groupby("fold"):
            if f["actual"].nunique() < 2:
                continue
            vals.append({
                "balanced_accuracy": balanced_accuracy_score(f["actual"].astype(str), f["predicted"].astype(str)),
                "accuracy": accuracy_score(f["actual"].astype(str), f["predicted"].astype(str)),
                "weighted_f1": f1_score(f["actual"].astype(str), f["predicted"].astype(str), average="weighted"),
            })
        if vals:
            m = pd.DataFrame(vals)
            rows.append({
                "dataset": keys[0],
                "label": keys[1],
                "validation": keys[2],
                "model": keys[3],
                "folds": len(m),
                "balanced_accuracy_mean": m["balanced_accuracy"].mean(),
                "balanced_accuracy_sd": m["balanced_accuracy"].std(ddof=1) if len(m) > 1 else 0,
                "accuracy_mean": m["accuracy"].mean(),
                "accuracy_sd": m["accuracy"].std(ddof=1) if len(m) > 1 else 0,
                "weighted_f1_mean": m["weighted_f1"].mean(),
                "weighted_f1_sd": m["weighted_f1"].std(ddof=1) if len(m) > 1 else 0,
            })
    out = pd.DataFrame(rows).sort_values(["dataset", "validation", "model"])
    out.to_csv(TAB / "table_fold_uncertainty_classification.csv", index=False)
    return out


def threshold_sensitivity():
    ucs = pd.read_csv(DATA_DIR / "final_reliable_UCS_dataset_v1.csv")
    cbr = pd.read_csv(DATA_DIR / "final_reliable_CBR_dataset_v1.csv")
    rows = []
    for thr in [1000, 1500, 2000, 2500]:
        y = (pd.to_numeric(ucs["ucs_kpa"], errors="coerce") >= thr).astype(int)
        rows.append({
            "target": "UCS",
            "threshold": thr,
            "unit": "kPa",
            "n": len(y),
            "positive_class_count": int(y.sum()),
            "positive_class_fraction": float(y.mean()),
            "interpretation": "higher threshold gives stricter high-strength class",
        })
    for thr in [5, 8, 10, 15]:
        y = (pd.to_numeric(cbr["cbr_pct"], errors="coerce") >= thr).astype(int)
        rows.append({
            "target": "CBR",
            "threshold": thr,
            "unit": "%",
            "n": len(y),
            "positive_class_count": int(y.sum()),
            "positive_class_fraction": float(y.mean()),
            "interpretation": "higher threshold gives stricter bearing-capacity class",
        })
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "table_threshold_sensitivity_class_balance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for target, g in out.groupby("target"):
        ax.plot(g["threshold"], g["positive_class_fraction"], marker="o", label=target)
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Positive-class fraction")
    ax.set_title("Threshold Sensitivity of Class Balance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig_threshold_sensitivity_class_balance.png", dpi=300)
    plt.close(fig)
    return out


def regression_baseline():
    reg = pd.read_csv(SYN_DIR / "synthetic_regression_vs_real_only_imported_v1.csv")
    base = reg[reg["augmentation_multiplier"].eq(0)].copy()
    best_base = base.sort_values(["dataset", "validation", "r2"], ascending=[True, True, False]).groupby(["dataset", "validation"]).head(1)
    best_aug = reg[reg["augmentation_multiplier"].gt(0)].sort_values(["dataset", "validation", "delta_r2_vs_same_model_baseline"], ascending=[True, True, False]).groupby(["dataset", "validation"]).head(1)
    out = pd.concat([
        best_base.assign(case="best_real_only"),
        best_aug.assign(case="best_augmented_training"),
    ], ignore_index=True)
    out.to_csv(TAB / "table_regression_baseline_and_augmented_comparison.csv", index=False)
    return out


def hyperparams():
    rows = [
        {"model": "Random Forest classifier/regressor", "implementation": "scikit-learn", "main_parameters": "n_estimators=500; min_samples_leaf=4 for main runs; max_features=0.7; random_state=42; n_jobs=-1", "role": "classification and baseline comparison"},
        {"model": "ExtraTrees classifier/regressor", "implementation": "scikit-learn", "main_parameters": "n_estimators=500-600; min_samples_leaf=3-4; max_features=0.7; random_state=42; n_jobs=-1", "role": "classification and feature importance"},
        {"model": "Histogram Gradient Boosting", "implementation": "scikit-learn", "main_parameters": "max_iter=220-300; learning_rate=0.04-0.05; l2_regularization=0.05-0.08; max_leaf_nodes=10-15; random_state fixed", "role": "regression and equation-guided synthetic target assignment"},
        {"model": "Preprocessing", "implementation": "scikit-learn", "main_parameters": "median imputation and robust scaling for numerical variables; most-frequent imputation and one-hot encoding for categorical variables", "role": "all tabular models"},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "table_model_hyperparameters_reproducibility.csv", index=False)
    return out


def write_note():
    note = """# Analyses Added from Review Report

Created: 2026-05-26

This package contains analyses requested in the external review report:

- Fold-level uncertainty table for classification metrics.
- Threshold sensitivity table and figure.
- Regression baseline versus augmentation comparison.
- Model/hyperparameter reproducibility table.

These files should be used in the revised manuscript methods and results sections.
"""
    (OUT / "README_REVIEW_REQUIRED_ANALYSES.md").write_text(note, encoding="utf-8")


def main():
    fold_uncertainty()
    threshold_sensitivity()
    regression_baseline()
    hyperparams()
    write_note()
    print(OUT)


if __name__ == "__main__":
    main()
