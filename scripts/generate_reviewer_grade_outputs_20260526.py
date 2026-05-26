from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


PROJECT = Path(r"D:\kec folder\my files\student project\Elsevier_Q2_ExternalOnly_StabilizedSoil_ML")
DATA_DIR = PROJECT / "Final_Reliable_Dataset_For_Elsevier_v1_2026-05-25"
DECISION_DIR = PROJECT / "Decision_Support_Framework_v1_2026-05-25"
SYN_DIR = PROJECT / "Synthetic_Augmentation_PrePaper_Audit_v1_2026-05-26"
OUT = PROJECT / "Reviewer_Grade_Figures_Tables_v1_2026-05-26"
FIG = OUT / "figures"
TAB = OUT / "tables"
for d in [OUT, FIG, TAB]:
    d.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "domain_family", "soil_class", "sand_pct", "silt_pct", "clay_pct", "gravel_pct",
    "ll_pct", "pl_pct", "pi_pct", "specific_gravity", "omc_pct", "mdd_gcm3",
    "water_content_pct", "dry_density_gcm3", "cement_pct", "lime_pct", "fly_ash_pct",
    "ggbs_pct", "rice_husk_ash_pct", "pond_ash_pct", "marble_powder_pct", "harha_pct",
    "fgd_gypsum_pct", "biopolymer_pct", "xanthan_gum_pct", "guar_gum_pct", "chitosan_pct",
    "lignin_pct", "fiber_pct", "polypropylene_pct", "jute_pct", "waste_plastic_pct",
    "sludge_pct", "coconut_shell_pct", "naoh_pct", "na2sio3_pct", "alkali_activator_pct",
    "water_binder_ratio", "compaction_rate_mm_min", "curing_days", "cbr_condition",
]
CAT = {"domain_family", "soil_class", "cbr_condition"}
NUM = [c for c in FEATURES if c not in CAT]


def load():
    ucs = pd.read_csv(DATA_DIR / "final_reliable_UCS_dataset_v1.csv")
    cbr = pd.read_csv(DATA_DIR / "final_reliable_CBR_dataset_v1.csv")
    preds = pd.read_csv(DECISION_DIR / "decision_support_classification_predictions_v1.csv")
    metrics = pd.read_csv(DECISION_DIR / "decision_support_classification_metrics_v1.csv")
    synth = pd.read_csv(SYN_DIR / "synthetic_classification_vs_real_only_v1.csv")
    pareto = pd.read_csv(DECISION_DIR / "pareto_front_real_mixtures_v1.csv")
    return ucs, cbr, preds, metrics, synth, pareto


def add_labels(df):
    df = df.copy()
    if "ucs_kpa" in df:
        u = pd.to_numeric(df["ucs_kpa"], errors="coerce")
        df["ucs_high_binary"] = np.where(u >= 1500, "high_or_above", "below_high")
        df["ucs_class"] = pd.cut(u, [-np.inf, 500, 1500, 3000, np.inf], labels=["low", "medium", "high", "very_high"]).astype(str)
    if "cbr_pct" in df:
        c = pd.to_numeric(df["cbr_pct"], errors="coerce")
        df["cbr_pass_8"] = np.where(c >= 8, "pass", "fail")
        df["cbr_class"] = pd.cut(c, [-np.inf, 5, 8, 20, np.inf], labels=["low_lt5", "marginal_5_8", "subbase_8_20", "high_gt20"]).astype(str)
    return df


def source_inventory(ucs, cbr):
    combo = pd.concat([ucs.assign(response="UCS"), cbr.assign(response="CBR")], ignore_index=True)
    inv = combo.groupby(["source_dataset", "source_doi", "domain_family"]).agg(
        total_rows=("universal_id", "count"),
        ucs_rows=("ucs_kpa", lambda s: s.notna().sum()),
        cbr_rows=("cbr_pct", lambda s: s.notna().sum()),
        variables_available=("universal_id", lambda s: ""),
    ).reset_index()
    feat_cols = [c for c in FEATURES if c in combo.columns]
    availability = combo.groupby("source_dataset")[feat_cols].apply(lambda g: int(g.notna().any().sum())).rename("n_features_with_any_data")
    inv = inv.merge(availability, on="source_dataset", how="left")
    inv["variables_available"] = inv["n_features_with_any_data"].astype(str) + " of " + str(len(feat_cols))
    inv.to_csv(TAB / "table_data_source_inventory.csv", index=False)
    return inv


def plot_source_distribution(inv):
    plot = inv.sort_values("total_rows", ascending=True)
    fig, ax = plt.subplots(figsize=(9.2, max(4.5, 0.35 * len(plot))))
    ax.barh(plot["source_dataset"], plot["ucs_rows"], label="UCS rows", color="#315C7A")
    ax.barh(plot["source_dataset"], plot["cbr_rows"], left=plot["ucs_rows"], label="CBR rows", color="#8B1E3F")
    ax.set_xlabel("Number of real rows")
    ax.set_title("Real Dataset Contribution by Literature Source")
    ax.legend()
    fig.tight_layout()
    p = FIG / "fig_source_distribution_by_response.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


def plot_missingness(ucs, cbr):
    combo = pd.concat([ucs.assign(response="UCS"), cbr.assign(response="CBR")], ignore_index=True)
    groups = {
        "Gradation": ["sand_pct", "silt_pct", "clay_pct", "gravel_pct"],
        "Plasticity": ["ll_pct", "pl_pct", "pi_pct"],
        "Compaction": ["omc_pct", "mdd_gcm3", "water_content_pct", "dry_density_gcm3"],
        "Binders": ["cement_pct", "lime_pct", "fly_ash_pct", "ggbs_pct", "rice_husk_ash_pct", "pond_ash_pct", "harha_pct"],
        "Fibres": ["fiber_pct", "polypropylene_pct", "jute_pct", "waste_plastic_pct"],
        "Activators": ["naoh_pct", "na2sio3_pct", "alkali_activator_pct"],
        "Curing/testing": ["curing_days", "cbr_condition"],
    }
    rows = []
    for src, g in combo.groupby("source_dataset"):
        row = {"source_dataset": src}
        for name, cols in groups.items():
            present = [c for c in cols if c in g.columns]
            vals = g[present].notna().mean().mean() if present else np.nan
            row[name] = vals
        rows.append(row)
    heat = pd.DataFrame(rows).set_index("source_dataset")
    heat.to_csv(TAB / "table_feature_group_availability_by_source.csv")
    fig, ax = plt.subplots(figsize=(8.6, max(4.5, 0.32 * len(heat))))
    im = ax.imshow(heat.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_title("Feature Availability by Literature Source")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean non-missing fraction")
    fig.tight_layout()
    p = FIG / "fig_feature_availability_heatmap.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


def confusion_figs(preds):
    outputs = []
    tasks = [
        ("UCS_high_binary", "ucs_high_binary", "group_by_source_5fold", "ExtraTrees"),
        ("CBR_pass_8_binary", "cbr_pass_8", "group_by_source_3fold", "RF"),
        ("UCS_multiclass", "ucs_class", "group_by_source_5fold", "RF"),
    ]
    for dataset, label, validation, model in tasks:
        sub = preds[(preds["dataset"] == dataset) & (preds["label"] == label) & (preds["validation"] == validation) & (preds["model"] == model)].copy()
        if sub.empty:
            continue
        labels = sorted(pd.unique(pd.concat([sub["actual"], sub["predicted"]]).astype(str)))
        cm = confusion_matrix(sub["actual"].astype(str), sub["predicted"].astype(str), labels=labels)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(TAB / f"confusion_{dataset}_{validation}_{model}.csv")
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title(f"{dataset.replace('_', ' ')}\n{validation}, {model}")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        p = FIG / f"fig_confusion_{dataset}_{validation}_{model}.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        outputs.append(p)
    return outputs


def make_pre(df):
    cols = [c for c in FEATURES if c in df.columns]
    num = [c for c in cols if c in NUM]
    cat = [c for c in cols if c not in num]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", RobustScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=2))]), cat),
    ])
    return cols, pre


def model_importance(df, target_label, outstem):
    df = add_labels(df.copy())
    df = df[df[target_label].notna()].copy()
    y = df[target_label].astype(str)
    cols, pre = make_pre(df)
    clf = ExtraTreesClassifier(random_state=42, n_estimators=600, min_samples_leaf=3, max_features=0.7, n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("model", clf)])
    pipe.fit(df[cols], y)
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pipe.named_steps["model"].feature_importances_
    rows = []
    for name, val in zip(names, imp):
        raw = name.split("__", 1)[1]
        base = raw.split("_", 1)[0] if raw.startswith("domain") else raw
        # Recover one-hot original feature names more conservatively.
        for f in cols:
            if raw == f or raw.startswith(f + "_"):
                base = f
                break
        rows.append({"feature": base, "importance": val})
    agg = pd.DataFrame(rows).groupby("feature", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    agg.to_csv(TAB / f"feature_importance_{outstem}.csv", index=False)
    top = agg.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.barh(top["feature"], top["importance"], color="#315C7A")
    ax.set_xlabel("Aggregated ExtraTrees feature importance")
    ax.set_title(f"Model Explainability: {outstem.replace('_', ' ')}")
    fig.tight_layout()
    p = FIG / f"fig_feature_importance_{outstem}.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


def synthetic_ratio_plot(synth):
    sub = synth[synth["validation"].str.contains("group")].copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for task, g in sub.groupby("dataset"):
        best_model = g[g["augmentation_multiplier"] == 0].sort_values("balanced_accuracy", ascending=False)["model"].iloc[0]
        gg = g[g["model"] == best_model].sort_values("augmentation_multiplier")
        ax.plot(gg["augmentation_multiplier"], gg["balanced_accuracy"], marker="o", label=f"{task} ({best_model})")
    ax.set_xlabel("Synthetic-to-real training multiplier")
    ax.set_ylabel("Grouped-source balanced accuracy on real validation")
    ax.set_title("Synthetic Augmentation Ratio Sensitivity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "fig_synthetic_ratio_sensitivity.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


def pareto_plot(pareto):
    cols = pareto.columns
    carbon = "kgco2e_per_100kg_soil" if "kgco2e_per_100kg_soil" in cols else None
    target = "gain" if "gain" in cols else ("ucs_kpa" if "ucs_kpa" in cols else None)
    if not carbon or not target:
        return None
    sub = pareto[pd.to_numeric(pareto[carbon], errors="coerce").notna() & pd.to_numeric(pareto[target], errors="coerce").notna()].copy()
    sub[carbon] = pd.to_numeric(sub[carbon], errors="coerce")
    sub[target] = pd.to_numeric(sub[target], errors="coerce")
    sub.to_csv(TAB / "table_pareto_real_mixtures_for_plot.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    colors = sub["pareto_response"].map({"UCS": "#315C7A", "CBR": "#8B1E3F"}).fillna("#315C7A")
    ax.scatter(sub[carbon], sub[target], c=colors, s=30, alpha=0.8)
    ax.set_xlabel("Estimated kg CO2e per 100 kg soil")
    ax.set_ylabel("Performance gain over source baseline")
    ax.set_title("Real Pareto-Efficient Mixtures: Strength Gain Versus Carbon Burden")
    fig.tight_layout()
    p = FIG / "fig_real_pareto_carbon_strength.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


def expanded_references():
    refs = [
        "Ahmad, S., Ghazi, M. S. A., Syed, M., and Al-Osta, M. A. 2024. Utilization of fly ash with and without secondary additives for stabilizing expansive soils: A review. Results in Engineering, 22, 102079.",
        "Almuaythir, S., Zaini, M. S. I., and Lodhi, R. H. 2025. Predicting soil compaction parameters in expansive soils using advanced machine learning models: A comparative study. Scientific Reports, 15, 24018.",
        "ASTM D638. 2014. Standard test method for tensile properties of plastics. ASTM International.",
        "Barrera-Animas, A. Y., and Davila Delgado, J. M. 2023. Generating real-world-like labelled synthetic datasets for construction site applications. Automation in Construction, 151, 104850.",
        "Bureau of Indian Standards. 1970. IS 1498: Classification and identification of soils for general engineering purposes. BIS.",
        "Bureau of Indian Standards. 2016. IS 2720: Methods of test for soils. BIS.",
        "Chauhan, K. S., and Rajesh, J. 2015. Effect of fly ash and fibre on index properties of black cotton soil. International Journal of Science Technology and Engineering, 2, 1-6.",
        "Chavan, G. A., and Savoikar, P. 2023. Characterization of Black Cotton soil in Kolhapur region. Indian Geotechnical Journal, 53, 1454-1467.",
        "Degirmenci, N., Okucu, A., and Turabi, A. 2007. Application of phosphogypsum in soil stabilization.",
        "Deshpande, S. S., and Puranik, M. M. 2017. Effect of fly ash and polypropylene on the engineering properties of black cotton soil. SSRG International Journal of Civil Engineering, 4, 49-52.",
        "Dixit, A., Nigam, M., and Mishra, R. 2016. Effect of fly ash on geotechnical properties of soil. International Journal of Engineering Technology Management and Research, 3, 7-14.",
        "Dixit, M. S. 2017. Optimum use of polypropylene fibers improves soil properties. International Journal of Civil Engineering and Technology, 8, 149-154.",
        "Eyo, E. U., and Onyekpe, U. 2021. Data on one-dimensional vertical free swelling potential of soils and related soil properties. Data in Brief, 39, 107608.",
        "Eyo, E. U., Abbey, S. J., Lawrence, T. T., and Tetteh, F. K. 2022. Improved prediction of clay soil expansion using machine learning algorithms and meta-heuristic dichotomous ensemble classifiers. Geoscience Frontiers, 13, 101296.",
        "Ge, Q., Li, J., Lacasse, S., Sun, H., and Liu, Z. 2024. Data-augmented landslide displacement prediction using generative adversarial network. Journal of Rock Mechanics and Geotechnical Engineering, 16, 4017-4033.",
        "Gautam, K. K., and Patel, S. 2025. Experimental investigation to stabilize black cotton soil using lime, waste quarry dust, and polyester fiber. Lecture Notes in Civil Engineering.",
        "Huang, K., Zhang, Z., Sun, Y., Shi, X., Wang, C., Guo, S., and Fan, C. 2025. Characterization of engineering properties in stabilized expansive soil: Integrating experimental investigation and machine learning modeling. Construction and Building Materials, 485, 141960.",
        "Indian Roads Congress. 2012. IRC: 37: Guidelines for the design of flexible pavements. IRC.",
        "Indiramma, P., Sudharani, C., and Needhidasan, S. 2020. Utilization of fly ash and lime to stabilize expansive soil. Materials Today: Proceedings, 22, 694-700.",
        "Jayashree, J., and Roja, S. Y. 2019. Stabilization of expansive soil using rice husk ash and lime. International Journal of Recent Technology and Engineering, 8, 2661-2665.",
        "Katti, R. K. 1978. Search for solutions to problems in black cotton soils. Indian Institute of Technology.",
        "Li, H., Chen, W., and Tan, X. 2025. Back analysis of geomechanical parameters based on a data augmentation algorithm and machine learning technique. Underground Space, 21, 215-231.",
        "Luhar, I., and Luhar, S. 2022. A comprehensive review on fly ash-based geopolymer. Journal of Composites Science, 6, 219.",
        "Luo, Z., Xue, X., and Xiong, C. 2025. Data-driven prediction of unconfined compressive strength in stabilized soils using machine learning. Discover Applied Sciences, 7, 1122.",
        "Mai-Bade, U. Y., Chinade, A. U., Batari, A., and Saeed, S. M. 2021. Stabilization of black cotton soil using various admixtures: A review. Composite Materials, 5, 37-45.",
        "Mamuye, Y., and Geremew, A. 2019. Stabilization of expansive soil using waste materials.",
        "Mazhar, S., and GuhaRay, A. 2020. Stabilization of expansive clay by fibre-reinforced alkali-activated binder: Experimental investigation and prediction modelling. International Journal of Geotechnical Engineering, 15, 977-993.",
        "Mitchell, J. K., and Soga, K. 2005. Fundamentals of soil behavior. John Wiley & Sons.",
        "Moghaddas, S. A., and Bao, Y. 2025. Explainable machine learning framework for predicting concrete abrasion depth. Case Studies in Construction Materials, 22, e04686.",
        "Nanda, R. P., and Priya, N. 2024. Geopolymer as stabilising materials in pavement constructions: A review. Cleaner Waste Systems, 7, 100134.",
        "Navagire, O. P., Srinivasan, V., Patel, A., and Gowrisankar, D. 2025. Stabilization of expansive soils using polypropylene fiber for enhanced engineering properties. Indian Geotechnical Journal.",
        "Parihar, N. S., and Gupta, A. K. 2024. Stabilization of expansive soils using non-conventional waste stabilizers: A review. Indian Geotechnical Journal, 54, 971-997.",
        "Patil, S. C., Mathada, D. V. S., and Bharamagoud, B. 2018. Comparative study on black cotton soil stabilization using lime and sisal fiber.",
        "Phani Kumar, B. R., and Sharma, R. S. 2004. Effect of fly ash on engineering properties of expansive soils. Journal of Geotechnical and Geoenvironmental Engineering, 130, 764-767.",
        "Rao, A. U., Kiran, D., Kumar, A. G. S., Prakash, K. G., and Maddodi, B. S. 2023. Effect of polypropylene macro fiber on geotechnical characteristics of black cotton soil. Engineered Science, 21, 1-13.",
        "Rokade, S., Kumar, R., and Jain, P. K. 2017. Effect of inclusion of fly ash and nylon fiber on strength characteristics of black cotton soil.",
        "Saad, A. H., Nahazanan, H., Yusuf, B., Toha, S. F., Alnuaim, A., El-Mouchi, A., Elseknidy, M., and Mohammed, A. A. 2023. Systematic review of ML applications in soil improvement using green materials. Sustainability, 15, 9738.",
        "Sharma, K., and Kumar, A. 2020. Utilization of industrial waste-based geopolymers as a soil stabilizer: A review. Innovative Infrastructure Solutions, 5, 97.",
        "Shukla, N. K., and Sharma, A. K. 2025. Recent advances in stabilized expansive soils.",
        "Syed, M., Agarwal, S., and GuhaRay, A. 2021. Stabilization of expansive black cotton soil using alkali-activated binder with glass and polypropylene fiber. Lecture Notes in Civil Engineering.",
        "Syed, M., GuhaRay, A., Ghadge, H., and Chikte, A. 2025. ANN estimation of compressive strength of glass fiber reinforced expansive soil in alkaline activated binder. Lecture Notes in Civil Engineering.",
        "Thapa, I., and Ghani, S. 2024. Enhancing UCS prediction in nano-silica stabilized soil: Ensemble and deep learning models. Modeling Earth Systems and Environment, 10, 5079-5102.",
        "Thomas, G. E., and John, J. 2016. Stabilization of expansive soil using waste additives.",
        "Tiwari, D. K., Dixit, R. K., and Roy, S. 2016. Stabilization of black cotton soil using stone dust and polypropylene fibers.",
        "Wang, H., Liu, T., Yan, C., and Wang, J. 2023. Expansive soil stabilization using alkali-activated fly ash. Processes, 11, 1550.",
        "Yang, H., Hu, G., Liu, L., Li, Y., Deng, Y., and Wu, J. 2025. A physics-informed and SHAP-enhanced modeling framework for predicting strength of cement-stabilized soil. Case Studies in Construction Materials, 23, e05280.",
        "Yang, C., Jiang, Y., Li, Z., Huang, Y., and Yue, J. 2025. Pavement performance of slag/fly ash-based geopolymer-stabilized soil. Materials, 18, 3173.",
        "Zeini, H. A., Al-Jeznawi, D., Imran, H., Bernardo, L. F. A., Al-Khafaji, Z., and Ostrowski, K. A. 2023. Random forest algorithm for strength prediction of geopolymer stabilized clayey soil. Sustainability, 15, 1408.",
        "Zong, B., Li, J., Yuan, T., Wang, J., and Yuan, R. 2025. Recent progress on machine learning with limited materials data. Journal of Materiomics, 11, 100916.",
    ]
    pd.DataFrame({"reference": refs}).to_csv(TAB / "expanded_reference_backbone_50_items.csv", index=False)
    (OUT / "EXPANDED_REFERENCE_BACKBONE_50_ITEMS.md").write_text("\n".join(f"{i+1}. {r}" for i, r in enumerate(refs)), encoding="utf-8")


def roadmap_note(paths):
    note = f"""# Reviewer-Grade Output Package

Created: 2026-05-26

This package was generated after auditing recent Elsevier papers. These outputs should replace the simple figures in the earlier draft.

## Generated figures

{chr(10).join('- ' + p.name for p in paths if p)}

## Generated tables

{chr(10).join('- ' + p.name for p in sorted(TAB.glob('*')))}

## Manuscript implication

The manuscript should now be rewritten around these outputs:

1. Data curation and source inventory.
2. Feature availability and data limitations.
3. Internal versus grouped-source validation.
4. Confusion matrices for engineering decision tasks.
5. Explainability using model-based feature importance.
6. Synthetic augmentation as training-only sensitivity analysis.
7. Real-only carbon-strength Pareto screening.

Do not present the earlier simple bar-chart draft as final.
"""
    (OUT / "README_REVIEWER_GRADE_OUTPUTS.md").write_text(note, encoding="utf-8")


def main():
    ucs, cbr, preds, metrics, synth, pareto = load()
    inv = source_inventory(ucs, cbr)
    paths = [
        plot_source_distribution(inv),
        plot_missingness(ucs, cbr),
        *confusion_figs(preds),
        model_importance(ucs, "ucs_high_binary", "ucs_high_binary"),
        model_importance(cbr, "cbr_pass_8", "cbr_pass_8"),
        synthetic_ratio_plot(synth),
        pareto_plot(pareto),
    ]
    expanded_references()
    roadmap_note(paths)
    print(OUT)


if __name__ == "__main__":
    main()
