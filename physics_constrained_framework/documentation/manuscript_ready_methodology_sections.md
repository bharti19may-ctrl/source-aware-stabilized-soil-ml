# Manuscript-Ready Computational Methodology

## Section 1: Physics-Constrained Machine Learning Framework

A physics-constrained machine learning framework was developed for chemically stabilized soil systems. The framework compares an unconstrained gradient-boosting baseline with a monotonic physics-constrained model. The target hierarchy was defined as unconfined compressive strength, California bearing ratio, and resilient modulus. When more than one target was available, unconfined compressive strength, \(UCS\), was selected as the primary response because it was the most consistently reported mechanical property in the compiled database.

The constrained model was designed to reduce the risk of physically inconsistent prediction. Positive monotonic constraints were assigned to curing duration, maximum dry density, cement content, lime content, total binder content, and binder-to-clay ratio. Negative monotonic constraints were assigned to moisture deviation above optimum moisture content and high plasticity indicators. The constraint vector was generated from the processed feature names rather than hard-coded feature positions, which improves reproducibility when column availability changes across datasets.

## Section 2: Geotechnical Feature Engineering and Mineralogical Proxies

Mechanistically meaningful predictors were derived from the reported geotechnical and binder descriptors. Clay activity was defined as:

\[
A_{clay}=\frac{PI}{CF}
\]

where \(PI\) is plasticity index and \(CF\) is clay fraction. The reactive oxide ratio was defined as:

\[
\phi_{ox}=\frac{SiO_2+Al_2O_3}{CaO}
\]

Moisture deviation from optimum moisture content was calculated as:

\[
\Delta w=w-OMC
\]

Total binder content was calculated as:

\[
B_{total}=C_{cement}+C_{lime}+C_{other}
\]

The binder-to-clay ratio was calculated as:

\[
BCR=\frac{B_{total}}{CF}
\]

Curing maturity was represented as:

\[
M_c=\log(1+t_c)
\]

These variables act as mineralogical and reaction-progress proxies when direct X-ray diffraction, pore-water chemistry, and binder oxide composition are incompletely reported.

## Section 3: Grouped-Source Validation Strategy

Grouped-source validation was used to evaluate transferability across independent literature sources. Source labels, study identifiers, paper identifiers, or laboratory identifiers were used as grouping variables where available. Leave-one-group-out validation was applied when the number of sources was small, while grouped \(k\)-fold validation was used for larger source counts. If no grouping variable was available, the framework issued a warning and used random \(k\)-fold validation only as a fallback.

For each fold, \(R^2\), root mean squared error, mean absolute error, and mean absolute percentage error were computed. \(MAPE\) was calculated only when measured values were non-zero. Fold-wise metrics and mean \(\pm\) standard deviation summaries were exported to support transparent reporting.

## Section 4: Physical Consistency Verification

A controlled perturbation test was used to assess whether predictions followed expected geotechnical trends. Predictions were compared before and after increasing curing duration, maximum dry density, cement content, lime content, and moisture deviation above optimum. A physical violation was recorded when a predicted response contradicted the imposed geotechnical trend. The violation rate was calculated as:

\[
VR=\frac{N_{viol}}{N_{check}}\times100
\]

where \(N_{viol}\) is the number of inconsistent prediction pairs and \(N_{check}\) is the total number of perturbation checks. This verification does not replace external validation, but it strengthens reproducibility and reduces risk of physically inconsistent prediction.

## Section 5: Cradle-to-Gate Life Cycle Assessment

Cradle-to-gate embodied carbon was calculated for A1-A3 stages using \(1\,m^3\) of compacted stabilized soil as the functional unit. The screening equation was:

\[
EC_{mix}=\rho_d\sum_{j=1}^{n}(C_jEF_j)+EC_{transport}
\]

where \(EC_{mix}\) is embodied carbon in \(kg\,CO_2e/m^3\), \(\rho_d\) is dry density in \(kg/m^3\), \(C_j\) is dry mass fraction of material \(j\), \(EF_j\) is the emission factor of material \(j\), and \(EC_{transport}\) is transport-related emission. The default emission factors are editable placeholders and should be replaced with region-specific environmental product declarations or accepted life-cycle inventory database values before journal submission.

## Section 6: SHAP-Based Mechanistic Interpretation

The final trained model was interpreted using SHAP where the package and model compatibility allowed it. Summary, bar, and dependence plots were specified for curing duration, total binder content or cement content, moisture deviation, and maximum dry density. When SHAP was unavailable, permutation importance was used as a fallback. The fallback result still provides a reproducible global importance ranking but should be described more cautiously than full SHAP decomposition.

## Section 7: Reproducibility and Open-Science Availability

The framework fixes the random seed, standardizes column names, records missing variables, filters physically impossible records, performs leakage-safe preprocessing inside each validation fold, exports all intermediate and final tables, saves publication-quality figures, and archives the trained model. The output package includes a requirements file and a README-style reproduction guide. This structure improves reviewer confidence and supports independent verification of the reported computational results.
