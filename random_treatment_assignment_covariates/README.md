# README: Treatment Effect Estimation (Submission 4)

## Overview

This document describes the implementation for Submission 4. This version evolves from previous approaches by adopting a hybrid machine learning and Bayesian modeling strategy to enhance the precision of treatment effect estimates.

While treatment is randomly assigned, including predictive covariates in a regression model can reduce residual variance and produce more precise estimates. This implementation formalizes that process in a principled, two-step manner.

The core objectives of this script are to calculate:

1. **PATE** (Population Average Treatment Effect)
2. **subCATE** (Subgroup Conditional Average Treatment Effect) for the binary covariate `x12`
3. **iCATE** (Individual Conditional Average Treatment Effect) for each unit
4. **sCATE** (Sample Conditional Average Treatment Effect)

## Key Methodological Features

This implementation is distinct from previous versions, particularly in its estimation of average effects:

1. **Bayesian Regression for Average Effects**: PATE and subCATE are estimated using Bayesian regression models via the `bambi` library. This model-based approach controls for covariates to increase the statistical precision of the treatment effect estimates.

2. **Principled Covariate Selection with LASSO**: To avoid overfitting and select only the most relevant predictors, a two-stage process is used:
    * **Stage 1 (Selection):** A LASSO (`LassoCV`) regression is run on the full set of 40 covariates to identify a smaller, more informative subset that is most predictive of the outcome `y`.
    * **Stage 2 (Estimation):** Only this pre-selected subset of covariates is included in the final Bayesian models for PATE and subCATE estimation.

3. **Causal Forest for Individual Effects (iCATE)**: The robust `CausalForestDML` model from the `EconML` library is retained for estimating iCATEs. This ensures a powerful, non-linear approach to capturing individual-level heterogeneity, consistent with previous submissions.

4. **Heterogeneity-Adjusted CIs for sCATE**: The advanced method for calculating sCATE confidence intervals, which incorporates iCATE estimates to adjust for treatment effect heterogeneity, is also retained from the previous implementation.

## Implementation Details

The script iterates through each data file and performs the following sequence of estimations.

### 1. Covariate Selection (Automated Feature Engineering)

* **Method**: Before any effect estimation, the script runs a `LassoCV` model, regressing the outcome `y` on all 40 covariates. This automatically selects the most relevant features for inclusion in subsequent models.

### 2. PATE Estimation

* **Method**: A Bayesian regression model (`y ~ z + <selected_covariates>`) is fit using `bambi`. The model includes the treatment indicator `z` and the subset of covariates selected by LASSO. The coefficients for `z` from the model's posterior distribution provide the PATE estimates and their 95% credible intervals.
* **Output**: A `PATE_*.csv` file and a `BEST_PATE_*.csv` file.

### 3. subCATE Estimation

* **Method**: The data is stratified by the `x12` subgroup (`x12=0` and `x12=1`). For each subgroup, a separate Bayesian regression model (identical in structure to the PATE model) is fit on the subsetted data.
* **Output**: A `subCATE_*.csv` file containing the estimates for each treatment arm within each subgroup, and a corresponding `BEST_subCATE_*.csv` file.

### 4. iCATE Estimation

* **Method**: This process is consistent with the previous submission. A `CausalForestDML` model is trained, and its raw iCATE predictions are then adjusted to ensure their subgroup averages are consistent with the more robust subCATEs estimated via the Bayesian models.
* **Output**: An `iCATE_*.csv` file with adjusted individual-level estimates and a `BEST_iCATE_*.csv` file.

### 5. sCATE Estimation

* **Method**: This process is also consistent with the previous submission. The point estimate is taken from the PATE results, but the confidence intervals are derived from a specialized variance formula that adjusts for treatment effect heterogeneity using the iCATEs.
* **Output**: A `sCATE_*.csv` file with point estimates and heterogeneity-adjusted 95% confidence intervals, and a `BEST_sCATE_*.csv` file.

### 6. Consistency Checks

The script concludes by performing and printing several numerical checks to ensure the internal consistency of the estimates (e.g., PATE vs. weighted subCATE, subCATE vs. averaged iCATE).

## How to Run

1. Place the competition data files (e.g., `data_0001.csv`) in the `curated_data/` directory.
2. Configure the script by setting the global constants at the top of the file:

    ```python
    TEAM_ID = "0020"
    SUBMISSION_ID = "4"
    DATA_FOLDER = "curated_data"
    TREATMENT_ARMS = ["b", "c", "d", "e"]
    CONTROL_ARM = "a"
    ```

3. Run the Python script.

The results will be saved in a new directory named based on the `TEAM_ID` and `SUBMISSION_ID` (e.g., `0020_4/`).

## Dependencies

The script requires the following Python libraries:
* `pandas`
* `numpy`
* `scikit-learn`
* `econml`
* `lightgbm`
* `bambi`
* `matplotlib`
* `seaborn`
