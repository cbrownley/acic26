# README: Treatment Effect Estimation (Submission 3)

## Overview

This document describes the implementation for Submission 3, which estimates several causal quantities of interest from the provided datasets. This version marks a significant methodological shift from previous regression-based approaches.

Given that the treatment variable `z` is randomly assigned, this implementation leverages non-parametric and robust methods for estimation. This approach simplifies the estimation of average effects, reduces dependency on model specification, and focuses computational power on estimating heterogeneous effects.

The core objectives of this script are to calculate:

1. **PATE** (Population Average Treatment Effect)
2. **subCATE** (Subgroup Conditional Average Treatment Effect) for the binary covariate `x12`
3. **iCATE** (Individual Conditional Average Treatment Effect) for each unit
4. **sCATE** (Sample Conditional Average Treatment Effect)

## Key Methodological Features

This implementation is distinct from previous versions in three key ways:

1. **Direct Difference-in-Means for Average Effects**: For PATE and subCATE, the estimation relies on a direct difference-in-means calculation. This is a robust, unbiased estimator for average treatment effects in the context of a randomized controlled trial (RCT), which this dataset represents. This avoids potential misspecification issues inherent in regression models.

2. **Heterogeneity-Adjusted CIs for sCATE**: While the sCATE point estimate is also a difference-in-means, its confidence intervals are calculated using a specialized variance formula (`scate_variance_with_icates`). This advanced method incorporates the estimated iCATEs to adjust for treatment effect heterogeneity, providing more accurate and robust confidence intervals than standard formulas.

3. **Causal Forest for Individual Effects (iCATE)**: To capture treatment effect heterogeneity at the individual level, this implementation uses the `CausalForestDML` model from the `EconML` library. This powerful, tree-based model is specifically designed to estimate iCATEs in a robust, non-linear fashion.

## Implementation Details

The script iterates through each data file and performs the following sequence of estimations.

### 1. PATE Estimation

- **Method**: The PATE for each treatment arm (`b`, `c`, `d`, `e`) versus the control arm (`a`) is calculated directly using a **difference-in-means** approach via the `get_diff_in_means` function.
- **Output**: A `PATE_*.csv` file containing the point estimate and 95% confidence interval for each treatment arm. A `BEST_PATE_*.csv` identifies the arm with the highest estimated effect.

### 2. subCATE Estimation

- **Method**: The data is first stratified into subgroups based on the binary covariate `x12` (`x12=0` and `x12=1`). The same **difference-in-means** function is then applied to each subgroup to calculate the subCATEs.
- **Output**: A `subCATE_*.csv` file containing estimates and CIs for each treatment arm within each subgroup. A `BEST_subCATE_*.csv` identifies the best treatment arm for each subgroup.

### 3. iCATE Estimation

- **Method**: This is a multi-step process:
    1. **Model Training**: A `CausalForestDML` model is trained using `lightgbm` as the underlying nuisance model learner. It estimates the conditional expectation of the outcome (`Y`) and the treatment assignment (`T`) to isolate the causal effect.
    2. **Raw Estimation**: The trained model predicts the iCATE (and its confidence interval) for each individual and each treatment arm.
    3. **Adjustment**: The raw iCATEs are centered to ensure consistency with the more robust, non-parametric subCATE estimates. The average of the raw iCATEs within each `x12` subgroup is calculated, and the difference between this average and the subCATE estimate is applied as an adjustment factor to all individual iCATEs in that subgroup.
- **Output**: An `iCATE_*.csv` file containing the final, adjusted estimates and CIs for every individual. A `BEST_iCATE_*.csv` file identifies the optimal treatment for each individual.

### 4. sCATE Estimation

- **Method**:
    1. **Point Estimate**: The point estimate for the sCATE is the same as the PATE (simple difference-in-means).
    2. **Confidence Interval**: The standard error for the sCATE is calculated using a specialized variance formula that accounts for treatment effect heterogeneity by incorporating the previously estimated iCATEs. This provides a more precise CI than standard methods.
- **Output**: A `sCATE_*.csv` file containing the point estimates and the heterogeneity-adjusted 95% confidence intervals. A `BEST_sCATE_*.csv` file is also generated.

### 5. Consistency Checks

The script concludes by performing and printing several numerical checks to ensure the internal consistency of the estimates (e.g., PATE vs. weighted subCATE, subCATE vs. averaged iCATE).

## How to Run

1. Place the competition data files (e.g., `data_0001.csv`) in the `curated_data/` directory.
2. Configure the script by setting the global constants at the top of the file:

    ```python
    TEAM_ID = "0020"
    SUBMISSION_ID = "3"
    DATA_FOLDER = "curated_data"
    TREATMENT_ARMS = ["b", "c", "d", "e"]
    CONTROL_ARM = "a"
    ```

3. Run the Python script.

The results will be saved in a new directory named based on the `TEAM_ID` and `SUBMISSION_ID` (e.g., `0020_3/`).

## Dependencies

The script requires the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `econml`
- `lightgbm`
- `matplotlib`
- `seaborn`
