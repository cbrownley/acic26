# ACIC 2026 Data Challenge — Submission README

**Team ID:** XXXX  
**Submission ID:** 1  
**Track:** Curated (18 representative datasets)

---

## Summary

We submit a four-estimator variance-weighted ensemble built around doubly-robust (DR) and double-machine-learning (DML) methods, implemented in Python using EconML and scikit-learn. The approach is designed to produce accurate point estimates and well-calibrated uncertainty intervals for all four required estimands (iCATE, sCATE, subCATE, PATE) across the full range of treatment-effect heterogeneity encountered in the curated datasets.

---

## Data and Preprocessing

Each dataset contains $n$ observations with 40 covariates: six nominal variables (x1–x6, taking values such as "level 1" through "level 7"), 18 binary indicators (x7–x24), and 16 continuous variables (x25–x40). All nominal variables are one-hot encoded with the first level dropped; continuous variables are standardized; binary variables are passed through unchanged. Treatment assignment is completely at random within each population, so standard ignorability and SUTVA assumptions hold throughout.

---

## Nuisance Models

All nuisance functions — the conditional outcome mean $\mu(\mathbf{x}, z)$ and the propensity score $\pi(z \mid \mathbf{x})$ — are estimated using stacked ensemble learners.

**Outcome model** $\mu(\mathbf{x}, z)$: a two-level stacking ensemble whose level-1 learners are LightGBM (300 trees, 63 leaves, learning rate 0.05, 80% row and column subsampling), a random forest (150 trees), and ElasticNet with cross-validated regularization. The meta-learner is RidgeCV.

**Propensity model** $\pi(z \mid \mathbf{x})$: a stacking classifier with LightGBM and a random forest at level-1 and multinomial logistic regression as the meta-learner. Propensity scores are clipped to $[0.025,\, 0.975]$ after fitting to bound the maximum inverse-probability weight in the DR pseudo-outcomes at $1/0.025 = 40$, preventing extreme observations from dominating the cross-fitting step.

All nuisance models are fit using **5-fold cross-fitting**, so the pseudo-outcomes used in the CATE stage are always out-of-sample predictions.

---

## CATE Estimators

Four CATE estimators are fit independently and combined:

**1. Nonlinear DRLearner.** A doubly-robust learner whose final stage regresses the DR pseudo-outcomes on covariates using the same LightGBM-primary stacked ensemble described above (with lighter regularization). Pointwise 95% confidence intervals are obtained via 50-replicate bootstrap inference on the final CATE stage.

**2. LinearDRLearner.** A doubly-robust learner with a linear final stage (no featurizer, to avoid rank deficiency at $p \approx 54$ encoded features). The same stacked nuisance models are used. Confidence intervals are derived analytically from the HC1 heteroskedasticity-consistent sandwich variance of the weighted-least-squares CATE regression — equivalent to the influence-function estimator and essentially free to compute after fitting.

**3. CausalForestDML.** One honest causal forest fit per treatment arm (b, c, d, e) versus control (a), following the DML orthogonalization path. The treatment model is a LightGBM classifier with `discrete_treatment=True`, which is correct for the binary arm-versus-control contrasts. Confidence intervals use GRF-style honest variance without bootstrap.

**4. ForestDRLearner.** A DR learner whose final stage is an honest causal forest rather than a parametric model. This estimator shares the DR orthogonalization path with estimators 1 and 2 but uses the GRF forest final stage of estimator 3, occupying a genuinely different corner of the bias–variance–CI-method space. Confidence intervals are GRF-style without bootstrap.

---

## Variance-Weighted Ensemble

The four estimators are combined into a single set of iCATE estimates via **observation-level inverse-variance weighting**:

$$
\hat{\tau}_{\text{ens}}(\mathbf{x}_i) = \frac{\sum_k w_k(\mathbf{x}_i)\, \hat{\tau}_k(\mathbf{x}_i)}{\sum_k w_k(\mathbf{x}_i)}, \qquad w_k(\mathbf{x}_i) = \frac{1}{\hat{\sigma}_k^2(\mathbf{x}_i)}
$$

where $\hat{\sigma}_k(\mathbf{x}_i) = (U_{95,k}(\mathbf{x}_i) - L_{95,k}(\mathbf{x}_i)) / (2 \times 1.96)$ is recovered from each estimator's pointwise 95% interval. The ensemble variance is $\hat{\sigma}^2_{\text{ens}}(\mathbf{x}_i) = 1 / \sum_k w_k(\mathbf{x}_i)$.

Per-observation weights are floored at $0.1 \times \text{std}(\hat{\tau}_k)$ to cap the maximum precision ratio across observations at $100:1$, preventing artificially narrow CIs from a single estimator from dominating the ensemble at individual covariate points.

When bootstrap inference is disabled (e.g. for computational efficiency), the nonlinear DRLearner contributes its point estimate with a fixed weight of 0.25 and CIs are derived entirely from the three analytic estimators (LinearDRLearner, CausalForestDML, ForestDRLearner).

---

## Aggregated Estimands

- **iCATE**: the ensemble point estimate and 95% CI at each observed covariate vector, for each treatment arm versus control.
- **sCATE**: the sample-average iCATE per arm, with 95% CI derived via the delta method: $\widehat{\text{SE}}[\bar{\tau}] = \sqrt{\sum_i \hat{\sigma}_i^2}\, /\, n$.
- **subCATE**: the same aggregation restricted to subgroups defined by $X_{12} \in \{0, 1\}$, normalised by the full sample size $n$ so that subCATE$(z,0)$ + subCATE$(z,1)$ = sCATE$(z)$.
- **PATE**: the doubly-robust AIPW average treatment effect estimated by EconML's `ate_interval()`, which uses the efficient influence function of the DR estimator for $\sqrt{n}$-consistent inference.

Best-treatment files (argmax over point estimates) are provided for all four estimands.

---

## Software and Reproducibility

```
Python       3.11
econml       >= 0.15
lightgbm     >= 4.0
scikit-learn >= 1.5
numpy        >= 1.26
pandas       >= 2.0
scipy        >= 1.11
tqdm         >= 4.0
```

**Reproduce this submission** (curated track, parallel across all available cores, no bootstrap for speed):

```bash
python main.py \
    --batch curated \
    --parallel \
    --no-bootstrap \
    --team-id XXXX \
    --subm-id 1 \
    --data-dir curated_data/ \
    --out-dir  submissions/
```

**With bootstrap** (higher-quality CIs, slower):

```bash
python main.py \
    --batch curated \
    --parallel \
    --n-boot 50 \
    --team-id XXXX \
    --subm-id 1 \
    --data-dir curated_data/ \
    --out-dir  submissions/
```

All random seeds are fixed (`RANDOM_STATE = 42`). The pipeline is fully deterministic given the same dataset and seed.
