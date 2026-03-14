# SOURCE: https://github.com/stanford-msande228/winter23/blob/main/CATE-estimation.ipynb

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.special
from sklearn.linear_model import LassoCV, LinearRegression, ElasticNetCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.base import clone
import joblib
from statsmodels.api import OLS
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

from myflaml import auto_reg, auto_clf, auto_weighted_reg

# Define helper functions
def rmse(cate, preds):
    return np.sqrt(np.mean((cate - preds)**2))

# Set high-level parameters
time_budget = 600 # time budget for auto-ml in seconds (advisable at least 120)
verbose = 0 # verbosity of auto-ml
n_splits = 5 # cross-fitting and cross-validation splits
# data = '401k' # which dataset, one of {'401k', 'criteo', 'welfare', 'poverty', 'star'}
plot = True # whether to plot results
# xfeat = 'inc' # feature to use as x axis in plotting, e.g. for criteo 'f1', for 401k 'inc', for welfare 'polviews'
# Formula for the BLP of CATE regression.
# blp_formula = 'np.log(inc)' # e.g. 'f1' for criteo, np.log(inc)' for 401k, 'C(polviews)' for the welfare case.
# hetero_feats = ['inc'] # list of subset of features to be used for CATE model or the string 'all' for everything
binary_y = False

X, D, y, groups = get_data()

# Split Train and Validation and Test
# The training data will be used to fit the various CATE models. 
# The validation data will be used for scoring and selection of the best CATE model or best ensemble of CATE models. 
# The test data will be used for testing and evaluation of the performance of the best chosen model.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

if groups is None:
    X, Xval, D, Dval, y, yval = train_test_split(X, D, y, train_size=.6, shuffle=True, random_state=123)
    Xval, Xtest, Dval, Dtest, yval, ytest = train_test_split(Xval, Dval, yval, train_size=.5, shuffle=True, random_state=123)
    groupsval, groupstest = None, None
else:
    train, val = next(GroupShuffleSplit(n_splits=2, train_size=.6, random_state=123).split(X, y, groups=groups))
    X, Xval, D, Dval, y, yval = X.iloc[train], X.iloc[val], D[train], D[val], y[train], y[val]
    groups, groupsval = groups[train], groups[val]

    val, test = next(GroupShuffleSplit(n_splits=2, train_size=.5, random_state=123).split(Xval, yval, groups=groupsval))
    Xval, Xtest, Dval, Dtest, yval, ytest = Xval.iloc[val], Xval.iloc[test], Dval[val], Dval[test], yval[val], yval[test]
    groupsval, groupstest = groupsval[val], groupsval[test]
    
# Nuisance Model Selection
# Using the training data we will select the best model for each of the nuisance models that arise in meta learner CATE approaches.
# We will select the best hyperparameters/model type for each predictive problem using cross-validation, where the splits are also stratified by the treatment (so that we have balanced split of the treatment groups across folds).
if groups is None:
    split_type = 'auto'
else:
    split_type = GroupKFold(n_splits=n_splits)

# These function calls perform auto-ml hyperparameter tuning and return a "model class generator"
# i.e. a function that whenever called returns an instance of an un-fitted model with the best hyper-parameters
if binary_y:
    model_reg = auto_clf(np.column_stack((D, X)), y, groups=groups, n_splits=n_splits, split_type=split_type, 
                         verbose=verbose, time_budget=time_budget)
    model_y = auto_clf(X, y, n_splits=n_splits, split_type=split_type, 
                       verbose=verbose, time_budget=time_budget)
    model_reg_zero = auto_clf(X[D==0], y[D==0], groups=groups, n_splits=n_splits, split_type=split_type, 
                              verbose=verbose, time_budget=time_budget)
    model_reg_one = auto_clf(X[D==1], y[D==1], groups=groups, n_splits=n_splits, split_type=split_type, 
                             verbose=verbose, time_budget=time_budget)
else:
    model_reg = auto_reg(np.column_stack((D, X)), y, groups=groups, n_splits=n_splits, split_type=split_type, 
                         verbose=verbose, time_budget=time_budget)
    model_y = auto_reg(X, y, groups=groups, n_splits=n_splits, split_type=split_type, 
                       verbose=verbose, time_budget=time_budget)
    model_reg_zero = auto_reg(X[D==0], y[D==0], groups=groups, n_splits=n_splits, split_type=split_type, 
                              verbose=verbose, time_budget=time_budget)
    model_reg_one = auto_reg(X[D==1], y[D==1], groups=groups, n_splits=n_splits, split_type=split_type, 
                             verbose=verbose, time_budget=time_budget)
model_t = auto_clf(X, D, groups=groups, n_splits=n_splits, split_type=split_type, 
                   verbose=verbose, time_budget=time_budget)

# If you want to save or load these models from a previous run un-comment the following lines:
# joblib.dump([model_reg(), model_y(), model_t(), model_reg_zero(), model_reg_one()], 'nuisance.jbl')
# mreg, my, mt, mreg_zero, mreg_one = joblib.load('nuisance.jbl')
# model_reg = lambda: clone(mreg)
# model_y = lambda: clone(my)
# model_t = lambda: clone(mt)
# model_reg_zero = lambda: clone(mreg_zero)
# model_reg_one = lambda: clone(mreg_one)

# We now also evaluate the performance of the selected models in terms of R^2
if groups is None:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=123)
else:
    cv = GroupKFold(n_splits=n_splits)
    
score_reg = np.mean(cross_val_score(model_reg(), X, y, groups=groups, cv=cv, scoring='r2'))
print(f'model_reg: {score_reg:.3f}')
score_reg = np.mean(cross_val_score(model_y(), X, y, groups=groups, cv=cv, scoring='r2'))
print(f'model_y: {score_reg:.3f}')
score_reg = np.mean(cross_val_score(model_t(), X, D, groups=groups, cv=cv, scoring='r2'))
print(f'model_t: {score_reg:.3f}')
if groups is None:
    score_reg = np.mean(cross_val_score(model_reg_zero(), X[D==0], y[D==0], groups=None, cv=cv, scoring='r2'))
    print(f'model_reg_zero: {score_reg:.3f}')
    score_reg = np.mean(cross_val_score(model_reg_one(), X[D==1], y[D==1], groups=None, cv=cv, scoring='r2'))
    print(f'model_reg_one: {score_reg:.3f}')
else:
    score_reg = np.mean(cross_val_score(model_reg_zero(), X[D==0], y[D==0], groups=groups[D==0], cv=cv, scoring='r2'))
    print(f'model_reg_zero: {score_reg:.3f}')
    score_reg = np.mean(cross_val_score(model_reg_one(), X[D==1], y[D==1], groups=groups[D==1], cv=cv, scoring='r2'))
    print(f'model_reg_one: {score_reg:.3f}')
       
# Nuisance Cross-Fitted Estimation and Prediction
# After selecting the hyper-parameters for each of the nuisance models we perform cross-fitting to get out-of-fold predictions from each of these nuisance models. 
# At the end of this process, we will have for each sample, out-of-fold nuisance values.
if groups is None:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
    splits = list(cv.split(X, D))
else:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=123)
    splits = list(cv.split(X, D, groups=groups))

n = X.shape[0]
reg_preds = np.zeros(n)
reg_zero_preds = np.zeros(n)
reg_one_preds = np.zeros(n)
reg_preds_t = np.zeros(n)
reg_zero_preds_t = np.zeros(n)
reg_one_preds_t = np.zeros(n)

DX = np.column_stack((D, X))
for train, test in splits:
    reg = model_reg().fit(DX[train], y[train])
    reg_preds[test] = reg.predict(DX[test])
    reg_one_preds[test] = reg.predict(np.column_stack([np.ones(len(test)), X.iloc[test]]))
    reg_zero_preds[test] = reg.predict(np.column_stack([np.zeros(len(test)), X.iloc[test]]))

    reg_zero = model_reg_zero().fit(X.iloc[train][D[train]==0], y[train][D[train]==0])
    reg_one = model_reg_one().fit(X.iloc[train][D[train]==1], y[train][D[train]==1])
    reg_zero_preds_t[test] = reg_zero.predict(X.iloc[test])
    reg_one_preds_t[test] = reg_one.predict(X.iloc[test])
    reg_preds_t[test] = reg_zero_preds_t[test] * (1 - D[test]) + reg_one_preds_t[test] * D[test]

res_preds = cross_val_predict(model_y(), X, y, cv=splits)
prop_preds = cross_val_predict(model_t(), X, D, cv=splits)

# ATE Estimation
# For an RCT the coefficient associated with the treatment in the limit converges to the true ATE and provides correct inference for the true ATE.
dfX = X.copy()
dfX = dfX - dfX.mean(axis=0)
dfX['D'] = D
dfX['const'] = 1
if groups is None:
    print(OLS(y, dfX).fit(cov_type='HC1').summary())
else:
    print(OLS(y, dfX).fit(cov_type='cluster', cov_kwds={'groups': groups}).summary())

# Using plain OLS with the interactive model (after de-meaning X). 
# For an RCT the coefficient associated with the treatment in the limit converges to the true ATE and provides correct inference for the true ATE. 
# Moreover, the interactions should be giving us dimensions of heterogeneity.
from formulaic import Formula
dfX = X.copy()
dfX = dfX - dfX.mean(axis=0)
dfX['D'] = D
dfX = Formula('D * (' + '+'.join(X.columns) + ')').get_model_matrix(dfX)
dfX['const'] = 1
if groups is None:
    print(OLS(y, dfX).fit(cov_type='HC1').summary())
else:
    print(OLS(y, dfX).fit(cov_type='cluster', cov_kwds={'groups': groups}).summary())

# Using the residual on residual regression, which should be giving us a correct estimate of the ATE in an RCT, and 
# in observational settings a correct estimate of the ATE under a partially linear model 
# (otherwise only a weighted average of the CATEs; weighted by the conditional variance of the treatment)
yres = y - res_preds
Dres = D - prop_preds

if groups is None:
    print(OLS(yres, Dres).fit(cov_type='HC1').summary())
else:
    print(OLS(yres, Dres).fit(cov_type='cluster', cov_kwds={'groups': groups}).summary())
    
# Using the doubly robust method. This should be more efficient in the worst-case and should be returning a consistent estimate of the ATE 
# even beyond RCTs and will also correctly account for any imbalances or violations of the randomization assumption in an RCT.
dr_preds = reg_one_preds_t - reg_zero_preds_t
dr_preds += (y - reg_preds_t) * (D - prop_preds) / np.clip(prop_preds * (1 - prop_preds), .09, np.inf)

if groups is None:
    print(OLS(dr_preds, np.ones((len(dr_preds), 1))).fit(cov_type='HC1').summary())
else:
    print(OLS(dr_preds, np.ones((len(dr_preds), 1))).fit(cov_type='cluster', cov_kwds={'groups': groups}).summary())
    
# Best Linear CATE Predictor and Simultaneous (Joint) Confidence Intervals
# We can also use the doubly robust variables as pseudo-outcomes in an OLS regression, so as to estimate the best linear approximation of the true CATE. 
# In an RCT, these should be similar to the coefficients recovered in a plain interactive OLS regression.
dr_preds = reg_one_preds_t - reg_zero_preds_t
dr_preds += (y - reg_preds_t) * (D - prop_preds) / np.clip(prop_preds * (1 - prop_preds), .09, np.inf)

dfX = X.copy()
dfX = dfX - dfX.mean(axis=0)
dfX['const'] = 1
if groups is None:
    lr = OLS(dr_preds, dfX).fit(cov_type='HC1')
    cov = lr.get_robustcov_results(cov_type='HC1')
else:
    lr = OLS(dr_preds, dfX).fit(cov_type='cluster', cov_kwds={'groups': groups})
    cov = lr.get_robustcov_results(cov_type='cluster', groups=groups)
print(lr.summary())

# We can also perform joint inference on all these parameters controlling the joint probability of failure of the confidence intervals by 95%.
V = cov.cov_params()
S = np.diag(np.diagonal(V)**(-1/2))
epsilon = np.random.multivariate_normal(np.zeros(V.shape[0]), S @ V @ S, size=(1000))
critical = np.percentile(np.max(np.abs(epsilon), axis=1), 95)
stderr = np.diagonal(V)**(1/2)
lb = cov.params - critical * stderr
ub = cov.params + critical * stderr
jointsummary = pd.DataFrame({'coef': cov.params,
                             'std err': stderr,
                             'lb': lb,
                             'ub': ub,
                             'statsig': ['' if ((l <= 0) & (0 <= u)) else '**' for (l, u) in zip(lb, ub)]},
                            index=dfX.columns)
print(jointsummary)


# We can also produce confidence intervals for the predictions of the CATE at particular points
# grid = np.unique(np.percentile(dfX[xfeat], np.arange(0, 110, 20)))

# Zpd = pd.DataFrame(np.tile(np.median(dfX, axis=0, keepdims=True), (len(grid), 1)),
#                     columns=dfX.columns)
# Zpd[xfeat] = grid

# pred_df = lr.get_prediction(Zpd).summary_frame()
# preds, lb, ub = pred_df['mean'].values, pred_df['mean_ci_lower'].values, pred_df['mean_ci_upper'].values
# preds = preds.flatten()
# lb = lb.flatten()
# ub = ub.flatten()
# plt.errorbar(Zpd[xfeat], preds, yerr=(preds-lb, ub-preds))
# plt.xlabel(xfeat)
# plt.ylabel('Predicted CATE (at median value of other features)')
# plt.show()


# And even simultaneous inference on all these predictions that controls the joint failure probability of these confidence intervals to be at most 95%
# predsV = Zpd.values @ V @ Zpd.values.T
# predsS = np.diag(np.diagonal(predsV)**(-1/2))
# epsilon = np.random.multivariate_normal(np.zeros(predsV.shape[0]), predsS @ predsV @ predsS, size=(1000))
# critical = np.percentile(np.max(np.abs(epsilon), axis=1), 95)
# stderr = np.diagonal(predsV)**(1/2)
# lb = preds - critical * stderr
# ub = preds + critical * stderr

# plt.errorbar(Zpd[xfeat], preds, yerr=(preds-lb, ub-preds))
# plt.xlabel(xfeat)
# plt.ylabel('Predicted CATE (at median value of other features)')
# plt.show()


# CATE Model Estimation with Meta-Learners
# We specify which indices of the X variables we want to use for heterogeneity. Let's denote these subset of variables with Z
if hetero_feats == 'all':
    hetero_feats = X.columns
Z, Zval, Ztest = X[hetero_feats], Xval[hetero_feats], Xtest[hetero_feats]

# We specify a generic automl approach for training the final CATE model
model_final_fn = lambda Z, y: auto_reg(Z, y, groups=groups,
                                       n_splits=n_splits, split_type=split_type, 
                                       verbose=verbose, time_budget=time_budget)


# Single Learner (S-Learner)
slearner_best = model_final_fn(Z, reg_one_preds - reg_zero_preds)
slearner = slearner_best().fit(Z, reg_one_preds - reg_zero_preds)
slearner_cates = slearner.predict(Z)

# Two Learner (T-Learner)
tlearner_best = model_final_fn(Z, reg_one_preds_t - reg_zero_preds_t)
tlearner = tlearner_best().fit(Z, reg_one_preds_t - reg_zero_preds_t)
tlearner_cates = tlearner.predict(Z)

# Cross Learner (X-Learner)
tau1_preds = y[D==1] - reg_zero_preds_t[D==1]
tau0_preds = reg_one_preds_t[D==0] - y[D==0]
tau1 = model_final_fn(X[D==1], tau1_preds)().fit(X[D==1], tau1_preds)
tau0 = model_final_fn(X[D==0], tau0_preds)().fit(X[D==0], tau0_preds)
xtarget = prop_preds * tau0.predict(X) + (1 - prop_preds) * tau1.predict(X)
xlearner = model_final_fn(Z, xtarget)().fit(Z, xtarget)
xlearner_cates = xlearner.predict(Z)

# Doubly Robust Learner (DR-Learner)
# The DR-Learner (in particular, the variant based on the T-Learner) adds a de-biasing correction to the T-Learner using the propensity.
dr_preds = reg_one_preds_t - reg_zero_preds_t
dr_preds += (y - reg_preds_t) * (D - prop_preds) / np.clip(prop_preds * (1 - prop_preds), .09, np.inf)
drlearner_best = model_final_fn(Z, dr_preds)
drlearner = drlearner_best().fit(Z, dr_preds)
drlearner_cates = drlearner.predict(Z)

# Residual Learner (R-Learner)
yres = y - res_preds
Dres = D - prop_preds
Dres = np.clip(Dres, .001, np.inf) * (Dres >= 0) + np.clip(Dres, -np.inf, -.001) * (Dres < 0)

rlearner_fn = auto_weighted_reg(Z, yres / Dres, sample_weight=Dres**2, groups=groups,
                                n_splits=n_splits, verbose=verbose, time_budget=time_budget)
rlearner = rlearner_fn().fit(Z, yres / Dres, sample_weight=Dres**2)
rlearner_cates = rlearner.predict(Z)

# Constant Effect DR-Learner
# We also add a heavily regularized CATE model that predicts the ATE using the doubly robust pseudo outcomes.
drlearner_const = make_pipeline(PolynomialFeatures(degree=0, include_bias=True), 
                                LinearRegression(fit_intercept=False)).fit(Z, dr_preds)
drlearner_const_cates = drlearner_const.predict(Z)

# Causal Score Estimation and Definition
# We want to be able to select among all these different meta learners. 
# For this reason we will use scoring functions that can evaluate the performance of an arbitrary CATE function and is not tailored to any particular methodology. 
# This way we can evaluate all methods using the same score on the validation set and select the best among the methods, or ensemble the methods using this scoring metric. 
# We will use two such meta scores, the R-score and the DR-score.

# The Residual Score (R-Score)
# The R-score uses the residual loss that we used in the final stage of the R-Learner, to evaluate any CATE model.
 
# if groups is None:
#     cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
#     splits_val = list(cv.split(Xval, Dval))
# else:
#     cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=123)
#     splits_val = list(cv.split(Xval, Dval, groups=groupsval))
# yres_val = yval - cross_val_predict(model_y(), Xval, yval, cv=splits_val)
# Dres_val = Dval - cross_val_predict(model_t(), Xval, Dval, cv=splits_val)

yres_val = yval - model_y().fit(X, y).predict(Xval)
Dres_val = Dval - model_t().fit(X, D).predict(Xval)

overall_ate_val_r = np.mean(yres_val * Dres_val) / np.mean(Dres_val**2)

def rscore(cate_preds): 
    rscore_t = np.mean((yres_val - cate_preds * Dres_val)**2)
    rscore_b = np.mean((yres_val - overall_ate_val_r * Dres_val)**2)
    return 1 - rscore_t / rscore_b

# The Doubly Robust Score (DR-Score)
# The doubly robust score calculates in a cross-fitting manner, using only the validation set, the doubly robust proxy variables

# def calculate_dr_outcomes(Xtrain, Dtrain, ytrain, groupstrain, X, D, y, groups):
#     if groups is None:
#         cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
#         splits = list(cv.split(X, D))
#     else:
#         cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=123)
#         splits = list(cv.split(X, D, groups=groups))

#     n = X.shape[0]
#     reg_preds_t = np.zeros(n)
#     reg_zero_preds_t = np.zeros(n)
#     reg_one_preds_t = np.zeros(n)

#     for train, test in splits:
#         reg_zero = model_reg_zero().fit(X.iloc[train][D[train]==0], y[train][D[train]==0])
#         reg_one = model_reg_one().fit(X.iloc[train][D[train]==1], y[train][D[train]==1])
#         reg_zero_preds_t[test] = reg_zero.predict(X.iloc[test])
#         reg_one_preds_t[test] = reg_one.predict(X.iloc[test])
#         reg_preds_t[test] = reg_zero_preds_t[test] * (1 - D[test]) + reg_one_preds_t[test] * D[test]

#     prop_preds = cross_val_predict(model_t(), X, D, cv=splits)

#     dr = reg_one_preds_t - reg_zero_preds_t
#     reisz = (D - prop_preds) / np.clip(prop_preds * (1 - prop_preds), .09, np.inf)
#     dr += (y - reg_preds_t) * reisz

#     return dr

def calculate_dr_outcomes(Xtrain, Dtrain, ytrain, groupstrain, Xval, Dval, yval, groupsval):

    reg_zero = model_reg_zero().fit(Xtrain[Dtrain==0], ytrain[Dtrain==0])
    reg_one = model_reg_one().fit(Xtrain[Dtrain==1], ytrain[Dtrain==1])
    reg_zero_preds_t = reg_zero.predict(Xval)
    reg_one_preds_t = reg_one.predict(Xval)
    reg_preds_t = reg_zero_preds_t * (1 - Dval) + reg_one_preds_t * Dval
    prop_preds = model_t().fit(Xtrain, Dtrain).predict(Xval)

    dr = reg_one_preds_t - reg_zero_preds_t
    reisz = (Dval - prop_preds) / np.clip(prop_preds * (1 - prop_preds), .09, np.inf)
    dr += (yval - reg_preds_t) * reisz

    return dr

dr_val = calculate_dr_outcomes(X, D, y, groups, Xval, Dval, yval, groupsval)

overall_ate_val_dr = np.mean(dr_val)

def drscore(cate_preds):
    drscore_t = np.mean((dr_val - cate_preds)**2)
    drscore_b = np.mean((dr_val - overall_ate_val_dr)**2)
    return 1 - drscore_t / drscore_b


# Score CATE Models
# We first chose one of the scorers and we score all the CATE models.
scorer = drscore
score_name = 'DRscore'
names = ['slearner', 'tlearner', 'xlearner', 'drlearner', 'rlearner', 'drlearner_const']
models = [slearner, tlearner, xlearner, drlearner, rlearner, drlearner_const]

scores = [scorer(model.predict(Zval)) for model in models]
print([f'{name}: {score:.4f}' for name, score in zip(names, scores)])


# Plotting CATE
# if plot:
#     grid = np.unique(np.percentile(X[xfeat], np.arange(0, 105, 5)))
#     Xpd = pd.DataFrame(np.tile(np.median(X, axis=0, keepdims=True), (len(grid), 1)),
#                         columns=X.columns)
#     Xpd[xfeat] = grid
#     plt.figure(figsize=(20, 7))
#     for it, (name, model, score) in enumerate(zip(names, models, scores)):
#         plt.subplot(1, len(models), it + 1)
#         preds = model.predict(Xpd[hetero_feats])
#         plt.scatter(Xpd[xfeat], preds, label=name, alpha=.4)
#         if semi_synth:
#             plt.scatter(Xpd[xfeat], true_cate(Xpd), label='True')
#             plt.title(f'{score_name}={score:.4f}, True RMSE={rmse(true_cate(X), model.predict(Z)):.5f}')
#         else:
#             plt.title(f'{score_name}={score:.4f}, ATE={np.mean(model.predict(Z)):.3f}')
#         plt.legend()
#         plt.xlabel(xfeat)
#         plt.ylabel('Predicted CATE (at median value of other features)')
#     plt.tight_layout()
#     plt.show()


# Causal Model Selection and Ensembling
# We can also use these scores to create an ensemble CATE model of the different methods based on the score performance. 
# We want to create a new CATE model that is a weighted linear combination of all the cate models
from sklearn.base import BaseEstimator

class Ensemble(BaseEstimator):
    
    def __init__(self, names, models, weights, intercept=0):
        self.names = names
        self.models = models
        self.weights = weights
        self.intercept = intercept
    
    def predict(self, X):
        wcate = np.sum(self.weights.reshape((-1, 1)) * np.array([m.predict(X) for m in self.models]), axis=0)
        return self.intercept + wcate

eta_grid = np.logspace(-5, 5, 10)
ens = {}
for eta in eta_grid:
    weights = scipy.special.softmax(eta * np.array(scores))
    ensemble = Ensemble(names, models, weights)
    ens[eta] = (ensemble, scorer(ensemble.predict(Zval)))

score_best = -np.inf
for eta in eta_grid:
    if ens[eta][1] >= score_best:
        score_best = ens[eta][1]
        eta_best = eta

softmax_ensemble = ens[eta_best][0]
softmax_ensemble

# if plot:
#     grid = np.unique(np.percentile(X[xfeat], np.arange(0, 105, 5)))
#     Xpd = pd.DataFrame(np.tile(np.median(X, axis=0, keepdims=True), (len(grid), 1)),
#                         columns=X.columns)
#     Xpd[xfeat] = grid
#     plt.figure(figsize=(10, 5))
#     plt.scatter(Xpd[xfeat], softmax_ensemble.predict(Xpd[hetero_feats]),
#                 label=f'SoftMaxCATE(eta={eta_best:.3f})', alpha=.4)
#     if semi_synth:
#         plt.scatter(Xpd[xfeat], true_cate(Xpd), label='True')
#         plt.title(f'{score_name}={scorer(softmax_ensemble.predict(Zval)):.5f}, '
#                   f'True RMSE={rmse(true_cate(X), ens[eta_best][0].predict(Z)):.5f}')
#     else:
#         plt.title(f'{score_name}={scorer(softmax_ensemble.predict(Zval)):.5f}, '
#                   f'ATE={np.mean(ens[eta_best][0].predict(Z)):.5f}')
#     plt.xlabel(xfeat)
#     plt.ylabel('CATE predictions (at median value of other features)')
#     plt.legend()
#     plt.show()

# Alternatively we could have also performed Stacking by fitting a (potentially l1-penalized) linear model to minimize the corresponding loss. 
# For the case of the DR score this boils down to a penalized linear regression
from sklearn.linear_model import RidgeCV

F = np.array([m.predict(Zval) for m in models]).T
meansF = np.mean(F, axis=0)
F = F - meansF
# One of LassoCV(fit_intercept=False) or ElasticNetCV(fit_intercept=False) or
# or LinearRegression(fit_intercept=False) or LassoCV(positive=True, fit_intercept=False)
stacker = LassoCV(fit_intercept=False)
if score_name == 'DRscore':
    stacker.fit(F, dr_val - np.mean(dr_preds))
    intercept = np.mean(dr_preds) - meansF @ stacker.coef_
    stack_ensemble = Ensemble(names, models, stacker.coef_, intercept)
elif score_name == 'Rscore':
    # we will avoid penalizing the intercept of the CATE by multiplying the constant 1
    # by a large number; equivalently this divides the penalty for that parameter by that number
    stacker.fit(F * Dres_val.reshape(-1, 1), yres_val - np.mean(dr_preds) * Dres_val)
    intercept = np.mean(dr_preds) - meansF @ stacker.coef_
    stack_ensemble = Ensemble(names, models, stacker.coef_, intercept)

print(stack_ensemble)

# if plot:
#     grid = np.unique(np.percentile(X[xfeat], np.arange(0, 105, 5)))
#     Xpd = pd.DataFrame(np.tile(np.median(X, axis=0, keepdims=True), (len(grid), 1)),
#                         columns=X.columns)
#     Xpd[xfeat] = grid
#     plt.figure(figsize=(10, 5))
#     plt.scatter(Xpd[xfeat], stack_ensemble.predict(Xpd[hetero_feats]),
#                 label=f'StackedCATE', alpha=.4)
#     if semi_synth:
#         plt.scatter(Xpd[xfeat], true_cate(Xpd), label='True')
#         plt.title(f'{score_name}={scorer(stack_ensemble.predict(Zval)):.5f}, '
#                   f'True RMSE={rmse(true_cate(X), stack_ensemble.predict(Z)):.5f}')
#     else:
#         plt.title(f'{score_name}={scorer(stack_ensemble.predict(Zval)):.5f}, '
#                   f'ATE={np.mean(stack_ensemble.predict(Z)):.5f}')
#     plt.xlabel(xfeat)
#     plt.ylabel('CATE predictions (at median value of other features)')
#     plt.legend()
#     plt.show()


# We use one of these ensembles as our final best model

# overall_best = stack_ensemble
overall_best = softmax_ensemble


# Validation Tests on Test Data
# Now that we have a selected a winning CATE model (or ensemble), we can run a set of hypothesis tests and other diagnostic metrics on the test set, 
# to see if the model really picked up some dimensions of effect heterogeneity and satisfies some self-conistency checks.

# Hypothesis Test Based on Doubly Robust Best-Linear Predictor of CATE using model of CATE
dr_test = calculate_dr_outcomes(X, D, y, groups, Xtest, Dtest, ytest, groupstest)
cate_test = overall_best.predict(Ztest)
print(OLS(dr_test, np.stack((np.ones(len(dr_test)), cate_test), axis=-1)).fit().summary())


# Validation Based on Calibration
# We can measure whether each group defined by the quartile levels of CATE predictions is consistent with the out-of-sample Group ATE (GATE) 
# for the corresponding group based on the doubly robust GATE estimate. (standard errors here ignore cluster/group correlations)
cate_val = overall_best.predict(Zval)
qs = np.percentile(cate_val, np.arange(0, 101, 25))

gate, gate_std, group_prob = np.zeros(len(qs) - 1), np.zeros(len(qs) - 1), np.zeros(len(qs) - 1)
predicted_gate = np.zeros(len(qs) - 1)
for it in range(len(qs) - 1):
    # samples in the [q[it], q[it+1]) quantile group of predicted CATEs
    inds = (qs[it] <= cate_test) & (cate_test <= qs[it + 1]) 
    gate[it] = np.mean(dr_test[inds]) # DR estimate of group average treatment effect (GATE)
    gate_std[it] = np.std(dr_test[inds])/np.sqrt(np.sum(inds)) # standard error of GATE
    group_prob[it] = np.mean(inds) # probability mass of group
    predicted_gate[it] = np.mean(cate_test[inds]) # GATE as calculated from CATE model

# weighted average calibration error of cate model
cal = np.sum(group_prob * np.abs(gate - predicted_gate))
# weighted average calibration error of a constant cate model
calbase = np.sum(group_prob * np.abs(gate - np.mean(dr_test)))
# calibration score
calscore = 1 - cal/calbase
# plt.title(f'CalScore={calscore:.4f}')
# plt.errorbar(predicted_gate, gate, yerr=1.96*gate_std, fmt='o')
# plt.xlabel('Predicted GATE based on CATE model')
# plt.ylabel('Doubly Robust GATE estimate')
# plt.show()


# We can also try to interpret what are the differences of characteristics between the top and bottom CATE groups; 
# if we find that they have statistically significantly different GATEs. 
# We can do that by either reporting the mean values of the covariates in the two groups or 
# building some interpretable classification model that distinguishes between the two groups.
group1 = (qs[0] <= cate_test) & (cate_test < qs[1]) 
group2 = (qs[-2] <= cate_test) & (cate_test < qs[-1])
Ztest1 = Ztest[group1]
Ztest2 = Ztest[group2]

df = pd.DataFrame({'group1 means': np.mean(Ztest1), 
                   'group1 s.e.': np.std(Ztest1) / np.sqrt(Ztest1.shape[0]),
                   'group2 means': np.mean(Ztest2),
                   'group2 s.e.': np.std(Ztest2) / np.sqrt(Ztest2.shape[0]),
                   'group1 means - group2 means': np.mean(Ztest1) - np.mean(Ztest2),
                   'diff s.e.': np.std(Ztest1) / np.sqrt(Ztest1.shape[0]) + np.std(Ztest2) / np.sqrt(Ztest2.shape[0])})
print(df)

# tree = DecisionTreeClassifier(max_depth=2)
# tree.fit(pd.concat((Ztest1, Ztest2)),
#          np.concatenate((np.zeros(len(Ztest1)), np.ones(len(Ztest2)))))
# plot_tree(tree, filled=True, feature_names=Ztest1.columns, class_names=['group1', 'group2'])
# plt.show()


# Validation Based on Uplift Curves
# These curves are related to "prioritization" or "stratification" implications of the CATE model. 
# What if we target to treat a q-percentage of the population. Then if we trust and follow the CATE model, 
# then we should be offering the treatment to the parts of the population that the CATE model predicts 
# have a CATE larger than the 1-qth percentile of the CATE distribution as produced by the CATE.
# Based on out-of-sample CATE thresholds
ugrid = np.linspace(5, 95, 50)
qs = np.percentile(overall_best.predict(Zval), ugrid)

toc, toc_std, group_prob = np.zeros(len(qs)), np.zeros(len(qs)), np.zeros(len(qs))
true_toc = np.zeros(len(qs))
toc_psi = np.zeros((len(qs), dr_test.shape[0])) # influence function representation of the TOC at each quantile
n = len(dr_test)
ate = np.mean(dr_test)
for it in range(len(qs)):
    inds = (qs[it] <= cate_test) # group with larger CATE prediction than the q-th quantile
    group_prob = np.sum(inds) / n # fraction of population in this group
    toc[it]= np.mean(dr_test[inds]) - ate # tau(q) := E[Y(1) - Y(0) | tau(X) >= q[it]] - E[Y(1) - Y(0)]
    # influence function for the tau(q); it is a standard influence function of a "covariance"
    toc_psi[it, :] = (dr_test - ate) * (inds / group_prob - 1) - toc[it]
    toc_std[it] = np.sqrt(np.mean(toc_psi[it]**2) / n) # standard error of tau(q)
    if semi_synth:
        true_toc[it] = np.mean((true_cate(Xtest) - np.mean(true_cate(Xtest))) * (inds * n / np.sum(inds) - 1))

# plt.errorbar(100 - ugrid, toc, yerr=1.96*toc_std, fmt='o', label='Est. TOC')
# plt.plot(100 - ugrid, np.zeros(len(ugrid)))
# if semi_synth:
#     plt.plot(100 - ugrid, true_toc, 'o', label='True TOC')
# plt.xlabel("Percentage treated")
# plt.ylabel("Gain in Average Effect of Treated by CATE over Random")
# plt.legend()
# plt.show()


# Uniform Confidence Band with Multiplier Bootstrap
# In fact the "1.96" is wrong if we want the confidence intervals to hold simultaneoulsy for the whole curve. 
# To have such "simultaneous coverage" guarantees we need to calculate a larger "critical value" than 1.96. 
# We can calculate the appropriate such constant using the multiplier bootstrap, which tries to estimate 
# the maximum deviation around the mean as a multiple of the standard deviation for each point.
# For computational reasons if dataset is too large, we should not be constructing
# an n_samples x n_bootstrap_samples matrix of multipliers due to memory issues;
# even though constructing such a matrix avoids a for loop in python, which is always advised.
if dr_test.shape[0] > 1e6:
    mboot = np.zeros((len(qs), 1000))
    for it in range(1000):
        w = np.random.normal(0, 1, size=(dr_test.shape[0],))
        mboot[:, it] = (toc_psi / toc_std.reshape(-1, 1)) @ w / n
else:
    w = np.random.normal(0, 1, size=(dr_test.shape[0], 1000))
    mboot = (toc_psi / toc_std.reshape(-1, 1)) @ w / n

max_mboot = np.max(np.abs(mboot), axis=0)
uniform_critical_value = np.percentile(max_mboot, 95)
print(uniform_critical_value)

# plt.errorbar(100 - ugrid, toc, yerr=uniform_critical_value*toc_std, fmt='o', label='Est. TOC')
# plt.plot(100 - ugrid, np.zeros(len(ugrid)))
# if semi_synth:
#     plt.plot(100 - ugrid, true_toc, 'o', label='True TOC')
# plt.xlabel("Percentage treated")
# plt.ylabel("Gain in Average Effect of Treated by CATE over Random")
# plt.legend()
# plt.show()


# Note that if there is any point that is above the zero line, with confidence, in this curve, then the CATE model has identified heterogeneity 
# in the effect in a statistically significant manner. To do this we need a one-sided confidence interval, as we only care that the quantities 
# are larger than some value with high confidence. We can then calculate the critical value for a uniform one-sided confidence interval 
# across all the points, using the multiplier bootstrap.
min_mboot = np.min(mboot, axis=0)
uniform_one_side_critical_value = np.abs(np.percentile(min_mboot, 5))
print(uniform_one_side_critical_value)

# plt.errorbar(100 - ugrid, toc,
#              yerr=[uniform_one_side_critical_value*toc_std, np.zeros(len(toc))], fmt='o', label='Est. TOC')
# plt.plot(100 - ugrid, np.zeros(len(ugrid)))
# if semi_synth:
#     plt.plot(100 - ugrid, true_toc, 'o', label='True TOC')
# plt.xlabel("Percentage treated")
# plt.ylabel("Gain in Average Effect of Treated by CATE over Random")
# plt.legend()
# plt.show()

print(f'Heterogeneity Statistic: {np.max(toc - uniform_one_side_critical_value*toc_std)}')

# We can also calcualte the area under the curve and the confidence interval for that area. 
# If the confidence interval does not contain zero, then we have again detected heterogeneity.
autoc_psi = np.sum(toc_psi[:-1] * np.diff(ugrid).reshape(-1, 1) / 100, 0)
autoc = np.sum(toc[:-1] * np.diff(ugrid) / 100)
autoc_stderr = np.sqrt(np.mean(autoc_psi**2) / n)
print(f'AUTOC: {autoc:.4f}, s.e.: {autoc_stderr:.4f}, '
      f'One-Sided 95% CI=[{autoc - scipy.stats.norm.ppf(.95) * autoc_stderr:.4f}, Infty]')

# Qini Curve
# Based on out-of-sample CATE thresholds
ugrid = np.linspace(5, 95, 50)
qs = np.percentile(overall_best.predict(Zval), ugrid)

toc, toc_std, group_prob = np.zeros(len(qs)), np.zeros(len(qs)), np.zeros(len(qs))
true_toc = np.zeros(len(qs))
toc_psi = np.zeros((len(qs), dr_test.shape[0]))
n = len(dr_test)
ate = np.mean(dr_test)
for it in range(len(qs)):
    inds = (qs[it] <= cate_test) # group with larger CATE prediction than the q-th quantile
    group_prob = np.sum(inds) / n # fraction of population in this group
    toc[it] = group_prob * (np.mean(dr_test[inds]) - ate) # tau(q) = q * E[Y(1) - Y(0) | tau(X) >= q[it]] - E[Y(1) - Y(0)]
    toc_psi[it, :] = (dr_test - ate) * (inds - group_prob) - toc[it] # influence function for the tau(q)
    toc_std[it] = np.sqrt(np.mean(toc_psi[it]**2) / n) # standard error of tau(q)
    if semi_synth:
        true_toc[it] = np.mean((true_cate(Xtest) - np.mean(true_cate(Xtest))) * (inds - group_prob))

# plt.errorbar(100 - ugrid, toc, yerr=1.96*toc_std, fmt='o', label='Est. QINI')
# plt.plot(100 - ugrid, np.zeros(len(ugrid)))
# if semi_synth:
#     plt.plot(100 - ugrid, true_toc, 'o', label='True QINI')
# plt.xlabel("Percentage treated")
# plt.ylabel("Gain in Policy Value over Random Treatment")
# plt.legend()
# plt.show()


# And for uniform coverage we can again use the multiplier bootstrap
if dr_test.shape[0] > 1e6:
    mboot = np.zeros((len(qs), 1000))
    for it in range(1000):
        w = np.random.normal(0, 1, size=(dr_test.shape[0],))
        mboot[:, it] = (toc_psi / toc_std.reshape(-1, 1)) @ w / n
else:
    w = np.random.normal(0, 1, size=(dr_test.shape[0], 1000))
    mboot = (toc_psi / toc_std.reshape(-1, 1)) @ w / n

max_mboot = np.max(np.abs(mboot), axis=0)
uniform_critical_value = np.percentile(max_mboot, 95)
print(uniform_critical_value)

# plt.errorbar(100 - ugrid, toc, yerr=uniform_critical_value*toc_std, fmt='o', label='Est. QINI')
# plt.plot(100 - ugrid, np.zeros(len(ugrid)))
# if semi_synth:
#     plt.plot(100 - ugrid, true_toc, 'o', label='True QINI')
# plt.xlabel("Percentage treated")
# plt.ylabel("Gain in Policy Valuee over Random Treatment")
# plt.legend()
# plt.show()


# Or the one-sided multiplier bootstrap
min_mboot = np.min(mboot, axis=0)
uniform_one_side_critical_value = np.abs(np.percentile(min_mboot, 5))
print(uniform_one_side_critical_value)

# plt.errorbar(100 - ugrid, toc, yerr=[uniform_one_side_critical_value*toc_std, np.zeros(len(toc))],
#              fmt='o', label='Est. QINI')
# plt.plot(100 - ugrid, np.zeros(len(ugrid)))
# if semi_synth:
#     plt.plot(100 - ugrid, true_toc, 'o', label='True QINI')
# plt.xlabel("Percentage treated")
# plt.ylabel("Gain in Average Effect of Treated by CATE over Random")
# plt.legend()
# plt.show()

print(f'Heterogeneity Statistic: {np.max(toc - uniform_one_side_critical_value*toc_std)}')

qini_psi = np.sum(toc_psi[:-1] * np.diff(ugrid).reshape(-1, 1) / 100, 0)
qini = np.sum(toc[:-1] * np.diff(ugrid) / 100)
qini_stderr = np.sqrt(np.mean(qini_psi**2) / n)
print(f'QINI: {qini:.4f}, s.e.: {qini_stderr:.4f}, '
      f'One-Sided 95% CI=[{qini - scipy.stats.norm.ppf(.95) * qini_stderr:.4f}, Infty]')

# Confidence Intervals on CATE Predictions
# We now move on to the subject of constructing confidence intervals for the predictions of CATE models. 
# Confidence intervals for CATE predictions is an inherently harder task. In its generality it is at least as hard as 
# constructing confidence intervals for the predictions of a non-parametric regression function; which is a statistically daunting task.

# Confidence Intervals on BLPs of CATE with the DRLearner
from statsmodels.formula.api import ols
df = X.copy()
df['dr'] = dr_preds
if groups is None:
    lr = ols('dr ~ ' + blp_formula, df).fit(cov_type='HC1')
else:
    lr = ols('dr ~ ' + blp_formula, df).fit(cov_type='cluster', cov_kwds={'groups': groups})
print(lr.summary())

# grid = np.unique(np.percentile(X[xfeat], np.arange(0, 110, 20)))
# Xpd = pd.DataFrame(np.tile(np.median(X, axis=0, keepdims=True), (len(grid), 1)),
#                     columns=X.columns)
# Xpd[xfeat] = grid
# pred_df = lr.get_prediction(Xpd).summary_frame(alpha=.1)

# plt.plot(Xpd[xfeat], pred_df['mean'])
# plt.fill_between(Xpd[xfeat], pred_df['mean_ci_lower'], pred_df['mean_ci_upper'], alpha=.4)
# plt.xlabel(xfeat + ' (other features fixed at median value)')
# plt.ylabel('Predicted CATE BLP: cate ~' + blp_formula)
# plt.show()


# Non-Parametric Confidence Intervals with Causal Forests (standard errors here ignore cluster/group correlations)
from econml.grf import CausalForest

yres = y - res_preds
Dres = D - prop_preds
cf = CausalForest(4000, criterion='het', max_depth=5, max_samples=.4, min_samples_leaf=50, min_weight_fraction_leaf=.0)
cf.fit(Z, Dres, yres)

top_feat = np.argsort(cf.feature_importances_)[-1]
print(Z.columns[top_feat])

# grid = np.unique(np.percentile(Z.iloc[:, top_feat], np.arange(0, 105, 5)))
# Zpd = pd.DataFrame(np.tile(np.median(Z, axis=0, keepdims=True), (len(grid), 1)),
#                     columns=Z.columns)
# Zpd.iloc[:, top_feat] = grid

# preds, lb, ub = cf.predict(Zpd, interval=True, alpha=.1)
# preds = preds.flatten()
# lb = lb.flatten()
# ub = ub.flatten()
# plt.errorbar(Zpd.iloc[:, top_feat], preds, yerr=(preds-lb, ub-preds))
# plt.xlabel(Zpd.columns[top_feat])
# plt.ylabel('Predicted CATE (at median value of other features)')
# plt.show()


# Non-Parametric Confidence Intervals with Doubly Robust Forests (standard errors here ignore cluster/group correlations)
from econml.grf import RegressionForest

drrf = RegressionForest(4000, max_depth=5, max_samples=.4, min_samples_leaf=50,
                       min_weight_fraction_leaf=.0)
drrf.fit(Z, dr_preds)

top_feat = np.argsort(drrf.feature_importances_)[-1]
print(Z.columns[top_feat])

# grid = np.unique(np.percentile(Z.iloc[:, top_feat], np.arange(0, 105, 5)))
# Zpd = pd.DataFrame(np.tile(np.median(Z, axis=0, keepdims=True), (len(grid), 1)),
#                     columns=Z.columns)
# Zpd.iloc[:, top_feat] = grid

# preds, lb, ub = drrf.predict(Zpd, interval=True, alpha=.1)
# preds = preds.flatten()
# lb = lb.flatten()
# ub = ub.flatten()
# plt.errorbar(Zpd.iloc[:, top_feat], preds, yerr=(preds-lb, ub-preds))
# plt.xlabel(Zpd.columns[top_feat])
# plt.ylabel('Predicted CATE (at median value of other features)')
# plt.show()
