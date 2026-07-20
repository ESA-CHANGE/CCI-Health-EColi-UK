# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: phyto-cci-pig
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ESA CHANGE - Experiments with data and Random Forest
#
# ### Installation
# - Run using this environment (for now): /data/abitibi1/scratch/scratch_disk/pim/miniforge3/envs/phyto-cci-pig

# %% [markdown]
# ## Initialisation

# %%
# Setup and constants

# Import external packages
import numpy as np
import xarray as xr
from   netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import pandas as pd
import math
import glob
import seaborn as sns
import sys
import os
from   tqdm.auto import tqdm
from   dask.diagnostics import ProgressBar
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from   pyproj import Transformer
import cdsapi
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, r2_score, brier_score_loss
from sklearn.feature_selection import RFE
from datetime import timedelta
import itertools
import zipfile
from scipy.ndimage import distance_transform_edt
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import subprocess
import optuna

# Constants
KELVIN_TO_CELSIUS = -273.15
MRLC_VERSION_MAP = {year: "v2_0_7cds" if year <= 2015 else "v2_1_1"
            for year in range(1992, 2030)}
TARGETS = {
    'ecoli': {
        'classifier':   'escherichiaColiQualifier',
        'regressor':    'escherichiaColiCount',
    },
   'ie': {
        'classifier':   'intestinalEnterococciQualifier',
        'regressor':    'intestinalEnterococciCount',
    },
}

obs_data_root = "/data/datasets/Projects/CHANGE/data"
plots_root = os.path.join(obs_data_root, 'outputs', 'plots')
era5_cache = os.path.join(obs_data_root, 'cache', 'era5')
landcov_cache = os.path.join(obs_data_root, 'cache', 'landcov')
mhw_cache = os.path.join(obs_data_root, 'cache', 'noaa_mhw')

# Initialisation
os.makedirs(plots_root, exist_ok=True)
tqdm.pandas()

# Slice for neighbourhood average
lat_pad = lon_pad = 0.02  # half width of neighbourhood in degrees, so 0.02 gives ~5x5 km.
time_pad = pd.Timedelta(days=2)

# Time lags
lag_precip = [1, 2, 4, 7]
lag_sst = lag_precip
lag_chl = lag_precip

# Random Forest decisions
rf_classifier = True    # Classifier or Regressor
rf_type = 'classifier' if rf_classifier else 'regressor'
pathogen = 'ecoli'     # ecoli or ie
target_name = TARGETS[pathogen][rf_type]
print(f"Configured for {pathogen} {rf_type}: {target_name}\n")

# Restore all variables from store
# %store -r
# %store

# %% [markdown]
# ### Function definitions


# %% [markdown]
# ## Random Forest - regressor or classifier
# To predict pathogen given multiple EO features

# %%
# Set up features and targets for training and testing data

# Get the features and lagged versions in a sensible order
lag_names = [f"{v}_lag_{d}d" for v, d in itertools.product(['sst', 'tp_mm_daily', 'mhw', 'sst_anom', 'tsm'], [0] + lag_precip)]
feature_names = [s.replace('_lag_0d', '') for s in lag_names]
feature_names += ['chl', 'land_cov_near']
# feature_names = ['sst', 'tp_mm_daily', 'chl', 'land_cov_near', 'mhw'] + lag_names
# feature_names = ['sst', 'tp_mm_daily', 'chl']
# feature_names = ['sst', 'tp_mm_daily', 'chl', 'tp_mm_daily_lag_7d'] + [f'{v}_lag_{d}d' for v, d in itertools.product(['tp_mm_daily'], lag_precip)]
# feature_names = ['tp_mm_daily']

print(f"Target: {target_name}")
print(f"Using features: {feature_names}")
obs_valid_df = obs_df.dropna(subset=feature_names + [target_name])

# Terminology: X is table of features, y is array of target values to train and predict
X = obs_valid_df[feature_names]
y = obs_valid_df[target_name].transpose()

# Split into training and test sets, ensuring stratified as per the the target categories
class_labels = y if rf_classifier else None
X_train, X_test, y_train, y_test_truth = train_test_split(
    X, y, stratify=class_labels,
)


# %% [markdown]
# ### Optimise RF parameters

# %%
# Generate and fit RF model to training data, inside function for optimisation 
def objective (trial):
    global rf_model, y_test_pred, features_this
    print(f"\n{pd.to_datetime(pd.Timestamp.now()).strftime('%H:%M')} Trial {trial.number}: ", end="")
    bonus = 0.0

    # Some parameters to optimise automatically
    use_rfe_selector = trial.suggest_categorical("use_rfe", [True, False])
    rfe_fract_features = trial.suggest_float("rfe_fract_features", 0.1, 0.9, step=0.1) if use_rfe_selector else None
    # target params, e.g. number of health-related thresholds to use.
    # feature selection params, e.g. what fraction of features to use
    # rf params, e.g. depth
    
    # Construct RF model instance
    if rf_classifier:
        rf_model = RandomForestClassifier(max_depth=None, random_state=0)
    else:
        rf_model = RandomForestRegressor(max_depth=None, random_state=0)

    # Feature selection using recursive feature elimination - takes couple of minutes.
    # Problem is that accuracy will be slightly lower than all features, but more robust.
    # Perhaps we could add a robustness contribution to the score? Based on the average importance of selected features.
    # https://scikit-learn.org/stable/modules/model_evaluation.html#
    # Need to return the selected features to global sometime
    X_train_this = X_train
    X_test_this = X_test
    features_this = feature_names
    num_features = len(features_this)
    if use_rfe_selector:
        print(f"RFE feature selection {rfe_fract_features:.0%}...")
        selector = RFE(rf_model, n_features_to_select=rfe_fract_features)
        selector = selector.fit(X_train, y_train)

        # Output the trimmed feature selection
        X_train_this = selector.transform(X_train_this)
        X_test_this = selector.transform(X_test_this)
        features_this = selector.get_feature_names_out()
        num_features = len(features_this)

        # Display the rankings and threshold
        rankings = pd.DataFrame({'orig_index':range(len(feature_names)), 'feature': feature_names, 'ranking': selector.ranking_})
        sorted = rankings.sort_values('ranking')
        sorted.reset_index(drop=True, inplace=True)
        print(f"{sorted.head(num_features)}\n{'-'*40}\n{sorted.tail(-num_features)}")

        # Increase score for robustness, i.e. reduced num of features. Hack alert!
        bonus += (1.0 - (num_features / len(feature_names))) * bonus_weight_robustness

    # Fit/train a Random Forest classifier/regressor
    print(f"Fitting Random Forest {rf_type} to {len(X_train_this)} training samples of {num_features} features: {", ".join(features_this)}...")
    rf_model.fit(X_train_this, y_train)

    # Predict pathogen counts/category using the trained model
    print(f"Predicting for {len(X_test)} test samples of {num_features} features: {", ".join(features_this)}...")
    y_test_pred = rf_model.predict(X_test_this)

    # Metric to optimize. Accuracy may not be the best, perhaps AUC?
    # brier score may not work for regressor
    score = accuracy_score(y_test_truth, y_test_pred) + (bonus * opt_dir_val)
    return score


# %%
# Initialise and process the optimisation study (Optuna)
num_trials = 10
metric = 'accuracy'
opt_direction = 'maximize'
opt_dir_val = +1 if opt_direction == 'maximise' else -1
bonus_weight_robustness = 0.01          # Hack to favour smaller feature sets

print(f"RF parameter optimisation study using Optuna: {opt_direction} {metric}, {num_trials} trials")
study = optuna.create_study(study_name=f"{rf_type} for {pathogen}", direction=opt_direction, sampler=optuna.samplers.RandomSampler(seed=42))
study.optimize(objective, n_trials=num_trials)

print(f"\n{'-'*80}\n\nOptimisation study finished")

# %%
# Select the optimal pameters found during the trial

# Print the best parameters found 
trial = study.best_trial
print(f"Best trial {trial.number} score {trial.value:.4f}")
print("Params: ")
for key, value in trial.params.items():
    if isinstance(value, float):
        value = np.round(value, 4)
    print(f"    {key}: {value}")

# Refit the model using optimum parameters, for subsequent analysis
print(f"\nRefitting RF model using best params from trial {study.best_trial.number}")
best_score = objective(trial)
num_features = len(features_this)

# Then store the rf_model and trial for later
# %store rf_model trial

# %% [markdown]
# ### Evaluate RF model

# %%
# Explore particular trial even if it wasn't selected. Remember to rerun to reset features after and comment this out
# trials = study.get_trials()
# print(objective(trials[7]))
# num_features = len(features_this)

# %%
# Quick plots of metrics for optimised classifer

# Quick plot appropriate to RF type
if rf_classifier:
    # Classifier: Heatmap confusion matrix
    ax = plt.axes()
    disp = ConfusionMatrixDisplay.from_predictions(y_test_truth, y_test_pred, cmap='Greens', ax=ax)
    ax.set_title(f"Predicted vs actual {target_name} - samples ({num_features} features)")
    plt.show()
    ax = plt.axes()
    disp = ConfusionMatrixDisplay.from_predictions(y_test_truth, y_test_pred, cmap='Greens', normalize='true', ax=ax, )
    ax.set_title(f"Predicted vs actual {target_name} - normalised by true labels")
    plt.show()

else:
    # Regressor: Scatterplot of predicted vs actual target values
    plt.scatter(y_test_truth, y_test_pred, alpha=0.5)
    plt.xlabel(f"Actual {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"Predicted vs actual {target_name} ({num_features} features)")
    plt.grid(True)
    plt.show()

# Feature importance
importances = rf_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
forest_importances = pd.Series(importances, index=features_this)

# Plot feature importances with error bars
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title(f"Feature importance for {target_name} using mean decrease in impurity (MDI)")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

# %%
# Now need to evaluate the model performance, e.g. using R^2, MAE, etc.
print(f"Performance for {pathogen} {rf_type} on {target_name} ({num_features} features)")
if rf_classifier:
    acc = accuracy_score(y_test_truth, y_test_pred)
    print(f"Accuracy score: {acc:.3f}")

else:
    r2 = r2_score(y_test_truth, y_test_pred)
    print(f"R^2 score on test set: {r2:.3f}")
    mae = np.mean(np.abs(y_test_truth - y_test_pred))
    print(f"Mean Absolute Error on test set: {mae:.3f}")

# %%
# Display confusion matrix metrics
# Source - https://stackoverflow.com/a/43331484, Posted by lucidv01d

if rf_classifier:
    confusion_mx = confusion_matrix(y_test_truth, y_test_pred)
    FP = confusion_mx.sum(axis=0) - np.diag(confusion_mx)  
    FN = confusion_mx.sum(axis=1) - np.diag(confusion_mx)
    TP = np.diag(confusion_mx)
    TN = confusion_mx.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    print(f"Classifier confusion matrix metrics for {pathogen} {target_name}:\nLabels: {rf_model.classes_}")
    print(f"TPR={TPR}, FPR={FPR}, TNR={TNR}, FNR={FNR}, \nPPV={PPV}, NPV={NPV}, FDR={FDR}, ACC={ACC}")

# %% [markdown]
# ## Misc commands

# %%
# Display matchup data
display (obs_df)

# %%
plt.scatter(obs_df['tp_mm_daily_lag_7d'], obs_df['tp_mm_daily'], alpha=0.1)

times_df = tp_mm_daily_da['time'].values
point = tp_mm_daily_da

# %%
# Display what is in variable store
# %store
