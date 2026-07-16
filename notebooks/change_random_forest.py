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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, r2_score
from datetime import timedelta
import itertools
import zipfile
from scipy.ndimage import distance_transform_edt
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import subprocess

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

obs_data_root = '/data/datasets/Projects/CHANGE/data'
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
print(f'Configured for {pathogen} {rf_type}: {target_name}\n')

# Restore all variables from store
# %store -r
# %store

# %% [markdown]
# ### Function definitions


# %% [markdown]
# ## Random Forest - regressor or classifier

# %%
# Random Forest to predict pathogen given multiple EO features

# Get the features and lagged versions in a sensible order
lag_names = [f"{v}_lag_{d}d" for v, d in itertools.product(['sst', 'tp_mm_daily', 'mhw', 'sst_anom'], [0] + lag_precip)]
feature_names = [s.replace('_lag_0d', '') for s in lag_names] + ['chl', 'land_cov_near']
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

# Fit/train a Random Forest classifier/regressor
print(f"Fitting Random Forest {rf_type} to {len(X_train)} training samples of {", ".join(feature_names)}...")
if rf_classifier:
    rf_model = RandomForestClassifier(max_depth=None, random_state=0)
else:
    rf_model = RandomForestRegressor(max_depth=None, random_state=0)
rf_model.fit(X_train, y_train)

# %%
# Predict pathogen counts/category using the trained model

print(f"Predicting for {len(X_test)} test samples of {", ".join(feature_names)}...")
y_test_pred = rf_model.predict(X_test)

# Quick plot appropriate to RF type
if rf_classifier:
    # Classifier: Heatmap confusion matrix
    ax = plt.axes()
    disp = ConfusionMatrixDisplay.from_predictions(y_test_truth, y_test_pred, cmap='Greens', ax=ax)
    ax.set_title(f"Predicted vs actual {target_name} - samples")
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
    plt.title(f"Predicted vs actual {target_name}")
    plt.grid(True)
    plt.show()

# Feature importance
importances = rf_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
forest_importances = pd.Series(importances, index=feature_names)

# Plot feature importances with error bars
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title(f"Feature importance for {target_name} using mean decrease in impurity (MDI)")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

# %%
# Now need to evaluate the model performance, e.g. using R^2, MAE, etc.
print(f"Performance for {pathogen} {rf_type} on {target_name}")
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

    print(f"")
    print(f"TPR={TPR}, FPR={FPR}, TNR={TNR}, FNR={FNR}, PPV={PPV}, NPV={NPV}, FDR={FDR}, ACC={ACC}")

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
