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
# ## Installation
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
from sklearn.ensemble import RandomForestClassifier

# Import PML packages
# sys.path.append("/users/rsg/anla/code/satellite/match-maker/")
# from match_maker import extract

# Constants
KELVIN_TO_CELSIUS = -273.15

plots_root = '/data/datasets/Projects/CHANGE/data/outputs/plots'
era5_cache = '/data/datasets/Projects/CHANGE/data/cache/era5'

# Initialisation
os.makedirs(plots_root, exist_ok=True)
tqdm.pandas()

# Slice for neighbourhood average
lat_pad = lon_pad = 0.02  # half width of neighbourhood in degrees, so 0.02 gives ~5x5 km.
time_pad = pd.Timedelta(days=2)


# %%
# Functions for extracting ERA5 datasets on the fly from Copernicus Data Store.
# Note that you need to have cached username/password credentials set up for this to work.

def download_era5_subset(date, area, filename):
    """
    Download ERA5 precipitation for a given day and bounding box.
    area = [North, West, South, East] (degrees, lon in 0-360)
    Returns True on failure, False on success.
    """
    c = cdsapi.Client(quiet=True, timeout=60,)   # Suppress verbose output from the API client

    try: 
        c.retrieve(
            "reanalysis-era5-single-levels",
            # "reanalysis-era5-single-levels-timeseries",   # Zarr version, supposedly faster but not working for some reason
            {
                "product_type": "reanalysis",
                "variable": "total_precipitation",
                "year": date.strftime("%Y"),
                "month": date.strftime("%m"),
                "day": date.strftime("%d"),
                # Requests all the hours of day for good temporal matching, though this may be slowing it down?
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": [
                    float(area[0]),
                    float(area[1]),
                    float(area[2]),
                    float(area[3]),
                ],
                "format": "netcdf",
            },
            filename
        )
    except Exception as e:
        print(f"Warning: Failed to download ERA5 data for {date} with area {area}. Error: {e}")
        return True  # Indicate failure
    
    return False  # Indicate success


def extract_era5_matchups(df, cache_dir="/tmp/change/era5_cache", buffer=0.25, daily=False):
    """
    Vectorized ERA5 extraction with spatial subsetting.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: 'time', 'lat', 'lon'
    cache_dir : str
        Folder to cache downloaded ERA5 files
    buffer : float
        Spatial padding (degrees)
    day : boolean
        Return daily average instead of instantaneous (hourly)

    Returns
    -------
    DataFrame with tp_mm column
    """

    os.makedirs(cache_dir, exist_ok=True)
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    # Works better using standard lon (-180..180), gave errors when using 0..360.
    # df["lon_era5"] = df["lon"] % 360    # ERA5 uses 0–360 for longitude, so convert if necessary.

    results = []

    # Process per day, with progress bar (couldn't get that to work)
    # for group in df.groupby(df["time"].dt.date).progress_apply(lambda g: g):
    # Perhaps would have been quicker to group by month, as CDSAPI has a high overhead per request
    for date, group in df.groupby(df["time"].dt.date):

        # if pd.Timestamp(date) == pd.Timestamp('2016-05-17'):
        #     print (f"Skipping {date} due to known ERA5 download issues")
        #     continue
        group = group.copy()

        # Spatial subset bounds (tight bounding box) 
        # Care to calculate min lon before it is converted to 0-360, to avoid issues around grenwich meridian
        lat_max = group["lat"].max() + buffer
        lat_min = group["lat"].min() - buffer
        lon_max = group["lon"].max() + buffer
        lon_min = group["lon"].min() - buffer

        area = [lat_max, lon_min, lat_min, lon_max]

        fname = os.path.join(cache_dir, f"era5_{date}.nc")

        # Download once per day (cached)
        if not os.path.exists(fname):
            
            print(f"{pd.to_datetime(pd.Timestamp.now()).strftime('%H:%M:%S')} Downloading {date} for {len(group)} points...")
            if download_era5_subset(pd.Timestamp(date), area, fname):
                print(f"Failed to download ERA5 data for {date}. Skipping.")
                continue

        ds = xr.open_dataset(fname)

        # Vectorized selection
        times = xr.DataArray(group["time"].values, dims="points")
        lats  = xr.DataArray(group["lat"].values, dims="points")
        lons  = xr.DataArray(group["lon"].values, dims="points")

        matched = ds.sel(
            valid_time=times,
            latitude=lats,
            longitude=lons,
            method="nearest"
        )

        # Convert rainfall from m to mm
        tp_mm = matched["tp"].values * 1000
        if daily:
            group["tp_mm_daily"] = group["tp_mm"].groupby(group["time"].dt.date).transform("sum")
        else:
            group["tp_mm"] = tp_mm

        results.append(group)
        ds.close()

    return pd.concat(results).reset_index(drop=True)


# %% [markdown]
# ## Load global datasets ready for matchups

# %%
# # %%script false --no-raise-error     # Disables this cell

# Open environmental multi-file datasets

env_file_patt = '/data/datasets/sst/esa-cci-sst/v3.0.1/global/1d/20[12]?/??/??/*SST*.nc'
#from dask.distributed import Client
#client = Client(n_workers=20, threads_per_worker=2, memory_limit='10GB')

# Estimate number of files included
print ("Compiling list of dataset files...")
env_file_list = glob.glob(env_file_patt)
print (f'Env dataset contains {len(env_file_list)} files')

print ('Opening multi-file env data, this may take several minutes...')
with ProgressBar():
    env_ds = xr.open_mfdataset(env_file_patt, combine='by_coords', data_vars='all')

# Tried alternative but it still doesn't show progress bar
#with xr.open_mfdataset(env_file_patt, combine='by_coords', data_vars='all', parallel=True, chunks=dict(index=1)) as env_ds, ProgressBar():
#    env_ds.load()  # Force loading to ensure we catch any issues with the files now, while the progress bar is active.
print ('Done')

# match_maker.extract.gridded.nearest(ds, points, k=1, return_distance=False, units='km')

# Tips for using Xarray for matchups:
# https://gis.stackexchange.com/questions/225100/extract-time-series-values-from-a-3d-lon-lat-time-netcdf-file-using-python


print(f'Variables: {list(env_ds.keys())}')
sst_da = env_ds['analysed_sst'] + KELVIN_TO_CELSIUS
sst_da

# Do we have to invert the array? 

# %%
# Open Chl-a global dataset, 4km daily - takes 25 mins
# I could go for 1km data, but the other datasets are >= 4km resolution.

env_file_patt = '/data/datasets/CCI/v6.0-release/geographic/netcdf/daily/chlor_a/20[12]?/*.nc'
data_var = 'chlor_a'

# Estimate number of files included
print ("Compiling list of dataset files...")
env_file_list = glob.glob(env_file_patt)
print (f'Env dataset contains {len(env_file_list)} files')

print ('Opening multi-file env data, this may take several minutes...')
with ProgressBar():
    env_ds = xr.open_mfdataset(env_file_patt, combine='by_coords', data_vars=[data_var])
env_ds = env_ds.chunk(time=30)
print(f'Variables: {list(env_ds.keys())}')
chl_da = env_ds[data_var]
chl_da = chl_da.sortby('lat')
chl_da

# %% [markdown]
# ## Analysis of gastoenteritis cases (SaS)

# %%
# Load observed locations from CSV file - Surfers Against Sewage data on gastro cases
obs_data_root = '/data/datasets/Projects/CHANGE/data'
obs_data_name = os.path.join(obs_data_root, 'Surfers_Against_Sewage', 'sas_gastro_all_cases_23feb2026.csv')

obs_df = pd.read_csv(obs_data_name)
# Looks like 'n/a' are correctly turned into 'NaN' for missing data.

# Rename columns
obs_df.rename(columns={'Long':'lon', 'Lat':'lat'}, inplace=True)

# Correct data types, sometimes floats are read as strings due to n/a values
obs_df['lon'] = np.float32(obs_df['lon'])
obs_df['lat'] = np.float32(obs_df['lat'])

# Remove invalid rows, with missing lat/lon
obs_df = obs_df.dropna(subset=['lon', 'lat'])

# Add time variable, at midday for dates
obs_df['time'] = pd.to_datetime(obs_df['dateenteredwater'], dayfirst=True) + pd.Timedelta(hours=12)
obs_df

# %%
# Try a match-up with SST data

# Build coordinate arrays aligned to the DataFrame index
times = xr.DataArray(pd.DatetimeIndex(obs_df['time']), dims="points")
lats  = xr.DataArray(obs_df['lat'].values, dims="points")
lons  = xr.DataArray(obs_df['lon'].values, dims="points")

# Vectorised extraction
# NB produces a lot of 271.35 (=-2.0 'C?), perhaps many are too far inland (700 missing).
# 271.35 is temperature of Arctic water, I guess at (0,0) if lat/lon are NaN. -54.53K is missing.
print('Extracting matchups from dataset...')
method = 'nearest'
values = sst_da.sel(
    time=times,
    lat=lats,
    lon=lons,
    method=method,
)

# Select then interpolate doesn't work for me (3441 missing?)
# values = (
#     sst_da.sel(time=times, method=method)
#     .interp(lat=lats, lon=lons)
# )

obs_df['sst'] = values
print(f'Missing values: {obs_df['sst'].isnull().sum().sum() / len(obs_df):.1%}')

# Just show selected columns
print(f'Selected columns:')
print_df = pd.concat([obs_df.iloc[:, :6], obs_df.iloc[:, 16:]], axis='columns')
print_df

# %% magic_args="false --no-raise-error     # Disables this cell" language="script"
#
# # Uses slice for neighbourhood average, takes ~20 mins
# lat_pad = lon_pad = 0.02  # half width of neighbourhood in degrees, so 0.02 gives ~5x5 km.
# time_pad = pd.Timedelta(days=2)
#
# # You can't use nearest in time if also using a slice in coordinates, so loop through each row of observations.
# # Then use time and coords slice. But we should not average in time.
# # Actually, why are we adding a slice in time when there should be identical merged coverage every day?
# # Then it is taking the mean over 5 days.
# # tqdm makes a progress bar according to the rows of points completed.
# results = []
# for _, row in tqdm(obs_df.iterrows(), total=len(obs_df), desc='Extracting matchups using slice'):
#     nhood = sst_da.sel(
#         time=slice(row['time'] - time_pad, row['time'] + time_pad),
#         lat=slice(row['lat'] - lat_pad, row['lat'] + lat_pad),
#         lon=slice(row['lon'] - lon_pad, row['lon'] + lon_pad),
#     )
#     results.append(float(nhood.mean()))
#
# # Select time first, then slice coords. No, that will use the closest time whether or not region is missing.
# # for _, row in tqdm(obs_df.iterrows(), total=len(obs_df), desc='Extracting matchups using xxx'):
# #     nhood = (
# #         sst_da.sel(time=row['time'], method=method)
# #         .sel(
# #             lat=slice(row['lat'] - lat_pad, row['lat'] + lat_pad),
# #             lon=slice(row['lon'] - lon_pad, row['lon'] + lon_pad),
# #         )
# #     )
# #     results.append(float(nhood.mean()))
#
# # Select time first, then slice coords. No, that will use the closest time whether or not region is missing.
# # Nope, doesn't work due to problems with not monotonically increasing?
# # for _, row in tqdm(obs_df.iterrows(), total=len(obs_df), desc='Extracting matchups using xxx'):
# #     nhood = (
# #         sst_da.sel(time=row['time'], method=method, tolerance=pd.Timedelta(days=2))
# #         .sel(lat=row['lat'], lon=row['lon'], method=method)
# #     )
# #     results.append(float(nhood.mean()))
#
# obs_df['sst2'] = results
#
# print(f'Missing values: {obs_df['sst2'].isnull().sum().sum() / len(obs_df):.1%}')
#

# %%
# Simple plot of the matchup SST values
plt.plot(obs_df['sst'])
min(obs_df['sst'])


# %%
# Plotting the matchups points on a map, coloured by the SST value
fig, ax = plt.subplots(figsize=(12, 8),
                       subplot_kw={'projection': ccrs.PlateCarree()})

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3)
ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

sc = ax.scatter(obs_df['lon'], obs_df['lat'],
                c=obs_df['sst'],     # colour by this column
                cmap='turbo',        # colormap
                s=100,               # marker size
                alpha=0.8,
                transform=ccrs.PlateCarree(),
                zorder=5)

plt.colorbar(sc, ax=ax, label='SST (°C)', shrink=0.6)
ax.set_title('SST-coloured points matchups with SAS gastro cases on map')
plt.tight_layout()
plt.savefig(os.path.join(plots_root, 'sst_gastro_on_map.png'), dpi=150)
plt.show()

# %%
out_data_name = os.path.join(obs_data_root, 'outputs', 'sas_23feb_test_sst.csv')
obs_df.to_csv(out_data_name)

# %% [markdown]
# ## Analysis of E coli monitoring (EA)

# %%
# Load observed locations from CSV file - EA monitoring data on E coli
obs_data_root = '/data/datasets/Projects/CHANGE/data'
obs_data_name = os.path.join(obs_data_root, 'EA_Ecoli', 'ALL_BW_data_2012-2025.csv')

obs_df = pd.read_csv(obs_data_name)

# Transform UK National Grid easting, northing coordinates to lat, lon
# What is difference between 'ngrx' and 'ngrx (values)' columns?
transformer = Transformer.from_crs("epsg:27700", "epsg:4326", always_xy=True)
lons, lats = transformer.transform(obs_df['ngrx'].values, obs_df['ngry'].values)
obs_df['lat'], obs_df['lon'] = lats, lons

# Remove invalid rows, with missing lat/lon
obs_df = obs_df.dropna(subset=['lon', 'lat'])

# Add time variable, at midday for dates
obs_df['time'] = pd.to_datetime(obs_df['sampleTime'])
obs_df

# Restrict data to coastal sites only (not transitional, rivers, etc.)
obs_df = obs_df[obs_df['siteType'] == 'Coastal']

# Store the data with the lat, lon, time columns
out_data_name = os.path.join(obs_data_root, 'EA_Ecoli', 'ALL_BW_data_2012-2025_latlon.csv')
obs_df.to_csv(out_data_name)

# %%
# Match-up EA E coli with SST data - vectorised (~5 mins)
# Missing values: 8.6% that's not bad, so the coords must usually hit the sea pixels. 

# Build coordinate arrays aligned to the DataFrame index
times = xr.DataArray(pd.DatetimeIndex(obs_df['time']), dims="points")
lats  = xr.DataArray(obs_df['lat'].values, dims="points")
lons  = xr.DataArray(obs_df['lon'].values, dims="points")

print('Extracting matchups from dataset, this may take a while...')

# Vectorised extraction
method = 'nearest'
values = sst_da.sel(
    time=times,
    lat=lats,
    lon=lons,
    method=method,
)
obs_df['sst'] = values

print(f'Missing values: {obs_df['sst'].isnull().sum().sum() / len(obs_df):.1%}')

# Just show selected columns
print(f'Selected columns:')
print_df = pd.concat([obs_df.iloc[:, :6], obs_df.iloc[:, 16:]], axis='columns')
print_df

# %% magic_args="false --no-raise-error     # Disables this cell, takes too long" language="script"
#
# # Match-up EA E coli with SST data
# # Day-at-a-time rather than vectorised, to allow neighbourhood average
#
# print('Extracting matchups from dataset...')
#
# # Slightly dodgy as assuming all rows will be processed in time order within groupby.
# # So changed to write row by row, but now will take 60h...
# obs_df['sst'] = np.nan
# grouped = obs_df.groupby(obs_df["time"].dt.date)
# for date, group in tqdm(grouped, total=grouped.ngroups, desc='Extracting matchups using slice'):
#     param_group = sst_da.sel(time=date, method='nearest', tolerance=pd.Timedelta(days=2))
#     for _, row in group.iterrows():
#         nhood = param_group.sel(
#             lat=slice(row['lat'] - lat_pad, row['lat'] + lat_pad),
#             lon=slice(row['lon'] - lon_pad, row['lon'] + lon_pad),
#         )
#         row['sst'] = float(nhood.mean())
#
# print(f'Missing values: {obs_df['sst'].isnull().sum().sum() / len(obs_df):.1%}')
#
# # Just show selected columns
# print(f'Selected columns:')
# print_df = pd.concat([obs_df.iloc[:, :6], obs_df.iloc[:, 16:]], axis='columns')
# print_df

# %%
# Matchup EA data with rainfall
print('Caching and extracting matchups with ERA5 rainfall data, this may take a while...')
obs_df = extract_era5_matchups(obs_df, cache_dir=era5_cache) 
print(f'Missing values: {obs_df['tp_mm'].isnull().sum().sum() / len(obs_df):.1%}')


# %%
# Matchup EA data with rainfall - daily sum
print('Caching and extracting matchups with ERA5 daily rainfall data, this may take a while...')
obs_df = extract_era5_matchups(obs_df, cache_dir=era5_cache, daily=True) 
print(f'Missing values: {obs_df["tp_mm_daily"].isnull().sum().sum() / len(obs_df):.1%}')

# %%
# Match-up EA E coli with chl-a data - vectorised (~? mins)
# Missing values: ??

# Build coordinate arrays aligned to the DataFrame index
times = xr.DataArray(pd.DatetimeIndex(obs_df['time']), dims="points")
lats  = xr.DataArray(obs_df['lat'].values, dims="points")
lons  = xr.DataArray(obs_df['lon'].values, dims="points")
method = 'nearest'

# Chop out subscene
print('Making subscene...')
region_da = chl_da.sel(
    time=slice(obs_df['time'].min() - time_pad, obs_df['time'].max() + time_pad),
    lat=slice(obs_df['lat'].min() - lat_pad, obs_df['lat'].max() + lat_pad),
    lon=slice(obs_df['lon'].min() - lon_pad, obs_df['lon'].max() + lon_pad),
)

# Vectorised extraction
# Dask delayed run to enable use of progress bar
print('Extracting matchups from dataset, this may take a while...')
interp_delayed = region_da.sel(
    time=times,
    lat=lats,
    lon=lons,
    method=method,
).load(compute=False)
with ProgressBar():
    values = interp_delayed.compute()

obs_df['chl'] = values

print(f'Missing values: {obs_df['chl'].isnull().sum().sum() / len(obs_df):.1%}')

# Just show selected columns
print(f'Selected columns:')
print_df = pd.concat([obs_df.iloc[:, :6], obs_df.iloc[:, 16:]], axis='columns')
print_df

# %%
# Abritrary threshold of EA E coli data for plotting test
high_ecoli = {'=', '>'}
obs_filtered_df = obs_df[obs_df['escherichiaColiQualifier'].isin(high_ecoli)]
obs_filtered_df

# %%
# Plotting the E coli matchups points on a map, coloured by the SST value
fig, ax = plt.subplots(figsize=(12, 8),
                       subplot_kw={'projection': ccrs.PlateCarree()})

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3)
ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

sc = ax.scatter(obs_filtered_df['lon'], obs_filtered_df['lat'],
                c=obs_filtered_df['sst'],     # colour by this column
                cmap='turbo',        # colormap
                s=100,               # marker size
                alpha=0.5,
                transform=ccrs.PlateCarree(),
                zorder=5)

plt.colorbar(sc, ax=ax, label='SST (°C)', shrink=0.6)
ax.set_title('SST-coloured points matchups with EA E coli cases on map')
plt.tight_layout()
plt.savefig(os.path.join(plots_root, 'sst_ecoli_on_map.png'), dpi=150)
plt.show()

# %%
# Plotting the E coli matchups points on a map, coloured by the precipitation value
fig, ax = plt.subplots(figsize=(12, 8),
                       subplot_kw={'projection': ccrs.PlateCarree()})

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3)
ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

sc = ax.scatter(obs_filtered_df['lon'], obs_filtered_df['lat'],
                c=obs_filtered_df['tp_mm'],     # colour by this column
                cmap='turbo',        # colormap
                s=100,               # marker size
                alpha=0.5,
                transform=ccrs.PlateCarree(),
                zorder=5)

plt.colorbar(sc, ax=ax, label='Total precipitation (mm)', shrink=0.6)
ax.set_title('Precipitation-coloured points matchups with EA E coli cases on map')
plt.tight_layout()
plt.savefig(os.path.join(plots_root, 'tp_ecoli_on_map.png'), dpi=150)
plt.show()

# %%
# Plotting the E coli matchups points on a map, coloured by the chl value
fig, ax = plt.subplots(figsize=(12, 8),
                       subplot_kw={'projection': ccrs.PlateCarree()})

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3)
ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

sc = ax.scatter(obs_filtered_df['lon'], obs_filtered_df['lat'],
                c=obs_filtered_df['chl'],     # colour by this column
                cmap='turbo',        # colormap
                s=100,               # marker size
                alpha=0.5,
                transform=ccrs.PlateCarree(),
                zorder=5,
                norm=colors.LogNorm())

cb = plt.colorbar(sc, ax=ax, label='Chl-a (mg m^-3)', shrink=0.6)
cb.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
ax.set_title('Chl-a-coloured points matchups with EA E coli cases on map')
plt.tight_layout()
plt.savefig(os.path.join(plots_root, 'chl_ecoli_on_map.png'), dpi=150)
plt.show()

# %%
# Test extraction of ERA5 precipitation
df = pd.DataFrame({
    "time": ["2020-01-15 12:30", "2020-01-15 03:10"],
    "lat": [50.4, 51.0],
    "lon": [-4.1, -3.8]
})
out_df = extract_era5_matchups(df)

print(out_df)

# %% [markdown]
# ## Random Forest classification experiments

# %%
# Random Forest to predict E coli given SST and ...

# Remove invalid rows, with missing SST or E coli count
feature_names = ['sst', 'tp_mm', 'tp_mm_daily', 'chl']
obs_valid_df = obs_df.dropna(subset=feature_names + ['escherichiaColiCount'])

# Terminology: X is table of features, y is array of values to train and predict
X = obs_valid_df[feature_names]
y = obs_valid_df['escherichiaColiCount'].transpose()

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y
)

# Fit/train a Random Forest classifier
print(f'Fitting Random Forest classifier to {len(X_train)} training samples of {', '.join(feature_names)}...')
clf = RandomForestClassifier(max_depth=None, random_state=0)
clf.fit(X_train, y_train)

# %%
# Predict E coli counts using the trained model

print(f'Predicting for {len(X_test)} test samples of {', '.join(feature_names)}...')
y_result = clf.predict(X_test)

# Plot predicted vs actual E coli counts
plt.scatter(y_test, y_result, alpha=0.5)
plt.xlabel('Actual E coli count')
plt.ylabel('Predicted E coli count')
plt.title('Predicted vs actual E coli counts')
plt.grid(True)
plt.show()

# Feature importance
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
forest_importances = pd.Series(importances, index=feature_names)

# Plot feature importances with error bars
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importance using mean decrease in impurity (MDI)")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

# %%
# Now need to evaluate the model performance, e.g. using R^2, MAE, etc.
r2 = clf.score(X_test, y_test)
print(f'R^2 score on test set: {r2:.3f}') 
mae = np.mean(np.abs(y_test - y_result))
print(f'Mean Absolute Error on test set: {mae:.3f}')

# %% [markdown]
# ## Misc commands

# %%
# Store variables so we don't have to regenerate them
# %store obs_df obs_filtered_df sst_da chl_da

# %%
# Restore all variables from store
# %store -r
# %store
