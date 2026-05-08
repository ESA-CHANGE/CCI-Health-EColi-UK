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
# ## ESA CHANGE - Experiments with data and Random Forest 
#
# ### Installation
# - Run using this environment (for now): /data/abitibi1/scratch/scratch_disk/pim/miniforge3/envs/phyto-cci-pig

# %%
# Setup and constants

# Import external packages
import numpy as np
import xarray as xr
from   netCDF4 import Dataset
import matplotlib.pyplot as plt
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

# Import PML packages
# sys.path.append("/users/rsg/anla/code/satellite/match-maker/")
# from match_maker import extract

# Constants
KELVIN_TO_CELSIUS = -273.15

plots_root = '/data/datasets/Projects/CHANGE/data/outputs/plots'

# Initialisation
os.makedirs(plots_root, exist_ok=True)

# %%
# Open environmental multi-file datasets

env_file_patt = '/data/datasets/sst/esa-cci-sst/v3.0.1/global/1d/202?/??/??/*SST*.nc'

# Estimate number of files included
env_file_list = glob.glob(env_file_patt)
print (f'Env dataset contains {len(env_file_list)} files')

print ('Opening multi-file env data, this may take several minutes...')
with ProgressBar():
    env_ds = xr.open_mfdataset(env_file_patt, combine='by_coords', data_vars='all')
print ('Done')

# match_maker.extract.gridded.nearest(ds, points, k=1, return_distance=False, units='km')

# Tips for using Xarray for matchups:
# https://gis.stackexchange.com/questions/225100/extract-time-series-values-from-a-3d-lon-lat-time-netcdf-file-using-python


print(f'Variables: {list(env_ds.keys())}')
da = env_ds['analysed_sst'] + KELVIN_TO_CELSIUS
da

# Do we have to invert the array? 

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
obs_df['date_time'] = pd.to_datetime(obs_df['dateenteredwater'], dayfirst=True) + pd.Timedelta(hours=12)
obs_df

# %%
# Try a match-up with SST data

# Build coordinate arrays aligned to the DataFrame index
times = xr.DataArray(pd.DatetimeIndex(obs_df['date_time']), dims="points")
lats  = xr.DataArray(obs_df['lat'].values, dims="points")
lons  = xr.DataArray(obs_df['lon'].values, dims="points")

# Vectorised extraction
# NB produces a lot of 271.35 (=-2.0 'C?), perhaps many are too far inland (700 missing).
# 271.35 is temperature of Arctic water, I guess at (0,0) if lat/lon are NaN. -54.53K is missing.
print('Extracting matchups from dataset...')
method = 'nearest'
values = da.sel(
    time=times,
    lat=lats,
    lon=lons,
    method=method,
)

# Select then interpolate doesn't work for me (3441 missing?)
# values = (
#     da.sel(time=times, method=method)
#     .interp(lat=lats, lon=lons)
# )

obs_df['sst'] = values
print(f'Missing values: {obs_df['sst'].isnull().sum().sum() / len(obs_df):.1%}')

# Just show selected columns
print(f'Selected columns:')
print_df = pd.concat([obs_df.iloc[:, :6], obs_df.iloc[:, 16:]], axis='columns')
print_df

# %% magic_args="false --no-raise-error" language="script"
#
# # Uses slice for neighbourhood average, takes ~20 mins
# lat_pad = lon_pad = 0.02  # half width of neighbourhood in degrees, so 0.02 gives ~5x5 km.
# time_pad = pd.Timedelta(days=2)
# results = []
#
# # You can't use nearest in time if also using a slice in coordinates, so loop through each row of observations.
# # Then use time and coords slice. But we should not average in time.
# # tqdm makes a progress bar according to the rows of points completed.
# for _, row in tqdm(obs_df.iterrows(), total=len(obs_df), desc='Extracting matchups using slice'):
#     nhood = da.sel(
#         time=slice(row['date_time'] - time_pad, row['date_time'] + time_pad),
#         lat=slice(row['lat'] - lat_pad, row['lat'] + lat_pad),
#         lon=slice(row['lon'] - lon_pad, row['lon'] + lon_pad),
#     )
#     results.append(float(nhood.mean()))
#
# # Select time first, then slice coords. No, that will use the closest time whether or not region is missing.
# # for _, row in tqdm(obs_df.iterrows(), total=len(obs_df), desc='Extracting matchups using xxx'):
# #     nhood = (
# #         da.sel(time=row['date_time'], method=method)
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
# #         da.sel(time=row['date_time'], method=method, tolerance=pd.Timedelta(days=2))
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
obs_df['date_time'] = pd.to_datetime(obs_df['sampleTime'])
obs_df

# Restrict data to coastal sites only (not transitional, rivers, etc.)
obs_df = obs_df[obs_df['siteType'] == 'Coastal']

# Store the data with the lat, lon, time columns
out_data_name = os.path.join(obs_data_root, 'EA_Ecoli', 'ALL_BW_data_2012-2025_latlon.csv')
obs_df.to_csv(out_data_name)

# %%
# Abritrary threshold of EA E coli data for plotting test
high_ecoli = {'=', '>'}
obs_filtered_df = obs_df[obs_df['escherichiaColiQualifier'].isin(high_ecoli)]
obs_filtered_df

# %%
# Plotting the matchups points on a map, coloured by the SST value
fig, ax = plt.subplots(figsize=(12, 8),
                       subplot_kw={'projection': ccrs.PlateCarree()})

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3)
ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

sc = ax.scatter(obs_filtered_df['lon'], obs_filtered_df['lat'],
                # c=obs_filtered_df['sst'],     # colour by this column [Not there yet]
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
# Functions for extracting ERA5 datasets on the fly from Copernicus Data Store

def download_era5_subset(date, area, filename):
    """
    Download ERA5 precipitation for a given day and bounding box.
    area = [North, West, South, East] (degrees, lon in 0–360)
    """
    c = cdsapi.Client()

    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": "total_precipitation",
            "year": date.strftime("%Y"),
            "month": date.strftime("%m"),
            "day": date.strftime("%d"),
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [
                float(area[0]),
                float(area[1]),
                float(area[2]),
                float(area[3]),
            ],
            "format": "netcdf"
        },
        filename
    )


def extract_era5_matchups(df, cache_dir="era5_cache", buffer=0.25):
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

    Returns
    -------
    DataFrame with tp_mm column
    """

    # HERE: Change to temporary directory
    os.makedirs(cache_dir, exist_ok=True)

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["lon_era5"] = df["lon"] % 360

    results = []

    # Process per day
    for date, group in df.groupby(df["time"].dt.date):

        group = group.copy()

        # Spatial subset bounds (tight bounding box)
        lat_max = group["lat"].max() + buffer
        lat_min = group["lat"].min() - buffer
        lon_max = group["lon_era5"].max() + buffer
        lon_min = group["lon_era5"].min() - buffer

        area = [lat_max, lon_min, lat_min, lon_max]

        fname = os.path.join(cache_dir, f"era5_{date}.nc")

        # Download once per day (cached)
        if not os.path.exists(fname):
            download_era5_subset(pd.Timestamp(date), area, fname)

        ds = xr.open_dataset(fname)

        # Vectorized selection
        times = xr.DataArray(group["time"].values, dims="points")
        lats  = xr.DataArray(group["lat"].values, dims="points")
        lons  = xr.DataArray(group["lon_era5"].values, dims="points")

        matched = ds.sel(
            valid_time=times,
            latitude=lats,
            longitude=lons,
            method="nearest"
        )

        # Convert to mm
        tp_mm = matched["tp"].values * 1000

        group["tp_mm"] = tp_mm

        results.append(group)

        ds.close()

    return pd.concat(results).reset_index(drop=True)


# %%
# Test extraction of ERA5 precipitation
df = pd.DataFrame({
    "time": ["2020-01-15 12:30", "2020-01-15 03:10"],
    "lat": [50.4, 51.0],
    "lon": [-4.1, -3.8]
})
out_df = extract_era5_matchups(df)

print(out_df)
