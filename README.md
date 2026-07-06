# CHANGE

ESA CHANGE: Climate–Health Adaptation through New Generation Earth observations.  \
[Project website](https://climate.esa.int/es/supporting-the-paris-agreement/CHANGE/)  \
Involves Peter Miller and Dave Moffat at PML, with guidance from Gemma Kulk and Shubha Sathyendranath.  \
Oct. 2025 to Sep. 2028.


## To do list
- [ ] How to use SAS gastro data when they are all positive cases? I guess negatives are other regions for same dates?
- [ ] Try CCI matchup code, or Angus's.
- [x] Add more env vars: 
  - [x] Rainfall - instantaneous
  - [x] Rainfall - daily average
  - [x] Chl-a
  - [x] SST
  - [x] Land use
  - [x] Time lags
  - [x] Marine heatwave category
- [x] Calculate lat/lon coords for EA UKNG coords.
- [x] Try matchup using EA E coli data.
- [x] Dave re sample code for Random Forest.
- [ ] Try E coli category classification rather than regression (<, =, >)
- [ ] Optuna setup for fiddling with parameters.
- [ ] Sk-learn functions for feature selection.
- [x] Many MHW values are -9e18 but not missing, investigate.
- [ ] Try IE category classification.



## Installation
- Run Notebook using this environment: `/data/abitibi1/scratch/scratch_disk/pim/miniforge3/envs/phyto-cci-pig`
- We'll probably need to setup a new env for CHANGE later.

## Different matchup methods
### Xarray select or slice
- Select nearest point in time and coords, all at once, no tolerance.
  - 19%
- Select nearest time and slide coords
  - Doesn't let you.
- Loop through obs, then slice time and coords
  - 47% missing.
- Select time then interolate coords:
  - 71% missing!
- Select time first, then slice coords. 
  - 45% missing. No, that will use the closest time whether or not region is missing.

Then again for EA E coli vs. SST matchups:
- Select nearest point in time and coords, all at once, no tolerance.
  - 8.6% missing, not too bad.
- Loop through obs, then slice time and coords
  - ?

Then for EA E coli vs. chl-a matchups:
- Select nearest point in time and coords, all at once, no tolerance.
  - 90% missing... why?
- Interpolate instead of select, so as to avoid the frequent missing chl-a values.
  - 90.1% missing, 10m, no change.

This is getting silly and taking a lot of time. 
I think I should switch to CCI matchup Python script.

**NB** If you get no slice of latitutdes, that means the latitudes need to be changed to increasing order.

### CCI-type matchup
- Use from Round Robin scripts.
- Also Yanna has adapted for her FOCUS and PYROMAR projects for California Chl-a intercomparison.

### Angus Match-maker
- Doesn't search in time, only does neighbourhood on map pre-selected for the correct date.

## Misc commands
    ncdump -v longitude $file | grep '^ longitude ='

    foreach file ( *nc )
        echo -n "$file - "
    ncdump -h $file | tail +3 | head -3
    end

### Convert netCDF from int64 to int32 so ncview can handle it
    ncap2 --overwrite --script 'valid_time=int(valid_time)' input_file.nc output_file.nc

ERA5 rainfall data took 943 minutes... for half of the time period. +478 mins \
If we had got the Zarr dataset working, perhaps we could have extracted the whole UK region for the whole multi-year period, a lot quicker. It would have increased the size of the cache files, but, would have been more useful for further matchups.



## Lagging parameters
- https://stackoverflow.com/questions/37152723/how-to-auto-discover-a-lagging-of-time-series-data-in-scikit-learn-and-classify
- https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html
- Amy recommended [R package dlnm](http://www.ag-myresearch.com/package-dlnm.html), though can be called from [Python pydlnm module](https://github.com/aedessler/pydlnm), still needs R installed.


## Evaluation metrics
- So far r^2.
- [Evaluating a Random Forest model](https://medium.com/analytics-vidhya/evaluating-a-random-forest-model-9d165595ad56) on Medium, using sk-learn.


## Optimising feature selection
- Why are single features higher r^2 than several combined?
- Macarena recommended [Optuna - A hyperparameter optimization framework](https://optuna.org/) and helpful [Random Forest optimisation tutorial](https://medium.com/@ehsannabatchian/optimizing-random-forest-models-a-deep-dive-into-hyperparameter-tuning-with-optuna-b8e4fe7f3670).
- Here's a readable post on optimising: [How do I improve the performance of my Random Forest model on Databricks?](https://community.databricks.com/t5/machine-learning/how-do-i-improve-the-performance-of-my-random-forest-model-on/td-p/142507)

## CDS processing issues
- See your jobs in queue at: https://cds.climate.copernicus.eu/requests?tab=all

## Handling environmental datasets

### Land cover
- [Visualise LCCS map](https://climate.esa.int/de/projekte/land-cover/notebook-lc/)