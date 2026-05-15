# CHANGE

ESA CHANGE: Climate–Health Adaptation through New Generation Earth observations.  \
[Project website](https://climate.esa.int/es/supporting-the-paris-agreement/CHANGE/)  \
Involves Peter Miller and Dave Moffat at PML, with guidance from Gemma Kulk and Shubha Sathyendranath.  \
Oct. 2025 to Sep. 2028.


## To do list
- [ ] How to use SAS gastro data when they are all positive cases? I guess negatives are other regions for same dates?
- [ ] Try CCI matchup code, or Angus's.
- [ ] Add more env vars: rainfall, Chl-a, ...
- [ ] Calculate lat/lon coords for EA UKNG coords.
- [ ] Try matchup using EA E coli data.
- [ ] Dave re sample code for Random Forest.


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

### CCI-type matchup
- Use from Round Robin scripts.
- Also Yanna has adapted for her FOCUS and PYROMAR projects for California Chl-a intercomparison.

### Angus Match-maker
- Doesn't search in time, only does neighbourhood on map pre-selected for the correct date.

### Misc commands
    ncdump -v longitude $file | grep '^ longitude ='

    foreach file ( *nc )
        echo -n "$file - "
    ncdump -h $file | tail +3 | head -3
    end

### Convert netCDF from int64 to int32 so ncview can handle it
    ncap2 --overwrite --script 'valid_time=int(valid_time)' input_file.nc output_file.nc

ERA5 rainfall data took 943 minutes... for half of the time period. +478 mins \
If we had got the Zarr dataset working, perhaps we could have extracted the whole UK region for the whole multi-year period, a lot quicker. It would have increased the size of the cache files, but, would have been more useful for further matchups.
