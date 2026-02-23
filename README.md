# CHANGE

ESA CHANGE: Climate–Health Adaptation through New Generation Earth observations.  \
[Project website](https://climate.esa.int/es/supporting-the-paris-agreement/CHANGE/)  \
Involves Peter Miller and Dave Moffat at PML, with guidance from Gemma Kulk and Shubha Sathyendranath.  \
Oct. 2025 to Sep. 2028.


## To do list
- [ ] To do


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
- 

### Angus Match-maker
- Doesn't search in time, only does neighbourhood on map pre-selected for the correct date.