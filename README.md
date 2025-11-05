# NEE Satellite ML

Repository for downloading satellite predictors from Google Earth Engine (GEE) and training Leave-One-Season-Out (LOSO) models to predict NEE.

## Project layout
- `scripts/` — executable scripts for GEE export and modeling.  
- `data/` — place input CSV files here. Do not add large data files to the repository.  
- `results/` — metrics, saved models and other outputs produced by the modeling scripts.  
- `notebooks/` — optional notebooks for EDA or result visualization.

## Quick start

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Download predictors from Google Earth Engine:
```bash
# Authenticate GEE on the local machine
earthengine authenticate

# Edit scripts/download_predictors_30m.py to set AOI, date range and export folder
python scripts/download_predictors_30m.py
```
Run the export script to create export tasks in your Google Drive. After tasks complete, download the exported CSV files from Google Drive and place them in `data/`.

3. Run the modeling pipeline:
```bash
# Configure paths and parameters in config.yml
python scripts/best_model.py --config config.yml
```
Outputs (metrics CSVs and optional saved models) are written to `results/`.

## Configuration summary

`config.yml` must include:
- input data path(s) under `data`
- target column name (e.g. `NEE`)
- group column for LOSO (e.g. `season`)
- list of predictor variable names
- training/validation parameters
- `save_models: true|false`
- `output_dir` (defaults to `results/`)

Example fragment:
```yaml
data:
  path: "data/your_file.csv"

model:
  target: "NEE"
  predictors:
    - ndvi
    - lst
    - precip

cross_validation:
  group_column: "season"
  method: "LOSO"

save_models: true
output_dir: "results"
```

## Behavior and outputs
- The pipeline performs Leave-One-Group-Out (by season) cross-validation, tests multiple models and predictor combinations, and writes full metrics and a summary CSV to `results/`.  
- Set `save_models: true` in `config.yml` to save trained model artifacts to `results/`.  
- Keep `requirements.txt` synchronized for reproducibility.

## Data handling
- Do not commit large raw data files to this repository. Store large files locally or use Git LFS when necessary.  
- Store only the minimal metadata or example inputs in the repository; actual datasets belong outside the repo.


## Contact
Open an issue with reproducible steps, the `config.yml` used, and representative input examples placed in `data/`. Provide the exact commands you ran and any error messages for faster troubleshooting.
