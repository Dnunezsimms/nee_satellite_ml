# NEE Satellite ML

Repository for downloading satellite predictors (Google Earth Engine) and training LOSO (Leave-One-Season-Out) models to predict NEE.

Project layout
- scripts/: executable scripts (GEE export + modeling)
- data/: place your CSV files here (do not add large data to the repo)
- results/: metrics and saved models produced by the modeling script
- notebooks/: optional EDA or result notebooks

Quick start

1. Install Python dependencies:

    pip install -r requirements.txt

2. To download predictors from Google Earth Engine:

    - Authenticate Earth Engine locally: earthengine authenticate
    - Edit scripts/download_predictors_30m.py parameters if necessary (AOI, date range, export folder)
    - Run the script locally to create export tasks in your Google Drive:
    
        python scripts/download_predictors_30m.py

   After export, download the CSV files from your Google Drive into the local data/ folder.

3. To run the modeling pipeline:

    - Edit config.yml to point to your CSV files in data/
    - Run:
    
        python scripts/best_model.py --config config.yml

   Results (metrics CSVs and optional saved models) are written to results/.

Notes
- Do not commit large raw data files to GitHub. Keep data locally or use Git LFS if needed.
- The modeling script uses Leave-One-Group-Out (season) and tests multiple models and predictor combinations. It writes full metrics and a summary CSV.
- If you want models saved automatically set save_models: true in config.yml.
- License: MIT (change if needed).
