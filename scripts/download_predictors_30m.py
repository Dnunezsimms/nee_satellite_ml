#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Earth Engine script to export monthly predictor pixels to CSV.
- Configure EE project and AOI via environment variables or edit the placeholders below.
- Requires Earth Engine Python API and authentication: `earthengine authenticate`
- Exports table files to Google Drive (folder specified). Then download CSVs from Drive to local data/.
"""
import os
import ee
import datetime
import math

# Initialize Earth Engine using an optional environment variable EE_PROJECT.
# If EE_PROJECT is not set, initialize without specifying a project (local defaults).
ee_project = os.environ.get("EE_PROJECT")
if ee_project:
    ee.Initialize(project=ee_project)
else:
    ee.Initialize()

# PARAMETERS (adjust as needed)
# AOI can be provided via environment variable EE_AOI_ASSET (e.g. "users/your_user/your_asset"
# or "projects/your-project/assets/your_asset"). If not provided, replace the placeholder below.
AOI_ASSET = os.environ.get(
    "EE_AOI_ASSET",
    "projects/REPLACE_WITH_YOUR_PROJECT/assets/REPLACE_WITH_YOUR_ASSET"
)
try:
    AOI = ee.FeatureCollection(AOI_ASSET)
except Exception as e:
    raise RuntimeError(f"Failed to load AOI FeatureCollection '{AOI_ASSET}': {e}")

export_folder = os.environ.get("EE_EXPORT_FOLDER", "NEE_model_CSV")
scale = int(os.environ.get("EE_SCALE", 30))
crs = os.environ.get("EE_CRS", "EPSG:32719")
max_cloud = int(os.environ.get("EE_MAX_CLOUD", 30))
max_diff_days = int(os.environ.get("EE_MAX_DIFF_DAYS", 3))

def get_month_ranges(start_date, end_date):
    start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    ranges = []
    while start < end:
        month_start = start
        next_month = (month_start.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
        month_end = min(next_month - datetime.timedelta(days=1), end)
        ranges.append({'start': month_start.strftime('%Y-%m-%d'), 'end': month_end.strftime('%Y-%m-%d')})
        start = next_month
    return ranges

month_ranges = get_month_ranges(os.environ.get("EE_START_DATE", "2022-07-01"),
                                os.environ.get("EE_END_DATE", "2025-06-30"))

def export_month(rng, idx):
    print(f"Export month: {rng['start']}")
    # Implement your EE export logic here using AOI, export_folder, scale, etc.
    # This is a placeholder — copy your original functions here and replace hard-coded
    # project/asset names with AOI_ASSET or environment variables.
    pass

for i, rng in enumerate(month_ranges):
    export_month(rng, i)
