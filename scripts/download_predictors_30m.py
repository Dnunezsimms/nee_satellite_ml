#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Earth Engine script to export monthly predictor pixels to CSV.
- Edit parameters (AOI, date range, export_folder) before running.
- Requires Earth Engine Python API and authentication: `earthengine authenticate`
- Exports table files to Google Drive (folder specified). Then download CSVs from Drive to local data/.
"""
import ee
import datetime
import math

# Initialize (adjust project if needed)
ee.Initialize(project='dnunezsimms')

# PARAMETERS (adjust as needed)
AOI = ee.FeatureCollection("projects/dnunezsimms/assets/Matorral_SUCULENTAS_PNBFJ_UNI_TOPO")
export_folder = 'NEE_model_CSV'
scale = 30
crs = 'EPSG:32719'
max_cloud = 30
max_diff_days = 3

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

month_ranges = get_month_ranges('2022-07-01', '2025-06-30')

def export_month(rng, idx):
    print(f"Export month: {rng['start']}")
    # Implement your EE export logic here (copy your original functions)
    pass

for i, rng in enumerate(month_ranges):
    export_month(rng, i)
