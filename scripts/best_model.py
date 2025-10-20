#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Best model selection using Leave-One-Season-Out (LOSO) for NEE.
Reads configuration from a YAML file (--config)
Evaluates multiple models and predictor combinations
Saves metrics and optionally saves best models
"""
import argparse
import yaml
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime

def load_data(path_nee, paths_pred):
    nee_df = pd.read_csv(path_nee)
    # accept 'date' or 'fecha'
    if 'date' in nee_df.columns:
        nee_df['date'] = pd.to_datetime(nee_df['date'])
    elif 'fecha' in nee_df.columns:
        nee_df['date'] = pd.to_datetime(nee_df['fecha'])
        nee_df.rename(columns={'fecha': 'date'}, inplace=True)
    else:
        raise ValueError("NEE CSV must contain 'date' or 'fecha' column")

    pred_list = []
    for season, p in paths_pred:
        pred = pd.read_csv(p)
        if 'system:index' in pred.columns:
            pred['date'] = pd.to_datetime(pred['system:index'].str[:8], format='%Y%m%d', errors='coerce')
        elif 'date' in pred.columns:
            pred['date'] = pd.to_datetime(pred['date'])
        elif 'fecha' in pred.columns:
            pred['date'] = pd.to_datetime(pred['fecha'])
        else:
            raise ValueError(f"Predictor CSV {p} must contain 'system:index', 'date', or 'fecha' column")
        pred['season'] = season
        pred_list.append(pred)
    pred_df = pd.concat(pred_list, ignore_index=True)
    data = pd.merge(nee_df, pred_df, on='date', how='inner')
    if 'season' not in data.columns and 'temporada' in data.columns:
        data.rename(columns={'temporada': 'season'}, inplace=True)
    print(f"Merge completed. Rows: {data.shape[0]}")
    return data

def build_models(random_state=42):
    return {
        'Linear': Pipeline([('passthrough', 'passthrough'), ('est', LinearRegression())]),
        'Lasso': Pipeline([('scaler', StandardScaler()), ('est', Lasso(alpha=0.1, random_state=random_state))]),
        'RF': Pipeline([('passthrough', 'passthrough'), ('est', RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1))]),
        'SVR': Pipeline([('scaler', StandardScaler()), ('est', SVR(kernel='rbf'))]),
        'HGBR': Pipeline([('passthrough', 'passthrough'), ('est', HistGradientBoostingRegressor(loss='squared_error', learning_rate=0.1, max_iter=500, random_state=random_state))])
    }

def evaluate_loso(data, predictors, models, group_col='season', target='NEE_gC_m2_d', corr_threshold=0.1, min_pred=3, max_pred=5):
    for p in predictors:
        if p not in data.columns:
            raise KeyError(f"Predictor '{p}' not found in merged data columns")

    corr = data[predictors].corr().abs()
    def combo_ok(combo):
        for i in range(len(combo)):
            for j in range(i+1, len(combo)):
                if corr.loc[combo[i], combo[j]] > corr_threshold:
                    return False
        return True

    predictor_combos = []
    for k in range(min_pred, max_pred+1):
        predictor_combos.extend(list(combinations(predictors, k)))
    predictor_combos = [list(c) for c in predictor_combos if combo_ok(c)]
    print(f"Filtered predictor combinations: {len(predictor_combos)}")

    logo = LeaveOneGroupOut()
    results = []

    for combo in predictor_combos:
        X = data[combo].copy()
        y = data[target].copy()
        groups = data[group_col].values
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx].ffill().bfill(), X.iloc[test_idx].ffill().bfill()
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            if len(X_train) < 2 or len(X_test) < 1:
                continue
            for name, pipeline in models.items():
                try:
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                except Exception:
                    r2, rmse, mae = np.nan, np.nan, np.nan
                results.append({
                    'model': name,
                    'predictors': tuple(combo),
                    'test_group': int(groups[test_idx][0]) if len(test_idx) > 0 else None,
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'n_train': len(X_train),
                    'n_test': len(X_test)
                })
    results_df = pd.DataFrame(results)
    return results_df

def select_and_save_results(results_df, outdir, save_models=False, data=None, models=None):
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    metrics_file = os.path.join(outdir, f"metrics_loso_{ts}.csv")
    summary_file = os.path.join(outdir, f"summary_loso_{ts}.csv")
    results_df.to_csv(metrics_file, index=False)
    summary = results_df.groupby(['model', 'predictors']).mean(numeric_only=True).reset_index()
    summary.to_csv(summary_file, index=False)
    saved = []
    if save_models and models is not None and data is not None:
        for model_name in results_df['model'].unique():
            dfm = summary[summary['model'] == model_name]
            if dfm.empty:
                continue
            best_idx = dfm['r2'].idxmax()
            if pd.isna(best_idx):
                continue
            best = dfm.loc[best_idx]
            best_preds = list(best['predictors'])
            pipeline = models[model_name]
            X_full = data[best_preds].ffill().bfill()
            y_full = data['NEE_gC_m2_d']
            pipeline.fit(X_full, y_full)
            fname = os.path.join(outdir, f"best_{model_name}_{'_'.join(best_preds)}_{ts}.joblib")
            joblib.dump({'pipeline': pipeline, 'predictors': best_preds}, fname)
            saved.append(fname)
    return {'metrics_file': metrics_file, 'summary_file': summary_file, 'models_saved': saved}

def main(cfg):
    data = load_data(cfg['path_nee'], cfg['paths_pred'])
    if 'season' not in data.columns and 'temporada' in data.columns:
        data.rename(columns={'temporada': 'season'}, inplace=True)
    predictors = cfg.get('predictors', ['LST', 'NDVI', 'EVI', 'NDWI', 'RAD', 'RAD_topo', 'SWC_norm'])
    models = build_models(random_state=cfg.get('random_state', 42))
    results = evaluate_loso(data, predictors, models,
                            group_col=cfg.get('group_col', 'season'),
                            target=cfg.get('target', 'NEE_gC_m2_d'),
                            corr_threshold=cfg.get('corr_threshold', 0.1),
                            min_pred=cfg.get('min_pred', 3),
                            max_pred=cfg.get('max_pred', 5))
    out = select_and_save_results(results, cfg.get('outdir', 'results'),
                                  save_models=cfg.get('save_models', False),
                                  data=data, models=models)
    print("Saved results:", out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOSO modeling for NEE")
    parser.add_argument('--config', required=True, help="YAML config file path")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg)
