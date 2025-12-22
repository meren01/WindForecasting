#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT 1: TRAIN MODELS (DT, RF, ANN)
------------------------------------
1. Reads Training Data (feke_80.csv).
2. Generates 'Prev_Speed' (Lag Feature) internally for better R2.
3. Trains DT, RF, and ANN models using GridSearchCV.
4. SAVES the trained models to 'models/' directory.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# --- CONFIGURATION ---
TRAIN_FILE = r"C:\Users\erenf\feke_80.csv"
MODEL_SAVE_DIR = "./models"
EXCLUDE_COLS = {"WPD", "ID"}  # Columns to exclude

COMPASS_MAP = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5
}

def prepare_data(file_path, target_col="Speed"):
    """Reads and processes data. Adds Lag feature."""
    print(f"[TRAIN] Reading file: {file_path}")
    p = Path(file_path)
    
    # Read CSV or Excel
    try:
        df = pd.read_csv(p)
    except:
        df = pd.read_excel(p)
    
    df = df.copy()

    # 1. Sort by DateTime
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df = df.sort_values("DateTime").reset_index(drop=True)
        df["h"] = df["DateTime"].dt.hour.astype("Int64")

    # 2. Map Direction
    if "Direction" in df.columns:
        # Convert to numeric if possible, else map
        dir_num = pd.to_numeric(df["Direction"], errors="coerce")
        if dir_num.isna().mean() > 0.5:
            df["Direction_deg"] = df["Direction"].astype(str).str.strip().str.upper().map(COMPASS_MAP)
        else:
            df["Direction_deg"] = dir_num

    # 3. Clean Target
    if target_col in df.columns:
        if df[target_col].dtype == object:
            df[target_col] = df[target_col].astype(str).str.replace(",", ".").astype(float)

    # 4. Create Lag Feature (Prev_Speed)
    # This is the secret to high R2
    if target_col in df.columns:
        df["Prev_Speed"] = df[target_col].shift(1)
    
    # Drop NaN (First row has no history)
    df = df.dropna(subset=["Prev_Speed", "Direction_deg", target_col]).reset_index(drop=True)
    
    return df

def infer_features(df, target):
    """Selects numeric features automatically."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    bad_cols = EXCLUDE_COLS | {target}
    
    # Prioritize our specific features
    features = []
    priority = ["Prev_Speed", "Direction_deg", "h"]
    
    for p in priority:
        if p in num_cols and p not in bad_cols:
            features.append(p)
            
    # Add others if any
    for c in num_cols:
        if c not in features and c not in bad_cols:
            features.append(c)
            
    return features

def get_models_and_grids(random_state=42):
    """Returns the models and hyperparameter grids you specified."""
    models = {
        "DT": DecisionTreeRegressor(random_state=random_state),
        "RF": RandomForestRegressor(random_state=random_state, n_jobs=-1),
        "ANN": MLPRegressor(random_state=random_state, max_iter=1000, early_stopping=True)
    }
    grids = {
        "DT": {"model__max_depth": [5, 10, 20], "model__min_samples_split": [2, 5]},
        "RF": {"model__n_estimators": [100, 200], "model__max_depth": [10, 20]},
        "ANN": {"model__hidden_layer_sizes": [(64,), (64, 32)], "model__alpha": [1e-3]}
    }
    return models, grids

def main():
    target = "Speed"
    
    # 1. Prepare Data
    df_train = prepare_data(TRAIN_FILE, target)
    features = infer_features(df_train, target)
    
    print(f"[INFO] Features used: {features}")
    
    X = df_train[features]
    y = df_train[target]

    # 2. Setup Pipeline Components
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, features)],
        remainder="drop"
    )

    models, grids = get_models_and_grids()
    cv = TimeSeriesSplit(n_splits=5)
    
    outdir = Path(MODEL_SAVE_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    
    cv_summary = {}

    # 3. Train Loop
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
        
        gs = GridSearchCV(pipe, grids[name], cv=cv, scoring="r2", n_jobs=-1, refit=True, verbose=1)
        gs.fit(X, y)
        
        best = gs.best_estimator_
        joblib.dump(best, outdir / f"{name}.joblib")
        
        print(f"Saved: {outdir / f'{name}.joblib'}")
        print(f"Best CV R2: {gs.best_score_:.4f}")
        
        cv_summary[name] = {"best_params": gs.best_params_, "best_score_r2": float(gs.best_score_)}

    # Save metadata for the test script
    (outdir / "metadata.json").write_text(json.dumps({
        "target": target, "features": features
    }, indent=2), encoding="utf-8")
    
    print("\n[TRAIN] All models trained and saved.")

if __name__ == "__main__":
    main()