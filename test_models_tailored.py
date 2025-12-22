#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST SCRIPT (UPDATED FOR LAG FEATURE)
-------------------------------------
- Loads models trained with 'Prev_Speed'.
- Generates 'Prev_Speed' internally for the test file (No Leakage).
- Produces daily and combined plots with high accuracy.
"""
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

DEFAULT_TEST = r"C:\Users\erenf\feke_20.csv"
MODEL_DIR = "./models" 

COMPASS_MAP = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5
}

def _read_csv_smart(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        try:
            return pd.read_csv(p, sep=";")
        except Exception:
            import csv
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                sample = f.read(8192)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                    delim = dialect.delimiter
                except Exception:
                    delim = ","
            return pd.read_csv(p, sep=delim, engine="python")

def read_table(path: str, sheet: str | None = None) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in [".xlsx", ".xlsm", ".xls"]:
        try:
            import openpyxl
        except Exception as e:
            raise SystemExit("Excel okunamadı: openpyxl kurun: pip install openpyxl\nAyrıntı: %s" % e)
        engine = "openpyxl" if suf in [".xlsx", ".xlsm"] else None
        return pd.read_excel(p, sheet_name=(sheet if sheet else 0), engine=engine)
    else:
        return _read_csv_smart(p)

def add_time_and_direction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Ensure DateTime is sorted (Crucial for Lag)
    if "DateTime" in df.columns:
        try:
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            df = df.sort_values("DateTime").reset_index(drop=True) # SORT ADDED
            df["h"] = df["DateTime"].dt.hour.astype("Int64")
        except Exception:
            pass

    # 2. Map Direction
    if "Direction" in df.columns:
        dir_num = pd.to_numeric(df["Direction"], errors="coerce")
        if dir_num.isna().mean() > 0.5:
            dir_str = df["Direction"].astype(str).str.strip().str.upper()
            df["Direction_deg"] = dir_str.map(COMPASS_MAP)
        else:
            df["Direction_deg"] = dir_num
            
    return df

def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"MSE": float(mse), "RMSE": rmse, "R2_ACC": r2, "MAE": mae}

# ---- Plot Formatting Helpers ----
def format_daily_hour_axis(ax):
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(24)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 30]))
    ax.grid(True, which="major", linewidth=1.0)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.4)
    ax.tick_params(axis="x", which="major", labelsize=9)
    ax.tick_params(axis="x", which="minor", labelsize=0)
    plt.xticks(rotation=0)

def format_hourly_axis_span_days(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="x", which="major", labelsize=9, pad=8)
    ax.grid(True, which="major", linewidth=1.1)
    plt.xticks(rotation=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=DEFAULT_TEST, help="Test file")
    parser.add_argument("--sheet", default=None, help="Excel sheet")
    parser.add_argument("--models", default=MODEL_DIR, help="Models directory")
    parser.add_argument("--target", default=None, help="Target column")
    parser.add_argument("--timecol", default=None, help="Time column")
    parser.add_argument("--outdir", default="./plots", help="Output directory")
    parser.add_argument("--daily", action="store_true", help="Only daily plots")
    args = parser.parse_args()

    # 1. Read Data
    print(f"[INFO] Reading test file: {args.test}")
    df = read_table(args.test, args.sheet)
    df = add_time_and_direction(df)

    # 2. Load Metadata
    meta_path = Path(args.models) / "metadata.json"
    target = args.target
    features = None
    
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        target = target or meta.get("target")
        features = meta.get("features")
        print(f"[INFO] Metadata loaded. Target: {target}, Features: {features}")

    if not target:
        target = "Speed"
    
    if target not in df.columns:
        raise SystemExit(f"Target '{target}' not found in test file.")

    # 3. Clean Target & CREATE LAG FEATURE
    # This is the critical step to match the training logic
    if df[target].dtype == object:
        df[target] = df[target].astype(str).str.replace(",", ".").astype(float)
    
    # Generate Prev_Speed (Lag)
    df["Prev_Speed"] = df[target].shift(1)
    
    # Drop rows where Prev_Speed is NaN (First row)
    initial_len = len(df)
    df = df.dropna(subset=["Prev_Speed"]).reset_index(drop=True)
    
    if len(df) < initial_len:
        print(f"[INFO] Dropped {initial_len - len(df)} row(s) (Start of file, no history).")

    # 4. Feature Selection
    if not features:
        # Fallback if metadata missing
        features = ["h", "Direction_deg", "Prev_Speed"]
    
    # Ensure all features exist
    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        raise SystemExit(f"Missing features in test data: {missing_feats}")

    X_test = df[features]
    y_test = df[target].values

    # 5. X-Axis Setup
    timecol = args.timecol or ("DateTime" if "DateTime" in df.columns else None)
    if timecol and timecol in df.columns:
        try:
            x_axis = pd.to_datetime(df[timecol])
            x_label = "DateTime"
        except:
            x_axis = np.arange(len(df))
            x_label = "index"
    else:
        x_axis = np.arange(len(df))
        x_label = "index"

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    per_day_root = outdir / "per_day"; per_day_root.mkdir(parents=True, exist_ok=True)

    results = []
    preds_for_combo = {}

    # 6. Prediction Loop
    model_files = list(Path(args.models).glob("*.joblib"))
    if not model_files:
        print("[WARNING] No .joblib models found in", args.models)
    
    for model_file in model_files:
        name = model_file.stem
        print(f"Testing model: {name}")
        model = joblib.load(model_file)
        
        try:
            y_pred = model.predict(X_test).astype(float)
        except Exception as e:
            print(f"Error predicting with {name}: {e}")
            continue

        m = metrics(y_test, y_pred)
        m["Model"] = name
        print(f"  -> R2: {m['R2_ACC']:.4f}")
        results.append(m)
        preds_for_combo[name] = y_pred

        # --- General Plot (Full Timeline) ---
        if not args.daily:
            plt.figure(figsize=(12, 5))
            plt.plot(x_axis, y_test, label="Actual", color='black', alpha=0.7)
            plt.plot(x_axis, y_pred, label=f"Pred - {name}", color='red', alpha=0.8)
            
            if x_label == "DateTime":
                format_hourly_axis_span_days(plt.gca())
            
            plt.title(f"Actual vs Prediction - {name}")
            plt.xlabel("Time")
            plt.ylabel(target)
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / f"{name}_prediction.png", dpi=150)
            plt.close()

        # --- Daily Plots ---
        if x_label == "DateTime":
            day_index = pd.to_datetime(x_axis).dt.date
            unique_days = pd.unique(day_index)
            day_outdir = per_day_root / name
            day_outdir.mkdir(parents=True, exist_ok=True)

            for d in unique_days:
                mask = (day_index == d)
                if mask.sum() == 0: continue
                
                xd = x_axis[mask]
                yt = y_test[mask]
                yp = y_pred[mask]

                plt.figure(figsize=(12, 5))
                plt.plot(xd, yt, label="Actual", marker='.', markersize=5)
                plt.plot(xd, yp, label=f"Pred - {name}", marker='.', markersize=5)
                format_daily_hour_axis(plt.gca())
                plt.title(f"{name} — {d} (Hourly)")
                plt.xlabel("Hour")
                plt.ylabel(target)
                plt.legend()
                plt.tight_layout()
                plt.savefig(day_outdir / f"{name}_{d.isoformat()}.png", dpi=150)
                plt.close()

    # 7. Summary & Combined Plot
    if results:
        results_df = pd.DataFrame(results)[["Model", "RMSE", "MSE", "R2_ACC", "MAE"]].sort_values(by="RMSE")
        print("\n--- Final Results ---")
        print(results_df)
        results_df.to_csv(outdir / "error_table.csv", index=False)

    if preds_for_combo and not args.daily:
        plt.figure(figsize=(14, 6))
        plt.plot(x_axis, y_test, label="Actual", color="black", linewidth=2.5, alpha=0.7)
        
        colors = ["red", "blue", "green", "orange", "purple"]
        for i, (name, yp) in enumerate(preds_for_combo.items()):
            c = colors[i % len(colors)]
            plt.plot(x_axis, yp, label=f"{name}", color=c, linestyle="--", linewidth=1.5)
            
        if x_label == "DateTime":
            format_hourly_axis_span_days(plt.gca())
            
        plt.title("Combined Model Comparison")
        plt.xlabel("Time")
        plt.ylabel(target)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "combined_comparison.png", dpi=150)
        plt.close()

    print(f"\nDone. Results saved to: {outdir}")

if __name__ == "__main__":
    main()