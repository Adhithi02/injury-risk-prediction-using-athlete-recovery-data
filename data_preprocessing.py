import pandas as pd
import numpy as np
import os
from datetime import timedelta
from typing import Union, List

def load_csvs_from_folder(folder_path):
    """Load and merge all CSVs in a folder by date.

    Handles different CSV formats:
    - Wellness: first column is dates (unnamed), rest are player columns
    - Training: first column is "Date", rest are player columns  
    - Injury: has "timestamp" column
    """
    merged_df = None
    if not os.path.isdir(folder_path):
        return None

    for file in os.listdir(folder_path):
        if not file.endswith(".csv"):
            continue

        path = os.path.join(folder_path, file)
        df = pd.read_csv(path)
        
        # Find date column
        date_col = None
        if "date" in df.columns:
            date_col = "date"
        elif "Date" in df.columns:
            date_col = "Date"
        elif "timestamp" in df.columns:
            date_col = "timestamp"
        elif df.columns[0].lower() in ["date", "timestamp"] or df.columns[0] == "":
            # First column is likely dates (unnamed)
            date_col = df.columns[0]
        
        if date_col is None:
            continue
            
        # Rename date column to standard "date"
        if date_col != "date":
            df = df.rename(columns={date_col: "date"})

        # Parse dates
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])  # drop bad date rows

        # For wellness/training files: aggregate player columns into single metric
        # For injury files: create binary injury indicator
        metric_name = os.path.splitext(file)[0]
        
        if metric_name == "injury":
            # Injury file: create binary injury indicator per date
            df["injury"] = 1  # All rows in injury file represent injuries
            df = df[["date", "injury"]].groupby("date").max().reset_index()
        else:
            # Wellness/training files: aggregate across players
            value_cols = [c for c in df.columns if c != "date"]
            if len(value_cols) > 0:
                # Take mean across all players for each date
                df[metric_name] = df[value_cols].mean(axis=1)
                df = df[["date", metric_name]]

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="date", how="outer")

    return merged_df

def build_master_dataset(base_path="data/raw/"):
    wellness = load_csvs_from_folder(os.path.join(base_path, "wellness"))
    training = load_csvs_from_folder(os.path.join(base_path, "training-load"))
    injury = load_csvs_from_folder(os.path.join(base_path, "injury"))

    # Merge on date, handling missing groups
    parts = [p for p in [wellness, training, injury] if p is not None]
    if not parts:
        return None

    df = parts[0]
    for p in parts[1:]:
        df = df.merge(p, on="date", how="outer")

    # Sort and set index
    df = df.sort_values("date").reset_index(drop=True)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling averages, lag features, and stress-recovery index.

    The function is defensive to missing columns; it only computes features
    for columns that exist in the dataframe.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Identify likely load and wellness columns
    candidate_load_cols = [
        "daily_load", "weekly_load", "strain", "monotony", "acwr", "ATL", "CTL28", "CTL42",
        "atl", "ctl28", "ctl42"
    ]
    load_cols = [c for c in candidate_load_cols if c in df.columns]

    # Rolling means for loads
    for c in load_cols:
        df[f"{c}_roll7"] = df[c].rolling(window=7, min_periods=3).mean()
        df[f"{c}_roll14"] = df[c].rolling(window=14, min_periods=5).mean()

    # Lag features for selected wellness metrics
    wellness_cols = [
        "fatigue", "stress", "soreness", "sleep_quality", "sleep_duration", "readiness", "mood"
    ]
    for c in wellness_cols:
        if c in df.columns:
            df[f"{c}_lag1"] = df[c].shift(1)
            df[f"{c}_lag7"] = df[c].shift(7)

    # Stress-recovery index: higher means worse balance
    # (stress + soreness) / (sleep_quality + readiness)
    if {"stress", "soreness"}.issubset(df.columns) and (
        {"sleep_quality", "readiness"}.issubset(df.columns)
    ):
        denom = (df["sleep_quality"].clip(lower=1e-6) + df["readiness"].clip(lower=1e-6))
        df["stress_recovery_index"] = (df["stress"].fillna(0) + df["soreness"].fillna(0)) / denom

    # Ensure strictly increasing dates (drop duplicates if any)
    df = df.drop_duplicates(subset=["date"]).reset_index(drop=True)

    return df


def create_labels(df: pd.DataFrame, injury_column_guess: Union[str, None] = None,
                  min_days_ahead: int = 10, max_days_ahead: int = 15) -> pd.DataFrame:
    """Create binary label = 1 if an injury occurs within [min,max] days ahead.

    - Tries to detect the injury indicator column if not provided.
    - Expects injury column to be 1 on dates with injury, 0 otherwise.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Guess injury column based on files naming
    injury_col = injury_column_guess
    if injury_col is None:
        candidates = [c for c in df.columns if c.lower().startswith("injury")]
        if candidates:
            injury_col = candidates[0]

    if injury_col is None or injury_col not in df.columns:
        # No injury information found; create label of zeros
        df["label_next_10_15"] = 0
        return df

    # Normalize injury column to 0/1
    injury_signal = df[injury_col].fillna(0)
    if not np.issubdtype(injury_signal.dtype, np.number):
        injury_signal = (injury_signal.astype(str).str.lower().isin(["1", "true", "yes", "y"]))
    injury_signal = (injury_signal > 0).astype(int)

    df["label_next_10_15"] = 0

    dates = pd.to_datetime(df["date"]).reset_index(drop=True)
    # Build a fast lookup: indices where injury==1
    injury_indices = list(np.where(injury_signal.values == 1)[0])

    # Two-pointer sweep to mark windows [i+min_days, i+max_days]
    # We operate on index differences assuming daily rows. If dates skip days,
    # we fallback to calendar day logic.
    is_daily = dates.diff().dropna().value_counts().idxmax() == pd.Timedelta(days=1)
    if is_daily:
        for current_idx in range(len(df)):
            start_idx = current_idx + min_days_ahead
            end_idx = min(current_idx + max_days_ahead, len(df) - 1)
            if start_idx > end_idx:
                continue
            # check if any injury index falls in [start_idx, end_idx]
            if any(i for i in injury_indices if start_idx <= i <= end_idx):
                df.at[current_idx, "label_next_10_15"] = 1
    else:
        for current_idx, current_date in enumerate(dates):
            start_date = current_date + timedelta(days=min_days_ahead)
            end_date = current_date + timedelta(days=max_days_ahead)
            # find if any injury date falls within the interval
            future_mask = (dates >= start_date) & (dates <= end_date)
            if int((injury_signal[future_mask] > 0).any()):
                df.at[current_idx, "label_next_10_15"] = 1

    return df


def build_and_save_processed(base_path: str = "data/raw/", out_dir: str = "data/processed/") -> Union[str, None]:
    """Build master dataset, engineer features, create labels, and save to CSV.

    Returns the output file path or None if building failed.
    """
    df = build_master_dataset(base_path=base_path)
    if df is None or df.empty:
        return None

    df = engineer_features(df)
    df = create_labels(df)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "master_dataset.csv")
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    path = build_and_save_processed()
    if path:
        print(f"Saved processed dataset to: {path}")
    else:
        print("Failed to build processed dataset. Check input files.")
