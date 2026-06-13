"""
src/data_preprocessing.py
--------------------------
PART 1: DATA PREPROCESSING

PURPOSE:
    This file handles everything related to loading, merging, cleaning,
    and preparing the raw Kaggle data for analysis. Think of it as the
    "kitchen prep" step before cooking — we make the data safe and usable.

HOW TO USE:
    from src.data_preprocessing import load_and_preprocess
    df = load_and_preprocess()

INPUTS:
    data/raw/station_hour.csv   — hourly pollutant readings per station
    data/raw/stations.csv       — station metadata (city, state, lat, lon)

OUTPUT:
    A clean pandas DataFrame with datetime features and geographic info.
    Also saves: data/processed/northern_india_clean.csv
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# The five northern Indian states we care about.
TARGET_STATES = [
    "Delhi",
    "Punjab",
    "Haryana",
    "Uttar Pradesh",
    "Rajasthan",
]

# All pollutant columns present in station_hour.csv
POLLUTANT_COLS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
                  "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI"]

# Paths — relative to the project root
RAW_STATION_HOUR = os.path.join("data", "raw", "station_hour.csv")
RAW_STATIONS     = os.path.join("data", "raw", "stations.csv")
PROCESSED_PATH   = os.path.join("data", "processed", "northern_india_clean.csv")


# ─────────────────────────────────────────────
# STEP 1: LOAD RAW FILES
# ─────────────────────────────────────────────

def load_raw_data(station_hour_path=RAW_STATION_HOUR,
                  stations_path=RAW_STATIONS):
    """
    Load the two CSVs from Kaggle into DataFrames.

    WHY TWO FILES?
        station_hour.csv has the numbers (pollution readings) but only
        a StationId.  stations.csv has the location info (city, state).
        We need to JOIN them so every row knows which city/state it came from.

    Returns
    -------
    station_hour : pd.DataFrame
    stations     : pd.DataFrame
    """
    print("[1/6] Loading raw CSVs …")

    station_hour = pd.read_csv(station_hour_path, low_memory=False)
    stations     = pd.read_csv(stations_path)

    print(f"      station_hour  → {station_hour.shape[0]:,} rows, "
          f"{station_hour.shape[1]} cols")
    print(f"      stations      → {stations.shape[0]:,} rows, "
          f"{stations.shape[1]} cols")
    return station_hour, stations


# ─────────────────────────────────────────────
# STEP 2: MERGE ON StationId
# ─────────────────────────────────────────────

def merge_datasets(station_hour, stations):
    """
    Left-join station_hour with stations on 'StationId'.

    WHY LEFT JOIN?
        We keep all readings even if a station happens to be missing from
        the stations file (we'll drop those later).
    """
    print("[2/6] Merging datasets on StationId …")

    # Standardise the join key name just in case
    if "StationId" not in stations.columns and "StationCode" in stations.columns:
        stations = stations.rename(columns={"StationCode": "StationId"})

    merged = pd.merge(station_hour, stations, on="StationId", how="left")
    print(f"      Merged shape  → {merged.shape[0]:,} rows, {merged.shape[1]} cols")
    return merged


# ─────────────────────────────────────────────
# STEP 3: FILTER FOR NORTHERN STATES
# ─────────────────────────────────────────────

def filter_northern_states(df, target_states=TARGET_STATES):
    """
    Keep only rows where 'State' is one of our five target states.

    WHY FILTER?
        The full dataset covers all of India (~150 M rows).  Filtering to
        five states makes computation fast and keeps the project focused.
    """
    print(f"[3/6] Filtering for states: {target_states} …")

    filtered = df[df["State"].isin(target_states)].copy()
    print(f"      Rows after filter → {filtered.shape[0]:,}")
    return filtered


# ─────────────────────────────────────────────
# STEP 4: DATETIME PARSING & FEATURE CREATION
# ─────────────────────────────────────────────

def parse_datetime_features(df):
    """
    Convert the 'Datetime' column from a string to a proper datetime object
    and extract useful time-based features.

    WHY DATETIME FEATURES?
        Machine learning models cannot understand "2019-01-15 08:00:00" as
        a time — they need numbers.  Extracting Year, Month, Hour, etc.
        lets the model learn patterns like "pollution spikes in winter mornings".
    """
    print("[4/6] Parsing datetime and extracting features …")

    # Parse — handle multiple possible column names
    dt_col = "Datetime" if "Datetime" in df.columns else "Date"
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.rename(columns={dt_col: "Datetime"})

    # Drop rows where parsing failed
    before = len(df)
    df = df.dropna(subset=["Datetime"])
    print(f"      Dropped {before - len(df):,} rows with unparseable datetimes")

    # Extract components
    df["Year"]       = df["Datetime"].dt.year
    df["Month"]      = df["Datetime"].dt.month
    df["Day"]        = df["Datetime"].dt.day
    df["Hour"]       = df["Datetime"].dt.hour
    df["DayOfWeek"]  = df["Datetime"].dt.dayofweek   # 0 = Monday
    df["IsWeekend"]  = df["DayOfWeek"].isin([5, 6]).astype(int)

    # Season (Northern Hemisphere convention adapted for India)
    # Winter: Dec–Feb  | Spring: Mar–May | Monsoon: Jun–Sep | Autumn: Oct–Nov
    season_map = {
        12: "Winter", 1: "Winter",  2: "Winter",
        3:  "Spring", 4: "Spring",  5: "Spring",
        6:  "Monsoon",7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
        10: "Autumn", 11: "Autumn"
    }
    df["Season"] = df["Month"].map(season_map)

    # Crop burning season — Punjab & Haryana paddy harvest: Oct–Nov
    df["CropBurning"] = (
        (df["Month"].isin([10, 11])) &
        (df["State"].isin(["Punjab", "Haryana"]))
    ).astype(int)

    print(f"      Date range: {df['Datetime'].min()} → {df['Datetime'].max()}")
    return df


# ─────────────────────────────────────────────
# STEP 5: HANDLE MISSING VALUES
# ─────────────────────────────────────────────

def handle_missing_values(df):
    """
    Impute or drop missing pollutant readings intelligently.

    STRATEGY:
        - For each pollutant, forward-fill within each (Station, Date) group.
          This respects temporal continuity — if the sensor was reading 45 µg/m³
          and then went offline, 45 is a better guess than the station average.
        - After forward-fill, backward-fill any remaining leading NaNs.
        - Drop rows where ALL pollutants are still missing (sensor was broken).
        - Clip negative values to 0 (sensors sometimes report small negatives
          due to calibration drift — physically impossible for concentrations).
    """
    print("[5/6] Handling missing values …")

    poll_cols_present = [c for c in POLLUTANT_COLS if c in df.columns and c != "AQI"]

    before_nulls = df[poll_cols_present].isna().sum().sum()

    # Forward-fill then backward-fill, grouped by station
    df = df.sort_values(["StationId", "Datetime"])
    df[poll_cols_present] = (
        df.groupby("StationId")[poll_cols_present]
          .transform(lambda x: x.ffill().bfill())
    )

    after_nulls = df[poll_cols_present].isna().sum().sum()
    print(f"      Nulls before fill: {before_nulls:,}  →  after fill: {after_nulls:,}")

    # Clip negatives
    df[poll_cols_present] = df[poll_cols_present].clip(lower=0)

    # Drop rows where core pollutants are still all missing
    core = ["PM2.5", "PM10", "SO2", "NOx"]
    core_present = [c for c in core if c in df.columns]
    before_drop = len(df)
    df = df.dropna(subset=core_present, how="all")
    print(f"      Rows dropped (all core nulls): {before_drop - len(df):,}")

    return df


# ─────────────────────────────────────────────
# STEP 6: REMOVE DUPLICATES & FINALISE
# ─────────────────────────────────────────────

def remove_duplicates_and_finalise(df):
    """
    Drop exact duplicate rows and reset the index.

    WHY REMOVE DUPLICATES?
        Sometimes sensors submit the same reading twice (network retry).
        Duplicates inflate counts and bias averages.
    """
    print("[6/6] Removing duplicates and finalising …")

    before = len(df)
    df = df.drop_duplicates(subset=["StationId", "Datetime"])
    print(f"      Duplicates removed: {before - len(df):,}")

    # Sort chronologically by station
    df = df.sort_values(["StationId", "Datetime"]).reset_index(drop=True)

    # Reorder columns for readability
    id_cols   = ["StationId", "StationName", "City", "State", "Latitude",
                 "Longitude", "Datetime", "Year", "Month", "Day", "Hour",
                 "DayOfWeek", "IsWeekend", "Season", "CropBurning"]
    id_cols   = [c for c in id_cols if c in df.columns]
    poll_present = [c for c in POLLUTANT_COLS if c in df.columns]
    other_cols = [c for c in df.columns if c not in id_cols + poll_present]
    df = df[id_cols + poll_present + other_cols]

    print(f"\n{'='*50}")
    print(f"  CLEAN DATAFRAME SUMMARY")
    print(f"{'='*50}")
    print(f"  Shape     : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  States    : {sorted(df['State'].unique())}")
    print(f"  Cities    : {df['City'].nunique()} unique cities")
    print(f"  Date range: {df['Datetime'].min().date()} → {df['Datetime'].max().date()}")
    print(f"{'='*50}\n")

    return df


# ─────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────

def load_and_preprocess(station_hour_path=RAW_STATION_HOUR,
                        stations_path=RAW_STATIONS,
                        save=True):
    """
    Run the complete preprocessing pipeline in order and optionally
    save the result to data/processed/northern_india_clean.csv.

    Parameters
    ----------
    station_hour_path : str
    stations_path     : str
    save              : bool  — write processed CSV to disk?

    Returns
    -------
    df : pd.DataFrame  — clean, analysis-ready dataframe
    """
    station_hour, stations = load_raw_data(station_hour_path, stations_path)
    df = merge_datasets(station_hour, stations)
    df = filter_northern_states(df)
    df = parse_datetime_features(df)
    df = handle_missing_values(df)
    df = remove_duplicates_and_finalise(df)

    if save:
        os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)
        print(f"[✓] Saved clean data → {PROCESSED_PATH}")

    return df


# ─────────────────────────────────────────────
# QUICK-RUN (python src/data_preprocessing.py)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = load_and_preprocess()
    print(df.head())
    print("\nDtypes:\n", df.dtypes)
    print("\nNull counts:\n", df.isnull().sum())
