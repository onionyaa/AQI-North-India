import pandas as pd
import numpy as np
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

TARGET_STATES = [
    "Delhi",
    "Punjab",
    "Haryana",
    "Uttar Pradesh",
    "Rajasthan",
]

POLLUTANT_COLS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
                  "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI"]

RAW_STATION_HOUR = os.path.join(RAW_DIR, "station_hour.csv")
RAW_STATIONS     = os.path.join(RAW_DIR, "stations.csv")
PROCESSED_PATH   = os.path.join(PROCESSED_DIR, "northern_india_clean.csv")



def load_raw_data(station_hour_path=RAW_STATION_HOUR,
                  stations_path=RAW_STATIONS):

    Returns
    -------
    station_hour : pd.DataFrame
    stations     : pd.DataFrame
    """
    print(" Loading raw CSVs …")

    station_hour = pd.read_csv(station_hour_path, low_memory=False)
    stations     = pd.read_csv(stations_path)

    print(f"      station_hour  → {station_hour.shape[0]:,} rows, "
          f"{station_hour.shape[1]} cols")
    print(f"      stations      → {stations.shape[0]:,} rows, "
          f"{stations.shape[1]} cols")
    return station_hour, stations


def merge_datasets(station_hour, stations):
    print(" Merging datasets on StationId …")

    if "StationId" not in stations.columns and "StationCode" in stations.columns:
        stations = stations.rename(columns={"StationCode": "StationId"})

    merged = pd.merge(station_hour, stations, on="StationId", how="left")
    print(f"      Merged shape  → {merged.shape[0]:,} rows, {merged.shape[1]} cols")
    return merged

def filter_northern_states(df, target_states=TARGET_STATES):
    
    print(f"[3/6] Filtering for states: {target_states} …")

    filtered = df[df["State"].isin(target_states)].copy()
    print(f"      Rows after filter → {filtered.shape[0]:,}")
    return filtered

def parse_datetime_features(df):
    print("[4/6] Parsing datetime and extracting features …")

    dt_col = "Datetime" if "Datetime" in df.columns else "Date"
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.rename(columns={dt_col: "Datetime"})

    before = len(df)
    df = df.dropna(subset=["Datetime"])
    print(f"      Dropped {before - len(df):,} rows with unparseable datetimes")

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

    print(f" Date range: {df['Datetime'].min()} → {df['Datetime'].max()}")
    return df

def handle_missing_values(df):
    """
    Impute or drop missing pollutant readings.

    """
    print(" Handling missing values …")

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

    df[poll_cols_present] = df[poll_cols_present].clip(lower=0)

    # Drop rows where core pollutants are still all missing
    core = ["PM2.5", "PM10", "SO2", "NOx"]
    core_present = [c for c in core if c in df.columns]
    before_drop = len(df)
    df = df.dropna(subset=core_present, how="all")
    print(f"      Rows dropped (all core nulls): {before_drop - len(df):,}")

    return df


def remove_duplicates_and_finalise(df):
    print(" Removing duplicates and finalising …")

    before = len(df)
    df = df.drop_duplicates(subset=["StationId", "Datetime"])
    print(f"      Duplicates removed: {before - len(df):,}")

    df = df.sort_values(["StationId", "Datetime"]).reset_index(drop=True)

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

def load_and_preprocess(station_hour_path=RAW_STATION_HOUR,
                        stations_path=RAW_STATIONS,
                        save=True):
   
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

if __name__ == "__main__":
    df = load_and_preprocess()
    print(df.head())
    print("\nDtypes:\n", df.dtypes)
    print("\nNull counts:\n", df.isnull().sum())
