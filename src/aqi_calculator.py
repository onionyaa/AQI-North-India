import numpy as np
import pandas as pd

# PM2.5 in µg/m³, 24h avg
PM25_BREAKPOINTS = [
    [0, 30, 0, 50],
    [30, 60, 51, 100],
    [60, 90, 101, 200],
    [90, 120, 201, 300],
    [120, 250, 301, 400],
    [250, 500, 401, 500],
]

# PM10 in µg/m³, 24h avg
PM10_BREAKPOINTS = [
    [0, 50, 0, 50],
    [50, 100, 51, 100],
    [100, 250, 101, 200],
    [250, 350, 201, 300],
    [350, 430, 301, 400],
    [430, 600, 401, 500],
]

# SO2 in µg/m³, 24h avg
SO2_BREAKPOINTS = [
    [0, 40, 0, 50],
    [40, 80, 51, 100],
    [80, 380, 101, 200],
    [380, 800, 201, 300],
    [800, 1600, 301, 400],
    [1600, 2100, 401, 500],
]

# NOx in µg/m³, 24h avg
NOX_BREAKPOINTS = [
    [0, 40, 0, 50],
    [40, 80, 51, 100],
    [80, 180, 101, 200],
    [180, 280, 201, 300],
    [280, 400, 301, 400],
    [400, 800, 401, 500],
]

# NH3 in µg/m³, 24h avg
NH3_BREAKPOINTS = [
    [0, 200, 0, 50],
    [200, 400, 51, 100],
    [400, 800, 101, 200],
    [800, 1200, 201, 300],
    [1200, 1800, 301, 400],
    [1800, 2400, 401, 500],
]

# CO in mg/m³, 8h rolling max (note: mg not µg)
CO_BREAKPOINTS = [
    [0, 1.0, 0, 50],
    [1, 2.0, 51, 100],
    [2, 10.0, 101, 200],
    [10, 17.0, 201, 300],
    [17, 34.0, 301, 400],
    [34, 46.0, 401, 500],
]

# O3 in µg/m³, 8h rolling max
O3_BREAKPOINTS = [
    [0, 50, 0, 50],
    [50, 100, 51, 100],
    [100, 168, 101, 200],
    [168, 208, 201, 300],
    [208, 748, 301, 400],
    [748, 987, 401, 500],
]


def _sub_index(concentration, breakpoints):
    """Linear interpolation within a CPCB breakpoint range.

    Returns NaN if the value is NaN or below the lowest breakpoint.
    Caps at 500 if above the highest (CPCB "severe+" convention).
    """
    if pd.isna(concentration):
        return np.nan

    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= concentration <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo

    if concentration > breakpoints[-1][1]:
        return 500.0

    return np.nan


# vectorise so we can pass a whole numpy array at once
_sub_index_vec = np.vectorize(_sub_index, excluded=["breakpoints"])


def _rolling_per_station(df, col, window, agg="mean"):
    """Rolling mean or max, computed separately per station.

    Grouping by StationId prevents readings from one station bleeding
    into the next when they happen to be adjacent in the dataframe.
    min_periods is set to 50% of the window so we still get values
    near the start of each station's time series.
    """
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    min_p = int(window * 0.5)

    return df.groupby("StationId")[col].transform(
        lambda x: (
            x.rolling(window, min_periods=min_p).mean()
            if agg == "mean"
            else x.rolling(window, min_periods=min_p).max()
        )
    )


def calculate_aqi(df):
    
    print("Computing AQI...")
    df = df.sort_values(["StationId", "Datetime"]).copy()

    # rolling averages — 24h for particulates/gases, 8h for CO and O3
    df["PM25_24h"] = _rolling_per_station(df, "PM2.5", 24, "mean")
    df["PM10_24h"] = _rolling_per_station(df, "PM10", 24, "mean")
    df["SO2_24h"]  = _rolling_per_station(df, "SO2", 24, "mean")
    df["NOx_24h"]  = _rolling_per_station(df, "NOx", 24, "mean")
    df["NH3_24h"]  = _rolling_per_station(df, "NH3", 24, "mean")
    df["CO_8h"]    = _rolling_per_station(df, "CO", 8, "max")
    df["O3_8h"]    = _rolling_per_station(df, "O3", 8, "max")

    # sub-indices
    df["SI_PM25"] = _sub_index_vec(df["PM25_24h"].values, breakpoints=PM25_BREAKPOINTS)
    df["SI_PM10"] = _sub_index_vec(df["PM10_24h"].values, breakpoints=PM10_BREAKPOINTS)
    df["SI_SO2"]  = _sub_index_vec(df["SO2_24h"].values,  breakpoints=SO2_BREAKPOINTS)
    df["SI_NOx"]  = _sub_index_vec(df["NOx_24h"].values,  breakpoints=NOX_BREAKPOINTS)
    df["SI_NH3"]  = _sub_index_vec(df["NH3_24h"].values,  breakpoints=NH3_BREAKPOINTS)
    df["SI_CO"]   = _sub_index_vec(df["CO_8h"].values,    breakpoints=CO_BREAKPOINTS)
    df["SI_O3"]   = _sub_index_vec(df["O3_8h"].values,    breakpoints=O3_BREAKPOINTS)

    SI_COLS = ["SI_PM25", "SI_PM10", "SI_SO2", "SI_NOx", "SI_NH3", "SI_CO", "SI_O3"]

    df["AQI_Calculated"] = df[SI_COLS].max(axis=1)

    if "AQI" in df.columns:
        df["AQI_Final"] = df["AQI"].combine_first(df["AQI_Calculated"])
    else:
        df["AQI_Final"] = df["AQI_Calculated"]

    df["AQI_Final"] = df["AQI_Final"].round(2)

    df["AQI_Category"] = pd.cut(
        df["AQI_Final"],
        bins=[0, 50, 100, 200, 300, 400, 500],
        labels=["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"],
        include_lowest=True,
    )

    before = len(df)
    df = df.dropna(subset=["AQI_Final"])
    dropped = before - len(df)
    if dropped:
        print(f"  dropped {dropped:,} rows with no computable AQI")

    print(f"  done — {len(df):,} rows, AQI {df['AQI_Final'].min():.0f}-{df['AQI_Final'].max():.0f}, mean {df['AQI_Final'].mean():.0f}")
    return df


def get_dominant_pollutant(df):
    
    SI_COLS = ["SI_PM25", "SI_PM10", "SI_SO2", "SI_NOx", "SI_NH3", "SI_CO", "SI_O3"]
    name_map = {
        "SI_PM25": "PM2.5", "SI_PM10": "PM10", "SI_SO2": "SO2",
        "SI_NOx": "NOx", "SI_NH3": "NH3", "SI_CO": "CO", "SI_O3": "O3",
    }
    present = [c for c in SI_COLS if c in df.columns]
    df["DominantPollutant"] = df[present].idxmax(axis=1).map(name_map)
    return df

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_preprocessing import load_and_preprocess

    df = load_and_preprocess(save=False)
    df = calculate_aqi(df)
    df = get_dominant_pollutant(df)
    print(df[["City", "State", "Datetime", "AQI_Final", "AQI_Category", "DominantPollutant"]].head(20))
