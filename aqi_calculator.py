"""
src/aqi_calculator.py
----------------------
PART 2: AQI CALCULATION — CPCB METHODOLOGY

PURPOSE:
    Implements the official Central Pollution Control Board (India) formula
    for computing the Air Quality Index from raw pollutant concentrations.

HOW THE CPCB AQI FORMULA WORKS (plain English):
    1. For each pollutant, take the appropriate rolling average/maximum.
    2. Find which "breakpoint range" that average falls into (a lookup table
       defined by CPCB with 6 ranges, each with min/max concentration and
       min/max AQI values).
    3. Apply linear interpolation within that range to get a "sub-index".
    4. The FINAL AQI = the MAXIMUM sub-index across all pollutants.
       (The worst pollutant sets the AQI — not the average.)

REFERENCE:
    CPCB National Air Quality Index (2014 technical document)
    https://cpcb.nic.in/displaypdf.php?id=aqi/AQI_calculation_sheet.pdf

HOW TO USE:
    from src.aqi_calculator import calculate_aqi
    df = calculate_aqi(df)   # adds sub-index + AQI columns in place
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# BREAKPOINT TABLES  (concentration → sub-index mapping)
# Each row: [C_low, C_high, I_low, I_high]
#   C_low / C_high  = concentration breakpoints (µg/m³ or mg/m³)
#   I_low / I_high  = AQI sub-index breakpoints
# ─────────────────────────────────────────────────────────────

# PM2.5  (µg/m³, 24-hour average)
PM25_BREAKPOINTS = [
    [0,    30,    0,   50],
    [30,   60,   51,  100],
    [60,   90,  101,  200],
    [90,  120,  201,  300],
    [120, 250,  301,  400],
    [250, 500,  401,  500],
]

# PM10  (µg/m³, 24-hour average)
PM10_BREAKPOINTS = [
    [0,    50,    0,   50],
    [50,  100,   51,  100],
    [100, 250,  101,  200],
    [250, 350,  201,  300],
    [350, 430,  301,  400],
    [430, 600,  401,  500],
]

# SO2  (µg/m³, 24-hour average)
SO2_BREAKPOINTS = [
    [0,   40,    0,   50],
    [40,  80,   51,  100],
    [80,  380, 101,  200],
    [380, 800, 201,  300],
    [800, 1600,301,  400],
    [1600,2100,401,  500],
]

# NOx  (µg/m³, 24-hour average)
NOX_BREAKPOINTS = [
    [0,   40,    0,   50],
    [40,  80,   51,  100],
    [80,  180, 101,  200],
    [180, 280, 201,  300],
    [280, 400, 301,  400],
    [400, 800, 401,  500],
]

# NH3  (µg/m³, 24-hour average)
NH3_BREAKPOINTS = [
    [0,   200,   0,   50],
    [200, 400,  51,  100],
    [400, 800, 101,  200],
    [800,1200, 201,  300],
    [1200,1800,301,  400],
    [1800,2400,401,  500],
]

# CO  (mg/m³, 8-hour rolling maximum)
CO_BREAKPOINTS = [
    [0,  1.0,   0,   50],
    [1,  2.0,  51,  100],
    [2,  10.0,101,  200],
    [10, 17.0,201,  300],
    [17, 34.0,301,  400],
    [34, 46.0,401,  500],
]

# O3  (µg/m³, 8-hour rolling maximum)
O3_BREAKPOINTS = [
    [0,   50,    0,   50],
    [50,  100,  51,  100],
    [100, 168, 101,  200],
    [168, 208, 201,  300],
    [208, 748, 301,  400],
    [748, 987, 401,  500],
]


# ─────────────────────────────────────────────────────────────
# CORE INTERPOLATION FUNCTION
# ─────────────────────────────────────────────────────────────

def _sub_index(concentration, breakpoints):
    """
    Given a concentration value and a breakpoint table, compute the sub-index
    using CPCB linear interpolation formula:

        Ip = [(IHi - ILo) / (BPHi - BPLo)] × (Cp - BPLo) + ILo

    where:
        Ip   = sub-index for pollutant p
        Cp   = truncated concentration of pollutant p
        BPHi = concentration breakpoint ≥ Cp
        BPLo = concentration breakpoint ≤ Cp
        IHi  = AQI value corresponding to BPHi
        ILo  = AQI value corresponding to BPLo

    Parameters
    ----------
    concentration : float or np.ndarray
    breakpoints   : list of [C_low, C_high, I_low, I_high]

    Returns
    -------
    float sub-index (NaN if concentration is NaN or out of range)
    """
    if pd.isna(concentration):
        return np.nan

    for (c_lo, c_hi, i_lo, i_hi) in breakpoints:
        if c_lo <= concentration <= c_hi:
            # Linear interpolation
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo

    # Above max breakpoint — cap at 500 (CPCB convention for "severe+")
    if concentration > breakpoints[-1][1]:
        return 500.0

    return np.nan   # Below minimum (shouldn't happen after clipping)


# Vectorised version so it runs fast on a whole column
_sub_index_vec = np.vectorize(_sub_index, excluded=["breakpoints"])


# ─────────────────────────────────────────────────────────────
# ROLLING AVERAGES  (applied per station, preserving order)
# ─────────────────────────────────────────────────────────────

def _rolling_per_station(df, col, window, agg="mean"):
    """
    Compute a rolling statistic per station (so we don't bleed data
    across stations that happen to be adjacent in the dataframe).

    Parameters
    ----------
    df     : pd.DataFrame  (must be sorted by [StationId, Datetime])
    col    : str           column name
    window : int           number of hours
    agg    : "mean" or "max"

    Returns
    -------
    pd.Series aligned with df.index
    """
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    result = (
        df.groupby("StationId")[col]
          .transform(
              lambda x: x.rolling(window=window, min_periods=int(window * 0.5))
                         .mean()
              if agg == "mean"
              else x.rolling(window=window, min_periods=int(window * 0.5)).max()
          )
    )
    return result


# ─────────────────────────────────────────────────────────────
# MAIN AQI CALCULATION FUNCTION
# ─────────────────────────────────────────────────────────────

def calculate_aqi(df):
    """
    Full CPCB AQI pipeline applied to a DataFrame.

    Steps
    -----
    1. Compute rolling averages (24-h for most, 8-h for CO/O3)
    2. Calculate sub-index for each pollutant
    3. Final AQI = max sub-index across all pollutants
    4. Assign AQI category (Good → Severe)
    5. Drop rows where we could not compute any sub-index

    Parameters
    ----------
    df : pd.DataFrame  — must contain StationId, Datetime, and pollutant cols

    Returns
    -------
    df : pd.DataFrame  — original df + sub-index columns + AQI + AQI_Category
    """
    print("[AQI] Starting CPCB AQI calculation …")
    df = df.sort_values(["StationId", "Datetime"]).copy()

    # ── 1. Rolling averages ────────────────────────────────────
    print("      Computing rolling averages …")

    df["PM25_24h"]  = _rolling_per_station(df, "PM2.5", 24, "mean")
    df["PM10_24h"]  = _rolling_per_station(df, "PM10",  24, "mean")
    df["SO2_24h"]   = _rolling_per_station(df, "SO2",   24, "mean")
    df["NOx_24h"]   = _rolling_per_station(df, "NOx",   24, "mean")
    df["NH3_24h"]   = _rolling_per_station(df, "NH3",   24, "mean")
    df["CO_8h"]     = _rolling_per_station(df, "CO",    8,  "max")
    df["O3_8h"]     = _rolling_per_station(df, "O3",    8,  "max")

    # ── 2. Sub-indices ─────────────────────────────────────────
    print("      Computing pollutant sub-indices …")

    df["SI_PM25"] = _sub_index_vec(df["PM25_24h"].values,  breakpoints=PM25_BREAKPOINTS)
    df["SI_PM10"] = _sub_index_vec(df["PM10_24h"].values,  breakpoints=PM10_BREAKPOINTS)
    df["SI_SO2"]  = _sub_index_vec(df["SO2_24h"].values,   breakpoints=SO2_BREAKPOINTS)
    df["SI_NOx"]  = _sub_index_vec(df["NOx_24h"].values,   breakpoints=NOX_BREAKPOINTS)
    df["SI_NH3"]  = _sub_index_vec(df["NH3_24h"].values,   breakpoints=NH3_BREAKPOINTS)
    df["SI_CO"]   = _sub_index_vec(df["CO_8h"].values,     breakpoints=CO_BREAKPOINTS)
    df["SI_O3"]   = _sub_index_vec(df["O3_8h"].values,     breakpoints=O3_BREAKPOINTS)

    SI_COLS = ["SI_PM25", "SI_PM10", "SI_SO2", "SI_NOx", "SI_NH3", "SI_CO", "SI_O3"]

    # ── 3. Final AQI = maximum sub-index ──────────────────────
    print("      Aggregating to final AQI …")

    df["AQI_Calculated"] = df[SI_COLS].max(axis=1)

    # If the dataset already has an 'AQI' column from Kaggle, keep both.
    # Use calculated where Kaggle AQI is missing.
    if "AQI" in df.columns:
        df["AQI_Final"] = df["AQI"].combine_first(df["AQI_Calculated"])
    else:
        df["AQI_Final"] = df["AQI_Calculated"]

    # Round to 2 decimal places
    df["AQI_Final"] = df["AQI_Final"].round(2)

    # ── 4. AQI Category ───────────────────────────────────────
    print("      Assigning AQI categories …")
    df["AQI_Category"] = pd.cut(
        df["AQI_Final"],
        bins=      [0,  50, 100, 200, 300, 400, 500],
        labels=    ["Good", "Satisfactory", "Moderate", "Poor",
                    "Very Poor", "Severe"],
        include_lowest=True,
    )

    # ── 5. Drop rows with no computable AQI ───────────────────
    before = len(df)
    df = df.dropna(subset=["AQI_Final"])
    print(f"      Rows dropped (no AQI computable): {before - len(df):,}")

    # ── Summary ───────────────────────────────────────────────
    cat_counts = df["AQI_Category"].value_counts().sort_index()
    print(f"\n{'='*50}")
    print("  AQI CALCULATION COMPLETE")
    print(f"{'='*50}")
    print(f"  Rows with AQI  : {len(df):,}")
    print(f"  AQI range      : {df['AQI_Final'].min():.1f} – {df['AQI_Final'].max():.1f}")
    print(f"  Mean AQI       : {df['AQI_Final'].mean():.1f}")
    print("\n  Category distribution:")
    for cat, cnt in cat_counts.items():
        pct = 100 * cnt / len(df)
        print(f"    {cat:<14} : {cnt:>8,} ({pct:.1f}%)")
    print(f"{'='*50}\n")

    return df


# ─────────────────────────────────────────────────────────────
# DOMINANT POLLUTANT HELPER
# ─────────────────────────────────────────────────────────────

def get_dominant_pollutant(df):
    """
    Add a 'DominantPollutant' column showing which pollutant drove the AQI.
    Useful for policy analysis — tells you WHAT to target, not just the number.
    """
    SI_COLS = ["SI_PM25", "SI_PM10", "SI_SO2", "SI_NOx", "SI_NH3", "SI_CO", "SI_O3"]
    name_map = {
        "SI_PM25": "PM2.5", "SI_PM10": "PM10", "SI_SO2": "SO2",
        "SI_NOx": "NOx",   "SI_NH3": "NH3",   "SI_CO": "CO", "SI_O3": "O3"
    }
    present = [c for c in SI_COLS if c in df.columns]
    df["DominantPollutant"] = df[present].idxmax(axis=1).map(name_map)
    return df


# ─────────────────────────────────────────────────────────────
# QUICK-RUN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_preprocessing import load_and_preprocess

    df = load_and_preprocess(save=False)
    df = calculate_aqi(df)
    df = get_dominant_pollutant(df)
    print(df[["City", "State", "Datetime", "AQI_Final", "AQI_Category",
              "DominantPollutant"]].head(20))
