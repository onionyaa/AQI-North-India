"""
src/feature_engineering.py
----------------------------
PART 4: FEATURE ENGINEERING

PURPOSE:
    Take the clean AQI dataframe and engineer features that will help
    machine learning models predict future AQI values.

WHAT IS FEATURE ENGINEERING?
    Raw data often doesn't directly tell the model what it needs to know.
    For example, a model can't learn "yesterday's AQI predicts today's"
    unless we explicitly CREATE a column called "AQI_Lag1" that contains
    yesterday's value.  Feature engineering bridges that gap.

HOW TO USE:
    from src.feature_engineering import engineer_features
    df_feat = engineer_features(df)
"""

import pandas as pd
import numpy as np


def engineer_features(df):
    """
    Create all engineered features.  The input df must already contain
    AQI_Final, Datetime, State, and basic time columns from preprocessing.

    Returns
    -------
    df : pd.DataFrame  — original + new feature columns
    """
    print("[FE] Engineering features …")
    df = df.sort_values(["StationId", "Datetime"]).copy()

    # We group by StationId so that lag/rolling values don't "leak"
    # across stations.  Station A's AQI yesterday should not become
    # Station B's lag feature — they are physically separate sensors.
    g = df.groupby("StationId")["AQI_Final"]

    # ── LAG FEATURES ──────────────────────────────────────────────
    # WHY LAGS?
    #   Air quality has strong autocorrelation — today's AQI is heavily
    #   influenced by recent readings.  Pollution events (inversions,
    #   dust storms, crop fires) last for hours to days.  A lag-1 feature
    #   gives the model "memory" of what happened one hour ago.

    df["AQI_Lag1"]  = g.shift(1)   # 1 hour ago
    df["AQI_Lag3"]  = g.shift(3)   # 3 hours ago
    df["AQI_Lag6"]  = g.shift(6)   # 6 hours ago
    df["AQI_Lag24"] = g.shift(24)  # same hour yesterday

    # ── ROLLING AVERAGES ──────────────────────────────────────────
    # WHY ROLLING?
    #   A single lag is noisy (one bad reading).  A rolling average
    #   smooths out sensor spikes and captures the "background" pollution
    #   level over the past N hours.  The 24h rolling average essentially
    #   captures the day's pollution regime.

    df["AQI_Roll3"]  = g.transform(
        lambda x: x.rolling(3,  min_periods=2).mean()
    )
    df["AQI_Roll6"]  = g.transform(
        lambda x: x.rolling(6,  min_periods=3).mean()
    )
    df["AQI_Roll24"] = g.transform(
        lambda x: x.rolling(24, min_periods=12).mean()
    )

    # ── AQI RATE OF CHANGE ────────────────────────────────────────
    # WHY?
    #   Is AQI going up or down right now?  A rising AQI 3 hours ago
    #   is a stronger predictor of a high AQI now than a steady one.
    df["AQI_Delta1"]  = df["AQI_Final"] - df["AQI_Lag1"]
    df["AQI_Delta6"]  = df["AQI_Final"] - df["AQI_Lag6"]

    # ── TIME FEATURES ─────────────────────────────────────────────
    # WHY MONTH?
    #   Season proxy.  December in Delhi is categorically different
    #   from July — the model needs to know which month it is.
    if "Month" not in df.columns:
        df["Month"] = df["Datetime"].dt.month

    # WHY DAY OF WEEK?
    #   Traffic patterns (and therefore NOx/CO) differ between weekdays
    #   and weekends.  Industrial facilities often have maintenance
    #   shutdowns on Sundays, reducing SO2 emissions.
    if "DayOfWeek" not in df.columns:
        df["DayOfWeek"] = df["Datetime"].dt.dayofweek

    # WHY HOUR?
    #   Rush-hour (8 AM, 6 PM) spike is a real and reproducible pattern.
    #   Overnight (2–5 AM) often has the lowest AQI if no inversion present.
    if "Hour" not in df.columns:
        df["Hour"] = df["Datetime"].dt.hour

    # WHY SEASON (encoded as integer)?
    #   Winter pollution regime is fundamentally different from monsoon.
    #   Encoding as an integer lets tree models split on season cleanly.
    if "Season" not in df.columns:
        season_map = {12:0, 1:0, 2:0,   # Winter = 0
                      3:1, 4:1, 5:1,    # Spring = 1
                      6:2, 7:2, 8:2, 9:2,  # Monsoon = 2
                      10:3, 11:3}       # Autumn = 3
        df["Season_Enc"] = df["Month"].map(season_map)
    else:
        season_enc = {"Winter":0, "Spring":1, "Monsoon":2, "Autumn":3}
        df["Season_Enc"] = df["Season"].map(season_enc)

    # WHY WEEKEND INDICATOR?
    #   Binary: 1 if Saturday/Sunday, 0 otherwise.  Simple but powerful
    #   for capturing reduced industrial and traffic activity.
    if "IsWeekend" not in df.columns:
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    # WHY CROP BURNING?
    #   Hard binary variable: 1 if Oct–Nov AND state is Punjab/Haryana.
    #   This is the single largest episodic emission source in Northern India.
    if "CropBurning" not in df.columns:
        df["CropBurning"] = (
            (df["Month"].isin([10, 11])) &
            (df["State"].isin(["Punjab", "Haryana"]))
        ).astype(int)

    # ── CYCLICAL ENCODING (Month and Hour) ────────────────────────
    # WHY CYCLICAL?
    #   January (month 1) and December (month 12) are adjacent in time,
    #   but numerically they look far apart.  Encoding with sin/cos wraps
    #   the calendar into a circle so the model understands proximity.
    df["Month_Sin"]  = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"]  = np.cos(2 * np.pi * df["Month"] / 12)
    df["Hour_Sin"]   = np.sin(2 * np.pi * df["Hour"]  / 24)
    df["Hour_Cos"]   = np.cos(2 * np.pi * df["Hour"]  / 24)
    df["DoW_Sin"]    = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DoW_Cos"]    = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    # ── STATE ONE-HOT ENCODING ────────────────────────────────────
    # WHY ONE-HOT?
    #   "State" is a categorical variable.  Linear regression and XGBoost
    #   need numeric inputs.  One-hot creates a binary column per state.
    state_dummies = pd.get_dummies(df["State"], prefix="State", dtype=int)
    df = pd.concat([df, state_dummies], axis=1)

    # ── DROP ROWS WHERE CRITICAL LAG FEATURES ARE STILL NaN ───────
    critical = ["AQI_Lag1", "AQI_Roll3"]
    before = len(df)
    df = df.dropna(subset=critical)
    print(f"      Dropped {before - len(df):,} rows without lag features (first hours per station)")

    print(f"\n  New features added:")
    new_feats = [
        "AQI_Lag1", "AQI_Lag3", "AQI_Lag6", "AQI_Lag24",
        "AQI_Roll3", "AQI_Roll6", "AQI_Roll24",
        "AQI_Delta1", "AQI_Delta6",
        "Month_Sin", "Month_Cos", "Hour_Sin", "Hour_Cos", "DoW_Sin", "DoW_Cos",
        "Season_Enc", "IsWeekend", "CropBurning"
    ]
    new_feats += [c for c in df.columns if c.startswith("State_")]
    for f in new_feats:
        if f in df.columns:
            print(f"    ✓ {f}")

    print(f"\n  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def get_feature_columns(df, include_pollutants=True):
    """
    Return the list of feature columns to pass to the ML model.
    Excludes identifier and target columns.

    Parameters
    ----------
    df                 : pd.DataFrame
    include_pollutants : bool  — include raw pollutant readings as features?
                                 (True for richer model, False for AQI-only model)
    Returns
    -------
    feature_cols : list[str]
    target_col   : str
    """
    TARGET = "AQI_Final"

    # Always include these engineered features
    base_features = [
        "AQI_Lag1", "AQI_Lag3", "AQI_Lag6", "AQI_Lag24",
        "AQI_Roll3", "AQI_Roll6", "AQI_Roll24",
        "AQI_Delta1", "AQI_Delta6",
        "Month_Sin", "Month_Cos",
        "Hour_Sin",  "Hour_Cos",
        "DoW_Sin",   "DoW_Cos",
        "Season_Enc", "IsWeekend", "CropBurning", "Year",
    ]
    state_cols = [c for c in df.columns if c.startswith("State_")]

    if include_pollutants:
        poll_cols = [c for c in ["PM2.5", "PM10", "SO2", "NOx", "NH3", "CO", "O3"]
                     if c in df.columns]
        rolling_poll = [c for c in ["PM25_24h", "PM10_24h", "SO2_24h", "NOx_24h",
                                    "NH3_24h", "CO_8h", "O3_8h"]
                        if c in df.columns]
        all_features = base_features + state_cols + poll_cols + rolling_poll
    else:
        all_features = base_features + state_cols

    # Keep only columns that exist in df and have no NaNs
    feature_cols = [f for f in all_features if f in df.columns]
    return feature_cols, TARGET


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_preprocessing import load_and_preprocess
    from src.aqi_calculator import calculate_aqi

    df = load_and_preprocess(save=False)
    df = calculate_aqi(df)
    df = engineer_features(df)

    feature_cols, target = get_feature_columns(df)
    print(f"\nFeature columns ({len(feature_cols)}):")
    for f in feature_cols:
        print(f"  {f}")
    print(f"\nTarget: {target}")
    print(f"\nNull counts in features:\n{df[feature_cols].isnull().sum().sort_values(ascending=False).head(10)}")
