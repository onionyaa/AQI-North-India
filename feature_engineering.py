"""
Step 4 : Feature Engineering

Builds the feature set used for AQI forecasting - lags, rolling stats,
calendar features, and the crop-burning flag. Expects a dataframe that's
already been through preprocessing + AQI calculation (needs AQI_Final,
Datetime, State, StationId).

Usage:
    from src.feature_engineering import engineer_features
    df_feat = engineer_features(df)
"""

import pandas as pd
import numpy as np


def engineer_features(df):
    """
    Adds lag/rolling/calendar features to df and drops the warm-up rows
    that don't have enough history for the lag features yet.

    Returns the augmented dataframe.
    """
    print("[FE] engineering features...")
    df = df.sort_values(["StationId", "Datetime"]).copy()

    # group by station so lags don't leak across sensors - station A's
    # reading from an hour ago has nothing to do with station B
    g = df.groupby("StationId")["AQI_Final"]

    # lag features - air quality is heavily autocorrelated, pollution
    # events (inversions, stubble burning, dust) stick around for hours
    df["AQI_Lag1"] = g.shift(1)
    df["AQI_Lag3"] = g.shift(3)
    df["AQI_Lag6"] = g.shift(6)
    df["AQI_Lag24"] = g.shift(24)  # same hour, previous day

    # rolling means smooth out single bad readings and give a sense of
    # the "background" pollution level
    df["AQI_Roll3"] = g.transform(lambda x: x.rolling(3, min_periods=2).mean())
    df["AQI_Roll6"] = g.transform(lambda x: x.rolling(6, min_periods=3).mean())
    df["AQI_Roll24"] = g.transform(lambda x: x.rolling(24, min_periods=12).mean())

    # rate of change - rising vs falling AQI matters as much as the level
    df["AQI_Delta1"] = df["AQI_Final"] - df["AQI_Lag1"]
    df["AQI_Delta6"] = df["AQI_Final"] - df["AQI_Lag6"]

    # calendar features
    if "Month" not in df.columns:
        df["Month"] = df["Datetime"].dt.month
    if "DayOfWeek" not in df.columns:
        df["DayOfWeek"] = df["Datetime"].dt.dayofweek
    if "Hour" not in df.columns:
        df["Hour"] = df["Datetime"].dt.hour

    # season as an int (0=winter,1=spring,2=monsoon,3=autumn) - winter
    # pollution regime in N. India is a completely different beast from monsoon
    if "Season" not in df.columns:
        season_map = {
            12: 0, 1: 0, 2: 0,
            3: 1, 4: 1, 5: 1,
            6: 2, 7: 2, 8: 2, 9: 2,
            10: 3, 11: 3,
        }
        df["Season_Enc"] = df["Month"].map(season_map)
    else:
        season_enc = {"Winter": 0, "Spring": 1, "Monsoon": 2, "Autumn": 3}
        df["Season_Enc"] = df["Season"].map(season_enc)

    if "IsWeekend" not in df.columns:
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    # crop burning flag - Oct/Nov + Punjab/Haryana is the single biggest
    # episodic source of pollution up north, worth a dedicated feature
    # rather than hoping the model infers it from month + state alone
    if "CropBurning" not in df.columns:
        df["CropBurning"] = (
            (df["Month"].isin([10, 11])) & (df["State"].isin(["Punjab", "Haryana"]))
        ).astype(int)

    # cyclical encoding so e.g. Dec and Jan aren't treated as far apart
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["DoW_Sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DoW_Cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    # one-hot the state column for the linear/XGBoost models
    state_dummies = pd.get_dummies(df["State"], prefix="State", dtype=int)
    df = pd.concat([df, state_dummies], axis=1)

    # first ~24h per station won't have full lag history, drop those
    critical = ["AQI_Lag1", "AQI_Roll3"]
    before = len(df)
    df = df.dropna(subset=critical)
    print(f"  dropped {before - len(df):,} warm-up rows (no lag history yet)")

    new_feats = [
        "AQI_Lag1", "AQI_Lag3", "AQI_Lag6", "AQI_Lag24",
        "AQI_Roll3", "AQI_Roll6", "AQI_Roll24",
        "AQI_Delta1", "AQI_Delta6",
        "Month_Sin", "Month_Cos", "Hour_Sin", "Hour_Cos", "DoW_Sin", "DoW_Cos",
        "Season_Enc", "IsWeekend", "CropBurning",
    ]
    new_feats += [c for c in df.columns if c.startswith("State_")]
    added = [f for f in new_feats if f in df.columns]
    print(f"  added {len(added)} features")

    print(f"  final shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def get_feature_columns(df, include_pollutants=True):
    """
    Returns (feature_cols, target_col) for model training.

    include_pollutants=False gives an AQI-only feature set (lags/rolling/
    calendar), useful as a baseline or for forecasting horizons where raw
    pollutant readings won't be available at inference time.
    """
    TARGET = "AQI_Final"

    base_features = [
        "AQI_Lag1", "AQI_Lag3", "AQI_Lag6", "AQI_Lag24",
        "AQI_Roll3", "AQI_Roll6", "AQI_Roll24",
        "AQI_Delta1", "AQI_Delta6",
        "Month_Sin", "Month_Cos",
        "Hour_Sin", "Hour_Cos",
        "DoW_Sin", "DoW_Cos",
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
    print(f"\n{len(feature_cols)} feature columns:")
    for f in feature_cols:
        print(f"  {f}")
    print(f"\ntarget: {target}")
    print(f"\nnull counts:\n{df[feature_cols].isnull().sum().sort_values(ascending=False).head(10)}")
