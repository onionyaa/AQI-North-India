import pandas as pd
import numpy as np

def engineer_features(df):

    print("[FE] engineering features...")
    df = df.sort_values(["StationId", "Datetime"]).copy()
    g = df.groupby("StationId")["AQI_Final"]

    df["AQI_Lag1"] = g.shift(1)
    df["AQI_Lag3"] = g.shift(3)
    df["AQI_Lag6"] = g.shift(6)
    df["AQI_Lag24"] = g.shift(24)  # same hour, previous day

    df["AQI_Roll3"] = g.transform(lambda x: x.rolling(3, min_periods=2).mean())
    df["AQI_Roll6"] = g.transform(lambda x: x.rolling(6, min_periods=3).mean())
    df["AQI_Roll24"] = g.transform(lambda x: x.rolling(24, min_periods=12).mean())

    df["AQI_Delta1"] = df["AQI_Final"] - df["AQI_Lag1"]
    df["AQI_Delta6"] = df["AQI_Final"] - df["AQI_Lag6"]

    # calendar features
    if "Month" not in df.columns:
        df["Month"] = df["Datetime"].dt.month
    if "DayOfWeek" not in df.columns:
        df["DayOfWeek"] = df["Datetime"].dt.dayofweek
    if "Hour" not in df.columns:
        df["Hour"] = df["Datetime"].dt.hour

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

    if "CropBurning" not in df.columns:
        df["CropBurning"] = (
            (df["Month"].isin([10, 11])) & (df["State"].isin(["Punjab", "Haryana"]))
        ).astype(int)

    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["DoW_Sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DoW_Cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    state_dummies = pd.get_dummies(df["State"], prefix="State", dtype=int)
    df = pd.concat([df, state_dummies], axis=1)

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
