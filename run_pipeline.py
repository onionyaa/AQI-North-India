"""
run_pipeline.py
----------------
ONE COMMAND TO RUN THE ENTIRE PROJECT END-TO-END.

Usage:
    python run_pipeline.py

What it does:
    1. Preprocesses raw data
    2. Calculates AQI (CPCB methodology)
    3. Generates all 15 EDA visualisations
    4. Engineers ML features
    5. Trains and compares Linear Regression, Random Forest, XGBoost
    6. Saves the best model for the dashboard
    7. Exports all results and figures
"""

import os
import sys
import time

# Make sure src/ is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_preprocessing  import load_and_preprocess
from src.aqi_calculator       import calculate_aqi, get_dominant_pollutant
from src.eda_visualizations   import run_full_eda
from src.feature_engineering  import engineer_features, get_feature_columns
from src.ml_pipeline          import run_ml_pipeline

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   AQI ANALYSIS & FORECASTING — NORTHERN INDIA  2015–2020    ║
║   Complete Data Science Pipeline                             ║
╚══════════════════════════════════════════════════════════════╝
"""

def main():
    print(BANNER)
    start = time.time()

    # ── PART 1: DATA PREPROCESSING ──────────────────────────────
    print("\n" + "─"*60)
    print("  PART 1 · DATA PREPROCESSING")
    print("─"*60)
    df = load_and_preprocess(save=True)

    # ── PART 2: AQI CALCULATION ─────────────────────────────────
    print("\n" + "─"*60)
    print("  PART 2 · AQI CALCULATION (CPCB METHODOLOGY)")
    print("─"*60)
    df = calculate_aqi(df)
    df = get_dominant_pollutant(df)

    # Save AQI-enriched version for dashboard
    aqi_path = os.path.join("data", "processed", "northern_india_aqi.csv")
    df.to_csv(aqi_path, index=False)
    print(f"[✓] AQI data saved → {aqi_path}")

    # ── PART 3: EDA VISUALISATIONS ──────────────────────────────
    print("\n" + "─"*60)
    print("  PART 3 · EXPLORATORY DATA ANALYSIS (15 PLOTS)")
    print("─"*60)
    run_full_eda(df)

    # ── PART 4: FEATURE ENGINEERING ─────────────────────────────
    print("\n" + "─"*60)
    print("  PART 4 · FEATURE ENGINEERING")
    print("─"*60)
    df = engineer_features(df)
    feature_cols, target_col = get_feature_columns(df, include_pollutants=True)
    print(f"  Total features for ML: {len(feature_cols)}")

    # ── PART 5: MACHINE LEARNING ─────────────────────────────────
    print("\n" + "─"*60)
    print("  PART 5 · MACHINE LEARNING")
    print("─"*60)
    results_df, best_model, df_test = run_ml_pipeline(df, feature_cols, target_col)

    # ── DONE ─────────────────────────────────────────────────────
    elapsed = time.time() - start
    print("\n" + "═"*60)
    print(f"  ✅ PIPELINE COMPLETE  ({elapsed:.0f} seconds)")
    print("═"*60)
    print("""
  Next steps:
    1. Launch dashboard  →  streamlit run dashboard/app.py
    2. View figures      →  outputs/figures/
    3. View ML results   →  outputs/reports/ml_results.csv
    4. Push to GitHub    →  git add . && git commit -m "Add AQI project"
  """)


if __name__ == "__main__":
    main()
