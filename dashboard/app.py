"""
dashboard/app.py
-----------------
PART 6: STREAMLIT DASHBOARD — 6 PAGES

HOW TO RUN:
    cd AQI_Project
    streamlit run dashboard/app.py

PAGES:
    1. Project Overview     — About the project, methodology, data stats
    2. AQI Trends           — Interactive time-series charts
    3. State Comparison     — Side-by-side state analysis
    4. Pollutant Analysis   — Per-pollutant deep dive
    5. AQI Prediction       — Enter today's readings, get a forecast
    6. Download Results     — Export cleaned data and model results
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AQI Northern India | Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

STATE_COLORS = {
    "Delhi":         "#e74c3c",
    "Punjab":        "#3498db",
    "Haryana":       "#2ecc71",
    "Uttar Pradesh": "#f39c12",
    "Rajasthan":     "#9b59b6",
}
CATEGORY_COLORS = {
    "Good":        "#27ae60",
    "Satisfactory":"#f1c40f",
    "Moderate":    "#e67e22",
    "Poor":        "#e74c3c",
    "Very Poor":   "#8e44ad",
    "Severe":      "#2c3e50",
}
POLLUTANTS = ["PM2.5", "PM10", "SO2", "NOx", "NH3", "CO", "O3"]

# ─────────────────────────────────────────────────────────────
# DATA LOADING  (cached so it only runs once per session)
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data …")
def load_data():
    """Load the processed dataset.  Run the pipeline first if not present."""
    processed = os.path.join("data", "processed", "northern_india_aqi.csv")
    if not os.path.exists(processed):
        # Fallback: re-run pipeline on the fly
        from src.data_preprocessing import load_and_preprocess
        from src.aqi_calculator import calculate_aqi
        from src.feature_engineering import engineer_features
        df = load_and_preprocess()
        df = calculate_aqi(df)
        df = engineer_features(df)
        os.makedirs(os.path.dirname(processed), exist_ok=True)
        df.to_csv(processed, index=False)
    else:
        df = pd.read_csv(processed, low_memory=False)
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    return df


@st.cache_resource(show_spinner="Loading model …")
def load_model():
    """Load the best-performing saved model."""
    model_path = os.path.join("models", "best_model.pkl")
    feat_path  = os.path.join("models", "feature_cols.pkl")
    if os.path.exists(model_path) and os.path.exists(feat_path):
        model       = joblib.load(model_path)
        feature_cols = joblib.load(feat_path)
        return model, feature_cols
    return None, None


@st.cache_data(show_spinner="Loading model results …")
def load_model_results():
    path = os.path.join("outputs", "reports", "ml_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

def render_sidebar(df):
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/"
            "Flag_of_India.svg/320px-Flag_of_India.svg.png",
            width=80,
        )
        st.title("🌿 AQI Dashboard")
        st.markdown("**Northern India · 2015–2020**")
        st.divider()

        # Global filters
        states = sorted(df["State"].dropna().unique())
        selected_states = st.multiselect(
            "Filter by State", options=states, default=states
        )

        if "Year" in df.columns:
            year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
            year_range = st.slider(
                "Year range", year_min, year_max,
                (year_min, year_max), step=1
            )
        else:
            year_range = (2015, 2020)

        st.divider()
        st.markdown("**Navigate**")
        page = st.radio(
            "Page",
            options=[
                "1 · Project Overview",
                "2 · AQI Trends",
                "3 · State Comparison",
                "4 · Pollutant Analysis",
                "5 · AQI Prediction",
                "6 · Download Results",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Data: CPCB via Kaggle\nBuilt with Streamlit + Plotly")

    return page, selected_states, year_range


# ─────────────────────────────────────────────────────────────
# PAGE 1 — PROJECT OVERVIEW
# ─────────────────────────────────────────────────────────────

def page_overview(df):
    st.title("🌿 Air Quality Index Analysis & Forecasting")
    st.subheader("Northern India · 2015–2020")

    st.markdown("""
    This project analyses air quality data from five northern Indian states
    using the **official CPCB AQI methodology**, builds predictive ML models,
    and surfaces policy-relevant insights through this interactive dashboard.
    """)

    # KPI cards
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        ("📍 Stations",   df["StationId"].nunique() if "StationId" in df.columns else "–"),
        ("🏙️ Cities",     df["City"].nunique() if "City" in df.columns else "–"),
        ("📊 Records",    f"{len(df):,}"),
        ("📅 Date Range", f"{df['Datetime'].dt.year.min()}–{df['Datetime'].dt.year.max()}"),
        ("🏆 Mean AQI",   f"{df['AQI_Final'].mean():.0f}" if "AQI_Final" in df.columns else "–"),
    ]
    for col, (label, val) in zip([col1,col2,col3,col4,col5], metrics):
        col.metric(label, val)

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 📚 Methodology")
        st.markdown("""
        **Step 1 — Data Preprocessing**
        - Merged `station_hour.csv` with `stations.csv` on StationId
        - Filtered 5 northern states
        - Forward-fill imputation + outlier clipping

        **Step 2 — AQI Calculation (CPCB)**
        - 24-hour rolling averages: PM2.5, PM10, SO₂, NOx, NH₃
        - 8-hour rolling maxima: CO, O₃
        - Sub-index per pollutant → max sub-index = final AQI

        **Step 3 — EDA** (15 publication-quality visualisations)

        **Step 4 — Feature Engineering**
        - Lag features, rolling averages, cyclical time encoding

        **Step 5 — ML Models**
        - Linear Regression, Random Forest, XGBoost
        """)

    with col_r:
        st.markdown("### 📈 AQI Category Distribution (All States)")
        if "AQI_Category" in df.columns:
            cat_counts = df["AQI_Category"].value_counts()
            fig = px.pie(
                values=cat_counts.values,
                names=cat_counts.index,
                color=cat_counts.index,
                color_discrete_map=CATEGORY_COLORS,
                hole=0.4,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(showlegend=False, height=320,
                              margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🗺️ States Covered")
    geo_data = pd.DataFrame({
        "State": ["Delhi", "Punjab", "Haryana", "Uttar Pradesh", "Rajasthan"],
        "Lat":   [28.61,    31.15,    29.06,    26.85,           27.02],
        "Lon":   [77.21,    75.34,    76.08,    80.99,           74.22],
        "AQI":   [
            df[df["State"]=="Delhi"]["AQI_Final"].mean()         if "AQI_Final" in df.columns else 250,
            df[df["State"]=="Punjab"]["AQI_Final"].mean()        if "AQI_Final" in df.columns else 180,
            df[df["State"]=="Haryana"]["AQI_Final"].mean()       if "AQI_Final" in df.columns else 190,
            df[df["State"]=="Uttar Pradesh"]["AQI_Final"].mean() if "AQI_Final" in df.columns else 200,
            df[df["State"]=="Rajasthan"]["AQI_Final"].mean()     if "AQI_Final" in df.columns else 160,
        ],
    })
    fig_map = px.scatter_mapbox(
        geo_data, lat="Lat", lon="Lon", color="AQI",
        size="AQI", hover_name="State",
        color_continuous_scale="RdYlGn_r",
        mapbox_style="carto-positron",
        zoom=4.5, center={"lat": 28.5, "lon": 77.0},
        size_max=40, height=350,
    )
    fig_map.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig_map, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 2 — AQI TRENDS
# ─────────────────────────────────────────────────────────────

def page_aqi_trends(df, selected_states, year_range):
    st.title("📈 AQI Trends")

    df_f = df[
        (df["State"].isin(selected_states)) &
        (df["Year"].between(year_range[0], year_range[1]))
    ].copy() if "Year" in df.columns else df[df["State"].isin(selected_states)].copy()

    if df_f.empty:
        st.warning("No data for the selected filters.")
        return

    # Monthly median
    monthly = (
        df_f.groupby(["State", pd.Grouper(key="Datetime", freq="ME")])
            ["AQI_Final"].median().reset_index()
    )

    tab1, tab2, tab3 = st.tabs(["Monthly Trend", "Hourly Pattern", "Year-over-Year"])

    with tab1:
        fig = px.line(
            monthly, x="Datetime", y="AQI_Final", color="State",
            color_discrete_map=STATE_COLORS,
            title="Monthly Median AQI by State",
            labels={"AQI_Final": "Median AQI", "Datetime": "Month"},
        )
        fig.add_hline(y=100, line_dash="dash", line_color="orange",
                      annotation_text="Moderate (100)")
        fig.add_hline(y=200, line_dash="dash", line_color="red",
                      annotation_text="Poor (200)")
        fig.update_layout(hovermode="x unified", height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        hourly = df_f.groupby(["State","Hour"])["AQI_Final"].mean().reset_index() if "Hour" in df_f.columns else pd.DataFrame()
        if not hourly.empty:
            fig2 = px.line(
                hourly, x="Hour", y="AQI_Final", color="State",
                color_discrete_map=STATE_COLORS,
                title="Average AQI by Hour of Day",
                labels={"AQI_Final": "Mean AQI", "Hour": "Hour (0–23)"},
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
            st.info("💡 Rush-hour peaks (8 AM and 6–8 PM) are visible for Delhi and UP "
                    "due to traffic emissions.")

    with tab3:
        yearly = df_f.groupby(["State","Year"])["AQI_Final"].mean().reset_index() if "Year" in df_f.columns else pd.DataFrame()
        if not yearly.empty:
            fig3 = px.bar(
                yearly, x="Year", y="AQI_Final", color="State",
                color_discrete_map=STATE_COLORS, barmode="group",
                title="Annual Mean AQI by State",
                labels={"AQI_Final": "Mean AQI"},
            )
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
            st.info("💡 The 2020 drop is due to COVID-19 lockdowns (March–May 2020), "
                    "reducing traffic and industrial emissions sharply.")


# ─────────────────────────────────────────────────────────────
# PAGE 3 — STATE COMPARISON
# ─────────────────────────────────────────────────────────────

def page_state_comparison(df, selected_states, year_range):
    st.title("🗺️ State Comparison")

    df_f = df[
        (df["State"].isin(selected_states)) &
        (df["Year"].between(year_range[0], year_range[1]))
    ].copy() if "Year" in df.columns else df[df["State"].isin(selected_states)].copy()

    col1, col2 = st.columns(2)
    with col1:
        # Box plot
        fig = px.box(
            df_f, x="State", y="AQI_Final",
            color="State", color_discrete_map=STATE_COLORS,
            title="AQI Distribution by State",
            labels={"AQI_Final": "AQI"},
            notched=True,
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # AQI category stacked bar
        if "AQI_Category" in df_f.columns:
            cat_pct = (
                df_f.groupby(["State", "AQI_Category"]).size()
                    .reset_index(name="count")
            )
            totals = cat_pct.groupby("State")["count"].transform("sum")
            cat_pct["pct"] = 100 * cat_pct["count"] / totals
            cat_order = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]
            fig2 = px.bar(
                cat_pct, x="State", y="pct", color="AQI_Category",
                color_discrete_map=CATEGORY_COLORS,
                category_orders={"AQI_Category": cat_order},
                title="AQI Category Share by State (%)",
                labels={"pct": "% of Hours", "AQI_Category": "Category"},
                barmode="stack",
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)

    # Summary table
    st.subheader("Summary Statistics")
    summary = (
        df_f.groupby("State")["AQI_Final"]
            .agg(Mean="mean", Median="median", Std="std",
                 Min="min", Max="max", Count="count")
            .round(1).reset_index()
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Season comparison
    st.subheader("Seasonal AQI by State")
    if "Season" in df_f.columns:
        season_avg = df_f.groupby(["State","Season"])["AQI_Final"].mean().reset_index()
        fig3 = px.bar(
            season_avg, x="Season", y="AQI_Final", color="State",
            color_discrete_map=STATE_COLORS, barmode="group",
            category_orders={"Season":["Winter","Spring","Monsoon","Autumn"]},
            title="Mean AQI by Season and State",
            labels={"AQI_Final": "Mean AQI"},
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 4 — POLLUTANT ANALYSIS
# ─────────────────────────────────────────────────────────────

def page_pollutant_analysis(df, selected_states, year_range):
    st.title("💨 Pollutant Analysis")

    df_f = df[
        (df["State"].isin(selected_states)) &
        (df["Year"].between(year_range[0], year_range[1]))
    ].copy() if "Year" in df.columns else df[df["State"].isin(selected_states)].copy()

    poll_present = [p for p in POLLUTANTS if p in df_f.columns]

    selected_poll = st.selectbox("Select Pollutant", options=poll_present)

    col1, col2 = st.columns(2)

    with col1:
        # Monthly trend
        monthly_poll = (
            df_f.groupby(["State", pd.Grouper(key="Datetime", freq="ME")])
                [selected_poll].median().reset_index()
        )
        fig = px.line(
            monthly_poll, x="Datetime", y=selected_poll, color="State",
            color_discrete_map=STATE_COLORS,
            title=f"Monthly {selected_poll} Trend",
            labels={selected_poll: f"{selected_poll} (µg/m³)"},
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Distribution by state
        fig2 = px.violin(
            df_f[df_f[selected_poll] <= df_f[selected_poll].quantile(0.99)],
            y=selected_poll, x="State", color="State",
            color_discrete_map=STATE_COLORS, box=True,
            title=f"{selected_poll} Distribution by State",
        )
        fig2.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # Correlation heatmap
    st.subheader("Pollutant Correlation Matrix")
    corr_cols = ["AQI_Final"] + [p for p in poll_present if p in df_f.columns]
    corr = df_f[corr_cols].corr().round(2)
    fig3 = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Pearson Correlation — AQI & Pollutants",
        aspect="auto",
    )
    fig3.update_layout(height=450)
    st.plotly_chart(fig3, use_container_width=True)

    # Crop burning analysis
    st.subheader("🔥 Crop Burning Season (Oct–Nov)")
    burn = df_f[df_f["Month"].isin([9,10,11,12])].copy() if "Month" in df_f.columns else pd.DataFrame()
    if not burn.empty and selected_poll in burn.columns:
        daily_burn = burn.groupby(["State", pd.Grouper(key="Datetime", freq="D")])[selected_poll].median().reset_index()
        fig4 = px.line(
            daily_burn, x="Datetime", y=selected_poll, color="State",
            color_discrete_map=STATE_COLORS,
            title=f"Daily {selected_poll} — Autumn (Oct–Nov burning window highlighted)",
        )
        for yr in range(2015, 2021):
            fig4.add_vrect(
                x0=f"{yr}-10-15", x1=f"{yr}-11-30",
                fillcolor="orange", opacity=0.08, line_width=0,
                annotation_text="Burning" if yr == 2015 else "",
                annotation_position="top left",
            )
        fig4.update_layout(height=380)
        st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 5 — AQI PREDICTION
# ─────────────────────────────────────────────────────────────

def page_aqi_prediction(df):
    st.title("🔮 AQI Prediction")
    st.markdown("Enter current sensor readings to get a predicted AQI using the trained ML model.")

    model, feature_cols = load_model()

    if model is None:
        st.error("No trained model found.  Please run `python run_pipeline.py` first "
                 "to train and save the model.")
        return

    st.success(f"Model loaded: `models/best_model.pkl`  |  Features: {len(feature_cols)}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Location & Time")
        state    = st.selectbox("State", ["Delhi","Punjab","Haryana","Uttar Pradesh","Rajasthan"])
        month    = st.slider("Month",   1, 12, 1)
        hour     = st.slider("Hour",    0, 23, 12)
        dow      = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 0)
        is_weekend = 1 if dow >= 5 else 0

    with col2:
        st.subheader("💨 Recent AQI Readings")
        aqi_lag1  = st.number_input("AQI 1 hour ago",    0, 500, 150)
        aqi_lag3  = st.number_input("AQI 3 hours ago",   0, 500, 145)
        aqi_lag6  = st.number_input("AQI 6 hours ago",   0, 500, 140)
        aqi_lag24 = st.number_input("AQI 24 hours ago",  0, 500, 160)
        aqi_roll3  = (aqi_lag1 + aqi_lag3) / 2
        aqi_roll6  = (aqi_lag1 + aqi_lag3 + aqi_lag6) / 3
        aqi_roll24 = (aqi_lag1 + aqi_lag3 + aqi_lag6 + aqi_lag24) / 4

    if st.button("🚀 Predict AQI", type="primary"):
        # Build input row
        states = ["Delhi","Punjab","Haryana","Uttar Pradesh","Rajasthan"]
        season_map = {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2,9:2, 10:3,11:3}

        input_data = {
            "AQI_Lag1":    aqi_lag1,
            "AQI_Lag3":    aqi_lag3,
            "AQI_Lag6":    aqi_lag6,
            "AQI_Lag24":   aqi_lag24,
            "AQI_Roll3":   aqi_roll3,
            "AQI_Roll6":   aqi_roll6,
            "AQI_Roll24":  aqi_roll24,
            "AQI_Delta1":  aqi_lag1 - aqi_lag3,
            "AQI_Delta6":  aqi_lag1 - aqi_lag6,
            "Month_Sin":   np.sin(2 * np.pi * month / 12),
            "Month_Cos":   np.cos(2 * np.pi * month / 12),
            "Hour_Sin":    np.sin(2 * np.pi * hour / 24),
            "Hour_Cos":    np.cos(2 * np.pi * hour / 24),
            "DoW_Sin":     np.sin(2 * np.pi * dow / 7),
            "DoW_Cos":     np.cos(2 * np.pi * dow / 7),
            "Season_Enc":  season_map.get(month, 0),
            "IsWeekend":   is_weekend,
            "CropBurning": 1 if month in [10,11] and state in ["Punjab","Haryana"] else 0,
            "Year":        2024,
        }
        # State dummies
        for s in states:
            input_data[f"State_{s}"] = 1 if s == state else 0

        # Build feature vector in the exact order the model expects
        input_vec = []
        for feat in feature_cols:
            input_vec.append(input_data.get(feat, 0.0))
        input_array = np.array([input_vec])

        prediction = float(np.clip(model.predict(input_array)[0], 0, 500))

        # Determine category
        if   prediction <= 50:  cat, emoji, color = "Good",         "🟢", "#27ae60"
        elif prediction <= 100: cat, emoji, color = "Satisfactory", "🟡", "#f1c40f"
        elif prediction <= 200: cat, emoji, color = "Moderate",     "🟠", "#e67e22"
        elif prediction <= 300: cat, emoji, color = "Poor",         "🔴", "#e74c3c"
        elif prediction <= 400: cat, emoji, color = "Very Poor",    "🟣", "#8e44ad"
        else:                   cat, emoji, color = "Severe",       "⚫", "#2c3e50"

        st.divider()
        col_p1, col_p2 = st.columns([1, 2])
        with col_p1:
            st.markdown(
                f"""
                <div style="background-color:{color}22; border-left: 6px solid {color};
                 padding: 1.5rem; border-radius: 8px; text-align:center">
                    <h1 style="color:{color}; margin:0">{prediction:.0f}</h1>
                    <h3 style="color:{color}">{emoji} {cat}</h3>
                    <p style="color:gray; margin:0">Predicted AQI · {state}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_p2:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                gauge={
                    "axis":  {"range": [0, 500]},
                    "bar":   {"color": color},
                    "steps": [
                        {"range": [0,   50],  "color": "#27ae6033"},
                        {"range": [50,  100], "color": "#f1c40f33"},
                        {"range": [100, 200], "color": "#e67e2233"},
                        {"range": [200, 300], "color": "#e74c3c33"},
                        {"range": [300, 400], "color": "#8e44ad33"},
                        {"range": [400, 500], "color": "#2c3e5033"},
                    ],
                },
                title={"text": "Predicted AQI"},
            ))
            fig.update_layout(height=280, margin=dict(t=40, b=10, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 6 — DOWNLOAD RESULTS
# ─────────────────────────────────────────────────────────────

def page_download_results(df):
    st.title("📥 Download Results")

    st.markdown("Export processed data and model results for further analysis.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Processed Dataset")
        st.dataframe(df.head(500), use_container_width=True, height=300)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Full Dataset (CSV)",
            data=csv,
            file_name="northern_india_aqi_clean.csv",
            mime="text/csv",
            type="primary",
        )

    with col2:
        st.subheader("🤖 Model Results")
        results = load_model_results()
        if results is not None:
            st.dataframe(results, use_container_width=True)
            csv2 = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Model Comparison (CSV)",
                data=csv2,
                file_name="ml_results.csv",
                mime="text/csv",
            )
        else:
            st.info("Run `python run_pipeline.py` to generate model results.")

        st.subheader("📈 City-level Summary")
        city_summary = (
            df.groupby(["City", "State"])["AQI_Final"]
              .agg(Mean="mean", Median="median", Max="max", Count="count")
              .round(1).reset_index()
              .sort_values("Mean", ascending=False)
        ) if "City" in df.columns else pd.DataFrame()
        if not city_summary.empty:
            st.dataframe(city_summary, use_container_width=True, height=280)
            csv3 = city_summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download City Summary (CSV)",
                data=csv3,
                file_name="city_aqi_summary.csv",
                mime="text/csv",
            )


# ─────────────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────────────

def main():
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Could not load data: {e}\n\nRun `python run_pipeline.py` first.")
        st.stop()

    page, selected_states, year_range = render_sidebar(df)

    if not selected_states:
        st.warning("Please select at least one state in the sidebar.")
        st.stop()

    if   page.startswith("1"): page_overview(df)
    elif page.startswith("2"): page_aqi_trends(df, selected_states, year_range)
    elif page.startswith("3"): page_state_comparison(df, selected_states, year_range)
    elif page.startswith("4"): page_pollutant_analysis(df, selected_states, year_range)
    elif page.startswith("5"): page_aqi_prediction(df)
    elif page.startswith("6"): page_download_results(df)


if __name__ == "__main__":
    main()
