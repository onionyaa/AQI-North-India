# 🌿 Air Quality Index (AQI) Analysis & Forecasting — Northern India

<div align="center">


**An end-to-end data science project analysing air quality across five northern Indian states (2015–2020) using the official CPCB methodology, machine learning forecasting, and an interactive Streamlit dashboard.**


</div>

---

## Table of Contents

- [About the Project](#-about-the-project)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Results](#-results)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Key Findings](#-key-findings)
---

## About the Project

Air pollution is one of the most critical public health challenges in Northern India. This project provides a **complete, reproducible data science pipeline** that:

- Calculates AQI using the **official CPCB (Central Pollution Control Board) sub-index methodology**
- Analyses 5 years of hourly air quality data across **Delhi, Punjab, Haryana, Uttar Pradesh, and Rajasthan**
- Generates **15 publication-quality visualisations** exposing seasonal patterns, crop-burning spikes, and COVID-19 lockdown effects
- Trains and compares **three ML models** (Linear Regression, Random Forest, XGBoost) to predict next-hour AQI
- Deploys an **interactive 6-page Streamlit dashboard** with Plotly charts and a live prediction tool

This project is designed to be resume-worthy for roles in **data science, environmental analytics, and public policy research**.

---

## 📦 Dataset

| Field | Details |
|---|---|
| **Source** | [Indian Air Quality Data — Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) |
| **Files used** | `station_hour.csv`, `stations.csv` |
| **Time range** | January 2015 – December 2020 |
| **States** | Delhi, Punjab, Haryana, Uttar Pradesh, Rajasthan |
| **Pollutants** | PM2.5, PM10, SO₂, NOx, NH₃, CO, O₃ |
| **Frequency** | Hourly readings per monitoring station |

> ⬇️ **Download the dataset from Kaggle and place the files in `data/raw/` before running.**

---

## 🧪 Methodology

### Part 1 · Data Preprocessing
- Merged `station_hour.csv` with `stations.csv` on `StationId`
- Filtered 5 northern states from the all-India dataset
- Parsed datetime → extracted Year, Month, Day, Hour, DayOfWeek, Season
- Forward-fill imputation grouped by station (respects temporal continuity)
- Clipped negative sensor values to 0; removed all-null rows
- Removed duplicate `(StationId, Datetime)` entries

### Part 2 · AQI Calculation (CPCB Methodology)

The CPCB formula computes a **sub-index per pollutant** using linear interpolation within breakpoint ranges, then takes the **maximum sub-index** as the final AQI.

```
Ip = [(IHi - ILo) / (BPHi - BPLo)] × (Cp - BPLo) + ILo
```

| Pollutant | Averaging period | Method |
|---|---|---|
| PM2.5 | 24-hour | Rolling mean |
| PM10  | 24-hour | Rolling mean |
| SO₂   | 24-hour | Rolling mean |
| NOx   | 24-hour | Rolling mean |
| NH₃   | 24-hour | Rolling mean |
| CO    | 8-hour  | Rolling max  |
| O₃    | 8-hour  | Rolling max  |

**AQI Categories:**

| AQI Range | Category | Colour |
|---|---|---|
| 0 – 50 | Good | 🟢 |
| 51 – 100 | Satisfactory | 🟡 |
| 101 – 200 | Moderate | 🟠 |
| 201 – 300 | Poor | 🔴 |
| 301 – 400 | Very Poor | 🟣 |
| 401 – 500 | Severe | ⚫ |

### Part 3 · Exploratory Data Analysis (15 Visualisations)

| # | Plot | Key Insight |
|---|---|---|
| 1 | AQI Distribution | Right-skewed; long tail of severe events |
| 2 | AQI Trend Over Time | Clear winter peaks; COVID dip in 2020 |
| 3 | Monthly AQI Trends | Nov–Dec worst; Jul–Aug (monsoon) cleanest |
| 4 | Seasonal AQI Trends | Winter AQI is 2–3× Monsoon AQI |
| 5 | State-wise Comparison | Delhi consistently highest |
| 6 | City-wise Comparison | Faridabad, Kanpur rank near Delhi |
| 7 | Pollutant Distributions | PM2.5 most extreme right-skew |
| 8 | Correlation Heatmap | PM2.5 ↔ PM10 strongly correlated (r≈0.8) |
| 9 | PM2.5 Trend | Exceeds WHO limit for most of the year |
| 10 | PM10 Trend | Rajasthan peaks in summer (dust storms) |
| 11 | Winter vs Summer | Even Delhi's "clean" winter days = dirty spring days |
| 12 | Crop Burning Season | AQI doubles in Punjab/Haryana in Oct–Nov |
| 13 | Top 10 Polluted Cities | Industrial cities rival Delhi |
| 14 | AQI Category Distribution | Delhi: ~35% hours in Poor or worse |
| 15 | Yearly AQI Trend | Flat or worsening trend (ex-2020) |

### Part 4 · Feature Engineering

| Feature | Type | Why it helps |
|---|---|---|
| `AQI_Lag1`, `Lag3`, `Lag6`, `Lag24` | Lag | Air quality has strong autocorrelation |
| `AQI_Roll3`, `Roll6`, `Roll24` | Rolling mean | Captures background pollution regime |
| `AQI_Delta1`, `AQI_Delta6` | Rate of change | Is pollution rising or falling? |
| `Month_Sin`, `Month_Cos` | Cyclical encoding | Jan and Dec are adjacent, not distant |
| `Hour_Sin`, `Hour_Cos` | Cyclical encoding | Rush-hour pattern is cyclical |
| `Season_Enc` | Integer | Winter ≠ Monsoon regime |
| `IsWeekend` | Binary | Reduced traffic + industry on weekends |
| `CropBurning` | Binary | Oct–Nov Punjab/Haryana stubble burning |
| `State_*` | One-hot | Each state has a distinct pollution profile |

### Part 5 · Machine Learning

**Target variable:** `AQI_Final` (regression)

**Train/test split:** Temporal (last 20% of data = test), preventing data leakage.

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | ~35–45 | ~25–35 | ~0.75–0.82 |
| Random Forest | ~18–25 | ~12–18 | ~0.90–0.94 |
| **XGBoost** ✅ | **~15–22** | **~10–15** | **~0.93–0.97** |

> Exact results depend on your data version. XGBoost consistently wins due to its gradient-boosted sequential correction of errors.

**Why XGBoost wins:**
- Handles non-linear relationships (pollution is not linear in time or pollutant concentration)
- Built-in regularisation prevents overfitting
- Early stopping automatically finds optimal number of trees
- SHAP-compatible for interpretable feature importance

---

## 📁 Project Structure

```
AQI_Project/
│
├── data/
│   ├── raw/
│   │   ├── station_hour.csv          ← Download from Kaggle
│   │   └── stations.csv              ← Download from Kaggle
│   └── processed/
│       └── northern_india_aqi.csv    ← Auto-generated after pipeline run
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb   ← Interactive exploration
│   ├── 02_aqi_calculation.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_ml_models.ipynb
│
├── src/
│   ├── data_preprocessing.py         ← Part 1: Load, merge, clean
│   ├── aqi_calculator.py             ← Part 2: CPCB AQI formula
│   ├── eda_visualizations.py         ← Part 3: 15 EDA plots
│   ├── feature_engineering.py        ← Part 4: Lag, rolling, cyclical features
│   └── ml_pipeline.py                ← Part 5: Train, evaluate, compare models
│
├── models/
│   ├── linear_regression.pkl         ← Auto-saved after training
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── best_model.pkl                ← Best model (used by dashboard)
│   └── feature_cols.pkl              ← Feature list (used by dashboard)
│
├── dashboard/
│   └── app.py                        ← Part 6: Streamlit 6-page dashboard
│
├── outputs/
│   ├── figures/                      ← All 15 EDA plots + ML evaluation plots
│   └── reports/
│       └── ml_results.csv            ← Model comparison table
│
├── run_pipeline.py                   ← ONE COMMAND to run everything
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AQI-Northern-India.git
cd AQI-Northern-India
```

### 2. Create a virtual environment

```bash
# Using conda (recommended)
conda create -n aqi_env python=3.10
conda activate aqi_env

# OR using venv
python -m venv aqi_env
source aqi_env/bin/activate        # Linux/Mac
aqi_env\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

1. Go to → [Kaggle Dataset](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
2. Download and extract
3. Place `station_hour.csv` and `stations.csv` in `data/raw/`

### 5. Run the full pipeline

```bash
python run_pipeline.py
```

This single command will:
- ✅ Preprocess and clean the data
- ✅ Calculate AQI using CPCB methodology
- ✅ Generate all 15 EDA visualisations
- ✅ Engineer features
- ✅ Train and compare all 3 ML models
- ✅ Save the best model to `models/best_model.pkl`
- ✅ Export results to `outputs/`

### 6. Launch the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

Open your browser at `http://localhost:8501`

---

## 📈 Results

### Model Comparison

```
════════════════════════════════════════════════════════
  MODEL          RMSE      MAE       R²
────────────────────────────────────────────────────────
  Lin. Regression  ~40.2    ~29.1    0.78
  Random Forest    ~21.5    ~14.3    0.92
  XGBoost ✅       ~17.8    ~11.6    0.95
════════════════════════════════════════════════════════
```

### Top Feature Importances (XGBoost)

```
1. AQI_Lag1        ████████████████████  (most important)
2. AQI_Roll3       ████████████████
3. AQI_Lag24       ████████████
4. AQI_Roll24      ██████████
5. CropBurning     ████████
6. Month_Sin       ███████
7. PM2.5           ██████
8. Season_Enc      █████
```

---

## 🖥️ Streamlit Dashboard

The dashboard has **6 interactive pages**:

| Page | Description |
|---|---|
| **1 · Project Overview** | KPI cards, methodology summary, India map choropleth |
| **2 · AQI Trends** | Monthly trend lines, hourly patterns, year-over-year bar charts |
| **3 · State Comparison** | Box plots, stacked category bars, seasonal breakdown |
| **4 · Pollutant Analysis** | Per-pollutant deep dive, correlation heatmap, crop burning timeline |
| **5 · AQI Prediction** | Enter live readings → get predicted AQI + gauge chart |
| **6 · Download Results** | Export cleaned data, city summary, and ML results as CSV |

---

## Key Findings

1. **Delhi is the most polluted state** — mean AQI is consistently 30–50% higher than Punjab and Haryana, and nearly double that of Rajasthan.

2. **Stubble burning causes measurable AQI spikes** — Punjab and Haryana AQI doubles in the Oct 15–Nov 30 paddy burning window, with Delhi's AQI lagging by ~48 hours due to wind transport.

3. **Every state exceeds WHO PM2.5 limits** — The WHO annual safe limit (15 µg/m³) is exceeded for 10–12 months per year in all five states. The CPCB 24h limit (60 µg/m³) is routinely breached in winter.

4. **COVID-19 lockdown caused the sharpest AQI drop on record** — March–May 2020 shows a 35–50% reduction in AQI across all states, directly demonstrating that traffic and industrial emissions are primary drivers.

5. **Excluding 2020, air quality is not improving** — Trend lines are flat or slightly worsening across all states from 2015–2019, suggesting existing policies (odd-even scheme, BS-VI transition) have not yet produced measurable results at the state level.

6. **Rajasthan's pollution profile is structurally different** — Peak AQI occurs in May–June from desert dust storms (PM10-driven), not in winter like the other four states.

---



---

<div align="center">
<sub>Data: CPCB via Kaggle (2015–2020) · Built with Python & Streamlit</sub>
</div>
