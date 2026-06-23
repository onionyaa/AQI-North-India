# AQI Analysis & Forecasting — Northern India

Air quality pipeline covering five northern Indian states (2015-2020). Calculates official CPCB AQI from raw pollutant readings, does a full EDA pass, trains ML models to forecast next-hour AQI, and wraps it all in a Streamlit dashboard.


## Table of Contents

- [About](#about)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Results](#results)
- [Dashboard](#dashboard)
- [Key Findings](#key-findings)

## About

Northern India—especially Delhi, Punjab, Haryana, Uttar Pradesh, and Rajasthan—faces some of the worst air quality in the world. Yet the issue is often reduced to "Delhi smog" headlines each winter. This project analyzes five years of hourly air quality data to uncover the key factors driving pollution across the region.

Main Objectives:
- Calculate AQI using the official CPCB sub-index methodology.
- Analyze five years of air quality data to identify seasonal trends, crop-burning impacts, and the COVID-19 pollution dip.
- Develop Random Forest and XGBoost models for next-hour AQI prediction.
- Build a six-page Streamlit dashboard featuring interactive visualizations and a live AQI prediction tool.

## Dataset

| Field | Details |
|---|---|
| Source | [Air Quality Data in India — Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) |
| Files used | `station_hour.csv`, `stations.csv` |
| Time range | Jan 2015 – Jul 2020 |
| States | Delhi, Punjab, Haryana, Uttar Pradesh, Rajasthan |
| Pollutants | PM2.5, PM10, SO2, NOx, NH3, CO, O3 |
| Frequency | Hourly, per station |

## Methodology

### 1. Data Preprocessing

Merged `station_hour.csv` with `stations.csv` on `StationId`, filtered down to the 5 northern states out of the full all-India dataset. Parsed datetime into Year/Month/Day/Hour/DayOfWeek/Season, forward-filled missing readings grouped by station, clipped negative sensor readings to 0, dropped fully-null rows, and removed duplicate `(StationId, Datetime)` pairs.

### 2. AQI Calculation (CPCB Methodology)

AQI was calculated using the official CPCB methodology, with pollutant concentrations converted to sub-indices through linear interpolation and the final AQI defined as the maximum sub-index.
```
Ip = [(IHi - ILo) / (BPHi - BPLo)] × (Cp - BPLo) + ILo
```

Averaging periods per pollutant:

| Pollutant | Averaging period | Method |
|---|---|---|
| PM2.5 | 24-hour | Rolling mean |
| PM10  | 24-hour | Rolling mean |
| SO2   | 24-hour | Rolling mean |
| NOx   | 24-hour | Rolling mean |
| NH3   | 24-hour | Rolling mean |
| CO    | 8-hour  | Rolling max  |
| O3    | 8-hour  | Rolling max  |

AQI categories (standard CPCB bands):

| Range | Category |
|---|---|
| 0–50 | Good |
| 51–100 | Satisfactory |
| 101–200 | Moderate |
| 201–300 | Poor |
| 301–400 | Very Poor |
| 401–500 | Severe |

### 3. EDA

15 plots total, a few highlights below — full list and code in `src/eda_visualizations.py`.

| Plot | What it shows |
|---|---|
| AQI distribution | Right-skewed, long tail of severe events |
| AQI over time | Clear winter peaks, sharp COVID dip in 2020 |
| Monthly trends | Nov-Dec worst, Jul-Aug (monsoon) cleanest |
| Seasonal comparison | Winter AQI runs 2-3x monsoon AQI |
| State comparison | Delhi consistently on top |
| Correlation heatmap | PM2.5 and PM10 strongly correlated (r ≈ 0.8) |
| PM10 by state | Rajasthan spikes in summer — dust storms, not winter smog |
| Crop burning window | AQI roughly doubles in Punjab/Haryana, Oct-Nov |
| Yearly trend | Flat-to-worsening 2015-2019, excluding the COVID year |

## 4. Feature Engineering

| Feature | Type | Reasoning |
|---|---|---|
| `AQI_Lag1/3/6/24` | Lag | AQI is strongly autocorrelated hour to hour |
| `AQI_Roll3/6/24` | Rolling mean | Smooths spikes, captures the background pollution level |
| `AQI_Delta1/6` | Rate of change | Is it trending up or down right now |
| `Month_Sin/Cos`, `Hour_Sin/Cos` | Cyclical encoding | Dec and Jan are adjacent, not far apart numerically |
| `Season_Enc` | Integer | Winter regime is nothing like monsoon regime |
| `IsWeekend` | Binary | Traffic/industrial activity drops on weekends |
| `CropBurning` | Binary | Oct-Nov + Punjab/Haryana stubble burning flag |
| `State_*` | One-hot | Each state has a distinct baseline pollution profile |

## 5. Machine Learning

Target: `AQI_Final`, framed as regression.

Split is temporal — last 20% of the timeline held out as test set — rather than a random split, since a random split would let the model "see the future" via nearby timestamps.

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Random Forest | 9.28  | 1.65  | .09939 |
| XGBoost | 8.97 | 1.45 | 0.9943 |


## Project Structure

```
AQI_Project/
│
├── data/
│   ├── raw/
│   │   ├── station_hour.csv          ← from Kaggle
│   │   └── stations.csv              ← from Kaggle
│   └── processed/
│       └── northern_india_aqi.csv    ← generated by the pipeline
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_aqi_calculation.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_ml_models.ipynb
│
├── src/
│   ├── data_preprocessing.py         ← load, merge, clean
│   ├── aqi_calculator.py             ← CPCB AQI formula
│   ├── eda_visualizations.py         ← the 15 plots
│   ├── feature_engineering.py        ← lag/rolling/cyclical features
│   └── ml_pipeline.py                ← train, evaluate, compare
│
├── models/
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── best_model.pkl                ← used by the dashboard
│   └── feature_cols.pkl
│
├── dashboard/
│   └── app.py
│
├── outputs/
│   ├── figures/
│   └── reports/
│       └── ml_results.csv
│
├── run_pipeline.py                   ← runs the whole thing end to end
├── requirements.txt
└── README.md
```

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/AQI-Northern-India.git
cd AQI-Northern-India
python -m venv aqi_env && source aqi_env/bin/activate   # Windows: aqi_env\Scripts\activate
pip install -r requirements.txt
```

Download the [dataset from Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) and put `station_hour.csv` + `stations.csv` in `data/raw/`.

Then:

```bash
python run_pipeline.py
```

Runs the full pipeline — preprocessing through model training — and saves the best model to `models/best_model.pkl`.

Then launch the dashboard:

```bash
streamlit run dashboard/app.py
```

## Results

Model comparison table (filled in once the pipeline's been run on the full dataset):

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Random Forest | 9.28  | 1.65  | .09939 |
| XGBoost | 8.97 | 1.45 | 0.9943 |

## Dashboard

6 pages:

| Page | What's on it |
|---|---|
| Overview | KPI cards, methodology summary, India choropleth |
| AQI Trends | Monthly/hourly patterns, year-over-year bars |
| State Comparison | Box plots, category breakdowns by season |
| Pollutant Analysis | Per-pollutant deep dive, correlation heatmap, crop burning timeline |
| AQI Prediction | Punch in live readings, get a predicted AQI + gauge |
| Download Results | Export cleaned data / summaries / ML results as CSV |

## Key Findings

1. **Delhi is the worst of the five states** — mean AQI runs 30-50% higher than Punjab/Haryana, and close to double Rajasthan's.

2. **Stubble burning has a measurable, lagged effect on Delhi** — Punjab/Haryana AQI roughly doubles in the Oct 15–Nov 30 paddy-burning window, and Delhi's AQI spikes about 48 hours later, consistent with wind transport.

3. **All five states blow past WHO PM2.5 limits** — the WHO annual limit (15 µg/m³) is exceeded 10-12 months a year across the board, and the CPCB 24h limit (60 µg/m³) gets routinely breached every winter.

4. **COVID lockdown caused the biggest AQI drop in the dataset** — March-May 2020 shows a 35-50% reduction across all five states, which is about as clean a natural experiment as you'll get for "traffic and industry are the main drivers."

5. **Outside of 2020, things aren't improving** — trend lines are flat or slightly worse from 2015-2019. Doesn't look like odd-even or the BS-VI rollout has moved the needle yet, at least not at this level of granularity.

6. **Rajasthan doesn't follow the winter-smog pattern** — its peak AQI hits in May-June, driven by PM10 from desert dust storms, not the same mechanism as the other four states.

---

Data: CPCB via Kaggle (2015–2020). Built with Python and Streamlit.
