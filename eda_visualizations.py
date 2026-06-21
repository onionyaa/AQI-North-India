"""
Step 3: EDA visualizations for the Northern India AQI project.
Generates 15 plots and saves them to outputs/figures/.

Run directly or import run_full_eda(df) after preprocessing + AQI calculation.
"""

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

FIG_DIR = os.path.join("outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
matplotlib.rc("font", **{"family": "DejaVu Sans", "size": 11})

# one color per state, reused across plots
STATE_COLORS = {
    "Delhi": "#e74c3c",
    "Punjab": "#3498db",
    "Haryana": "#2ecc71",
    "Uttar Pradesh": "#f39c12",
    "Rajasthan": "#9b59b6",
}

CATEGORY_COLORS = {
    "Good": "#27ae60",
    "Satisfactory": "#f1c40f",
    "Moderate": "#e67e22",
    "Poor": "#e74c3c",
    "Very Poor": "#8e44ad",
    "Severe": "#2c3e50",
}

CATEGORY_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
SOURCE_TEXT = "Source: CPCB via Kaggle (2015-2020)  |  States: Delhi, Punjab, Haryana, UP, Rajasthan"


def _save(fig, filename):
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved -> {path}")
    return path


def _source(ax):
    ax.annotate(SOURCE_TEXT, xy=(0.5, -0.08), xycoords="axes fraction",
                ha="center", fontsize=8, color="gray", style="italic")


def plot_aqi_distribution(df):
    """Histogram + KDE of overall AQI. Expect a right skew."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["AQI_Final"].dropna(), bins=60, kde=True,
                 color="#e74c3c", edgecolor="white", linewidth=0.3,
                 ax=ax, stat="density")
    ax.axvline(df["AQI_Final"].mean(), color="#3498db", lw=2, ls="--",
               label=f"Mean = {df['AQI_Final'].mean():.0f}")
    ax.axvline(df["AQI_Final"].median(), color="#2ecc71", lw=2, ls="-.",
               label=f"Median = {df['AQI_Final'].median():.0f}")
    for lo, hi, cat in [(0,50,"Good"),(50,100,"Satisfactory"),
                        (100,200,"Moderate"),(200,300,"Poor"),
                        (300,400,"Very Poor"),(400,500,"Severe")]:
        ax.axvspan(lo, hi, alpha=0.07, color=CATEGORY_COLORS[cat])
    ax.set_title("AQI Distribution - Northern India (2015-2020)", fontsize=14, fontweight="bold")
    ax.set_xlabel("AQI Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend()
    _source(ax)
    return _save(fig, "01_aqi_distribution.png")


def plot_aqi_trend_over_time(df):
    """Monthly median AQI per state — shows seasonal cycles and the 2020 COVID dip."""
    monthly = (
        df.groupby(["State", pd.Grouper(key="Datetime", freq="ME")])["AQI_Final"]
        .median().reset_index()
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    for state, grp in monthly.groupby("State"):
        ax.plot(grp["Datetime"], grp["AQI_Final"],
                label=state, color=STATE_COLORS.get(state, "gray"), lw=2)
    ax.axhline(100, color="orange", ls="--", lw=1, alpha=0.7, label="Moderate threshold")
    ax.axhline(200, color="red", ls="--", lw=1, alpha=0.7, label="Poor threshold")
    ax.set_title("Monthly Median AQI by State (2015-2020)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Median AQI", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    _source(ax)
    return _save(fig, "02_aqi_trend_over_time.png")


def plot_monthly_aqi_trends(df):
    """Box plot by calendar month — Nov/Dec worst, monsoon months cleanest."""
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.boxplot(data=df, x="Month", y="AQI_Final", palette="coolwarm",
                notch=True, ax=ax, order=range(1, 13),
                flierprops={"marker": ".", "markersize": 2, "alpha": 0.3})
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels)
    ax.axhline(100, color="orange", ls="--", lw=1, alpha=0.6)
    ax.axhline(200, color="red", ls="--", lw=1, alpha=0.6)
    ax.set_title("Monthly AQI Distribution - All States", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("AQI", fontsize=12)
    _source(ax)
    return _save(fig, "03_monthly_aqi_trends.png")


def plot_seasonal_aqi_trends(df):
    """Violin plot by season. Winter often shows a bimodal shape."""
    season_order = ["Winter", "Spring", "Monsoon", "Autumn"]
    season_colors = {"Winter": "#3498db", "Spring": "#2ecc71",
                     "Monsoon": "#e67e22", "Autumn": "#e74c3c"}
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=df[df["Season"].isin(season_order)],
                   x="Season", y="AQI_Final", order=season_order,
                   palette=season_colors, inner="quartile", ax=ax)
    ax.set_title("Seasonal AQI Distribution - Northern India", fontsize=14, fontweight="bold")
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("AQI", fontsize=12)
    _source(ax)
    return _save(fig, "04_seasonal_aqi_trends.png")


def plot_statewise_aqi_comparison(df):
    """Mean AQI with 95% CI bars — Delhi is consistently worst."""
    state_stats = (
        df.groupby("State")["AQI_Final"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    state_stats["ci95"] = 1.96 * state_stats["std"] / np.sqrt(state_stats["count"])
    state_stats = state_stats.sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(state_stats["State"], state_stats["mean"],
                  color=[STATE_COLORS.get(s, "#95a5a6") for s in state_stats["State"]],
                  yerr=state_stats["ci95"], capsize=5, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, state_stats["mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(100, color="orange", ls="--", lw=1, alpha=0.7, label="Moderate (100)")
    ax.axhline(200, color="red", ls="--", lw=1, alpha=0.7, label="Poor (200)")
    ax.set_title("State-wise Mean AQI (2015-2020)", fontsize=14, fontweight="bold")
    ax.set_xlabel("State", fontsize=12)
    ax.set_ylabel("Mean AQI (±95% CI)", fontsize=12)
    ax.legend()
    _source(ax)
    return _save(fig, "05_statewise_aqi_comparison.png")


def plot_citywise_aqi_comparison(df):
    """Top 15 cities by mean AQI — industrial cities punch above their weight."""
    city_avg = (
        df.groupby(["City", "State"])["AQI_Final"]
        .mean().reset_index()
        .sort_values("AQI_Final", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [STATE_COLORS.get(s, "#95a5a6") for s in city_avg["State"]]
    bars = ax.barh(city_avg["City"][::-1], city_avg["AQI_Final"][::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.4)
    for bar, val in zip(bars, city_avg["AQI_Final"][::-1]):
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", fontsize=9)
    legend_elements = [Patch(facecolor=c, label=s) for s, c in STATE_COLORS.items()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.set_title("Top 15 Cities by Mean AQI", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean AQI", fontsize=12)
    _source(ax)
    return _save(fig, "06_citywise_aqi_comparison.png")


def plot_pollutant_distributions(df):
    """KDE + histogram for each pollutant, clipped at 99th pct to handle outliers."""
    poll_cols = [c for c in ["PM2.5","PM10","SO2","NOx","NH3","CO","O3"] if c in df.columns]
    ncols = 3
    nrows = int(np.ceil(len(poll_cols) / ncols))
    colors = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c","#e67e22"]

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(poll_cols):
        data = df[col].dropna()
        cap = data.quantile(0.99)
        data = data[data <= cap]
        axes[i].hist(data, bins=50, density=True, color=colors[i],
                     alpha=0.5, edgecolor="white", linewidth=0.3)
        data.plot.kde(ax=axes[i], color=colors[i], lw=2)
        axes[i].set_title(col, fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Concentration (µg/m³)")
        axes[i].set_ylabel("Density")
        axes[i].text(0.97, 0.95, f"Median: {data.median():.1f}",
                     transform=axes[i].transAxes, ha="right", va="top",
                     fontsize=8, color="gray")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Pollutant Distributions", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    return _save(fig, "07_pollutant_distributions.png")


def plot_correlation_heatmap(df):
    """Pearson correlations"""
    num_cols = [c for c in ["AQI_Final","PM2.5","PM10","SO2","NOx","NH3",
                             "CO","O3","Month","Hour","Year"] if c in df.columns]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, linecolor="white",
                annot_kws={"size": 9}, ax=ax)
    ax.set_title("Correlation Matrix - AQI & Pollutants", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "08_correlation_heatmap.png")


def plot_pm25_trend(df):
    """Monthly PM2.5 per state vs WHO (15) and CPCB (60) limits."""
    if "PM2.5" not in df.columns:
        print("  PM2.5 not found, skipping")
        return
    monthly = (
        df.groupby(["State", pd.Grouper(key="Datetime", freq="ME")])["PM2.5"]
        .median().reset_index()
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    for state, grp in monthly.groupby("State"):
        ax.plot(grp["Datetime"], grp["PM2.5"], label=state,
                color=STATE_COLORS.get(state, "gray"), alpha=0.7, lw=1.5)
    ax.axhline(60, color="#e67e22", ls="--", lw=1.5, label="CPCB 24h limit (60)")
    ax.axhline(15, color="#2ecc71", ls=":", lw=1.5, label="WHO annual limit (15)")
    ax.set_title("Monthly Median PM2.5 by State (2015-2020)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend(fontsize=9)
    _source(ax)
    return _save(fig, "09_pm25_trend.png")


def plot_pm10_trend(df):
    """Monthly PM10 — Rajasthan spikes in summer (dust), others in winter."""
    if "PM10" not in df.columns:
        print("  PM10 not found, skipping")
        return
    monthly = (
        df.groupby(["State", pd.Grouper(key="Datetime", freq="ME")])["PM10"]
        .median().reset_index()
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    for state, grp in monthly.groupby("State"):
        ax.plot(grp["Datetime"], grp["PM10"], label=state,
                color=STATE_COLORS.get(state, "gray"), lw=1.8)
    ax.axhline(100, color="#e67e22", ls="--", lw=1.5, label="CPCB 24h limit (100)")
    ax.set_title("Monthly Median PM10 by State (2015-2020)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM10 (µg/m³)")
    ax.legend(fontsize=9)
    _source(ax)
    return _save(fig, "10_pm10_trend.png")


def plot_winter_vs_summer(df):
    """Winter vs Spring AQI side by side — the gap is huge, especially Delhi."""
    df2 = df[df["Season"].isin(["Winter", "Spring"])].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df2, x="State", y="AQI_Final", hue="Season",
                hue_order=["Winter", "Spring"],
                palette={"Winter": "#3498db", "Spring": "#e67e22"},
                ax=ax, notch=True,
                flierprops={"marker": ".", "markersize": 2, "alpha": 0.3})
    ax.set_title("Winter vs Spring AQI by State", fontsize=14, fontweight="bold")
    ax.set_xlabel("State")
    ax.set_ylabel("AQI")
    ax.legend(title="Season")
    plt.xticks(rotation=15)
    _source(ax)
    return _save(fig, "11_winter_vs_summer.png")


def plot_crop_burning_season(df):
    """Daily AQI in Sep-Dec for Punjab, Haryana, Delhi with burning window shaded."""
    burn_states = ["Punjab", "Haryana", "Delhi"]
    burn_df = df[df["State"].isin(burn_states) & df["Month"].isin([9, 10, 11, 12])].copy()

    daily = burn_df.groupby(["State", "Datetime"])["AQI_Final"].median().reset_index()

    fig, ax = plt.subplots(figsize=(13, 5))
    for state, grp in daily.groupby("State"):
        ax.plot(grp["Datetime"], grp["AQI_Final"],
                label=state, color=STATE_COLORS.get(state, "gray"), lw=1.5, alpha=0.8)

    # shade Oct 15 - Nov 30 each year
    for yr in range(2015, 2021):
        try:
            ax.axvspan(pd.Timestamp(f"{yr}-10-15"), pd.Timestamp(f"{yr}-11-30"),
                       alpha=0.08, color="#e67e22")
        except Exception:
            pass

    ax.annotate("paddy burning window (Oct 15 - Nov 30)",
                xy=(0.50, 0.97), xycoords="axes fraction",
                ha="center", fontsize=8, color="#e67e22")
    ax.set_title("Crop Burning Season - Punjab, Haryana & Delhi", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Median AQI")
    ax.legend(fontsize=9)
    _source(ax)
    return _save(fig, "12_crop_burning_season.png")


def plot_top10_polluted_cities(df):
    """Top 10 cities by mean AQI with state color coding."""
    top10 = (
        df.groupby(["City", "State"])["AQI_Final"]
        .mean().reset_index()
        .sort_values("AQI_Final", ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(10), top10["AQI_Final"],
                  color=[STATE_COLORS.get(s, "#95a5a6") for s in top10["State"]],
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(10))
    ax.set_xticklabels(
        [f"{c}\n({s[:2]})" for c, s in zip(top10["City"], top10["State"])],
        fontsize=9
    )
    for bar, val in zip(bars, top10["AQI_Final"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    legend_elements = [Patch(facecolor=c, label=s) for s, c in STATE_COLORS.items()
                       if s in top10["State"].values]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.set_title("Top 10 Most Polluted Cities by Mean AQI", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean AQI")
    _source(ax)
    return _save(fig, "13_top10_polluted_cities.png")


def plot_aqi_category_distribution(df):
    """100% stacked bar — shows what fraction of hours each state spends in each category."""
    cat_counts = df.groupby(["State", "AQI_Category"]).size().reset_index(name="count")
    totals = cat_counts.groupby("State")["count"].transform("sum")
    cat_counts["pct"] = 100 * cat_counts["count"] / totals

    pivot = cat_counts.pivot(index="State", columns="AQI_Category", values="pct").fillna(0)
    for cat in CATEGORY_ORDER:
        if cat not in pivot.columns:
            pivot[cat] = 0
    pivot = pivot[CATEGORY_ORDER]

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax,
               color=[CATEGORY_COLORS[c] for c in CATEGORY_ORDER],
               edgecolor="white", linewidth=0.3, width=0.7)
    ax.set_title("AQI Category Distribution by State (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("State")
    ax.set_ylabel("% of Hours")
    ax.legend(title="Category", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    _source(ax)
    plt.tight_layout()
    return _save(fig, "14_aqi_category_distribution.png")


def plot_yearly_aqi_trend(df):
    """Annual mean AQI with error bars and linear trend line per state."""
    yearly = (
        df.groupby(["State", "Year"])["AQI_Final"]
        .agg(["mean", "sem"]).reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    for state, grp in yearly.groupby("State"):
        grp = grp.sort_values("Year")
        color = STATE_COLORS.get(state, "gray")
        ax.errorbar(grp["Year"], grp["mean"], yerr=1.96 * grp["sem"],
                    label=state, color=color, lw=2, marker="o",
                    markersize=7, capsize=4, capthick=1.5)
        if len(grp) > 2:
            z = np.polyfit(grp["Year"], grp["mean"], 1)
            ax.plot(grp["Year"], np.poly1d(z)(grp["Year"]),
                    color=color, ls="--", lw=1, alpha=0.5)
    ax.set_title("Yearly Mean AQI by State (2015-2020)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Mean AQI")
    ax.set_xticks(sorted(df["Year"].unique()))
    ax.legend(fontsize=9)
    _source(ax)
    return _save(fig, "15_yearly_aqi_trend.png")


def run_full_eda(df):
    """Run all 15 plots. Returns list of saved file paths."""
    plots = [
        ("01. AQI Distribution",         plot_aqi_distribution),
        ("02. AQI Trend Over Time",       plot_aqi_trend_over_time),
        ("03. Monthly AQI Trends",        plot_monthly_aqi_trends),
        ("04. Seasonal AQI Trends",       plot_seasonal_aqi_trends),
        ("05. State-wise Comparison",     plot_statewise_aqi_comparison),
        ("06. City-wise Comparison",      plot_citywise_aqi_comparison),
        ("07. Pollutant Distributions",   plot_pollutant_distributions),
        ("08. Correlation Heatmap",       plot_correlation_heatmap),
        ("09. PM2.5 Trend",              plot_pm25_trend),
        ("10. PM10 Trend",               plot_pm10_trend),
        ("11. Winter vs Summer",          plot_winter_vs_summer),
        ("12. Crop Burning Season",       plot_crop_burning_season),
        ("13. Top 10 Polluted Cities",    plot_top10_polluted_cities),
        ("14. AQI Category Distribution", plot_aqi_category_distribution),
        ("15. Yearly AQI Trend",          plot_yearly_aqi_trend),
    ]

    saved = []
    for name, func in plots:
        print(f"\n{name}")
        try:
            path = func(df)
            if path:
                saved.append(path)
        except Exception as e:
            print(f"  failed: {e}")

    print(f"\ndone — {len(saved)}/15 figures saved to {FIG_DIR}")
    return saved


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.aqi_calculator import calculate_aqi
    from src.data_preprocessing import load_and_preprocess

    df = load_and_preprocess(save=False)
    df = calculate_aqi(df)
    run_full_eda(df)
