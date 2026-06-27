import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model     import LinearRegression
from sklearn.ensemble         import RandomForestRegressor
from sklearn.model_selection  import train_test_split, cross_val_score, KFold
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline         import Pipeline

import xgboost as xgb

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR  = os.path.join(PROJECT_ROOT, "models")
FIG_DIR    = os.path.join(PROJECT_ROOT, "outputs", "figures")
REPORT_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def _evaluate(y_true, y_pred, model_name):
    """
    Compute RMSE, MAE, and R² for a model's predictions.

    WHY THESE THREE METRICS?
        RMSE  : Penalises large errors heavily.  A model that is usually
                right but occasionally very wrong gets a high RMSE.
        MAE   : Treats all errors equally.  Easier to interpret
                (average error in AQI units).
        R²    : How much variance in AQI does the model explain?
                1.0 = perfect, 0.0 = no better than predicting the mean,
                <0 = worse than the mean.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {model_name:<22} RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}")
    return {"Model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2}

def _cross_validate(model, X, y, cv=5, model_name=""):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, scoring="r2", cv=kf)
    print(f"  {model_name:<22} CV R² = {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean(), scores.std()

def prepare_ml_data(df, feature_cols, target_col="AQI_Final",
                    test_size=0.2, random_state=42):
    print("[ML] Preparing data …")
    if "Datetime" in df.columns:
        df = df.sort_values("Datetime")
    df_clean = df[needed].dropna()

    X = df_clean[feature_cols].values
    y = df_clean[target_col].values

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")
    print(f"  Features: {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test, df_clean.iloc[split_idx:].copy()


def train_linear_regression(X_train, y_train):
    print("\n[Model 1] Linear Regression")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_random_forest(X_train, y_train):

    print("\n[Model 2] Random Forest")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=10,
        min_samples_split=20,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, X_val=None, y_val=None):
  
    print("\n[Model 3] XGBoost")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        verbosity=0,
    )
    eval_set = [(X_val, y_val)] if X_val is not None else [(X_train, y_train)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    print(f"  Best iteration: {model.best_iteration}")
    return model


def plot_actual_vs_predicted(y_test, predictions_dict):
   
    n = len(predictions_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    for ax, (name, y_pred), color in zip(axes, predictions_dict.items(), colors):
        ax.scatter(y_test, y_pred, alpha=0.2, s=8, color=color)
        lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Perfect fit")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Actual AQI")
        ax.set_ylabel("Predicted AQI")
        r2 = r2_score(y_test, y_pred)
        ax.text(0.05, 0.92, f"R² = {r2:.3f}", transform=ax.transAxes,
                fontsize=10, color="black",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        ax.legend()
    plt.suptitle("Actual vs Predicted AQI", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "ml_actual_vs_predicted.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved → {path}")


def plot_residuals(y_test, predictions_dict):
    
    n = len(predictions_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]
    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    for ax, (name, y_pred), color in zip(axes, predictions_dict.items(), colors):
        residuals = y_test - y_pred
        sns.histplot(residuals, bins=60, kde=True, color=color, ax=ax)
        ax.axvline(0, color="black", lw=1.5, ls="--")
        ax.set_title(f"Residuals — {name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Residual (Actual − Predicted)")
        ax.set_ylabel("Count")
        ax.text(0.97, 0.95, f"Mean={residuals.mean():.1f}\nStd={residuals.std():.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "ml_residuals.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved → {path}")


def plot_feature_importance(model, feature_cols, model_name, top_n=20):
  
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        return   # Skip for linear model — use coefficients separately
    else:
        return

    idx = np.argsort(importances)[-top_n:][::-1]
    top_features = np.array(feature_cols)[idx]
    top_values   = importances[idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(top_n), top_values[::-1],
                   color=plt.cm.viridis(np.linspace(0.2, 0.9, top_n)))
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_title(f"Feature Importance — {model_name} (Top {top_n})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    path = os.path.join(FIG_DIR, f"ml_feature_importance_{safe_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved → {path}")


def plot_model_comparison(results_df):
   
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = ["RMSE", "MAE", "R2"]
    titles  = ["Root Mean Square Error ↓", "Mean Absolute Error ↓", "R² Score ↑"]
    colors  = ["#e74c3c", "#f39c12", "#2ecc71"]
    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        bars = ax.bar(results_df["Model"], results_df[metric],
                      color=color, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (results_df[metric].max() * 0.01),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "ml_model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved → {path}")


def run_ml_pipeline(df, feature_cols, target_col="AQI_Final"):
  
    print("\n" + "="*55)
    print("  MACHINE LEARNING PIPELINE")
    print("="*55)

    X_train, X_test, y_train, y_test, df_test_raw = prepare_ml_data(
        df, feature_cols, target_col
    )
   
    models = {
        "Linear Regression": train_linear_regression(X_train, y_train),
        "Random Forest":     train_random_forest(X_train, y_train),
        "XGBoost":           train_xgboost(X_tr, y_tr, X_val, y_val),
    }

    print("\n─── Test-set performance ───")
    results = []
    predictions = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0, 500)   # AQI physically bounded
        predictions[name] = y_pred
        results.append(_evaluate(y_test, y_pred, name))
        df_test_raw[f"Pred_{name.replace(' ', '_')}"] = y_pred

    print("\n─── 5-fold cross-validation (R²) ───")
    for name, model in models.items():
        if "XGBoost" in name:
            # XGBoost CV with built-in method for speed
            dtrain = xgb.DMatrix(X_train, label=y_train)
            cv_res = xgb.cv(
                model.get_xgb_params(), dtrain,
                num_boost_round=model.best_iteration,
                nfold=5, seed=42, verbose_eval=False
            )
            mean_rmse = cv_res["test-rmse-mean"].iloc[-1]
            ss_res = mean_rmse ** 2 * len(y_train)
            ss_tot = np.var(y_train) * len(y_train)
            mean_r2 = 1 - ss_res / ss_tot
            print(f"  {name:<22} CV R² ≈ {mean_r2:.4f}")
        else:
            _cross_validate(model, X_train, y_train, model_name=name)

    print("\n─── Generating evaluation plots ───")
    results_df = pd.DataFrame(results)
    plot_actual_vs_predicted(y_test, predictions)
    plot_residuals(y_test, predictions)
    plot_model_comparison(results_df)
    for name, model in models.items():
        plot_feature_importance(model, feature_cols, name)

    best_row   = results_df.loc[results_df["RMSE"].idxmin()]
    best_name  = best_row["Model"]
    best_model = models[best_name]

    print(f"\n{'='*55}")
    print(f"  BEST MODEL: {best_name}")
    print(f"    RMSE = {best_row['RMSE']:.2f}")
    print(f"    MAE  = {best_row['MAE']:.2f}")
    print(f"    R²   = {best_row['R2']:.4f}")
    print(f"{'='*55}")

    print("\n─── Saving models ───")
    name_file = {
        "Linear Regression": "linear_regression.pkl",
        "Random Forest":     "random_forest.pkl",
        "XGBoost":           "xgboost.pkl",
    }
    for name, model in models.items():
        path = os.path.join(MODEL_DIR, name_file[name])
        joblib.dump(model, path)
        print(f"  [✓] Saved → {path}")

    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    print(f"  [✓] Best model saved → models/best_model.pkl")

    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.pkl"))

    results_df.to_csv(os.path.join(REPORT_DIR, "ml_results.csv"), index=False)
    print(f"  [✓] Results → outputs/reports/ml_results.csv")

    return results_df, best_model, df_test_raw

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_preprocessing import load_and_preprocess
    from src.aqi_calculator import calculate_aqi
    from src.feature_engineering import engineer_features, get_feature_columns

    df = load_and_preprocess()
    df = calculate_aqi(df)
    df = engineer_features(df)
    feature_cols, target = get_feature_columns(df)

    results, best_model, df_test = run_ml_pipeline(df, feature_cols, target)
    print("\nModel comparison summary:")
    print(results.to_string(index=False))
