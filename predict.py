import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


MODELS_DIR = os.path.join("models")
PROCESSED_PATH = os.path.join("data", "processed", "master_dataset.csv")
PREDICTIONS_DIR = os.path.join("predictions")
PLOTS_DIR = os.path.join(PREDICTIONS_DIR, "plots")


def load_best_model():
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    best_path = os.path.join(MODELS_DIR, "model_best.joblib")
    if not os.path.exists(best_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Best model or metadata not found. Train the model first.")
    model = joblib.load(best_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    threshold = meta.get("best_threshold", 0.5)
    return model, meta["feature_columns"], float(threshold)


def generate_recommendations(row: pd.Series) -> list:
    recs = []
    # Simple heuristic rules
    acwr = row.get("acwr") or row.get("ACWR")
    if acwr is not None and pd.notna(acwr) and acwr > 1.5:
        recs.append("High ACWR: Reduce training intensity, monitor acute spikes.")

    if row.get("sleep_duration") is not None and pd.notna(row.get("sleep_duration")) and row.get("sleep_duration") < 7:
        recs.append("Low sleep duration: Target ≥ 7 hours, prioritize recovery.")

    if row.get("fatigue") is not None and pd.notna(row.get("fatigue")) and row.get("fatigue") >= 7:
        recs.append("High fatigue: Add low-intensity or rest day; monitor wellness.")

    if row.get("monotony") is not None and pd.notna(row.get("monotony")) and row.get("monotony") > 2.0:
        recs.append("High monotony: Increase training variation to lower strain.")

    if row.get("stress_recovery_index") is not None and pd.notna(row.get("stress_recovery_index")) and row.get("stress_recovery_index") > 1.2:
        recs.append("Imbalanced stress-recovery: Improve sleep and readiness before increasing load.")

    return recs


def main():
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError("Processed dataset missing. Run preprocessing first.")

    model, feature_cols, best_threshold = load_best_model()
    df = pd.read_csv(PROCESSED_PATH)

    # Prepare features identically to training
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    proba = model.predict_proba(X)[:, 1]

    out = df[["date"]].copy() if "date" in df.columns else pd.DataFrame({"row": np.arange(len(df))})
    out["injury_risk"] = proba
    out["risk_label"] = np.where(out["injury_risk"] >= best_threshold, 1, 0)

    # Weekwise aggregation
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]) 
        out["year_week"] = out["date"].dt.strftime("%G-%V")
        weekly = out.groupby("year_week")["injury_risk"].agg(["mean", "max"]).reset_index()
    else:
        weekly = pd.DataFrame()

    # Recommendations per day
    recs = []
    for idx, row in df.iterrows():
        suggestions = generate_recommendations(row)
        recs.append("; ".join(suggestions))
    out["recommendations"] = recs

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    daily_path = os.path.join(PREDICTIONS_DIR, "daily_predictions.csv")
    weekly_path = os.path.join(PREDICTIONS_DIR, "weekly_predictions.csv")

    out.to_csv(daily_path, index=False)
    if not weekly.empty:
        weekly.to_csv(weekly_path, index=False)

    # Plots: time-series of predicted risk
    try:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

        # Daily risk over time
        if "date" in out.columns:
            out_sorted = out.sort_values("date").reset_index(drop=True)
            out_sorted["risk_roll7"] = out_sorted["injury_risk"].rolling(7, min_periods=3).mean()

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(out_sorted["date"], out_sorted["injury_risk"], label="Daily risk", color="#1f77b4", alpha=0.6)
            ax.plot(out_sorted["date"], out_sorted["risk_roll7"], label="7d rolling avg", color="#ff7f0e", linewidth=2)
            ax.axhline(y=best_threshold, color="#d62728", linestyle="--", label=f"Threshold = {best_threshold:.2f}")
            ax.set_ylabel("Predicted injury risk (0-1)")
            ax.set_xlabel("Date")
            ax.set_title("Daily Predicted Injury Risk")
            ax.legend(loc="upper left")
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(os.path.join(PLOTS_DIR, "daily_risk.png"), dpi=150)
            plt.close(fig)

        # Weekly risk (mean and max)
        if not weekly.empty:
            # Compute high-risk day counts per week using threshold
            if "date" in out.columns:
                tmp = out.copy()
                tmp["date"] = pd.to_datetime(tmp["date"]) 
                tmp["year_week"] = tmp["date"].dt.strftime("%G-%V")
                weekly_counts = tmp.groupby("year_week")["risk_label"].sum().reset_index(name="high_risk_days")
                weekly = weekly.merge(weekly_counts, on="year_week", how="left").fillna({"high_risk_days": 0})

            fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
            axes[0].plot(weekly["year_week"], weekly["mean"], label="Weekly mean", color="#2ca02c")
            axes[0].axhline(y=best_threshold, color="#d62728", linestyle="--", linewidth=1)
            axes[0].set_ylabel("Mean risk")
            axes[0].set_title("Weekly Predicted Injury Risk - Mean")
            axes[0].legend(loc="upper left")

            axes[1].plot(weekly["year_week"], weekly["max"], label="Weekly max", color="#ff7f0e")
            axes[1].axhline(y=best_threshold, color="#d62728", linestyle="--", linewidth=1)
            axes[1].set_ylabel("Max risk")
            axes[1].set_title("Weekly Predicted Injury Risk - Max")
            axes[1].legend(loc="upper left")

            for ax in axes:
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_horizontalalignment("right")
            fig.tight_layout()
            fig.savefig(os.path.join(PLOTS_DIR, "weekly_risk.png"), dpi=150)
            plt.close(fig)

            # Weekly high-risk day counts (bar)
            if "high_risk_days" in weekly.columns:
                fig, ax = plt.subplots(figsize=(12, 3.5))
                ax.bar(weekly["year_week"], weekly["high_risk_days"], color="#9467bd")
                ax.set_ylabel("Days ≥ threshold")
                ax.set_xlabel("Year-Week")
                ax.set_title("High-Risk Days per Week")
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_horizontalalignment("right")
                fig.tight_layout()
                fig.savefig(os.path.join(PLOTS_DIR, "weekly_high_risk_days.png"), dpi=150)
                plt.close(fig)

        # Distribution of predicted risk (all days)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(out["injury_risk"], bins=30, kde=True, color="#1f77b4", ax=ax)
        ax.axvline(best_threshold, color="#d62728", linestyle="--", label=f"Threshold = {best_threshold:.2f}")
        ax.set_xlabel("Predicted injury risk")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Predicted Injury Risk")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "risk_distribution.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        # Keep predictions even if plotting fails
        print(f"Plot generation failed: {e}")

    print(f"Saved daily predictions to {daily_path}")
    if os.path.exists(weekly_path):
        print(f"Saved weekly predictions to {weekly_path}")


if __name__ == "__main__":
    main()


