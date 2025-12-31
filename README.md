## Injury Prediction Project

### Prerequisites
- Python 3.8+
- Windows PowerShell or terminal

### 1) Install
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2) Prepare Data
Place CSVs under `data/raw/` in the following folders:
- `wellness/` (fatigue.csv, stress.csv, soreness.csv, sleep_quality.csv, sleep_duration.csv, readiness.csv, mood.csv)
- `training-load/` (daily_load.csv, weekly_load.csv, strain.csv, monotony.csv, acwr.csv, atl.csv, ctl28.csv, ctl42.csv)
- `injury/` (injury.csv)

Build the processed dataset:
```bash
python data_preprocessing.py
```
This creates `data/processed/master_dataset.csv` with engineered features and label `label_next_10_15`.

### 3) Train Models
```bash
python train_model.py
```
What happens:
- Trains Logistic Regression and Random Forest with a time-based split
- Tunes the decision threshold for higher precision with a minimum recall floor
- Saves:
  - `models/model_best.joblib`, `models/model_<name>.joblib`
  - `models/metadata.json` (best_model, feature_columns, best_threshold)
  - `reports/metrics_<model>.json`, `reports/metrics_overall.json`
  - Plots to `reports/plots/`

### 4) CLI Predictions (optional)
```bash
python predict.py
```
Outputs:
- `predictions/daily_predictions.csv` (with recommendations)
- `predictions/weekly_predictions.csv`
- Chart images under `predictions/plots/`

### 5) Interactive App (Streamlit)
```bash
streamlit run app.py
```
Tabs:
- Quick Prediction: enter 4–6 inputs, see risk gauge, recommendations
- Historical Analysis: daywise/weekwise charts from processed data
- Batch Processing: upload CSV, score in bulk, download results
- Model Insights: metrics table, feature list, importance, confusion matrix
- Risk Dashboard: KPIs, distributions, temporal trends, correlation
- Scenario Testing: sweep a parameter and compare scenarios
- User Guide: role-based instructions (Coach, Athlete, Analyst, Admin)

### Notes
- To retrain, rerun step 3; the app picks up updated artifacts automatically.
- If you change features substantially, rebuild processed data (step 2) first.
### Interactive UI (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Features:
- Enter 4-6 key inputs (e.g., `acwr`, `monotony`, `strain`, `sleep_duration`, `fatigue`, `stress`)
- Ranges derived from historical data (5th–95th percentiles)
- Shows predicted risk, thresholded risk level, and tailored recommendations
- Clean plots for predictions are saved under `predictions/plots/`

Advanced Dashboard Tabs:
- Overview: What the app does, how to use it, current model and threshold.
- Single Prediction: Inputs, risk percent, progress bar, recommendations.
- Daywise & Weekwise: Compute over processed data with interactive charts and downloadable CSVs.
- Batch CSV: Upload your own CSV to score in bulk; optional `date` column for charts.
- Model Insights: Permutation importance to understand feature impact.
- Scenarios: Compare two sets of inputs side by side and see risk deltas.


