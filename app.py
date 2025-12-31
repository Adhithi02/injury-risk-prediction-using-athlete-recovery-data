import os
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
from scipy import stats

# Configuration
MODELS_DIR = os.path.join("models")
REPORTS_DIR = os.path.join("reports")
PROCESSED_PATH = os.path.join("data", "processed", "master_dataset.csv")

# Enhanced styling
st.set_page_config(
    page_title="Injury Risk Intelligence System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .risk-moderate {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    }
    .risk-low {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .recommendation-item {
        background-color: #fff3cd;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 3px solid #ffc107;
    }
    .success-box {
        background-color: #d4edda;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_processed() -> pd.DataFrame:
    """Load processed dataset with error handling"""
    try:
        if os.path.exists(PROCESSED_PATH):
            df = pd.read_csv(PROCESSED_PATH)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            return df
    except Exception as e:
        st.warning(f"Error loading data: {str(e)}")
    return pd.DataFrame()


def _load_threshold_for_model(model_name: str, default_threshold: float = 0.5) -> float:
    """Load optimal threshold for a specific model"""
    metrics_path = os.path.join(REPORTS_DIR, f"metrics_{model_name}.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            if "threshold" in m:
                return float(m["threshold"])
        except Exception:
            pass
    
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "best_threshold" in meta:
                return float(meta["best_threshold"])
        except Exception:
            pass
    return float(default_threshold)


@st.cache_resource
def load_model_and_meta(selected: str = "best") -> Tuple[object, List[str], float, str, Dict]:
    """Load model with comprehensive metadata"""
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    
    if selected == "best":
        path = os.path.join(MODELS_DIR, "model_best.joblib")
    else:
        path = os.path.join(MODELS_DIR, f"model_{selected}.joblib")
    
    if not os.path.exists(path):
        return None, [], 0.5, selected, {}
    
    model = joblib.load(path)
    feature_cols: List[str] = []
    metadata = {}
    
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            feature_cols = metadata.get("feature_columns", [])
        except Exception:
            feature_cols = []
    
    threshold = _load_threshold_for_model(selected)
    return model, feature_cols, float(threshold), selected, metadata


def list_trained_models() -> List[str]:
    """Return list of trained model names based on files under models/.
    Example files: model_logreg.joblib -> name 'logreg'. Always include 'best' if best exists.
    """
    names: List[str] = []
    try:
        if os.path.isdir(MODELS_DIR):
            for fn in os.listdir(MODELS_DIR):
                if fn.startswith("model_") and fn.endswith(".joblib") and fn != "model_best.joblib":
                    name = fn[len("model_"):-len(".joblib")]
                    names.append(name)
    except Exception:
        pass
    # Keep order deterministic
    names = sorted(list(set(names)))
    # Prepend 'best' if available
    if os.path.exists(os.path.join(MODELS_DIR, "model_best.joblib")):
        return ["best"] + names
    return names


def load_model_metrics(selected_model: str, metadata: Dict) -> Dict:
    """Load metrics JSON for selected model. If 'best', map to metadata['best_model']."""
    model_key = selected_model
    if selected_model == "best":
        model_key = metadata.get("best_model", "")
    metrics_path = os.path.join(REPORTS_DIR, f"metrics_{model_key}.json") if model_key else ""
    if metrics_path and os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def compute_live_metrics_from_data(model, feature_cols: List[str], data_df: pd.DataFrame, threshold: float) -> Dict:
    """Compute metrics on available processed data if labels exist.
    Used as a fallback when reports JSON lacks certain stats.
    """
    try:
        if data_df is None or data_df.empty or "label_next_10_15" not in data_df.columns:
            return {}
        X = pd.DataFrame(columns=feature_cols)
        for c in feature_cols:
            if c in data_df.columns and pd.api.types.is_numeric_dtype(data_df[c]):
                X[c] = data_df[c]
            else:
                X[c] = 0.0
        X = X.fillna(X.median(numeric_only=True))
        y_true = data_df["label_next_10_15"].astype(int).values
        y_score = model.predict_proba(X)[:, 1]
        y_pred = (y_score >= threshold).astype(int)

        # Compute metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        live = {}
        live["accuracy"] = float(accuracy_score(y_true, y_pred))
        live["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        live["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        live["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            live["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            pass
        return live
    except Exception:
        return {}


def calculate_risk_category(risk_score: float, threshold: float) -> Tuple[str, str]:
    """Categorize risk with color coding"""
    if risk_score >= threshold:
        return "HIGH RISK", "🔴"
    elif risk_score >= threshold * 0.7:
        return "MODERATE RISK", "🟡"
    else:
        return "LOW RISK", "🟢"


def generate_advanced_recommendations(row: pd.Series, risk_score: float, threshold: float) -> Dict[str, List[str]]:
    """Generate comprehensive, actionable recommendations"""
    recs = {
        "immediate": [],
        "short_term": [],
        "long_term": [],
        "positive": []
    }
    
    # ACWR Analysis
    acwr = row.get("acwr") or row.get("ACWR")
    if acwr is not None and pd.notna(acwr):
        if acwr > 1.5:
            recs["immediate"].append(f"⚠️ Critical ACWR spike detected ({acwr:.2f}). Reduce training volume by 20-30% immediately.")
        elif acwr > 1.3:
            recs["short_term"].append(f"📊 Elevated ACWR ({acwr:.2f}). Monitor closely and avoid intensity increases.")
        elif 0.8 <= acwr <= 1.3:
            recs["positive"].append(f"✅ Optimal ACWR range ({acwr:.2f}). Training load is well-balanced.")
    
    # Sleep Analysis
    sleep = row.get("sleep_duration")
    if sleep is not None and pd.notna(sleep):
        if sleep < 6:
            recs["immediate"].append(f"😴 Severe sleep deficit ({sleep:.1f}h). Prioritize 8+ hours tonight.")
        elif sleep < 7:
            recs["short_term"].append(f"🛏️ Suboptimal sleep ({sleep:.1f}h). Target 7-9 hours for proper recovery.")
        else:
            recs["positive"].append(f"✅ Good sleep duration ({sleep:.1f}h). Maintain this recovery pattern.")
    
    # Fatigue Management
    fatigue = row.get("fatigue")
    if fatigue is not None and pd.notna(fatigue):
        if float(fatigue) >= 8:
            recs["immediate"].append(f"🚨 Extreme fatigue ({fatigue}/10). Consider rest day or active recovery only.")
        elif float(fatigue) >= 7:
            recs["short_term"].append(f"😓 High fatigue ({fatigue}/10). Reduce training intensity by 30-40%.")
    
    # Monotony Analysis
    monotony = row.get("monotony")
    if monotony is not None and pd.notna(monotony):
        if monotony > 2.5:
            recs["long_term"].append(f"🔄 High training monotony ({monotony:.2f}). Introduce varied workouts and cross-training.")
        elif monotony > 2.0:
            recs["short_term"].append(f"📈 Elevated monotony ({monotony:.2f}). Add variety to training sessions.")
    
    # Stress-Recovery Balance
    sri = row.get("stress_recovery_index")
    if sri is not None and pd.notna(sri):
        if sri > 1.5:
            recs["immediate"].append(f"⚖️ Severe stress-recovery imbalance ({sri:.2f}). Implement recovery protocols now.")
        elif sri > 1.2:
            recs["short_term"].append(f"📉 Poor stress-recovery ratio ({sri:.2f}). Enhance recovery strategies.")
    
    # Overall Risk Guidance
    if risk_score >= threshold:
        recs["immediate"].append("🛑 HIGH RISK ALERT: Consider modifying or postponing high-intensity activities.")
    
    return recs


def create_risk_gauge(risk_score: float, threshold: float) -> go.Figure:
    """Create an interactive risk gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Injury Risk Score", 'font': {'size': 24}},
        delta={'reference': threshold * 100, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold * 70], 'color': '#d4edda'},
                {'range': [threshold * 70, threshold * 100], 'color': '#fff3cd'},
                {'range': [threshold * 100, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="white",
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_feature_importance_chart(model, feature_cols: List[str], X_sample: pd.DataFrame) -> go.Figure:
    """Create feature importance visualization"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return None
        
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=True).tail(10)
        
        fig = go.Figure(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker=dict(
                color=importance_df['Importance'],
                colorscale='Viridis'
            )
        ))
        
        fig.update_layout(
            title="Top 10 Risk Factors",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=400,
            showlegend=False
        )
        
        return fig
    except Exception:
        return None


def derive_ranges(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    """Derive realistic input ranges from data"""
    ranges: Dict[str, Dict[str, float]] = {}
    
    if df is None or df.empty:
        default_ranges = {
            'acwr': {'min': 0.5, 'max': 2.0, 'step': 0.01},
            'monotony': {'min': 1.0, 'max': 4.0, 'step': 0.1},
            'strain': {'min': 0.0, 'max': 10000.0, 'step': 100.0},
            'sleep_duration': {'min': 4.0, 'max': 12.0, 'step': 0.5},
            'fatigue': {'min': 1.0, 'max': 10.0, 'step': 1.0},
            'stress': {'min': 1.0, 'max': 10.0, 'step': 1.0}
        }
        for c in cols:
            ranges[c] = default_ranges.get(c, {'min': 0.0, 'max': 1.0, 'step': 0.01})
        return ranges
    
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            series = df[c].dropna()
            if len(series) == 0:
                ranges[c] = {'min': 0.0, 'max': 1.0, 'step': 0.01}
                continue
            
            lo = float(series.quantile(0.05))
            hi = float(series.quantile(0.95))
            
            if lo == hi:
                lo = float(series.min())
                hi = float(series.max())
            
            step = max(0.01, (hi - lo) / 100) if hi > lo else 0.01
            ranges[c] = {'min': lo, 'max': hi, 'step': step}
        else:
            ranges[c] = {'min': 0.0, 'max': 1.0, 'step': 0.01}
    
    return ranges


def create_trend_analysis(df: pd.DataFrame, risk_scores: np.ndarray) -> go.Figure:
    """Create comprehensive trend analysis"""
    if 'date' not in df.columns:
        return None
    
    trend_df = pd.DataFrame({
        'date': df['date'],
        'risk': risk_scores
    }).sort_values('date')
    
    trend_df['rolling_7'] = trend_df['risk'].rolling(7, min_periods=1).mean()
    trend_df['rolling_30'] = trend_df['risk'].rolling(30, min_periods=1).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trend_df['date'],
        y=trend_df['risk'],
        mode='markers',
        name='Daily Risk',
        marker=dict(size=4, color='lightblue', opacity=0.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=trend_df['date'],
        y=trend_df['rolling_7'],
        mode='lines',
        name='7-Day Average',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=trend_df['date'],
        y=trend_df['rolling_30'],
        mode='lines',
        name='30-Day Average',
        line=dict(color='purple', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Risk Trend Analysis",
        xaxis_title="Date",
        yaxis_title="Risk Score",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Injury Risk Intelligence System</h1>', unsafe_allow_html=True)
    st.markdown("**Advanced Predictive Analytics for Athlete Safety & Performance Optimization**")
    
    # Sidebar Configuration
    with st.sidebar:
        # Sidebar Header Card
        st.markdown("""
        <div style='background:linear-gradient(90deg,#667eea,#764ba2);padding:14px;border-radius:10px;color:white;'>
            <div style='font-weight:700;font-size:18px;'>Risk Intelligence Settings</div>
            <div style='opacity:0.9;font-size:12px;'>Configure model, threshold, and visuals</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(" ")

        st.header("⚙️ Configuration")

        # Only show trained models discovered on disk
        available_models = list_trained_models()
        if not available_models:
            available_models = ["best"]
        selected_model_name = st.selectbox(
            "Model Selection",
            available_models,
            index=0,
            help="Choose the trained model for predictions"
        )

        theme_choice = st.selectbox(
            "Theme",
            ["Light", "Dark"],
            index=0,
            help="Switch visualization theme"
        )
        
        st.markdown("---")
        st.subheader("🎯 Threshold Settings")
        
        override_thr = st.checkbox(
            "Custom Threshold",
            value=False,
            help="Override the optimized threshold"
        )
        
        custom_thr = st.slider(
            "Threshold Value",
            0.0, 1.0, 0.5, 0.01,
            disabled=not override_thr,
            help="Higher threshold = more conservative predictions"
        )
        
        st.markdown("---")
        st.subheader("📊 Display Options")
        
        show_confidence = st.checkbox("Show Confidence Intervals", value=True)
        show_insights = st.checkbox("Show AI Insights", value=True)
        
    # Load Model
    model, feature_cols, best_threshold, actual_model_name, metadata = load_model_and_meta(selected_model_name)
    data_df = load_processed()
    
    if model is None or not feature_cols:
        st.error("❌ **Model Not Found**")
        st.info("Please train the model first by running: `python train_model.py`")
        st.stop()
    
    threshold_in_use = float(custom_thr) if override_thr else float(best_threshold)
    plotly_template = "plotly_dark" if theme_choice == "Dark" else "plotly_white"
    
    # Model Info Banner
    # Pull metrics for accurate display
    metrics_for_selected = load_model_metrics(actual_model_name, metadata)
    # Fallback: live compute from data if some metrics are missing
    if data_df is not None and not data_df.empty:
        live = compute_live_metrics_from_data(model, feature_cols, data_df, threshold_in_use)
        for k, v in live.items():
            if k not in metrics_for_selected or metrics_for_selected.get(k) in [None, 0, "N/A"]:
                metrics_for_selected[k] = v

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active Model", actual_model_name.upper())
    with col2:
        st.metric("Decision Threshold", f"{threshold_in_use:.3f}")
    with col3:
        st.metric("Features", len(feature_cols))
    with col4:
        # Prefer per-model metrics if available
        acc = metrics_for_selected.get('accuracy') or metadata.get('accuracy')
        st.metric("Model Accuracy", f"{acc*100:.1f}%" if acc else "N/A")
    
    st.markdown("---")
    
    # Main Tabs
    tabs = st.tabs([
        "🎯 Quick Prediction",
        "📈 Historical Analysis",
        "📊 Batch Processing",
        "🔬 Model Insights",
        "📋 Risk Dashboard",
        "🧪 Scenario Testing",
        "📚 User Guide"
    ])
    
    # Prepare input ranges and medians
    candidate_inputs = ["acwr", "monotony", "strain", "sleep_duration", "fatigue", "stress"]
    available_inputs = [c for c in candidate_inputs if c in feature_cols]
    if not available_inputs:
        available_inputs = [c for c in candidate_inputs if c in (data_df.columns if not data_df.empty else candidate_inputs)]
    
    ranges = derive_ranges(data_df, available_inputs)
    
    feature_medians = {}
    if not data_df.empty:
        for c in feature_cols:
            if c in data_df.columns and pd.api.types.is_numeric_dtype(data_df[c]):
                feature_medians[c] = float(data_df[c].median())
            else:
                feature_medians[c] = 0.0
    else:
        feature_medians = {c: 0.0 for c in feature_cols}
    
    # TAB 1: QUICK PREDICTION
    with tabs[0]:
        st.subheader("⚡ Real-Time Risk Assessment")
        
        input_col, result_col = st.columns([1, 1])
        
        with input_col:
            st.markdown("### 📝 Input Parameters")
            inputs: Dict[str, float] = {}
            
            for c in available_inputs:
                r = ranges[c]
                default_val = float(np.clip((r["min"] + r["max"]) / 2, r["min"], r["max"]))
                
                help_texts = {
                    "acwr": "Acute:Chronic Workload Ratio | Optimal: 0.8-1.3",
                    "monotony": "Training Monotony | Lower is better",
                    "strain": "Training Strain (Monotony × Load)",
                    "sleep_duration": "Sleep Duration (hours) | Target: 7-9",
                    "fatigue": "Subjective Fatigue | Scale: 1 (fresh) - 10 (exhausted)",
                    "stress": "Perceived Stress | Scale: 1 (calm) - 10 (overwhelmed)"
                }
                
                inputs[c] = st.slider(
                    label=c.replace('_', ' ').title(),
                    min_value=float(r["min"]),
                    max_value=float(r["max"]),
                    value=default_val,
                    step=float(r["step"]),
                    help=help_texts.get(c)
                )
            
            predict_btn = st.button("🔮 Predict Risk", type="primary", use_container_width=True)
        
        with result_col:
            if predict_btn:
                # Build feature vector
                input_row = {c: feature_medians.get(c, 0.0) for c in feature_cols}
                for k, v in inputs.items():
                    if k in input_row:
                        input_row[k] = float(v)
                
                # Calculate derived features
                if all(col in feature_cols for col in ["stress", "soreness", "sleep_quality", "readiness"]):
                    denom = max(1e-6, input_row.get("sleep_quality", 0.0) + input_row.get("readiness", 0.0))
                    input_row["stress_recovery_index"] = (input_row.get("stress", 0.0) + input_row.get("soreness", 0.0)) / denom
                
                X_input = pd.DataFrame([input_row])[feature_cols]
                
                # Predict
                proba = float(model.predict_proba(X_input)[:, 1][0])
                risk_category, emoji = calculate_risk_category(proba, threshold_in_use)
                
                # Display Results
                st.markdown("### 🎯 Prediction Results")
                
                # Risk Gauge
                gauge_fig = create_risk_gauge(proba, threshold_in_use)
                gauge_fig.update_layout(template=plotly_template)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Risk Category
                if "HIGH" in risk_category:
                    st.error(f"{emoji} **{risk_category}** | Score: {proba:.1%}")
                elif "MODERATE" in risk_category:
                    st.warning(f"{emoji} **{risk_category}** | Score: {proba:.1%}")
                else:
                    st.success(f"{emoji} **{risk_category}** | Score: {proba:.1%}")
                
                # Recommendations
                st.markdown("### 💡 Personalized Recommendations")
                recs = generate_advanced_recommendations(pd.Series(inputs), proba, threshold_in_use)
                
                if recs["immediate"]:
                    st.markdown("**⚠️ Immediate Actions:**")
                    for rec in recs["immediate"]:
                        st.markdown(f'<div class="recommendation-item">{rec}</div>', unsafe_allow_html=True)
                
                if recs["short_term"]:
                    st.markdown("**📅 Short-Term (1-3 days):**")
                    for rec in recs["short_term"]:
                        st.markdown(f'<div class="info-box">{rec}</div>', unsafe_allow_html=True)
                
                if recs["long_term"]:
                    st.markdown("**🎯 Long-Term Strategy:**")
                    for rec in recs["long_term"]:
                        st.info(rec)
                
                if recs["positive"]:
                    st.markdown("**✅ Positive Indicators:**")
                    for rec in recs["positive"]:
                        st.markdown(f'<div class="success-box">{rec}</div>', unsafe_allow_html=True)

                # Fallback generic recommendations if none produced
                if not any([recs["immediate"], recs["short_term"], recs["long_term"], recs["positive"]]):
                    st.info("No specific drivers detected. General guidance: maintain ACWR in 0.8–1.3, target 7–9h sleep, reduce monotony, and monitor fatigue/stress.")
    
    # TAB 2: HISTORICAL ANALYSIS
    with tabs[1]:
        st.subheader("📈 Historical Risk Analysis")
        
        if data_df.empty:
            st.warning("⚠️ No historical data available. Please ensure processed data exists.")
        else:
            analyze_btn = st.button("🔍 Analyze Historical Data", type="primary")
            
            if analyze_btn:
                with st.spinner("Analyzing historical patterns..."):
                    # Prepare features
                    X_all = pd.DataFrame(columns=feature_cols)
                    for c in feature_cols:
                        if c in data_df.columns and pd.api.types.is_numeric_dtype(data_df[c]):
                            X_all[c] = data_df[c]
                        else:
                            X_all[c] = feature_medians.get(c, 0.0)
                    
                    X_all = X_all.fillna(X_all.median(numeric_only=True))
                    risk_scores = model.predict_proba(X_all)[:, 1]
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average Risk", f"{np.mean(risk_scores):.1%}")
                    with col2:
                        st.metric("Peak Risk", f"{np.max(risk_scores):.1%}")
                    with col3:
                        high_risk_days = np.sum(risk_scores >= threshold_in_use)
                        st.metric("High Risk Days", int(high_risk_days))
                    with col4:
                        st.metric("Total Days", len(risk_scores))
                    
                    # Trend Analysis
                    if 'date' in data_df.columns:
                        trend_fig = create_trend_analysis(data_df, risk_scores)
                        if trend_fig:
                            trend_fig.update_layout(template=plotly_template)
                            st.plotly_chart(trend_fig, use_container_width=True)
                    
                    # Distribution
                    dist_col1, dist_col2 = st.columns(2)
                    
                    with dist_col1:
                        fig_hist = px.histogram(
                            x=risk_scores,
                            nbins=50,
                            title="Risk Score Distribution",
                            labels={'x': 'Risk Score', 'y': 'Frequency'}
                        )
                        fig_hist.update_layout(template=plotly_template)
                        fig_hist.add_vline(x=threshold_in_use, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with dist_col2:
                        # Weekly aggregation
                        if 'date' in data_df.columns:
                            result_df = pd.DataFrame({
                                'date': data_df['date'],
                                'risk': risk_scores
                            })
                            result_df['week'] = result_df['date'].dt.strftime('%Y-%U')
                            weekly = result_df.groupby('week')['risk'].agg(['mean', 'max', 'count']).reset_index()
                            
                            fig_weekly = go.Figure()
                            fig_weekly.add_trace(go.Bar(
                                x=weekly['week'],
                                y=weekly['mean'],
                                name='Weekly Avg',
                                marker_color='lightblue'
                            ))
                            fig_weekly.update_layout(
                                title="Weekly Risk Trends",
                                xaxis_title="Week",
                                yaxis_title="Average Risk",
                                template=plotly_template
                            )
                            st.plotly_chart(fig_weekly, use_container_width=True)
    
    # TAB 3: BATCH PROCESSING
    with tabs[2]:
        st.subheader("📊 Batch CSV Processing")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with athlete data",
            type=['csv'],
            help="CSV should contain columns matching the model features"
        )
        
        if uploaded_file:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(batch_df)} records")
                
                st.dataframe(batch_df.head(10))
                
                if st.button("🚀 Process Batch", type="primary"):
                    with st.spinner("Processing batch predictions..."):
                        X_batch = pd.DataFrame(columns=feature_cols)
                        
                        for c in feature_cols:
                            if c in batch_df.columns:
                                X_batch[c] = batch_df[c]
                            else:
                                X_batch[c] = feature_medians.get(c, 0.0)
                        
                        X_batch = X_batch.fillna(feature_medians)
                        batch_risks = model.predict_proba(X_batch)[:, 1]
                        
                        batch_df['injury_risk_score'] = batch_risks
                        batch_df['risk_category'] = [
                            calculate_risk_category(r, threshold_in_use)[0]
                            for r in batch_risks
                        ]
                        
                        st.success("✅ Batch processing complete!")
                        st.dataframe(batch_df)
                        
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            "batch_predictions.csv",
                            "text/csv",
                            use_container_width=True
                        )
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    # TAB 4: MODEL INSIGHTS
    with tabs[3]:
        st.subheader("🔬 Model Performance & Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Model Information")
            # Load per-model metrics if available
            model_metrics = load_model_metrics(actual_model_name, metadata)
            # Live fallback for missing values
            if data_df is not None and not data_df.empty:
                live_m = compute_live_metrics_from_data(model, feature_cols, data_df, threshold_in_use)
                for k, v in live_m.items():
                    if k not in model_metrics or model_metrics.get(k) in [None, 0, "N/A"]:
                        model_metrics[k] = v
            acc = model_metrics.get('accuracy') or metadata.get('accuracy')
            prec = model_metrics.get('precision') or metadata.get('precision')
            rec = model_metrics.get('recall') or metadata.get('recall')
            f1 = model_metrics.get('f1') or metadata.get('f1')
            auc = model_metrics.get('roc_auc') or model_metrics.get('auc') or metadata.get('auc')

            metrics_data = {
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
                "Value": [
                    f"{acc*100:.2f}%" if acc is not None else "N/A",
                    f"{prec*100:.2f}%" if prec is not None else "N/A",
                    f"{rec*100:.2f}%" if rec is not None else "N/A",
                    f"{f1*100:.2f}%" if f1 is not None else "N/A",
                    f"{auc:.3f}" if auc is not None else "N/A"
                ]
            }
            st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)
            
            st.markdown("### 🎯 Feature List")
            feature_df = pd.DataFrame({
                "Feature": feature_cols,
                "Type": ["Numeric" for _ in feature_cols]
            })
            st.dataframe(feature_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Feature Importance")
            
            if not data_df.empty:
                X_sample = pd.DataFrame(columns=feature_cols)
                for c in feature_cols:
                    if c in data_df.columns:
                        X_sample[c] = data_df[c].head(100)
                    else:
                        X_sample[c] = feature_medians.get(c, 0.0)
                X_sample = X_sample.fillna(feature_medians)
                
                importance_fig = create_feature_importance_chart(model, feature_cols, X_sample)
                if importance_fig:
                    st.plotly_chart(importance_fig, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model type")
            else:
                st.info("Load historical data to view feature importance")
        
        # Confusion Matrix Visualization
        if metadata and 'confusion_matrix' in metadata:
            st.markdown("### 🎯 Model Confusion Matrix")
            cm = np.array(metadata['confusion_matrix'])
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Low', 'Predicted High'],
                y=['Actual Low', 'Actual High'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16}
            ))
            
            fig_cm.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=400
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
    
    # TAB 5: RISK DASHBOARD
    with tabs[4]:
        st.subheader("📋 Comprehensive Risk Dashboard")
        
        if data_df.empty:
            st.warning("⚠️ No data available for dashboard. Load historical data first.")
        else:
            dashboard_btn = st.button("🔄 Generate Dashboard", type="primary")
            
            if dashboard_btn:
                with st.spinner("Building comprehensive dashboard..."):
                    # Prepare predictions
                    X_all = pd.DataFrame(columns=feature_cols)
                    for c in feature_cols:
                        if c in data_df.columns and pd.api.types.is_numeric_dtype(data_df[c]):
                            X_all[c] = data_df[c]
                        else:
                            X_all[c] = feature_medians.get(c, 0.0)
                    
                    X_all = X_all.fillna(X_all.median(numeric_only=True))
                    risk_scores = model.predict_proba(X_all)[:, 1]
                    
                    # KPI Row
                    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                    
                    with kpi1:
                        avg_risk = np.mean(risk_scores)
                        st.metric(
                            "Average Risk",
                            f"{avg_risk:.1%}",
                            delta=f"{(avg_risk - threshold_in_use):.1%}",
                            delta_color="inverse"
                        )
                    
                    with kpi2:
                        high_risk_count = np.sum(risk_scores >= threshold_in_use)
                        high_risk_pct = high_risk_count / len(risk_scores) * 100
                        st.metric(
                            "High Risk Days",
                            f"{int(high_risk_count)}",
                            f"{high_risk_pct:.1f}%"
                        )
                    
                    with kpi3:
                        st.metric(
                            "Peak Risk",
                            f"{np.max(risk_scores):.1%}",
                            delta="Critical" if np.max(risk_scores) > 0.8 else "OK",
                            delta_color="inverse" if np.max(risk_scores) > 0.8 else "off"
                        )
                    
                    with kpi4:
                        st.metric(
                            "Lowest Risk",
                            f"{np.min(risk_scores):.1%}",
                            delta="Optimal" if np.min(risk_scores) < 0.2 else "OK",
                            delta_color="normal" if np.min(risk_scores) < 0.2 else "off"
                        )
                    
                    with kpi5:
                        risk_std = np.std(risk_scores)
                        st.metric(
                            "Risk Volatility",
                            f"{risk_std:.3f}",
                            delta="Stable" if risk_std < 0.15 else "Variable",
                            delta_color="normal" if risk_std < 0.15 else "inverse"
                        )
                    
                    st.markdown("---")
                    
                    # Risk Distribution Analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🎯 Risk Category Distribution")
                        categories = [calculate_risk_category(r, threshold_in_use)[0] for r in risk_scores]
                        cat_counts = pd.Series(categories).value_counts()
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=cat_counts.index,
                            values=cat_counts.values,
                            hole=.4,
                            marker=dict(colors=['#d4edda', '#fff3cd', '#f8d7da'])
                        )])
                        fig_pie.update_layout(
                            title="Risk Category Breakdown",
                            height=350,
                            template=plotly_template
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 📊 Risk Score Distribution")
                        
                        fig_box = go.Figure()
                        fig_box.add_trace(go.Box(
                            y=risk_scores,
                            name="Risk Scores",
                            boxmean='sd',
                            marker_color='lightblue'
                        ))
                        
                        fig_box.add_hline(
                            y=threshold_in_use,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Threshold: {threshold_in_use:.2f}"
                        )
                        
                        fig_box.update_layout(
                            title="Risk Score Statistics",
                            yaxis_title="Risk Score",
                            height=350,
                            showlegend=False,
                            template=plotly_template
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Time Series Analysis (if date available)
                    if 'date' in data_df.columns:
                        st.markdown("### 📅 Temporal Risk Analysis")
                        
                        temporal_df = pd.DataFrame({
                            'date': data_df['date'],
                            'risk': risk_scores
                        }).sort_values('date')
                        
                        temporal_df['month'] = temporal_df['date'].dt.to_period('M').astype(str)
                        monthly_stats = temporal_df.groupby('month')['risk'].agg(['mean', 'max', 'std']).reset_index()
                        
                        fig_monthly = go.Figure()
                        
                        fig_monthly.add_trace(go.Scatter(
                            x=monthly_stats['month'],
                            y=monthly_stats['mean'],
                            mode='lines+markers',
                            name='Monthly Average',
                            line=dict(color='blue', width=3)
                        ))
                        
                        fig_monthly.add_trace(go.Scatter(
                            x=monthly_stats['month'],
                            y=monthly_stats['max'],
                            mode='lines+markers',
                            name='Monthly Peak',
                            line=dict(color='red', width=2, dash='dot')
                        ))
                        
                        fig_monthly.update_layout(
                            title="Monthly Risk Trends",
                            xaxis_title="Month",
                            yaxis_title="Risk Score",
                            height=400,
                            hovermode='x unified',
                            template=plotly_template
                        )
                        
                        st.plotly_chart(fig_monthly, use_container_width=True)
                    
                    # Correlation Analysis
                    st.markdown("### 🔗 Feature Correlation with Risk")
                    
                    corr_data = []
                    for feat in available_inputs:
                        if feat in data_df.columns:
                            corr = np.corrcoef(data_df[feat].fillna(0), risk_scores)[0, 1]
                            corr_data.append({'Feature': feat, 'Correlation': corr})
                    
                    if corr_data:
                        corr_df = pd.DataFrame(corr_data).sort_values('Correlation', key=abs, ascending=False)
                        
                        fig_corr = go.Figure(go.Bar(
                            x=corr_df['Correlation'],
                            y=corr_df['Feature'],
                            orientation='h',
                            marker=dict(
                                color=corr_df['Correlation'],
                                colorscale='RdYlGn_r',
                                cmin=-1,
                                cmax=1
                            )
                        ))
                        
                        fig_corr.update_layout(
                            title="Feature Correlation with Injury Risk",
                            xaxis_title="Correlation Coefficient",
                            yaxis_title="Feature",
                            height=400,
                            template=plotly_template
                        )
                        
                        st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Export Dashboard Report
                    st.markdown("---")
                    st.markdown("### 📥 Export Dashboard Data")
                    
                    export_df = pd.DataFrame({
                        'date': data_df['date'] if 'date' in data_df.columns else range(len(risk_scores)),
                        'risk_score': risk_scores,
                        'risk_category': [calculate_risk_category(r, threshold_in_use)[0] for r in risk_scores]
                    })
                    
                    for feat in available_inputs:
                        if feat in data_df.columns:
                            export_df[feat] = data_df[feat]
                    
                    csv_export = export_df.to_csv(index=False)
                    st.download_button(
                        "📊 Download Dashboard Report (CSV)",
                        csv_export,
                        "risk_dashboard_report.csv",
                        "text/csv",
                        use_container_width=True
                    )
    
    # TAB 6: SCENARIO TESTING
    with tabs[5]:
        st.subheader("🧪 Scenario Testing & What-If Analysis")
        
        st.markdown("""
        Test different scenarios to understand how changes in specific parameters affect injury risk.
        This helps in strategic planning and intervention decisions.
        """)
        
        scenario_col1, scenario_col2 = st.columns([1, 1])
        
        with scenario_col1:
            st.markdown("### 🎛️ Scenario Configuration")
            
            # Base scenario
            st.markdown("**Base Scenario**")
            base_inputs = {}
            for c in available_inputs[:3]:  # Limit to 3 for simplicity
                r = ranges[c]
                default_val = float(np.clip((r["min"] + r["max"]) / 2, r["min"], r["max"]))
                base_inputs[c] = st.slider(
                    f"Base {c}",
                    float(r["min"]),
                    float(r["max"]),
                    default_val,
                    float(r["step"]),
                    key=f"base_{c}"
                )
            
            st.markdown("---")
            
            # Scenario variations
            st.markdown("**Test Variations**")
            vary_feature = st.selectbox("Feature to Vary", available_inputs)
            
            num_scenarios = st.slider("Number of Test Points", 5, 20, 10)
            
            run_scenario = st.button("🚀 Run Scenario Analysis", type="primary")
        
        with scenario_col2:
            if run_scenario:
                st.markdown("### 📊 Scenario Results")
                
                with st.spinner("Running scenario simulations..."):
                    # Generate scenario variations
                    r = ranges[vary_feature]
                    test_values = np.linspace(r["min"], r["max"], num_scenarios)
                    
                    scenario_risks = []
                    
                    for val in test_values:
                        # Build input vector
                        input_row = {c: feature_medians.get(c, 0.0) for c in feature_cols}
                        
                        # Apply base inputs
                        for k, v in base_inputs.items():
                            if k in input_row:
                                input_row[k] = float(v)
                        
                        # Apply variation
                        if vary_feature in input_row:
                            input_row[vary_feature] = float(val)
                        
                        # Calculate derived features
                        if all(col in feature_cols for col in ["stress", "soreness", "sleep_quality", "readiness"]):
                            denom = max(1e-6, input_row.get("sleep_quality", 0.0) + input_row.get("readiness", 0.0))
                            input_row["stress_recovery_index"] = (input_row.get("stress", 0.0) + input_row.get("soreness", 0.0)) / denom
                        
                        X_test = pd.DataFrame([input_row])[feature_cols]
                        risk = float(model.predict_proba(X_test)[:, 1][0])
                        scenario_risks.append(risk)
                    
                    # Visualization
                    fig_scenario = go.Figure()
                    
                    fig_scenario.add_trace(go.Scatter(
                        x=test_values,
                        y=scenario_risks,
                        mode='lines+markers',
                        name='Risk Score',
                        line=dict(color='blue', width=3),
                        marker=dict(size=8)
                    ))
                    
                    fig_scenario.add_hline(
                        y=threshold_in_use,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Risk Threshold: {threshold_in_use:.2f}"
                    )
                    
                    # Mark optimal zone
                    fig_scenario.add_hrect(
                        y0=0, y1=threshold_in_use * 0.5,
                        fillcolor="green", opacity=0.1,
                        annotation_text="Low Risk Zone"
                    )
                    
                    fig_scenario.add_hrect(
                        y0=threshold_in_use * 0.5, y1=threshold_in_use,
                        fillcolor="yellow", opacity=0.1,
                        annotation_text="Moderate Risk Zone"
                    )
                    
                    fig_scenario.add_hrect(
                        y0=threshold_in_use, y1=1.0,
                        fillcolor="red", opacity=0.1,
                        annotation_text="High Risk Zone"
                    )
                    
                    fig_scenario.update_layout(
                        title=f"Risk vs {vary_feature.replace('_', ' ').title()}",
                        xaxis_title=vary_feature.replace('_', ' ').title(),
                        yaxis_title="Predicted Risk Score",
                        height=500,
                        hovermode='x'
                    )
                    
                    st.plotly_chart(fig_scenario, use_container_width=True)
                    
                    # Insights
                    st.markdown("### 💡 Scenario Insights")
                    
                    min_risk_idx = np.argmin(scenario_risks)
                    max_risk_idx = np.argmax(scenario_risks)
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric(
                            "Lowest Risk",
                            f"{scenario_risks[min_risk_idx]:.1%}",
                            f"at {vary_feature} = {test_values[min_risk_idx]:.2f}"
                        )
                    
                    with col_b:
                        st.metric(
                            "Highest Risk",
                            f"{scenario_risks[max_risk_idx]:.1%}",
                            f"at {vary_feature} = {test_values[max_risk_idx]:.2f}"
                        )
                    
                    with col_c:
                        risk_range = max(scenario_risks) - min(scenario_risks)
                        st.metric(
                            "Risk Sensitivity",
                            f"{risk_range:.1%}",
                            "Impact of parameter"
                        )
                    
                    # Recommendations
                    st.markdown("### 🎯 Optimization Recommendations")
                    
                    if scenario_risks[min_risk_idx] < threshold_in_use * 0.5:
                        st.success(f"✅ Optimal {vary_feature} value: {test_values[min_risk_idx]:.2f} (Low Risk Zone)")
                    elif scenario_risks[min_risk_idx] < threshold_in_use:
                        st.warning(f"⚠️ Recommended {vary_feature} value: {test_values[min_risk_idx]:.2f} (Moderate Risk Zone)")
                    else:
                        st.error(f"🚨 All scenarios show elevated risk. Consider adjusting multiple parameters.")
                    
                    # Export scenario data
                    scenario_export = pd.DataFrame({
                        vary_feature: test_values,
                        'risk_score': scenario_risks,
                        'risk_category': [calculate_risk_category(r, threshold_in_use)[0] for r in scenario_risks]
                    })
                    
                    csv_scenario = scenario_export.to_csv(index=False)
                    st.download_button(
                        "📥 Download Scenario Results",
                        csv_scenario,
                        f"scenario_analysis_{vary_feature}.csv",
                        "text/csv"
                    )

    # TAB 7: USER GUIDE
    with tabs[6]:
        st.subheader("📚 Role-Based User Guide")
        role = st.selectbox("Select your role", ["Coach", "Athlete", "Analyst", "Admin"], index=0)

        if role == "Coach":
            st.markdown("""
            - Use Quick Prediction for daily check-ins with athletes.
            - Watch High Risk flags; adjust training plans accordingly.
            - In Historical Analysis, monitor trends and weekly peaks to plan deload weeks.
            - Use Scenario Testing to find safer parameter ranges (e.g., ACWR adjustments).
            """)
        elif role == "Athlete":
            st.markdown("""
            - Enter your current wellness metrics in Quick Prediction.
            - Follow recommendations: prioritize sleep and recovery on high-risk days.
            - Review your weekly charts to understand patterns impacting risk.
            """)
        elif role == "Analyst":
            st.markdown("""
            - Use Batch Processing to score large datasets and export results.
            - Model Insights provides feature importances and confusion matrix where available.
            - Correlation and Dashboard exports support reporting to staff.
            """)
        else:
            st.markdown("""
            - Manage model selection and thresholds in Settings.
            - Ensure data preprocessing and training are up to date.
            - Oversee access and coordinate model retraining when performance drifts.
            """)

        st.markdown("---")
        st.markdown("""
        Tips:
        - Threshold: Higher values reduce false positives (higher precision), lower values increase sensitivity.
        - Input Ranges: Sliders use 5th–95th percentile ranges from historical data.
        - Exports: Use CSV downloads for sharing with staff and for archiving.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Injury Risk Intelligence System</strong> v2.0</p>
        <p>Advanced Predictive Analytics for Athlete Safety & Performance</p>
        <p style='font-size: 0.8rem;'>⚠️ This system provides risk estimates for informational purposes. 
        Always consult with qualified medical professionals for health decisions.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()