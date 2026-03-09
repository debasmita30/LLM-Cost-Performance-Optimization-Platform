import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

st.set_page_config(
    page_title="LLM Optimization System",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKEND_URL = "https://llm-cost-performance-optimization.onrender.com"

# ── Backend & Data Source Status ───────────────────────────────
def check_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def check_snowflake():
    try:
        r = requests.get(f"{BACKEND_URL}/snowflake/status", timeout=5)
        return r.status_code == 200 and r.json().get("connected", False)
    except Exception:
        return False

backend_online = check_backend()
snowflake_online = check_snowflake()

# ── Header ─────────────────────────────────────────────────────
st.title("🚀 LLM Cost–Performance Optimization Platform")

col_s1, col_s2, col_s3 = st.columns(3)
col_s1.markdown(
    f"{'🟢 Backend: Online' if backend_online else '🟡 Backend: Offline — Local Mode'}"
)
col_s2.markdown(
    f"{'🟢 Snowflake: Connected' if snowflake_online else '🟡 Snowflake: Offline — CSV Mode'}"
)
col_s3.markdown("🟢 Dashboard: Active")

st.divider()

# ── Data Loading ───────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    if backend_online:
        try:
            r = requests.get(f"{BACKEND_URL}/results", timeout=10)
            if r.status_code == 200:
                df = pd.DataFrame(r.json())
                if not df.empty:
                    st.sidebar.success("Data source: Snowflake via Backend")
                    return df
        except Exception:
            pass
    st.sidebar.info("Data source: Local CSV")
    return pd.read_csv("results/simulation_results.csv")

@st.cache_data(ttl=300)
def load_run_history():
    if backend_online:
        try:
            r = requests.get(f"{BACKEND_URL}/run-history", timeout=10)
            if r.status_code == 200:
                return pd.DataFrame(r.json())
        except Exception:
            pass
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_best_historical():
    if backend_online:
        try:
            r = requests.get(f"{BACKEND_URL}/best-config", timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    return {}

df = load_data()
run_history = load_run_history()
best_historical = load_best_historical()

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.header("Control Panel")

selected_models = st.sidebar.multiselect(
    "Models",
    df["model"].unique(),
    default=df["model"].unique()
)

budget = st.sidebar.slider(
    "Budget",
    float(df["cost"].min()),
    float(df["cost"].max()),
    float(df["cost"].max())
)

lambda_value = st.sidebar.slider("Lambda", 0.0, 5.0, 1.0)
daily_queries = st.sidebar.number_input("Daily Queries", 100, 100000, 10000)
sla_latency = st.sidebar.slider("SLA Max Latency", 0.2, 3.0, 1.0)
presentation_mode = st.sidebar.checkbox("Presentation Mode")

if st.sidebar.button("🔄 Refresh from Snowflake"):
    st.cache_data.clear()
    st.rerun()

# ── Filter ─────────────────────────────────────────────────────
filtered_df = df[
    (df["model"].isin(selected_models)) &
    (df["cost"] <= budget)
].copy()

for col in filtered_df.columns:
    try:
        filtered_df[col] = pd.to_numeric(filtered_df[col])
    except Exception:
        pass

# ── KPI Metrics ────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Max Accuracy", f"{filtered_df['accuracy'].max():.3f}")
c2.metric("Min Cost", f"{filtered_df['cost'].min():.4f}")
c3.metric("Avg Latency", f"{filtered_df['latency'].mean():.3f}")
c4.metric("Configs", len(filtered_df))


st.divider()

# ── Historical Best from Snowflake ────────────────────────────
if best_historical:
    st.subheader("🏅 Best Historical Config from Snowflake")
    st.json(best_historical)
    st.divider()

# ── Optimal Configuration ──────────────────────────────────────
filtered_df.loc[:, "objective_lambda"] = (
    filtered_df["accuracy"] - lambda_value * filtered_df["cost"]
)
best_config = filtered_df.loc[filtered_df["objective_lambda"].idxmax()]
monthly_cost = best_config["cost"] * daily_queries * 30

st.subheader("🏆 Optimal Configuration (Lambda-Aware)")
st.dataframe(
    best_config[[
        "model", "prompt_length", "few_shot",
        "temperature", "accuracy", "cost", "latency"
    ]].to_frame().T.astype(str)
)

# ── Pareto Frontier ────────────────────────────────────────────
@st.cache_data
def compute_pareto_fast(df):
    df_sorted = df.sort_values(["cost", "accuracy"], ascending=[True, False])
    pareto = []
    max_accuracy = -np.inf
    for _, row in df_sorted.iterrows():
        if row["accuracy"] > max_accuracy:
            pareto.append(row)
            max_accuracy = row["accuracy"]
    return pd.DataFrame(pareto)

pareto_df = compute_pareto_fast(filtered_df)

c5.metric("Pareto Configs", len(pareto_df))

st.subheader("📊 Pareto Frontier")
fig = px.scatter(
    filtered_df, x="cost", y="accuracy", color="model",
    hover_data=["prompt_length", "few_shot", "temperature"]
)
fig.add_scatter(
    x=pareto_df["cost"], y=pareto_df["accuracy"],
    mode="markers",
    marker=dict(size=14, symbol="diamond"),
    name="Pareto"
)
st.plotly_chart(fig, use_container_width=True)

# ── Lambda Tradeoff ────────────────────────────────────────────
lambda_space = np.linspace(0, 5, 30)
lambda_results = []
for lam in lambda_space:
    temp_df = filtered_df.copy()
    temp_df.loc[:, "objective"] = temp_df["accuracy"] - lam * temp_df["cost"]
    best = temp_df.loc[temp_df["objective"].idxmax()]
    lambda_results.append({
        "lambda": lam,
        "accuracy": best["accuracy"],
        "cost": best["cost"]
    })
lambda_df = pd.DataFrame(lambda_results)

st.subheader("⚖️ Lambda Tradeoff Curve")
st.plotly_chart(
    px.line(lambda_df, x="lambda", y=["accuracy", "cost"], markers=True),
    use_container_width=True
)

# ── Model Radar ────────────────────────────────────────────────
st.subheader("🕸 Model Radar Comparison")
radar_df = filtered_df.groupby("model").agg({
    "accuracy": "mean", "cost": "mean", "latency": "mean"
}).reset_index()
fig_radar = go.Figure()
for _, row in radar_df.iterrows():
    fig_radar.add_trace(go.Scatterpolar(
        r=[row["accuracy"], row["cost"], row["latency"]],
        theta=["Accuracy", "Cost", "Latency"],
        fill='toself', name=row["model"]
    ))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)))
st.plotly_chart(fig_radar, use_container_width=True)

# ── Heatmap ────────────────────────────────────────────────────
st.subheader("🔥 Cost per Correct Heatmap")
heatmap_df = filtered_df.pivot_table(
    values="cost_per_correct", index="prompt_length", columns="few_shot"
)
st.plotly_chart(
    px.imshow(heatmap_df, text_auto=True, aspect="auto"),
    use_container_width=True
)

# ── SLA Risk ───────────────────────────────────────────────────
st.subheader("⚠️ SLA Risk Analysis")
filtered_df.loc[:, "sla_violation"] = filtered_df["latency"] > sla_latency
risk_rate = filtered_df["sla_violation"].mean() * 100
sla_counts = filtered_df["sla_violation"].value_counts().reset_index()
sla_counts.columns = ["Violation", "Count"]
st.metric("SLA Violation Rate (%)", f"{risk_rate:.2f}")
st.plotly_chart(
    px.pie(sla_counts, names="Violation", values="Count"),
    use_container_width=True
)

# ── Deployment Projection ──────────────────────────────────────
st.subheader("💰 Deployment Projection")
st.metric("Projected Monthly Cost", f"${monthly_cost:,.2f}")

# ── Monte Carlo ────────────────────────────────────────────────
st.subheader("🎲 Monte Carlo Stability")

@st.cache_data
def monte_carlo_simulation(df, runs=20):
    simulations = []
    for _ in range(runs):
        noisy = df.copy()
        noisy.loc[:, "accuracy"] += np.random.normal(0, 0.01, len(noisy))
        simulations.append(noisy)
    mc_df = pd.concat(simulations)
    return mc_df.groupby("model")["accuracy"].agg(["mean", "std"]).reset_index()

agg = monte_carlo_simulation(filtered_df)
st.dataframe(agg.astype(str))

# ── Dual Objective ─────────────────────────────────────────────
st.subheader("🎯 Advanced Dual-Objective Optimization")
lambda_cost = st.slider("λ₁ Cost Weight", 0.0, 5.0, 1.0)
lambda_latency = st.slider("λ₂ Latency Weight", 0.0, 5.0, 1.0)
norm_df = filtered_df.copy()
for col in ["accuracy", "cost", "latency"]:
    min_val, max_val = norm_df[col].min(), norm_df[col].max()
    norm_df[f"{col}_norm"] = (
        (norm_df[col] - min_val) / (max_val - min_val)
        if max_val != min_val else 0
    )
norm_df["dual_objective"] = (
    norm_df["accuracy_norm"]
    - lambda_cost * norm_df["cost_norm"]
    - lambda_latency * norm_df["latency_norm"]
)
dual_best = norm_df.loc[norm_df["dual_objective"].idxmax()]
st.dataframe(
    dual_best[[
        "model", "prompt_length", "few_shot", "temperature",
        "accuracy", "cost", "latency", "dual_objective"
    ]].to_frame().T.astype(str)
)

# ── SLA Constrained ────────────────────────────────────────────
sla_constrained_df = filtered_df[filtered_df["latency"] <= sla_latency].copy()
if not sla_constrained_df.empty:
    sla_constrained_df.loc[:, "objective_lambda"] = (
        sla_constrained_df["accuracy"] - lambda_value * sla_constrained_df["cost"]
    )
    sla_best = sla_constrained_df.loc[sla_constrained_df["objective_lambda"].idxmax()]
    st.subheader("✅ Best SLA-Compliant Configuration")
    st.dataframe(
        sla_best[[
            "model", "prompt_length", "few_shot",
            "temperature", "accuracy", "cost", "latency"
        ]].to_frame().T.astype(str)
    )
else:
    st.warning("No configuration satisfies current SLA threshold.")

# ── 3D Plot ────────────────────────────────────────────────────
st.subheader("🌐 3D Cost–Latency–Accuracy Surface")
fig_3d = px.scatter_3d(
    filtered_df, x="cost", y="latency", z="accuracy",
    color="model",
    hover_data=["prompt_length", "few_shot", "temperature"]
)
st.plotly_chart(fig_3d, use_container_width=True)

# ── Snowflake Run History ──────────────────────────────────────
if not run_history.empty:
    st.subheader("📜 Optimization Run History (Snowflake)")
    st.dataframe(run_history)
    st.plotly_chart(
        px.line(
            run_history.sort_values("run_timestamp"),
            x="run_timestamp",
            y="best_accuracy",
            title="Best Accuracy Over Time"
        ),
        use_container_width=True
    )

# ── Executive Summary ──────────────────────────────────────────
st.subheader("📋 Executive Summary")
top_model = best_config["model"]
acc = best_config["accuracy"]
cost = best_config["cost"]
lat = best_config["latency"]
efficiency_score = acc / cost
sla_status = (
    "High Risk" if risk_rate > 50
    else "Moderate Risk" if risk_rate > 20
    else "Low Risk"
)
st.markdown(f"""
### Optimization Outcome
Optimal Model: **{top_model}**
Accuracy: **{acc:.3f}**
Cost per Request: **{cost:.4f}**
Latency: **{lat:.3f}s**
Efficiency Score: **{efficiency_score:.2f}**

### Deployment Risk Assessment
SLA Violation Rate: **{risk_rate:.2f}%**
Risk Category: **{sla_status}**
Estimated Monthly Cost: **${monthly_cost:,.2f}**

### Data Source
Backend: **{"Snowflake via Render" if backend_online and snowflake_online else "Local CSV"}**
""")

# ── Presentation Mode ──────────────────────────────────────────
if presentation_mode:
    st.markdown("## 🎤 Presentation Mode Active")
    st.markdown("""
    This system demonstrates multi-objective optimization,
    SLA-aware deployment filtering, cost forecasting,
    Pareto efficiency, Snowflake data persistence,
    and production-level decision analytics.
    """)

st.divider()
st.markdown("## 📦 System Summary")
st.markdown("""
This platform integrates:
• Multi-objective optimization
• SLA-aware constrained solving
• Pareto efficiency extraction
• Monte Carlo robustness testing
• Deployment cost forecasting
• Snowflake cloud data persistence
• Render backend API integration
• Executive-level reporting

Designed for real-world LLM production deployment decisions.
""")

