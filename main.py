from optimizer.grid_search import run_grid_search
from optimizer.pareto import compute_pareto
from visualization.plots import plot_cost_vs_accuracy
from config import LAMBDA_VALUES
from snowflake_connector import save_results_to_snowflake, load_results_from_snowflake
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

print("=" * 60)
print("LLM Cost-Performance Optimization Platform")
print("=" * 60)

# ── Grid Search ────────────────────────────────────────────────
print("\n[1/6] Running grid search...")
df = run_grid_search()

for lam in LAMBDA_VALUES:
    df[f"objective_lambda_{lam:.2f}"] = df["accuracy"] - lam * df["cost"]

# ── Pareto Frontier ────────────────────────────────────────────
print("[2/6] Computing Pareto frontier...")
pareto_df = compute_pareto(df)
plot_cost_vs_accuracy(df, pareto_df)

print("\nTop configs by accuracy:")
print(df.sort_values("accuracy", ascending=False).head())
print("\nPareto optimal configs:")
print(pareto_df.sort_values("accuracy", ascending=False))

# ── Budget Filter ──────────────────────────────────────────────
BUDGET = 0.0015
budget_df = df[df["cost"] <= BUDGET]
if not budget_df.empty:
    best_under_budget = budget_df.sort_values("accuracy", ascending=False).iloc[0]
    print("\nBest config under budget:")
    print(best_under_budget)

# ── Lambda Tradeoff ────────────────────────────────────────────
print("\n[3/6] Running lambda tradeoff analysis...")
lambda_results = []
for lam in LAMBDA_VALUES:
    df["objective"] = df["accuracy"] - lam * df["cost"]
    best = df.loc[df["objective"].idxmax()]
    lambda_results.append({
        "lambda": lam,
        "best_accuracy": best["accuracy"],
        "best_cost": best["cost"]
    })
lambda_df = pd.DataFrame(lambda_results)
fig = px.line(
    lambda_df,
    x="lambda",
    y=["best_accuracy", "best_cost"],
    title="Lambda Tradeoff Curve"
)
fig.show()

# ── 3D Surface ─────────────────────────────────────────────────
print("[4/6] Rendering 3D surface plot...")
surface_df = df[df["model"] == "medium"]
pivot = surface_df.pivot_table(
    values="accuracy",
    index="prompt_length",
    columns="few_shot"
)
fig = go.Figure(
    data=[go.Surface(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index
    )]
)
fig.update_layout(
    title="3D Surface: Accuracy vs Prompt Length vs Few Shot",
    scene=dict(
        xaxis_title="Few Shot",
        yaxis_title="Prompt Length",
        zaxis_title="Accuracy"
    )
)
fig.show()

# ── Regression Analysis ────────────────────────────────────────
print("[5/6] Running regression analysis...")
reg_df = df.copy()
reg_df = pd.get_dummies(reg_df, columns=["model"], drop_first=True)
print(reg_df.columns)
feature_cols = ["prompt_length", "few_shot", "temperature"]
model_dummy_cols = [col for col in reg_df.columns if col.startswith("model_")]
X = reg_df[feature_cols + model_dummy_cols]
X = X.astype(float)
y = reg_df["accuracy"].astype(float)
X = sm.add_constant(X)
ols_model = sm.OLS(y, X).fit()
print("\nRegression Summary:")
print(ols_model.summary())

# ── Monte Carlo Stability ──────────────────────────────────────
print("[6/6] Running Monte Carlo stability test...")
NUM_RUNS = 50
all_runs = []
for i in range(NUM_RUNS):
    df_run = run_grid_search()
    df_run["run"] = i
    all_runs.append(df_run)
mc_df = pd.concat(all_runs)
agg = mc_df.groupby(
    ["model", "prompt_length", "few_shot", "temperature"]
)["accuracy"].agg(["mean", "std"]).reset_index()
print("\nMonte Carlo Stability (Top 5 by mean accuracy):")
print(agg.sort_values("mean", ascending=False).head())

# ── Build Final Results DataFrame ─────────────────────────────
print("\nPreparing final results for storage...")
df["objective_score"] = df["accuracy"] - 0.3 * df["cost"]
df["sla_compliant"] = df["latency_ms"] <= 200 if "latency_ms" in df.columns else True
df["model_name"] = df["model"] if "model" in df.columns else "unknown"
df["prompt_strategy"] = df["prompt_length"].astype(str) + "_shot_" + df["few_shot"].astype(str) if "prompt_length" in df.columns else "default"

results_df = df[[
    "model_name", "prompt_strategy", "temperature",
    "accuracy", "cost", "objective_score", "sla_compliant"
]].rename(columns={"cost": "cost_per_request"})

if "latency_ms" not in results_df.columns:
    results_df["latency_ms"] = np.nan

# ── Save to CSV ────────────────────────────────────────────────
results_df.to_csv("results/simulation_results.csv", index=False)
print("Results saved to CSV.")

# ── Save to Snowflake ──────────────────────────────────────────
print("Pushing results to Snowflake...")
save_results_to_snowflake(results_df)

# ── Load Historical from Snowflake ────────────────────────────
print("Loading historical runs from Snowflake...")
historical_df = load_results_from_snowflake()
if not historical_df.empty:
    print(f"Loaded {len(historical_df)} historical records from Snowflake.")
    print("\nTop 5 historical configs by accuracy:")
    print(historical_df.sort_values("accuracy", ascending=False).head())
else:
    print("No historical data found in Snowflake yet.")

print("\nPipeline complete.")
