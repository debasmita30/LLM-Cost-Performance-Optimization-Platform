from optimizer.grid_search import run_grid_search
from optimizer.pareto import compute_pareto
from visualization.plots import plot_cost_vs_accuracy
from config import LAMBDA_VALUES

import numpy as np

df = run_grid_search()


for lam in LAMBDA_VALUES:
    df[f"objective_lambda_{lam:.2f}"] = df["accuracy"] - lam * df["cost"]


pareto_df = compute_pareto(df)

plot_cost_vs_accuracy(df, pareto_df)

print("\nTop configs by accuracy:")
print(df.sort_values("accuracy", ascending=False).head())

print("\nPareto optimal configs:")
print(pareto_df.sort_values("accuracy", ascending=False))

BUDGET = 0.0015

budget_df = df[df["cost"] <= BUDGET]

if not budget_df.empty:
    best_under_budget = budget_df.sort_values("accuracy", ascending=False).iloc[0]
    print("\nBest config under budget:")
    print(best_under_budget)


import pandas as pd
import plotly.express as px

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

import plotly.graph_objects as go
import numpy as np


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

import statsmodels.api as sm
import pandas as pd
import numpy as np


reg_df = df.copy()

reg_df = pd.get_dummies(reg_df, columns=["model"], drop_first=True)
print(reg_df.columns)

feature_cols = ["prompt_length", "few_shot", "temperature"]

model_dummy_cols = [col for col in reg_df.columns if col.startswith("model_")]


X = reg_df[feature_cols + model_dummy_cols]
X = X.astype(float)

y = reg_df["accuracy"].astype(float)


X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print("\nRegression Summary:")
print(model.summary())

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