import plotly.express as px

def plot_cost_vs_accuracy(df, pareto_df):

    fig = px.scatter(
        df,
        x="cost",
        y="accuracy",
        color="model",
        hover_data=["prompt_length", "few_shot", "temperature"]
    )

    fig.add_scatter(
        x=pareto_df["cost"],
        y=pareto_df["accuracy"],
        mode="markers",
        marker=dict(size=12, symbol="diamond"),
        name="Pareto Frontier"
    )

    fig.show()