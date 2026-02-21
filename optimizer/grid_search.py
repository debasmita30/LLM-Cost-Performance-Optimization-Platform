import os
import itertools
import pandas as pd
from core.model_runner import ModelRunner
from config import CONFIG_SPACE


def run_grid_search():

    results = []

    keys, values = zip(*CONFIG_SPACE.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for config in combinations:

        runner = ModelRunner(config["model"])

        accuracy, tokens, cost, latency = runner.generate(
            config["prompt_length"],
            config["few_shot"],
            config["temperature"]
        )

        results.append({
            "model": config["model"],
            "prompt_length": config["prompt_length"],
            "few_shot": config["few_shot"],
            "temperature": config["temperature"],
            "accuracy": accuracy,
            "tokens": tokens,
            "cost": cost,
            "latency": latency,
            "cost_per_correct": cost / accuracy if accuracy > 0 else float("inf")
        })

    df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)

    df.to_csv("results/simulation_results.csv", index=False)

    return df