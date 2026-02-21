import numpy as np

CONFIG_SPACE = {
    "model": ["small", "medium", "large"],
    "prompt_length": [100, 300, 600, 900],
    "few_shot": [0, 2, 4],
    "temperature": [0.0, 0.3, 0.7]
}

MODEL_QUALITY = {
    "small": 0.65,
    "medium": 0.78,
    "large": 0.88
}

PRICE_PER_1K = {
    "small": 0.001,
    "medium": 0.003,
    "large": 0.008
}

BASE_LATENCY = {
    "small": 0.3,
    "medium": 0.6,
    "large": 1.2
}

LAMBDA_VALUES = np.linspace(0, 5, 30)