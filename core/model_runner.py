import numpy as np
from config import MODEL_QUALITY, PRICE_PER_1K, BASE_LATENCY

class ModelRunner:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate(self, prompt_length, few_shot, temperature):

        base_acc = MODEL_QUALITY[self.model_name]

        # Diminishing return for long prompts
        length_factor = 0.03 * np.log1p(prompt_length)

        # Few-shot saturates
        few_shot_factor = 0.05 * (1 - np.exp(-0.6 * few_shot))

        # Temperature reduces stability
        temp_penalty = -0.15 * temperature

        # Add stochastic noise
        noise = np.random.normal(0, 0.02)

        accuracy = base_acc + length_factor + few_shot_factor + temp_penalty + noise
        accuracy = np.clip(accuracy, 0.5, 0.99)

        tokens_used = prompt_length + 120 + (few_shot * 40)

        cost = (tokens_used / 1000) * PRICE_PER_1K[self.model_name]

        latency = BASE_LATENCY[self.model_name] + 0.0015 * prompt_length

        return accuracy, tokens_used, cost, latency