from config import PRICE_PER_1K

def compute_cost(tokens):
    return (tokens / 1000) * PRICE_PER_1K