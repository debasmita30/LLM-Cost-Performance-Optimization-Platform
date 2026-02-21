def compute_accuracy(pred, gold):
    return int(gold.lower() in pred.lower())