def build_prompt(question, few_shot_examples=None, max_length=None):
    prompt = ""

    if few_shot_examples:
        for ex in few_shot_examples:
            prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"

    prompt += f"Q: {question}\nA:"

    if max_length:
        prompt = prompt[:max_length]

    return prompt