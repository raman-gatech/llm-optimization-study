import re

def extract_answer(text):
    # extract after ####
    match = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if match:
        return match.group(1)

    # fallback: last number
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]

    return None
