import os
import yaml
from typing import List
from sentence_transformers import CrossEncoder

# Load cross-encoder model
_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Load labeled examples
EXAMPLE_PATH = os.path.join(os.path.dirname(__file__), '../data/hazard_codes.yaml')
with open(EXAMPLE_PATH, 'r') as f:
    hazard_examples = yaml.safe_load(f)

# Flatten hazard examples for matching
hazard_pairs = []
hazard_labels = []
for code, examples in hazard_examples.items():
    for ex in examples:
        hazard_pairs.append((code, ex))
        hazard_labels.append(code)


def classify_hazards(prompt: str, threshold: float = 0.5) -> List[str]:
    pairs = []
    labels = []

    for code, examples in hazard_examples.items():
        for ex in examples:
            pairs.append((prompt, ex))
            labels.append(code)

    scores = _model.predict(pairs)
    results = {}

    for score, code in zip(scores, labels):
        if score >= threshold:
            results.setdefault(code, []).append(score)

    # Optionally return only top-k or average score per hazard
    return sorted(results.keys())
