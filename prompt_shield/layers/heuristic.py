import os
import joblib
import numpy as np
from typing import List
import re

def sentence_chunks(prompt: str, max_sentences: int = 2) -> List[str]:
    """
    Split a prompt into chunks of up to `max_sentences` sentences using regex.
    """
    sentences = re.split(r'(?<=[.!?])\s+', prompt.strip())
    return [
        ' '.join(sentences[i:i + max_sentences])
        for i in range(0, len(sentences), max_sentences)
    ]

# Load model and vectorizer
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../data/xgb_model.joblib')
VECTORIZER_PATH = os.path.join(BASE_DIR, '../data/vectorizer.joblib')

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def score_prompt(prompt: str, config: dict) -> float:
    """
    Vectorizes and classifies prompt chunks using a trained XGBoost model.

    Returns:
        float: max risk score across all chunks [0, 1]
    """
    chunker = config.get("chunking_method", "sentence")
    max_sentences = config.get("max_sentences", 2)

    if chunker == "sentence":
        chunks = sentence_chunks(prompt, max_sentences=max_sentences)
    else:
        chunks = [prompt]

    if not chunks:
        return 0.0

    X = vectorizer.transform(chunks)
    probs = model.predict_proba(X)[:, 1]
    return float(np.max(probs))
