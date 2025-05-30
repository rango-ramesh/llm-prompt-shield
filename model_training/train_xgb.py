import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os

# Load and combine Hugging Face Parquet files
splits = {
    "train": "data/train-00000-of-00001-9564e8b05b4757ab.parquet",
    "test": "data/test-00000-of-00001-701d16158af87368.parquet"
}

print("Loading dataset from Hugging Face Parquet files...")
train_df = pd.read_parquet("hf://datasets/deepset/prompt-injections/" + splits["train"])
test_df = pd.read_parquet("hf://datasets/deepset/prompt-injections/" + splits["test"])

# Combine and clean
df = pd.concat([train_df, test_df])
df = df.dropna(subset=["text", "label"])
df = df[["text", "label"]]

# Save locally for reproducibility
# df.to_csv("prompt_injections.csv", index=False)
# print("Saved combined dataset to prompt_injections.csv")

# Prepare data
X = df["text"].values
y = df["label"].astype(int).values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ("clf", XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=3.0  # try 1.5 to 3.0
)
)
])

# Train
print("Training model...")
pipeline.fit(X_train, y_train)

# Evaluate
print("Evaluating model...")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save components separately
SAVE_DIR = "prompt_shield/data"
print("Saving model and vectorizer...")
os.makedirs(SAVE_DIR, exist_ok=True)

tfidf = pipeline.named_steps["tfidf"]
model = pipeline.named_steps["clf"]

joblib.dump(tfidf, os.path.join(SAVE_DIR, "vectorizer.joblib"))
joblib.dump(model, os.path.join(SAVE_DIR, "xgb_model.joblib"))

print("Done.")
