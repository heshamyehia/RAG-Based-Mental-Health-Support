"""
Module 1 — Language Detection
==============================
Dataset : papluca/language-identification (HuggingFace)
          90k samples | 20 languages | train/validation/test splits
Model   : TF-IDF (char n-grams, 2-4) + best ML classifier (auto-selected)
Output  : language_detector/language_detector.joblib
          language_detector/language_detector_meta.joblib

Run:
    pip install scikit-learn datasets joblib matplotlib seaborn
    python train_language_detector.py
"""

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from datasets import load_dataset

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 1 — Loading dataset")
print("=" * 60)

hf = load_dataset("papluca/language-identification")

train_df = hf["train"].to_pandas().rename(columns={"labels": "language"})
val_df   = hf["validation"].to_pandas().rename(columns={"labels": "language"})
test_df  = hf["test"].to_pandas().rename(columns={"labels": "language"})

full_train = pd.concat([train_df, val_df], ignore_index=True)

X_train, y_train = full_train["text"], full_train["language"]
X_test,  y_test  = test_df["text"],    test_df["language"]

print(f"  Train : {len(X_train):,} samples")
print(f"  Test  : {len(X_test):,} samples")
print(f"  Languages ({y_train.nunique()}): {sorted(y_train.unique())}\n")

# ════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 2 — Preprocessing")
print("=" * 60)

def preprocess(text: str) -> str:
    """Strip and collapse whitespace. Keep all characters — script
    features (diacritics, CJK, Cyrillic) are discriminative for language ID."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())

X_train = X_train.apply(preprocess)
X_test  = X_test.apply(preprocess)

# Drop near-empty rows
mask_train = X_train.str.len() > 3
mask_test  = X_test.str.len()  > 3
X_train, y_train = X_train[mask_train], y_train[mask_train]
X_test,  y_test  = X_test[mask_test],   y_test[mask_test]

print(f"  Train after cleaning: {len(X_train):,}")
print(f"  Test  after cleaning: {len(X_test):,}\n")

# ════════════════════════════════════════════════════════════════════════════
# 3. DEFINE CANDIDATE PIPELINES
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 3 — Defining candidate pipelines")
print("=" * 60)

TFIDF_CHAR = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 4),
    max_features=150_000,
    sublinear_tf=True,
    strip_accents=None,   
    min_df=2,
)

COUNT_CHAR = CountVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 4),
    max_features=150_000,
    min_df=2,
)

PIPELINES = {
    "Linear SVM": Pipeline([
        ("vec", TFIDF_CHAR),
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, random_state=42), cv=3
        )),
    ]),
    "Logistic Regression": Pipeline([
        ("vec", TFIDF_CHAR),
        ("clf", LogisticRegression(
            C=5.0, solver="saga", max_iter=1000,
            n_jobs=-1, random_state=42
        )),
    ]),
    "SGD Classifier": Pipeline([
        ("vec", TFIDF_CHAR),
        ("clf", SGDClassifier(
            loss="modified_huber", alpha=1e-4,
            max_iter=200, n_jobs=-1, random_state=42
        )),
    ]),
    "Complement NaiveBayes": Pipeline([
        ("vec", COUNT_CHAR),
        ("clf", ComplementNB(alpha=0.1)),
    ]),
    "Multinomial NaiveBayes": Pipeline([
        ("vec", COUNT_CHAR),
        ("clf", MultinomialNB(alpha=0.1)),
    ]),
}

for name in PIPELINES:
    print(f"  • {name}")
print()

# ════════════════════════════════════════════════════════════════════════════
# 4. CROSS-VALIDATION BENCHMARK
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 4 — Cross-validation benchmark (5-fold)")
print("=" * 60)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

print(f"\n  {'Model':<25} {'CV Acc':>9} {'± Std':>8} {'Time':>8}")
print("  " + "-" * 55)

for name, pipe in PIPELINES.items():
    t0 = time.time()
    scores = cross_val_score(pipe, X_train, y_train, cv=CV,
                             scoring="accuracy", n_jobs=-1)
    elapsed = time.time() - t0
    cv_results.append({
        "model": name, "pipe": pipe,
        "mean": scores.mean(), "std": scores.std(),
        "time": elapsed,
    })
    print(f"  {name:<25} {scores.mean():>9.4f} {scores.std():>8.4f} {elapsed:>7.1f}s")

cv_results.sort(key=lambda x: -x["mean"])

# ════════════════════════════════════════════════════════════════════════════
# 5. TRAIN BEST MODEL ON FULL TRAINING SET
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 5 — Training best model")
print("=" * 60)

best      = cv_results[0]
best_name = best["model"]
best_pipe = best["pipe"]

print(f"\n  Best model : {best_name}")
print(f"  CV accuracy: {best['mean']:.4f} ± {best['std']:.4f}")
print("\n  Training on full training set...")

t0 = time.time()
best_pipe.fit(X_train, y_train)
print(f"  Done in {time.time() - t0:.1f}s\n")

# ════════════════════════════════════════════════════════════════════════════
# 6. EVALUATE ON TEST SET
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 6 — Test set evaluation")
print("=" * 60)

y_pred   = best_pipe.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
test_f1  = f1_score(y_test, y_pred, average="weighted")

print(f"\n  Test accuracy : {test_acc:.4%}")
print(f"  Weighted F1   : {test_f1:.4%}")
print("\n  Per-language report:")
print(classification_report(y_test, y_pred))


# ════════════════════════════════════════════════════════════════════════════
# 7. SAVE MODEL
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 7 — Saving model")
print("=" * 60)

os.makedirs("language_detector", exist_ok=True)
MODEL_PATH = "language_detector/language_detector.joblib"
META_PATH  = "language_detector/language_detector_meta.joblib"

joblib.dump(best_pipe, MODEL_PATH, compress=3)
joblib.dump({
    "model_name"   : best_name,
    "classes"      : labels,
    "languages"    : labels,
    "test_accuracy": test_acc,
    "test_f1"      : test_f1,
    "train_samples": len(X_train),
}, META_PATH)

print(f"  Saved → {MODEL_PATH}  ({os.path.getsize(MODEL_PATH)/1e6:.1f} MB)")
print(f"  Saved → {META_PATH}")
print(f"\n  ✔  Done!  Test accuracy = {test_acc:.4%}")
print("=" * 60)