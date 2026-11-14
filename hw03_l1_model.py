"""
hw03_l1_model.py
Same pipeline as hw03_l2_baseline.py (same data, same TF-IDF settings),
with ONE change: the logistic regression uses an L1 (lasso) penalty.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")  # early-modern texts have unusual characters

DATA_DIR = Path("data")
N_TOP_WORDS = 15
N_ERRORS_TO_SHOW = 10  # total misclassified chunks to print


# ---------- 1. Load the saved train/test sets ----------
with open(DATA_DIR / "train_core_vs_neg.json", encoding="utf-8") as f:
    train_data = json.load(f)
with open(DATA_DIR / "test_core_vs_neg.json", encoding="utf-8") as f:
    test_data = json.load(f)

train_texts = [text for text, label in train_data]
y_train = [label for text, label in train_data]
test_texts = [text for text, label in test_data]
y_test = [label for text, label in test_data]

print(f"Train chunks: {len(train_texts)} | Test chunks: {len(test_texts)}")

# ---------- 2. TF-IDF (same settings as the Week 09 tutorial) ----------
vectorizer = TfidfVectorizer(lowercase=True, min_df=5, max_df=0.9)
X_train = vectorizer.fit_transform(train_texts)  # learn vocabulary on train only
X_test = vectorizer.transform(test_texts)

# ---------- 3. Train the model with an L1 penalty ----------
# l1_ratio=1 means pure L1 in scikit-learn >= 1.8; liblinear is a solver that supports L1
clf = LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=2000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]  # probability of CORE

# ---------- 4. Evaluation ----------
print("\n=== L1 MODEL: EVALUATION ===")
print("Confusion matrix (rows = true NEG/CORE, cols = predicted NEG/CORE):")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(
    classification_report(
        y_test, y_pred, target_names=["NEG (0)", "CORE (1)"], digits=3
    )
)

print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 3))

# ---------- 5. Sparsity ----------
coefs = clf.coef_[0]
n_nonzero = int(np.sum(coefs != 0))
print(
    f"\nNon-zero coefficients: {n_nonzero} of {len(coefs)} "
    f"({100 * n_nonzero / len(coefs):.1f}%)"
)

# ---------- 6. Top words ----------
words = vectorizer.get_feature_names_out()
order = np.argsort(coefs)  # most negative first

print(f"\nTop {N_TOP_WORDS} positive-weight words (predict CORE = 1):")
for i in order[::-1][:N_TOP_WORDS]:
    if coefs[i] > 0:
        print(f"  {words[i]:<20} {coefs[i]:8.3f}")

print(f"\nTop {N_TOP_WORDS} negative-weight words (predict NEG = 0):")
for i in order[:N_TOP_WORDS]:
    if coefs[i] < 0:
        print(f"  {words[i]:<20} {coefs[i]:8.3f}")

# ---------- 7. Misclassified examples (for the label-noise question) ----------
y_test_arr = np.array(y_test)
false_pos = np.where((y_test_arr == 0) & (y_pred == 1))[0]
false_neg = np.where((y_test_arr == 1) & (y_pred == 0))[0]
print(f"\nFalse positives: {len(false_pos)} | False negatives: {len(false_neg)}")

# up to 5 false positives, then fill the rest of the 10 with false negatives
n_fp = min(5, len(false_pos))
n_fn = N_ERRORS_TO_SHOW - n_fp
for name, idx in [
    ("FALSE POSITIVES (label NEG, predicted CORE)", false_pos[:n_fp]),
    ("FALSE NEGATIVES (label CORE, predicted NEG)", false_neg[:n_fn]),
]:
    print(f"\n--- {name} ---")
    for i in idx:
        print(f"\n[test #{i}] P(core) = {y_prob[i]:.2f}")
        print(test_texts[i][:600])
