# IDS 570 – Homework 3: L1 vs. L2 Logistic Regression

Compares a TF-IDF + logistic regression classifier (CORE = merchant/trade passages vs. NEG)
using the default L2 penalty and an L1 (lasso) penalty.

## Files
- `build_datasets.py` – rebuilds `data/train_core_vs_neg.json` and `data/test_core_vs_neg.json`
  from the Week 09 tutorial steps 7 and 9 (keyword-tier labeling, balancing, 80/20 split, seed 42).
- `hw03_l2_baseline.py` – baseline model (default L2 penalty).
- `hw03_l1_model.py` – same pipeline with `l1_ratio=1, solver="liblinear"`.
- `outputs_l2.txt`, `outputs_l1.txt` – printed results from each script.

## How to run
1. Put the course `.txt` files (Canvas: Train_Text_Documents) in a folder named `Train_Test_Documents/`.
2. `pip install nltk gensim scikit-learn numpy`
3. `python build_datasets.py`
4. `python hw03_l2_baseline.py` and `python hw03_l1_model.py`

The texts and generated data are not included in this repo (see `.gitignore`).
Tested with Python 3.12 and scikit-learn 1.9.1.
