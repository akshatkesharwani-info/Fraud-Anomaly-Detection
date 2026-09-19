# Fraud / Anomaly Detection

A fraud classifier trained on real, extremely imbalanced transaction data -- where fraud is
0.17% of all transactions, so getting this right means handling imbalance correctly, not just
training a model.

## Problem
A card issuer needs to flag fraud in near real time. A naive model predicting "not fraud" for
everything would already be 99.8% accurate and completely useless -- accuracy is the wrong metric
here entirely.

## What It Does
- Quantifies the exact class imbalance before choosing a strategy
- Applies SMOTE oversampling on the training set only (never touches the test set)
- Trains a supervised XGBoost classifier alongside an unsupervised Isolation Forest baseline
  for comparison
- Evaluates on precision-recall AUC and a full confusion matrix, not accuracy

## Real Results (real dataset, 284,807 transactions, 0.1727% fraud rate)
- **PR-AUC: 0.869**
- **Recall on fraud: 86%** -- caught 84 of 98 actual fraudulent transactions in the test set
- Precision on fraud: 69%
- The unsupervised Isolation Forest baseline, for comparison, only reached 33% recall --
  showing clearly why the supervised + SMOTE approach was worth the extra complexity

## Tech Stack
Python, Scikit-learn, XGBoost, imbalanced-learn (SMOTE), Matplotlib

## How to Run
Open in Google Colab, run all cells. Dataset auto-downloads via `kagglehub` with a synthetic
fallback if it ever fails.
