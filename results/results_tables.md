# Experimental results (auto-generated)

Evaluation batch: N = 10,000 transactions (9,900 legitimate / 100 fraud).

## Confusion matrices

| Algorithm | TP | FN | FP | TN |
|---|---|---|---|---|
| XGBoost (Proposed) | 82 | 18 | 25 | 9875 |
| Random Forest | 80 | 20 | 10 | 9890 |
| SVM (RBF) | 53 | 47 | 69 | 9831 |
| Decision Tree | 72 | 28 | 21 | 9879 |
| Logistic Regression | 70 | 30 | 56 | 9844 |
| MLP | 66 | 34 | 15 | 9885 |
| Isolation Forest | 38 | 62 | 115 | 9785 |
| Rule Engine | 25 | 75 | 2 | 9898 |
| Hybrid Risk Engine | 73 | 27 | 1 | 9899 |

## Comparative performance

| Algorithm | Acc. | Prec. | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|---|
| XGBoost (Proposed) | 99.57% | 0.766 | 0.820 | 0.792 | 0.25% | 18.0% |
| Random Forest | 99.70% | 0.889 | 0.800 | 0.842 | 0.10% | 20.0% |
| SVM (RBF) | 98.84% | 0.434 | 0.530 | 0.477 | 0.70% | 47.0% |
| Decision Tree | 99.51% | 0.774 | 0.720 | 0.746 | 0.21% | 28.0% |
| Logistic Regression | 99.14% | 0.556 | 0.700 | 0.619 | 0.57% | 30.0% |
| MLP | 99.51% | 0.815 | 0.660 | 0.729 | 0.15% | 34.0% |
| Isolation Forest | 98.23% | 0.248 | 0.380 | 0.300 | 1.16% | 62.0% |
| Rule Engine | 99.23% | 0.926 | 0.250 | 0.394 | 0.02% | 75.0% |
| Hybrid Risk Engine | 99.72% | 0.986 | 0.730 | 0.839 | 0.01% | 27.0% |

## Per-transaction latency

| Engine | mean (ms) | p50 (ms) | p95 (ms) |
|---|---|---|---|
| Rule Engine | 0.12 | 0.12 | 0.13 |
| XGBoost (Proposed) | 0.08 | 0.07 | 0.08 |
| Isolation Forest | 2.01 | 2.02 | 2.06 |
| Random Forest | 14.70 | 14.78 | 15.61 |
| SVM (RBF) | 0.13 | 0.13 | 0.14 |
| Hybrid (sequential 3 engines + policy) | 2.19 | 2.19 | 2.26 |
| Hybrid (parallel bound = slowest branch) | 2.01 | 2.02 | 2.06 |

## Policy decision distribution (hybrid engine)

| Decision | Count |
|---|---|
| Approve | 9879 |
| Alert | 52 |
| Challenge | 47 |
| Block | 22 |

## Global SHAP importance (mean |phi|)

| Feature | mean abs SHAP |
|---|---|
| Merchant_Type | 3.8809 |
| Inter_Txn_Gap | 1.2030 |
| Txn_Amount | 1.1607 |
| z_amt | 1.0366 |
| Distance_From_Home | 0.7063 |
| Hour_Of_Day | 0.6926 |
| Device_Fingerprint | 0.6703 |
| Account_History_Idx | 0.5560 |
| Txn_Count_1Hr | 0.3081 |
| Sender_Receiver_Rel | 0.1317 |
