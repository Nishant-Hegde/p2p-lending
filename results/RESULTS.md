# Results: replication vs paper

Paper: Li & Han, *LendingClub Loan Default and Profitability Prediction* (Stanford CS229), `paper.pdf`. "Mine" = this repository, produced by `python src/run_all.py`. All metrics are on the **test set** unless stated. Underlying tables: `results/tables/`; decisions where the paper is vague: `ASSUMPTIONS.md`.

## 1. Dataset and preprocessing (Section III)

| Quantity | Mine | Paper |
|---|---:|---:|
| Loans (2012-2015, final status) | 786,820 | 745,529 |
| Positive (Fully Paid) | 81.4% | ~81% |
| Negative (Charged Off + Default) | 18.6% | ~19% |
| Train / test rows | 550,774 / 236,046 | 521,870 / 223,659 |
| Features after preprocessing | 1,126 | 1,097 |
| PCA components for 95% variance | 951 (84% of features) | ~900 (82% of features) |

Correlation heatmap (paper Fig. 1):

![Fig. 1](figures/fig1_correlation_heatmap.png)

## 2. Classification (Section V)

Precision / recall / F1 on the test set.

| Model | Class | Mine P / R / F1 | Paper P / R / F1 |
|---|---|---|---|
| Logistic Regression | Default | 0.31 / 0.67 / 0.42 | 0.63 / 0.88 / 0.73 |
| Logistic Regression | Paid Off | 0.90 / 0.66 / 0.76 | 0.97 / 0.88 / 0.92 |
| Logistic Regression | weighted avg | 0.79 / 0.66 / 0.70 | 0.90 / 0.88 / 0.88 |
| Neural Network (10,10,5,3) | Default | 0.49 / 0.12 / 0.19 | 0.72 / 0.71 / 0.72 |
| Neural Network (10,10,5,3) | Paid Off | 0.83 / 0.97 / 0.89 | 0.93 / 0.93 / 0.93 |
| Neural Network (10,10,5,3) | weighted avg | 0.77 / 0.81 / 0.76 | 0.89 / 0.89 / 0.89 |
| Random Forest (200 trees) | Default | 0.58 / 0.06 / 0.11 | 0.76 / 0.65 / 0.70 |
| Random Forest (200 trees) | Paid Off | 0.82 / 0.99 / 0.90 | 0.92 / 0.95 / 0.94 |
| Random Forest (200 trees) | weighted avg | 0.78 / 0.82 / 0.75 | 0.89 / 0.89 / 0.89 |

| Model | Train weighted P / R / F1 (mine) | Train weighted (paper) | Test AUC (mine) | Test AUC (paper) |
|---|---|---|---:|---:|
| Logistic Regression | 0.79 / 0.66 / 0.70 | not reported |  |  |
| Neural Network (10,10,5,3) | 0.79 / 0.82 / 0.78 | 0.91 / 0.91 / 0.91 |  |  |
| Random Forest (200 trees) | 1.00 / 1.00 / 1.00 | 0.97 / 0.96 / 0.96 | 0.720 | 0.94 |

Test confusion matrices (rows = true class, columns = predicted class):

| Model | True class | Mine: pred. Default | Mine: pred. Paid Off | Paper: pred. Default | Paper: pred. Paid Off |
|---|---|---:|---:|---:|---:|
| Logistic Regression | True Default | 29,330 | 14,660 | 37,632 | 5,135 |
| Logistic Regression | True Paid Off | 65,357 | 126,699 | 22,448 | 158,444 |
| Neural Network (10,10,5,3) | True Default | 5,323 | 38,667 | 30,545 | 12,222 |
| Neural Network (10,10,5,3) | True Paid Off | 5,541 | 186,515 | 11,864 | 169,028 |
| Random Forest (200 trees) | True Default | 2,641 | 41,349 | 27,760 | 15,007 |
| Random Forest (200 trees) | True Paid Off | 1,899 | 190,157 | 8,750 | 172,142 |

Train confusion matrices and full reports: `results/tables/03_*`. ROC curve: ![ROC](figures/roc_random_forest_test.png)

## 3. Regression on net annualized return (Sections VI-VII)

NAR = (total_pymnt / loan_amnt)^(365/D) - 1. Rows where D had to be set to the 30-day minimum: last_pymnt_d missing: 587; D <= 0: 1,130; 0 < D < 30: 496; total rows with D set to 30: 2,213.

Paper claim "most NARs of defaulted loans are well below -0.5": in this data **34.0%** of Charged Off / Default loans have NAR < -0.5 (53.9% are below -0.25). Distribution: `results/tables/04_nar_distribution.md`, figure `figures/nar_distribution.png`.

| Model | Train MSE mine / paper | Train R2 mine / paper | Test MSE mine / paper | Test R2 mine / paper |
|---|---|---|---|---|
| Linear Regression | 0.054 / 0.040 | 0.038 / 0.243 | 0.053 / 5.014 | 0.034 / -9.494e+22 |
| Ridge (alpha=1) | 0.054 / 0.040 | 0.038 / 0.243 | 0.053 / 0.040 | 0.034 / 0.238 |
| Neural Network (20,10,5,3) | 0.050 / 0.036 | 0.110 / 0.324 | 0.055 / 0.037 | 0.012 / 0.306 |
| Random Forest, depth 4 | 0.055 / 0.037 | 0.033 / 0.298 | 0.054 / 0.037 | 0.029 / 0.295 |
| Random Forest, depth 8 | 0.053 / 0.035 | 0.069 / 0.329 | 0.053 / 0.036 | 0.038 / 0.312 |
| Random Forest, depth 10 | 0.051 / 0.034 | 0.090 / 0.356 | 0.053 / 0.036 | 0.040 / 0.315 |

## 4. Loan selection strategy (Section VIII), RF depth-10 predictions

| Split | Loans with predicted NAR > 0.132 (mine) | Mean actual NAR of those loans (mine) | Mean NAR of all loans (mine) | Paper |
|---|---:|---:|---:|---|
| train | 0.01% (29 loans) | 179.7% | 0.2% | ~1.76% of loans, ~15% NAR |
| test | 0.01% (26 loans) | 2.9% | 0.2% | ~1.76% of loans, ~15% NAR |

Sweep over the threshold M (share of loans selected and their mean actual NAR):

| M | Train % selected | Train NAR | Test % selected | Test NAR |
|---:|---:|---:|---:|---:|
| 0.05 | 0.09% | 19.8% | 0.10% | 4.6% |
| 0.06 | 0.01% | 80.1% | 0.02% | 0.8% |
| 0.07 | 0.01% | 93.0% | 0.02% | 0.4% |
| 0.08 | 0.01% | 104.4% | 0.02% | 1.4% |
| 0.09 | 0.01% | 112.4% | 0.02% | 1.3% |
| 0.10 | 0.01% | 126.6% | 0.01% | 1.6% |
| 0.11 | 0.01% | 142.5% | 0.01% | 3.1% |
| 0.12 | 0.01% | 154.3% | 0.01% | 3.1% |
| 0.13 | 0.01% | 179.7% | 0.01% | 3.1% |
| 0.14 | 0.00% | 192.5% | 0.01% | 2.6% |
| 0.15 | 0.00% | 199.4% | 0.01% | 2.3% |
| 0.16 | 0.00% | 224.3% | 0.01% | 1.7% |
| 0.17 | 0.00% | 224.3% | 0.01% | 1.5% |
| 0.18 | 0.00% | 233.7% | 0.01% | 1.5% |
| 0.19 | 0.00% | 244.4% | 0.01% | 8.7% |
| 0.20 | 0.00% | 257.4% | 0.01% | 12.3% |

Annualized return vs share of loans invested (paper Fig. 2):

![Fig. 2](figures/fig2_return_vs_pct_invested.png)

## 5. Where the results differ from the paper, and why

Large gaps found in this run:

- Logistic Regression: weighted F1 0.70 vs 0.88
- Neural Network (10,10,5,3): weighted F1 0.76 vs 0.89
- Random Forest (200 trees): weighted F1 0.75 vs 0.89
- Random Forest AUC 0.720 vs 0.94
- Ridge (alpha=1): test R2 0.034 vs 0.238
- Neural Network (20,10,5,3): test R2 0.012 vs 0.306
- Random Forest, depth 4: test R2 0.029 vs 0.295
- Random Forest, depth 8: test R2 0.038 vs 0.312
- Random Forest, depth 10: test R2 0.040 vs 0.315
- selection at M=0.132: 0.01% of test loans, NAR 2.9% vs 1.76% / 15%

**What matches.** The dataset definition (years, final statuses, class balance), the size of the
feature space, the PCA finding (95% of the variance needs most of the components, so no useful reduction) and the
scale of the NAR label (its variance is close to the ~0.053 implied by the paper's MSE and R2) are all close to the paper.

**Classification and regression quality are much lower than the paper's.** The paper does not list which
columns it dropped. This replication removes every outcome / post-issuance column (payments received, recoveries,
last payment date and amount, last FICO, outstanding principal, hardship and settlement fields; full list in
`ASSUMPTIONS.md`), so the models only see what was known when the loan was issued. With those features the
models are only moderately better than the base rate. The paper's numbers (AUC 0.94, 88% recall on defaults for a
linear model, R2 of 0.24 for ridge regression on returns) are far above that. The most likely explanation is that
some post-issuance columns were still among the paper's 1,097 features, which would make the targets partly
predictable from their own consequences. **This is a hypothesis, not something this replication has tested**:
testing it would mean deliberately adding leaking columns back, which is outside the "replicate only" scope.
Smaller contributors that cannot produce a gap this large on their own: a newer data snapshot (more 2015 loans have
reached a final status), unknown imputation groups and max-set factor, unknown logistic-regression strength,
and the unknown number of trees and `max_features` of the regression forests.

**Linear regression did **not** blow up here** (test R2 = 0.034; paper: about -1e23). In this pipeline
the scaler is fit on the training rows only, so a one-hot column that is constant in train stays exactly zero
after centring and receives a zero coefficient, and scikit-learn's least-squares solver returns the minimum-norm
solution for the collinear one-hot columns. A blow-up of the paper's size needs a column that is numerically
almost, but not exactly, constant in train and takes a large value in a few test rows (for example a rare zip
code after scaling with statistics from the whole dataset); such a column gets an enormous coefficient that only
shows up on the test set. That mechanism is an explanation consistent with the paper's description
("high-variance problem"), not one that was reproduced here. Note also that the paper's own pair of numbers
(test MSE 5.014 with R2 = -9.5e22) cannot both be right for a label with variance ~0.05.

**NAR of defaulted loans.** The paper says most defaulted loans have NAR well below -0.5; here 34% do.
Many charged-off borrowers pay for a year or more before stopping, which gives a moderate negative annualized
return rather than a near-total loss.

**Loan selection.** Because the regressors explain little of the NAR variance, predicted NARs are compressed
towards the mean and very few loans exceed the paper's threshold M = 0.132, so the selected share and its
return differ from the paper's 1.76% / 15% and are noisy; the sweep table shows how the selection behaves at
lower thresholds.
The very high *train* returns in the table come from a handful of loans repaid within a month or two with
fees (NAR above 1, up to 22), which the forest partly memorizes; they do not carry over to the test set.

## 6. Runtime of this run

| Stage | Step | Time | Peak RAM of the process (GB) |
|---|---|---:|---:|
| 01_load_filter | total | 2.3 min | 2.54 |
| 02_preprocess | total | 2.8 min | 8.66 |
| 03_classification | fit logreg | 3.0 min | 7.10 |
| 03_classification | fit mlp | 8.5 min | 7.10 |
| 03_classification | fit rf | 21.8 min | 7.10 |
| 04_regression | fit linear | 1.6 min | 14.16 |
| 04_regression | fit mlp | 4.1 min | 14.16 |
| 04_regression | fit rf10 | 29.9 min | 14.16 |
| 04_regression | fit rf4 | 38.1 min | 14.16 |
| 04_regression | fit rf8 | 58.6 min | 14.16 |
| 04_regression | fit ridge | 45.5s | 14.16 |
| 05_loan_selection | total | 0.7s | 0.18 |

