# Assumptions and decisions where the paper is vague

Section numbers refer to Li & Han (`paper.pdf`). Constants named in `code font` live in `src/config.py`
or at the top of the stage script. Tables referenced here are written to `results/tables/` by each run.

## Environment
1. **Python 3.13.0**, not 3.10 / 3.11: it was the only interpreter on the machine that produced the committed
   results. The pinned packages in `requirements.txt` also install on 3.10 and 3.11, but only 3.13 has been run.
2. **Memory.** The build machine had 31 GB RAM (about 13-15 GB free). The design matrix is stored as **dense
   float32**. Sparse matrices are not used: the paper standardizes every feature to zero mean, and centring a
   one-hot column makes it dense anyway. `LogisticRegression`, `LinearRegression` and `Ridge` are fit on a
   float64 copy (`LINEAR_DTYPE`), which is the numerically safest choice for least squares on about 1,100 partly
   collinear columns and is what a default pandas/NumPy pipeline would use; the neural networks and forests use
   float32. The full run peaked at 14.2 GB (stage 04, linear models). Setting `LINEAR_DTYPE = "float32"` should
   roughly halve that but can change those models' numbers slightly; it has not been run on the full data.

## Data (Section III-A)
3. **Filter.** `issue_d` parsed with `%b-%Y`, year in 2012-2015; `loan_status` exactly "Fully Paid" (label 1),
   "Charged Off" or "Default" (label 0). Statuses such as "Does not meet the credit policy. Status:Fully Paid"
   are not final statuses in the paper's sense and are excluded.
4. **Row count** is 786,820 vs the paper's 745,529 (+5.5%), class balance 81.4% / 18.6% vs ~81% / ~19%. The
   Kaggle snapshot is newer than the authors' download, so more loans have reached a final status. Only 2 loans
   have status "Default".
5. **`--sample f`** keeps a uniform random fraction `f` of the *filtered* loans (seed 42). Everything a sampled
   run writes goes to `sample_<f>` sub-folders. If the full `data/processed/filtered.parquet` already exists, a
   sampled stage 01 samples from it instead of re-reading the raw file (same rows either way).

## Preprocessing (Section III-B)
6. **"Empty for most rows" = more than 90% missing** (`MISSING_THRESHOLD`), not 50%. Reason: in this slice
   `mths_since_last_delinq` is 50.7% missing and `mths_since_last_record` 84.0%, and the paper uses exactly these
   "months since" fields as its max-set example, so it must have kept them; a 50% cut would delete them. The data
   has a clean gap: every column between 50% and 90% missing is a "months since ..." field (missing because the
   event never happened), and the next columns are at least 97.6% missing (fields LendingClub introduced in late
   2015, joint-application and `sec_app_*` fields). Missingness is measured on all filtered rows (label-free).
7. **Columns dropped** (the exact list with reasons is written to `results/tables/02_dropped_columns.md`):
   * *Free text / ID (6):* `id, member_id, url, desc, title, emp_title`. (`zip_code` and `addr_state` are kept.)
   * *Leakage, outcome or post-issuance (38 raw columns + the helper `label`):* `loan_status` (after the labels
     are made), `total_pymnt, total_pymnt_inv, total_rec_prncp, total_rec_int, total_rec_late_fee, recoveries,
     collection_recovery_fee, last_pymnt_d, last_pymnt_amnt, next_pymnt_d, out_prncp, out_prncp_inv,
     last_credit_pull_d, last_fico_range_high, last_fico_range_low`; all `hardship_*` (12 columns), all
     `settlement_*` (5) and `debt_settlement_*` (2); plus `deferral_term, payment_plan_start_date,
     orig_projected_additional_accrued_interest`, which are hardship-plan fields without the `hardship_` prefix.
   * *More than 90% missing (30):* `annual_inc_joint, dti_joint, verification_status_joint, revol_bal_joint`, the
     13 `sec_app_*` columns, `open_acc_6m, open_act_il, open_il_12m, open_il_24m, mths_since_rcnt_il,
     total_bal_il, il_util, open_rv_12m, open_rv_24m, max_bal_bc, all_util, inq_fi, total_cu_tl, inq_last_12m`.
   * *Constant (3):* `pymnt_plan, policy_code, disbursement_method` (checked after the other drops).
   * Kept: everything known at issuance, including `funded_amnt, funded_amnt_inv, installment, int_rate, grade,
     sub_grade, fico_range_low/high, issue_d` and the credit-bureau fields. 74 columns remain (63 numeric
     including 2 dates, 11 categorical).
   * `loan_amnt, total_pymnt, issue_d, last_pymnt_d` are copied aside **before** dropping and saved as
     `data/processed/regression_cols_{train,test}.parquet`, row-aligned with `X_train` / `X_test`, only to build
     the regression label. `loan_amnt` and `issue_d` also remain ordinary features.
8. **Missing-value groups.** The paper names one example per group and no full list; the grouping is mine. Every
   numeric column with missing values is assigned by `missing_group()` in `02_preprocess.py`, which raises an
   error for an unassigned column instead of guessing. Assignment and fill values of a run:
   `results/tables/02_missing_value_groups.md`.
   * **mean-set** (fill = train mean): `dti, revol_util, bc_util, pct_tl_nvr_dlq, percent_bc_gt_75` (ratios) and
     the dollar-level fields `tot_cur_bal, avg_cur_bal, bc_open_to_buy, total_rev_hi_lim, tot_hi_cred_lim,
     total_bal_ex_mort, total_bc_limit, total_il_high_credit_limit`. Reason, as in the paper's DTI example: 0 would
     read as "no debt / no credit at all", an extreme value rather than a neutral one. (`annual_inc, int_rate,
     installment, revol_bal, earliest_cr_line` are also registered as mean-set but have no missing values here.)
   * **max-set** (fill = `MAX_SET_FACTOR` x train max): every column starting with `mths_since` or `mo_sin`:
     `mths_since_last_delinq, mths_since_last_record, mths_since_last_major_derog, mths_since_recent_bc,
     mths_since_recent_bc_dlq, mths_since_recent_inq, mths_since_recent_revol_delinq, mo_sin_old_il_acct,
     mo_sin_old_rev_tl_op, mo_sin_rcnt_rev_tl_op, mo_sin_rcnt_tl`.
   * **zero-set** (fill = 0): counts and amounts where missing most plausibly means none on record:
     `tot_coll_amt, acc_open_past_24mths, mort_acc` and every `num_*` column (`num_accts_ever_120_pd,
     num_actv_bc_tl, num_actv_rev_tl, num_bc_sats, num_bc_tl, num_il_tl, num_op_rev_tl, num_rev_accts,
     num_rev_tl_bal_gt_0, num_sats, num_tl_120dpd_2m, num_tl_30dpd, num_tl_90g_dpd_24m, num_tl_op_past_12m`).
   * **`MAX_SET_FACTOR = 2.0`.** The paper says "a constant factor" without a value. 2 x max places "never
     happened" clearly beyond the longest observed gap. The authors' poster says missing values are replaced
     with "the max" of the column, which suggests a factor of 1; the paper's "constant factor multiplied with the
     maximum" is followed here with 2.0, so this is a possible difference from what the authors ran.
   * `emp_length` (categorical, 5.3% missing) gets its own level "Unknown" rather than a numeric fill.
9. **Categoricals, one-hot, no level dropped:** `term, grade, sub_grade, emp_length, home_ownership,
   verification_status, purpose, zip_code, addr_state, initial_list_status, application_type`. `term` and
   `emp_length` are treated as categories because the paper only describes one-hot encoding. The level list is
   taken from all rows of the run (label-free), so the feature count does not depend on the split; a level that
   occurs only in test rows is an all-zero column in train.
10. **Dates** `issue_d` and `earliest_cr_line`: `%b-%Y` -> first day of that month -> days since 1970-01-01.
11. **Split: 70/30, stratified by label, `random_state=42`.** (The paper says "random split"; stratification was
    requested for this replication and changes the class balance per split by well under a percentage point.)
    The split is made **before** imputation and scaling.
12. **Standardization is fit on the training rows only** (mean and standard deviation per column, accumulated in
    float64) and applied to both splits; the same holds for the mean-set and max-set fill values. Columns with
    zero variance in train are left with divisor 1.
13. **Feature count.** 1,126 on the full data (63 numeric + 1,063 one-hot) vs the paper's 1,097. The paper does
    not list its columns, so the 29-feature difference cannot be traced.
14. **PCA** is fit on the standardized training matrix: `PCA(n_components=0.95, svd_solver="covariance_eigh")`,
    an exact eigen-decomposition of the feature covariance. It is used only to report the number of
    components; all original features are used for modelling, as in the paper. Note that the paper calls ~900
    components "close to 95% of the total number of features"; 900 of 1,097 is 82%.
15. **Heatmap (Fig. 1)** uses the raw, un-imputed, un-scaled values of the 14 listed columns over all rows of the
    run (pairwise-complete Pearson correlation), with `loan_status` coded 1 = Fully Paid, 0 = Charged Off / Default.

## Classification (Section V)
16. Hyper-parameters are those listed in the paper; anything it does not state is the scikit-learn 1.6 default:
    `LogisticRegression` solver `lbfgs`, `C=1.0` (the paper's lambda is not given); `MLPClassifier`
    `max_iter=200`, no early stopping, `tol=1e-4`; the forest uses bootstrap sampling.
    * **The classifier MLP uses the Adam solver.** The paper only says "Gradient Descent-based solver" for the
      classifier (it names Adam for the regression network). Adam is assumed because the paper's mini-batch size
      (200), learning rate (0.001) and L2 penalty (0.0001) are exactly scikit-learn's defaults for Adam.
    * **The Random Forest classifier has unlimited tree depth** (`max_depth=None`, `min_samples_leaf=1`), because
      the paper states no limit. It therefore fits the training set perfectly. The paper's own train table
      (Default recall 0.82, weighted F1 0.96) suggests smaller trees, but the authors' poster shows a perfect
      train confusion matrix (99,163 / 0 / 0 / 422,707) like this replication's; the two sources disagree and
      neither states a depth or leaf-size limit.
17. `class_weight="balanced"` is applied to logistic regression only, as in the paper.
18. Class names follow the paper's tables: "Default" = label 0 (Charged Off + Default), "Paid Off" = label 1.
19. `n_jobs=-1` is set for the random forests and `LinearRegression`. `LogisticRegression` (binary, lbfgs),
    `Ridge` and the MLPs have no useful `n_jobs`; they use multi-threaded BLAS automatically.

## Regression (Sections VI-VII)
20. **NAR = (total_pymnt / loan_amnt) ^ (365 / D) - 1.** In the PDF the exponent is typeset as `1/(365/D)`,
    which taken literally equals `D/365` and is not an annualization. `365/D` is used. Supporting check from the
    data: the paper's own numbers imply a label variance of MSE / (1 - R2) of about 0.053; the `365/D` label has
    variance 0.056 over all loans (0.055 after the exclusion in #21), the literal `D/365` label 1.36.
21. **D** = days from `issue_d` to `last_pymnt_d`. Both are month-level, so D is a multiple of about 30.
    * **Loans without a usable last-payment date are excluded from the regression and the loan selection**
      (train and test): `last_pymnt_d` missing, or not after `issue_d` (D <= 0). On the full data that is 1,717
      loans (1,216 train, 501 test): 587 with a missing date, all Charged Off, and 1,130 with D <= 0, of which
      1,129 are Fully Paid (repaid in the month of issue) and 1 is Charged Off. 785,103 loans remain. The paper
      does not say how it treated such loans. The exact counts of a run are in
      `results/tables/04_nar_exclusions.md`.
    * Why they are excluded and not given a default D: an earlier version set D to 30 days for them. 455 of the
      587 loans with a missing date have `total_pymnt > 0` (money came in although no payment date is
      recorded), and for those that had paid back more than their principal the 30-day rule produced impossible
      labels, up to NAR = 22. These were artifacts of the rule, not real returns; the regression forest partly
      memorized them and they inflated the train-set selection return.
    * For 0 < D < 30 (496 loans, e.g. a 28-day February) D is set to `MIN_DAYS = 30`.
    * NAR is not clipped or winsorized. After the exclusion 8 loans still have NAR above 1 (largest 3.18): they
      are Charged Off loans whose recorded last payment is one or two months after issue although more than the
      principal was paid back, so their recorded dates are probably also unreliable; they are kept because their
      D is defined.
22. Same features and the same train/test assignment as the classifiers, minus the loans excluded in #21.
    The classifiers (stages 01-03) still use all 786,820 loans.
23. `LinearRegression`, `Ridge(alpha=1)`: scikit-learn defaults. `MLPRegressor`: `max_iter=200`, no early
    stopping, target not rescaled.
24. **Regression forests: 100 trees** (`RF_REG_TREES`; the paper gives no count) and **`max_features=1.0`**
    (`RF_REG_MAX_FEATURES`, all features at every split). The paper states `max_features=50` for the classifier
    only; for the regressors it gives nothing, and all-features is the scikit-learn default for regression
    forests (also in the 2018 release the authors would have used). This is the slowest part of the pipeline.

## Loan selection (Section VIII)
25. Predictions come from the depth-10 regression forest. Each selected loan receives the same $25, so the
    portfolio return is the plain mean of the selected loans' actual NAR; there is no weighting by term or amount
    and no reinvestment.
26. Fig. 2 ranks loans by predicted NAR (high to low) and plots the mean actual NAR of the top-k loans for
    k >= 100 against k / N, where N is the number of loans kept for the regression in that split (#21).
27. The threshold M = 0.132 is the paper's value, applied as given ("predicted NAR > M"), not re-tuned.
28. The abstract says the strategy picks 1.76% of loans; Section VIII says 1.7%. 1.76% is used for comparison.
