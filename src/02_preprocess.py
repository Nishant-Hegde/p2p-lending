"""Stage 2 (paper Section III-B): column cleaning, missing values, one-hot, dates, 70/30 split,
standardization, PCA check and the correlation heatmap (paper Fig. 1)."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import config
from utils import (Dirs, parse_args, record_runtime, rel, require, save_figure, save_json,
                   save_table, step)

STAGE = "02_preprocess"

FREE_TEXT_ID = ["id", "member_id", "url", "desc", "title", "emp_title"]
# outcome / post-issuance columns that would leak the label
LEAK_EXACT = ["loan_status", "label", "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
              "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
              "next_pymnt_d", "out_prncp", "out_prncp_inv", "last_credit_pull_d", "last_fico_range_high",
              "last_fico_range_low",
              # hardship-plan fields that do not carry the hardship_ prefix
              "deferral_term", "payment_plan_start_date", "orig_projected_additional_accrued_interest"]
LEAK_PREFIX = ("hardship_", "settlement_", "debt_settlement_")
REGRESSION_COLS = ["loan_amnt", "total_pymnt", "issue_d", "last_pymnt_d"]   # set aside for the NAR label (stage 04)
DATE_COLS = ["issue_d", "earliest_cr_line"]
HEATMAP_COLS = ["loan_amnt", "int_rate", "annual_inc", "loan_status", "dti", "delinq_2yrs", "fico_range_low",
                "fico_range_high", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc"]

# ---- missing-value groups (paper: mean-set / max-set / zero-set)
# max-set: every "months since ..." column (prefix rule below)
MAX_SET_PREFIX = ("mths_since", "mo_sin")
# mean-set: ratios and dollar levels, where 0 would be a misleadingly strong value
MEAN_SET = {"dti", "revol_util", "bc_util", "pct_tl_nvr_dlq", "percent_bc_gt_75", "bc_open_to_buy", "avg_cur_bal",
            "tot_cur_bal", "total_rev_hi_lim", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
            "total_il_high_credit_limit", "annual_inc", "int_rate", "installment", "revol_bal", "earliest_cr_line"}
# zero-set: counts (0 = none on record)
ZERO_SET_PREFIX = ("num_",)
ZERO_SET = {"tot_coll_amt", "acc_open_past_24mths", "mort_acc", "delinq_2yrs", "inq_last_6mths", "open_acc",
            "pub_rec", "total_acc", "collections_12_mths_ex_med", "acc_now_delinq", "chargeoff_within_12_mths",
            "delinq_amnt", "pub_rec_bankruptcies", "tax_liens"}


def missing_group(col):
    if col.startswith(MAX_SET_PREFIX):
        return "max"
    if col in MEAN_SET:
        return "mean"
    if col in ZERO_SET or col.startswith(ZERO_SET_PREFIX):
        return "zero"
    raise KeyError(f"Column '{col}' has missing values but no mean/max/zero group. "
                   f"Add it to one of the sets at the top of 02_preprocess.py and to ASSUMPTIONS.md.")


def days_since_epoch(s):
    d = pd.to_datetime(s, format="%b-%Y", errors="coerce")
    return (d - pd.Timestamp("1970-01-01")).dt.days.astype("float64")


def drop_columns(df):
    """Return (kept columns, table of dropped columns with reason)."""
    miss = df.isna().mean()
    rows, kept = [], []
    for c in df.columns:
        if c in FREE_TEXT_ID:
            rows.append((c, "free text / id", miss[c]))
        elif c in LEAK_EXACT or c.startswith(LEAK_PREFIX):
            rows.append((c, "leakage (outcome / post-issuance)", miss[c]))
        elif miss[c] > config.MISSING_THRESHOLD:
            rows.append((c, f"more than {config.MISSING_THRESHOLD:.0%} missing", miss[c]))
        else:
            kept.append(c)
    nunique = df[kept].nunique(dropna=True)
    for c in list(kept):
        if nunique[c] <= 1:
            rows.append((c, "constant", miss[c]))
            kept.remove(c)
    dropped = pd.DataFrame(rows, columns=["column", "reason", "pct_missing"])
    dropped["pct_missing"] = (100 * dropped["pct_missing"]).round(2)
    return kept, dropped.sort_values(["reason", "column"]).reset_index(drop=True)


def main():
    args = parse_args(__doc__)
    dirs = Dirs(args.sample)
    src = require(dirs.processed / "filtered.parquet", "01_load_filter.py", dirs)

    with step("stage 02 total") as total:
        with step(f"load {rel(src)}"):
            df = pd.read_parquet(src)
        n = len(df)
        y = df["label"].to_numpy(dtype=np.int8)
        print(f"     {n:,} rows x {df.shape[1]} columns")

        # ---- set aside the columns needed for the regression label, before anything is dropped
        reg = df[REGRESSION_COLS].copy()
        reg["issue_d"] = pd.to_datetime(reg["issue_d"], format="%b-%Y")
        reg["last_pymnt_d"] = pd.to_datetime(reg["last_pymnt_d"], format="%b-%Y", errors="coerce")
        reg["label"] = y

        # ---- Fig. 1: correlation heatmap on raw values (loan_status: 1 = Fully Paid, 0 = Charged Off/Default)
        with step("correlation heatmap (paper Fig. 1)"):
            heat = df[[c for c in HEATMAP_COLS if c != "loan_status"]].copy()
            heat["loan_status"] = y
            corr = heat[HEATMAP_COLS].corr()
            fig, ax = plt.subplots(figsize=(11, 9))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True,
                        annot_kws={"size": 7}, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title("Feature correlation heatmap (loan_status: 1 = Fully Paid, 0 = Charged Off / Default)")
            save_figure(fig, "fig1_correlation_heatmap", dirs)
            save_table(corr.round(4).rename_axis("feature").reset_index(), "02_heatmap_correlations", dirs)

        # ---- drop columns
        with step("drop columns"):
            kept, dropped = drop_columns(df)
            for reason, g in dropped.groupby("reason"):
                print(f"     dropped {len(g):>2} [{reason}]: {', '.join(g['column'])}")
            save_table(dropped, "02_dropped_columns", dirs, floatfmt="{:.2f}")
            X_df = df[kept].copy()
            del df

        # ---- dates -> days since epoch; split column types
        for c in DATE_COLS:
            X_df[c] = days_since_epoch(X_df[c])
        cat_cols = [c for c in kept if X_df[c].dtype == object]
        num_cols = [c for c in kept if c not in cat_cols]
        print(f"     kept {len(kept)} columns: {len(num_cols)} numeric (incl. {len(DATE_COLS)} dates), "
              f"{len(cat_cols)} categorical: {', '.join(cat_cols)}")

        # ---- 70/30 stratified split (done first so fill values and the scaler come from train rows only)
        idx_train, idx_test = train_test_split(np.arange(n), test_size=config.TEST_SIZE, stratify=y,
                                               random_state=config.RANDOM_STATE)
        print(f"     split: train {len(idx_train):,} ({y[idx_train].mean():.2%} positive) | "
              f"test {len(idx_test):,} ({y[idx_test].mean():.2%} positive)")

        # ---- missing values
        with step("fill missing values (mean-set / max-set / zero-set)"):
            rows = []
            for c in num_cols:
                n_miss = int(X_df[c].isna().sum())
                if n_miss:
                    g = missing_group(c)
                    tr = X_df[c].iloc[idx_train]
                    fill = {"mean": tr.mean(), "max": config.MAX_SET_FACTOR * tr.max(), "zero": 0.0}[g]
                    X_df[c] = X_df[c].fillna(fill)
                    rows.append((c, f"{g}-set", n_miss, round(100 * n_miss / n, 2), fill))
            cat_missing = {c: int(X_df[c].isna().sum()) for c in cat_cols if X_df[c].isna().any()}
            for c, n_miss in cat_missing.items():
                rows.append((c, "categorical -> own level 'Unknown'", n_miss, round(100 * n_miss / n, 2), np.nan))
            groups = pd.DataFrame(rows, columns=["column", "group", "n_missing", "pct_missing", "fill_value"])
            groups = groups.sort_values(["group", "column"]).reset_index(drop=True)
            for g, sub in groups.groupby("group"):
                print(f"     {g} ({len(sub)}): {', '.join(sub['column'])}")
            save_table(groups, "02_missing_value_groups", dirs)

        # ---- design matrix: numeric block + one-hot block (dense float32, see ASSUMPTIONS.md)
        with step("one-hot encode + build design matrix"):
            names, levels = list(num_cols), {}
            for c in cat_cols:
                X_df[c] = X_df[c].fillna("Unknown").astype(str)
                levels[c] = sorted(X_df[c].unique())
                names += [f"{c}={v}" for v in levels[c]]
            p = len(names)
            X = np.zeros((n, p), dtype=config.X_DTYPE)
            X[:, :len(num_cols)] = X_df[num_cols].to_numpy(dtype=config.X_DTYPE)
            offset = len(num_cols)
            for c in cat_cols:
                codes = pd.Categorical(X_df[c], categories=levels[c]).codes.astype(np.int64)
                X[np.arange(n), offset + codes] = 1.0
                offset += len(levels[c])
            del X_df
            print(f"     one-hot levels: " + ", ".join(f"{c}={len(v)}" for c, v in levels.items()))
        print(f"\n     FINAL FEATURE COUNT: {p:,} = {len(num_cols)} numeric + {p - len(num_cols):,} one-hot"
              f"   (paper: 1,097)\n")

        # ---- standardize with train statistics, save
        with step("standardize (fit on train only) + save"):
            X_train = X[idx_train]
            mu = X_train.mean(axis=0, dtype=np.float64)
            ss = np.zeros(p, dtype=np.float64)
            for i in range(0, len(X_train), 50_000):
                ss += ((X_train[i:i + 50_000].astype(np.float64) - mu) ** 2).sum(axis=0)
            sd = np.sqrt(ss / len(X_train))
            n_const = int((sd == 0).sum())
            sd[sd == 0] = 1.0                      # level seen only in test rows: column stays 0 in train
            mu32, sd32 = mu.astype(config.X_DTYPE), sd.astype(config.X_DTYPE)
            X_train -= mu32
            X_train /= sd32
            X_test = X[idx_test]
            del X
            X_test -= mu32
            X_test /= sd32
            np.save(dirs.processed / "X_train.npy", X_train)
            np.save(dirs.processed / "X_test.npy", X_test)
            np.save(dirs.processed / "y_train.npy", y[idx_train])
            np.save(dirs.processed / "y_test.npy", y[idx_test])
            np.save(dirs.processed / "idx_train.npy", idx_train)
            np.save(dirs.processed / "idx_test.npy", idx_test)
            save_json(names, dirs.processed / "feature_names.json")
            reg.iloc[idx_train].reset_index(drop=True).to_parquet(dirs.processed / "regression_cols_train.parquet")
            reg.iloc[idx_test].reset_index(drop=True).to_parquet(dirs.processed / "regression_cols_test.parquet")
            gb = (X_train.nbytes + X_test.nbytes) / 1024 ** 3
            print(f"     X_train {X_train.shape}, X_test {X_test.shape}, {config.X_DTYPE}, {gb:.2f} GB in {rel(dirs.processed)}/")
            print(f"     columns with zero variance in train (left unscaled): {n_const}")

        # ---- PCA check; all original features are kept for modelling, as in the paper
        with step(f"PCA at {config.PCA_VARIANCE:.0%} explained variance (fit on train)"):
            pca = PCA(n_components=config.PCA_VARIANCE, svd_solver="covariance_eigh",
                      random_state=config.RANDOM_STATE).fit(X_train)
            k = int(pca.n_components_)
        print(f"\n     PCA: {k} components reach {pca.explained_variance_ratio_.sum():.2%} of the variance "
              f"= {k / p:.0%} of the {p:,} features   (paper: ~900 of 1,097)\n")

        summary = pd.DataFrame({
            "quantity": ["train rows", "test rows", "features", "numeric features", "one-hot features",
                         "PCA components for 95% variance", "PCA components / features (%)"],
            "mine": [len(idx_train), len(idx_test), p, len(num_cols), p - len(num_cols), k, round(100 * k / p, 1)],
            "paper": ["521,870", "223,659", "1,097", "", "", "~900", "~82 (900 of 1,097)"]})
        save_table(summary, "02_preprocess_summary", dirs)
    record_runtime(dirs, STAGE, "total", total["seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
