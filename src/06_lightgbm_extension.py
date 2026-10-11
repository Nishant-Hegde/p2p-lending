"""Stage 6 - PROJECT EXTENSION (not in the paper): LightGBM gradient boosting.

Adds a LightGBM classifier (Section V), a LightGBM NAR regressor (Sections VI-VII), the paper's loan
selection on its predictions (Section VIII) and SHAP explanations. It only READS the cached outputs of
stages 02-05 and writes to results/extension/, models/extension/ and data/processed/extension/.

    python src/06_lightgbm_extension.py                 all four tasks
    python src/06_lightgbm_extension.py --task clf      classifier (+ ROC vs the replication's Random Forest)
    python src/06_lightgbm_extension.py --task reg      NAR regressor
    python src/06_lightgbm_extension.py --task select   loan selection (needs --task reg)
    python src/06_lightgbm_extension.py --task shap     SHAP explanations (needs --task clf)
"""
import importlib
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

import config
from utils import (Dirs, load_feature_names, load_xy, parse_args, record_runtime, rel, require, save_figure,
                   save_table, step)

# Reuse the replication's own code so the extension is evaluated in exactly the same way:
cls03 = importlib.import_module("03_classification")    # evaluate(): confusion matrix + classification report
reg04 = importlib.import_module("04_regression")        # nar_label(): NAR label and excluded-loans rule
sel05 = importlib.import_module("05_loan_selection")    # top_k_curve(), at_threshold(), M_GRID

STAGE = "06_lightgbm_extension"
TASKS = ("clf", "reg", "select", "shap")

# ---- tuning setup (see ASSUMPTIONS.md, "Extension")
VAL_SIZE = 0.2            # validation split carved out of the TRAINING rows only; the test set is never used
MAX_TREES = 2000          # upper bound for n_estimators; early stopping picks the actual number
EARLY_STOPPING_ROUNDS = 50
GRID = [                  # small fixed grid: (num_leaves, learning_rate, min_child_samples)
    {"num_leaves": 31, "learning_rate": 0.10, "min_child_samples": 20},
    {"num_leaves": 63, "learning_rate": 0.10, "min_child_samples": 20},
    {"num_leaves": 127, "learning_rate": 0.10, "min_child_samples": 20},
    {"num_leaves": 31, "learning_rate": 0.10, "min_child_samples": 200},
    {"num_leaves": 63, "learning_rate": 0.10, "min_child_samples": 200},
    {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 200},
]
SHAP_SAMPLE = 10_000
TOP_N = 20


class ExtDirs:
    """Extension folders: results/extension, models/extension, data/processed/extension
    (inside the sample_<fraction> folders for sampled runs). `base` are the replication's folders."""

    def __init__(self, sample=None):
        self.base = Dirs(sample)
        self.sample, self.flag = sample, self.base.flag
        self.results = self.base.results / "extension"
        self.tables, self.figures = self.results / "tables", self.results / "figures"
        self.models = self.base.models / "extension"
        self.processed = self.base.processed / "extension"
        for d in (self.tables, self.figures, self.models, self.processed):
            d.mkdir(parents=True, exist_ok=True)


def import_lightgbm():
    try:
        import lightgbm
        return lightgbm
    except ImportError:
        sys.exit("\nERROR: lightgbm is not installed. Run:  pip install -r requirements.txt\n")


def make_model(kind, **params):
    lgb = import_lightgbm()
    common = dict(random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1, deterministic=True, force_col_wise=True)
    if kind == "clf":
        return lgb.LGBMClassifier(objective="binary", class_weight="balanced", **common, **params)
    return lgb.LGBMRegressor(objective="regression", **common, **params)


def tune(kind, X, y, dirs):
    """Small grid search with early stopping on a validation split of the training rows.
    Returns the chosen parameters including n_estimators (= best iteration of the winning configuration)."""
    lgb = import_lightgbm()
    metric, higher_better = ("auc", True) if kind == "clf" else ("l2", False)
    i_fit, i_val = train_test_split(np.arange(len(y)), test_size=VAL_SIZE, random_state=config.RANDOM_STATE,
                                    stratify=y if kind == "clf" else None)
    X_fit, X_val, y_fit, y_val = X[i_fit], X[i_val], y[i_fit], y[i_val]
    print(f"     tuning on {len(i_fit):,} training rows, early stopping on {len(i_val):,} validation rows "
          f"(both from the training set), metric: validation {metric}")
    rows = []
    for params in GRID:
        with step(f"  {params}"):
            model = make_model(kind, n_estimators=MAX_TREES, **params)
            model.fit(X_fit, y_fit, eval_X=X_val, eval_y=y_val, eval_metric=metric,
                      callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False)])
            score = float(model.best_score_["valid_0"][metric])
            rows.append({**params, "best_iteration": int(model.best_iteration_), f"validation_{metric}": score})
            print(f"       best iteration {model.best_iteration_}, validation {metric} = {score:.5f}")
    table = pd.DataFrame(rows)
    col = f"validation_{metric}"
    best_i = int(table[col].idxmax() if higher_better else table[col].idxmin())
    table["chosen"] = ["yes" if i == best_i else "" for i in range(len(table))]
    save_table(table, f"06_{kind}_tuning", dirs, floatfmt="{:.5g}")
    best = {k: rows[best_i][k] for k in ("num_leaves", "learning_rate", "min_child_samples")}
    best["n_estimators"] = rows[best_i]["best_iteration"]
    print(f"     chosen: {best}")
    return best


# ----------------------------------------------------------------------------- Random Forest reference
def rf_reference(dirs):
    """Test-set probabilities and feature importances of the replication's Random Forest classifier
    (cached in data/processed/extension after the first call; the 0.5 GB model is loaded only once)."""
    p_path, i_path = dirs.processed / "rf_test_proba.npy", dirs.processed / "rf_feature_importances.npy"
    if not (p_path.exists() and i_path.exists()):
        model_path = require(dirs.base.models / "rf_classifier.joblib", "03_classification.py --model rf", dirs)
        with step("load the replication's Random Forest classifier and score the test set"):
            rf = joblib.load(model_path)
            X_test, _ = load_xy(dirs.base, "test", config.X_DTYPE)
            np.save(p_path, rf.predict_proba(X_test)[:, 1])
            np.save(i_path, rf.feature_importances_)
    return np.load(p_path), np.load(i_path)


# ----------------------------------------------------------------------------- task: clf
def task_clf(dirs):
    X_train, y_train = load_xy(dirs.base, "train", config.X_DTYPE)
    with step("tune LightGBM classifier") as t_tune:
        best = tune("clf", X_train, y_train, dirs)
    model = make_model("clf", **best)
    with step(f"fit LightGBM classifier on all {len(y_train):,} training rows") as fit:
        model.fit(X_train, y_train)
    cm_tr, rep_tr = cls03.evaluate(model, X_train, y_train, "train")
    del X_train
    X_test, y_test = load_xy(dirs.base, "test", config.X_DTYPE)
    cm_te, rep_te = cls03.evaluate(model, X_test, y_test, "test")
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    np.save(dirs.processed / "lgbm_test_proba.npy", proba)
    save_table(pd.concat([cm_tr, cm_te], ignore_index=True), "06_lgbm_confusion_matrix", dirs)
    save_table(pd.concat([rep_tr, rep_te], ignore_index=True), "06_lgbm_classification_report", dirs,
               floatfmt="{:.4f}")
    save_table(pd.DataFrame([{"model": "lgbm", "test_auc": auc, **best}]), "06_lgbm_clf_info", dirs, floatfmt="{:.6g}")
    joblib.dump(model, dirs.models / "lgbm_classifier.joblib", compress=3)
    print(f"     saved model  {rel(dirs.models / 'lgbm_classifier.joblib')}")

    rf_proba, _ = rf_reference(dirs)
    rf_auc = roc_auc_score(y_test, rf_proba)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, p, a, color in (("LightGBM (extension)", proba, auc, "#d62728"),
                              ("Random Forest (replication)", rf_proba, rf_auc, "#1f77b4")):
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name}, AUC = {a:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves on the test set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    save_figure(fig, "roc_lightgbm_vs_random_forest", dirs)
    print(f"\n     test AUC: LightGBM {auc:.4f} | Random Forest (replication) {rf_auc:.4f}")
    record_runtime(dirs, STAGE, "clf: tuning", t_tune["seconds"])
    record_runtime(dirs, STAGE, "clf: final fit", fit["seconds"])


# ----------------------------------------------------------------------------- task: reg
def nar_labels(dirs):
    """NAR label and kept-row masks, built by 04_regression.nar_label (same rule, same excluded loans)."""
    y, keep = {}, {}
    for split in ("train", "test"):
        reg = pd.read_parquet(require(dirs.base.processed / f"regression_cols_{split}.parquet",
                                      "02_preprocess.py", dirs))
        nar, keep[split], info = reg04.nar_label(reg, split)
        y[split] = nar[keep[split]]
        print(f"     {split}: {info['kept for regression']:,} loans kept, {info['excluded total']:,} excluded "
              f"(no usable last-payment date), as in 04_regression.py")
    return y, keep


def task_reg(dirs):
    y, keep = nar_labels(dirs)
    X_train, _ = load_xy(dirs.base, "train", config.X_DTYPE, rows=keep["train"])
    with step("tune LightGBM regressor") as t_tune:
        best = tune("reg", X_train, y["train"], dirs)
    model = make_model("reg", **best)
    with step(f"fit LightGBM regressor on all {len(y['train']):,} kept training rows") as fit:
        model.fit(X_train, y["train"])
    pred = {"train": model.predict(X_train)}
    del X_train
    X_test, _ = load_xy(dirs.base, "test", config.X_DTYPE, rows=keep["test"])
    pred["test"] = model.predict(X_test)
    metrics = pd.DataFrame({"model": "lgbm", "split": ["train", "test"],
                            "n_loans": [len(pred["train"]), len(pred["test"])],
                            "mse": [mean_squared_error(y[s], pred[s]) for s in ("train", "test")],
                            "r2": [r2_score(y[s], pred[s]) for s in ("train", "test")]})
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    save_table(metrics, "06_lgbm_reg_metrics", dirs, floatfmt="{:.5g}")
    save_table(pd.DataFrame([{"model": "lgbm", **best}]), "06_lgbm_reg_info", dirs, floatfmt="{:.6g}")
    for split in ("train", "test"):
        np.savez(dirs.processed / f"lgbm_predictions_{split}.npz", predicted_nar=pred[split], actual_nar=y[split])
    joblib.dump(model, dirs.models / "lgbm_regressor.joblib", compress=3)
    print(f"     saved model and predictions ({rel(dirs.models)}/, {rel(dirs.processed)}/)")
    record_runtime(dirs, STAGE, "reg: tuning", t_tune["seconds"])
    record_runtime(dirs, STAGE, "reg: final fit", fit["seconds"])


# ----------------------------------------------------------------------------- task: select
def task_select(dirs):
    data = {}
    for model, folder, name, script in (
            ("RF depth 10 (replication)", dirs.base.processed, "rf10_predictions", "04_regression.py --model rf10"),
            ("LightGBM (extension)", dirs.processed, "lgbm_predictions", "06_lightgbm_extension.py --task reg")):
        data[model] = {}
        for split in ("train", "test"):
            z = np.load(require(folder / f"{name}_{split}.npz", script, dirs))
            data[model][split] = (z["predicted_nar"], z["actual_nar"])
    rows, sweep = [], []
    for model, d in data.items():
        for split, (p, a) in d.items():
            rows.append({"model": model, "split": split, "M": config.SELECTION_M,
                         **sel05.at_threshold(p, a, config.SELECTION_M), "mean_nar_all_loans": float(a.mean())})
        for M in sel05.M_GRID:
            row = {"model": model, "M": M}
            for split, (p, a) in d.items():
                r = sel05.at_threshold(p, a, M)
                row.update({f"{split}_n_selected": r["n_selected"], f"{split}_pct_selected": r["pct_selected"],
                            f"{split}_mean_actual_nar": r["mean_actual_nar"]})
            sweep.append(row)
    at_m, sweep = pd.DataFrame(rows), pd.DataFrame(sweep)
    print(f"\n     At M = {config.SELECTION_M}   (paper: ~1.76% of loans selected, ~15% return)")
    print(at_m.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    save_table(at_m, "06_selection_at_paper_threshold", dirs)
    save_table(sweep, "06_threshold_sweep", dirs)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (model, d), color in zip(data.items(), ("#1f77b4", "#d62728")):
        x, yv = sel05.top_k_curve(*d["test"])
        ax.plot(x, yv, lw=1.8, color=color, label=model)
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel=f"% of test loans invested (ranked by predicted NAR, high to low; from "
                  f"{config.SELECTION_MIN_LOANS} loans, ${config.STAKE_USD} each)",
           ylabel="mean actual NAR of selected loans",
           title="Test set: annualized return vs % of loans invested")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    save_figure(fig, "return_vs_pct_invested_rf10_vs_lightgbm", dirs)

    # how good is each ranking? mean actual NAR of the top 1%, 1.76% (the paper's share), 5%, 10%, 25% of test loans
    rows = []
    for model, d in data.items():
        p, a = d["test"]
        order = np.argsort(-p, kind="stable")
        row = {"model": model}
        for share in (0.01, 0.0176, 0.05, 0.10, 0.25, 1.0):
            row[f"top {share:.2%}"] = float(a[order[:max(int(share * len(p)), 1)]].mean())
        rows.append(row)
    top = pd.DataFrame(rows)
    print("\n     Mean actual NAR of the best-ranked test loans:")
    print(top.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    save_table(top, "06_top_share_returns_test", dirs)


# ----------------------------------------------------------------------------- task: shap
def task_shap(dirs):
    try:
        import shap
    except ImportError:
        sys.exit("\nERROR: shap is not installed. Run:  pip install -r requirements.txt\n")
    model = joblib.load(require(dirs.models / "lgbm_classifier.joblib", "06_lightgbm_extension.py --task clf", dirs))
    names = load_feature_names(dirs.base)
    X_test = np.load(require(dirs.base.processed / "X_test.npy", "02_preprocess.py", dirs), mmap_mode="r")
    rng = np.random.RandomState(config.RANDOM_STATE)
    rows = np.sort(rng.choice(X_test.shape[0], size=min(SHAP_SAMPLE, X_test.shape[0]), replace=False))
    Xs = np.asarray(X_test[rows])
    with step(f"SHAP TreeExplainer on {len(rows):,} test loans") as t:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sv = shap.TreeExplainer(model).shap_values(Xs)
        if isinstance(sv, list):          # older shap versions: one array per class
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:                  # (rows, features, classes)
            sv = sv[:, :, 1]
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)[:TOP_N]
    _, rf_imp = rf_reference(dirs)
    rf_order = np.argsort(-rf_imp)[:TOP_N]
    table = pd.DataFrame({"rank": np.arange(1, TOP_N + 1),
                          "lightgbm_feature": [names[i] for i in order], "mean_abs_shap": mean_abs[order],
                          "random_forest_feature": [names[i] for i in rf_order],
                          "rf_feature_importance": rf_imp[rf_order]})
    table["also_in_rf_top20"] = ["yes" if names[i] in set(table["random_forest_feature"]) else "" for i in order]
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    save_table(table, "06_shap_top20", dirs)

    shap.summary_plot(sv, Xs, feature_names=names, max_display=TOP_N, show=False)
    fig = plt.gcf()
    fig.axes[0].set_xlabel("SHAP value (log-odds of 'Paid Off')")
    save_figure(fig, "shap_beeswarm", dirs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([names[i] for i in order][::-1], mean_abs[order][::-1], color="#d62728")
    ax.set(xlabel="mean |SHAP value| (log-odds)", title=f"LightGBM classifier: top {TOP_N} features by mean |SHAP|")
    save_figure(fig, "shap_top20_bar", dirs)
    record_runtime(dirs, STAGE, "shap", t["seconds"])


# ----------------------------------------------------------------------------- EXTENSION.md
def read(folder, name):
    path = folder / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def write_extension_md(dirs):
    """Build results/extension/EXTENSION.md from the saved tables (replication tables are read, never written)."""
    T, B = dirs.tables, dirs.base.tables
    L = []
    add = L.append
    sampled = dirs.sample is not None
    add("# Extension: LightGBM gradient boosting" + (f" (SAMPLE RUN, {dirs.sample:g} of the loans - debug only)"
                                                      if sampled else ""))
    add("")
    add("This goes beyond the paper, which uses logistic regression, neural networks and random forests but no "
        "gradient boosting. LightGBM is trained on exactly the same features, train/test split and labels as the "
        "replication, tuned with a small grid and early stopping on a validation split of the training data "
        "(setup and chosen values: `ASSUMPTIONS.md`, section \"Extension\"). The replication results in "
        "`results/RESULTS.md` are not changed by it. All numbers are on the test set unless stated.\n")
    summary = []

    # ---- classification
    add("## 1. Classification\n")
    info = read(T, "06_lgbm_clf_info")
    if info is None:
        add("_Not run yet: `python src/06_lightgbm_extension.py --task clf`._\n")
    else:
        add("| Model | Default P / R / F1 | Paid Off P / R / F1 | Weighted F1 | Test AUC |\n|---|---|---|---:|---:|")
        f1 = {}
        for name, folder, table, key in (("Logistic Regression", B, "03_logreg_classification_report", "logreg"),
                                         ("Neural Network", B, "03_mlp_classification_report", "mlp"),
                                         ("Random Forest", B, "03_rf_classification_report", "rf"),
                                         ("**LightGBM (extension)**", T, "06_lgbm_classification_report", "lgbm")):
            rep = read(folder, table)
            if rep is None:
                continue
            te = rep[rep["split"] == "test"].set_index("class")
            f1[key] = te.loc["weighted avg", "f1"]
            auc = ""
            if key == "lgbm":
                auc = f"{info['test_auc'].iloc[0]:.3f}"
            elif key == "rf" and read(B, "03_rf_info") is not None:
                auc = f"{read(B, '03_rf_info')['test_auc'].iloc[0]:.3f}"
            elif key in ("logreg", "mlp"):
                auc = "not computed"
            cells = [" / ".join(f"{te.loc[c, m]:.2f}" for m in ("precision", "recall", "f1")) for c in ("Default", "Paid Off")]
            add(f"| {name} | {cells[0]} | {cells[1]} | {f1[key]:.2f} | {auc} |")
        add("\nThe paper reports weighted F1 of 0.88 to 0.89 for its three classifiers and the poster an AUC of 0.94; "
            "see `results/RESULTS.md` for why the replication is far below that.\n")
        tr = read(T, "06_lgbm_classification_report")
        w = tr[(tr["split"] == "train") & (tr["class"] == "weighted avg")].iloc[0]
        add(f"LightGBM train weighted F1: {w['f1']:.2f}. Chosen hyperparameters: num_leaves {int(info['num_leaves'].iloc[0])}, "
            f"learning_rate {info['learning_rate'].iloc[0]:g}, min_child_samples {int(info['min_child_samples'].iloc[0])}, "
            f"n_estimators {int(info['n_estimators'].iloc[0])} (tuning table: `tables/06_clf_tuning.md`).\n")
        add("![ROC](figures/roc_lightgbm_vs_random_forest.png)\n")
        rf_info = read(B, "03_rf_info")
        if rf_info is not None:
            a_l, a_r = info["test_auc"].iloc[0], rf_info["test_auc"].iloc[0]
            summary.append(f"Classification: test AUC {a_l:.3f} for LightGBM against {a_r:.3f} for the replication's "
                           f"Random Forest ({a_l - a_r:+.3f}).")
        if "rf" in f1:
            summary.append(f"Weighted F1 is {f1['lgbm']:.2f} for LightGBM against {f1['rf']:.2f} for the Random Forest; "
                           "note that LightGBM uses balanced class weights like the logistic regression, so its "
                           "precision/recall trade-off is closer to the logistic regression's than to the forest's.")

    # ---- regression
    add("## 2. Regression on net annualized return\n")
    lg = read(T, "06_lgbm_reg_metrics")
    if lg is None:
        add("_Not run yet: `python src/06_lightgbm_extension.py --task reg`._\n")
    else:
        add("| Model | Train MSE | Train R2 | Test MSE | Test R2 |\n|---|---:|---:|---:|---:|")
        best_rep = None
        for name, folder, table in [("Linear Regression", B, "04_linear_metrics"), ("Ridge (alpha=1)", B, "04_ridge_metrics"),
                                    ("Neural Network", B, "04_mlp_metrics"), ("Random Forest, depth 4", B, "04_rf4_metrics"),
                                    ("Random Forest, depth 8", B, "04_rf8_metrics"),
                                    ("Random Forest, depth 10", B, "04_rf10_metrics"),
                                    ("**LightGBM (extension)**", T, "06_lgbm_reg_metrics")]:
            t = read(folder, table)
            if t is None:
                continue
            tr, te = t[t["split"] == "train"].iloc[0], t[t["split"] == "test"].iloc[0]
            add(f"| {name} | {tr['mse']:.4f} | {tr['r2']:.3f} | {te['mse']:.4f} | {te['r2']:.3f} |")
            if folder is B and (best_rep is None or te["r2"] > best_rep[1]):
                best_rep = (name, te["r2"])
        ri = read(T, "06_lgbm_reg_info")
        add(f"\nSame NAR label and the same excluded loans as `04_regression.py`. Chosen hyperparameters: num_leaves "
            f"{int(ri['num_leaves'].iloc[0])}, learning_rate {ri['learning_rate'].iloc[0]:g}, min_child_samples "
            f"{int(ri['min_child_samples'].iloc[0])}, n_estimators {int(ri['n_estimators'].iloc[0])} "
            "(tuning table: `tables/06_reg_tuning.md`). The paper's best test R2 is 0.315.\n")
        if best_rep:
            r_l = lg[lg["split"] == "test"]["r2"].iloc[0]
            summary.append(f"Regression: test R2 {r_l:.3f} for LightGBM against {best_rep[1]:.3f} for the best "
                           f"replication model ({best_rep[0]}), a difference of {r_l - best_rep[1]:+.3f}.")

    # ---- selection
    add("## 3. Loan selection\n")
    at_m, sweep, top = read(T, "06_selection_at_paper_threshold"), read(T, "06_threshold_sweep"), read(T, "06_top_share_returns_test")
    if at_m is None:
        add("_Not run yet: `python src/06_lightgbm_extension.py --task select` (after `--task reg`)._\n")
    else:
        add(f"Same logic as `05_loan_selection.py`: invest $25 in every loan with predicted NAR above M.\n")
        add(f"| Model | Split | Loans with predicted NAR > {config.SELECTION_M} | Mean actual NAR of those loans |\n|---|---|---:|---:|")
        for _, r in at_m.iterrows():
            nar = "n/a" if pd.isna(r["mean_actual_nar"]) else f"{r['mean_actual_nar']:.1%}"
            add(f"| {r['model']} | {r['split']} | {r['pct_selected']:.3f}% ({int(r['n_selected']):,} loans) | {nar} |")
        add("\nPaper: about 1.76% of loans selected with about 15% return.\n")
        add("Threshold sweep on the test set:\n")
        models = list(sweep["model"].unique())
        add("| M | " + " | ".join(f"{m}: % selected | {m}: NAR" for m in models) + " |\n|---:|" + "---:|---:|" * len(models))
        for M in sorted(sweep["M"].unique()):
            cells = []
            for m in models:
                r = sweep[(sweep["model"] == m) & (sweep["M"] == M)].iloc[0]
                nar = "n/a" if pd.isna(r["test_mean_actual_nar"]) else f"{r['test_mean_actual_nar']:.1%}"
                cells += [f"{r['test_pct_selected']:.3f}%", nar]
            add(f"| {M:.2f} | " + " | ".join(cells) + " |")
        if top is not None:
            add("\nBecause few predictions reach these thresholds, the ranking is easier to judge by fixed shares: mean "
                "actual NAR of the best-ranked test loans.\n")
            cols = [c for c in top.columns if c != "model"]
            add("| Model | " + " | ".join(c.replace("top 100.00%", "all loans") for c in cols) + " |\n|---|" + "---:|" * len(cols))
            for _, r in top.iterrows():
                add(f"| {r['model']} | " + " | ".join(f"{r[c]:.1%}" for c in cols) + " |")
            l, rf = top[top["model"].str.startswith("LightGBM")].iloc[0], top[top["model"].str.startswith("RF")].iloc[0]
            summary.append(f"Loan selection: the best-ranked 1.76% of test loans (the share the paper invests in) return "
                           f"{l['top 1.76%']:.1%} with LightGBM against {rf['top 1.76%']:.1%} with RF depth 10; investing "
                           f"in every loan returns {l['top 100.00%']:.1%}. The paper reports about 15%.")
        add("\n![selection](figures/return_vs_pct_invested_rf10_vs_lightgbm.png)\n")

    # ---- shap
    add("## 4. Explainability (SHAP)\n")
    sh = read(T, "06_shap_top20")
    if sh is None:
        add("_Not run yet: `python src/06_lightgbm_extension.py --task shap` (after `--task clf`)._\n")
    else:
        add(f"SHAP TreeExplainer on the LightGBM classifier for a random sample of up to {SHAP_SAMPLE:,} test loans. "
            "Positive SHAP values push the prediction towards \"Paid Off\". Feature values are standardized, and "
            "one-hot features are named `column=level`.\n")
        add("![beeswarm](figures/shap_beeswarm.png)\n")
        add("![bar](figures/shap_top20_bar.png)\n")
        add("| Rank | LightGBM feature | Mean \\|SHAP\\| | Also in RF top 20 | Random Forest feature | RF importance |\n|---:|---|---:|---|---|---:|")
        for _, r in sh.iterrows():
            also = r["also_in_rf_top20"] if isinstance(r["also_in_rf_top20"], str) else ""
            add(f"| {int(r['rank'])} | {r['lightgbm_feature']} | {r['mean_abs_shap']:.4f} | {also} | "
                f"{r['random_forest_feature']} | {r['rf_feature_importance']:.4f} |")
        n_common = int((sh["also_in_rf_top20"] == "yes").sum())
        add(f"\n{n_common} of LightGBM's top {TOP_N} features are also in the Random Forest's top {TOP_N}. The two "
            "measures are not the same quantity: SHAP measures the average size of a feature's contribution to "
            "individual predictions, while the forest's importance measures how much the feature reduces Gini impurity.\n")
        if not sampled:
            add(SHAP_INTERPRETATION)

    add("## 5. Summary: how much does LightGBM improve on the paper's models?\n")
    if sampled:
        add("_Sampled debug run: the numbers below are not meaningful for conclusions._\n")
    for s in summary:
        add(f"- {s}")
    if not sampled:
        add(SUMMARY_TEXT)
    rt = read(T, "runtimes")
    if rt is not None:
        add("\n## 6. Runtime of this run\n")
        add("| Step | Seconds |\n|---|---:|")
        for _, r in rt.iterrows():
            add(f"| {r['step']} | {r['seconds']:.0f} |")
    out = dirs.results / "EXTENSION.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {rel(out)}")


# Interpretation of the committed full-data run (written after looking at its tables; not shown for sampled runs).
SHAP_INTERPRETATION = """**Reading the plots.** The interest rate is by far the strongest driver: high rates push a loan
towards "Default", low rates towards "Paid Off". A 36-month term pushes towards "Paid Off" (so 60-month loans
towards default). Later issue dates, more accounts opened in the past 24 months, a higher debt-to-income ratio,
renting and an unknown employment length push towards default; higher income, a higher FICO score and grade A push
towards "Paid Off". These directions are what credit-risk intuition predicts, with one caveat: the effect of
`issue_d` partly reflects how the dataset was built, because recently issued loans are only in the "final status"
sample if they ended early. The Random Forest's impurity importances are spread almost evenly over the
continuous credit-bureau variables (all between 0.016 and 0.029) and contain no one-hot level, while SHAP gives
weight to `term`, `grade` and `home_ownership` levels; impurity importance is known to favour continuous features
with many split points, so the two rankings should not be expected to agree.
"""
SUMMARY_TEXT = """
**Overall.** LightGBM beats the replication's Random Forest on AUC, every replication regressor on test R2, and
RF depth 10 on the return of the best-ranked loans, but the gains are small and nowhere near the paper's reported numbers (AUC 0.94 on the
poster, test R2 0.315, 15% return on 1.76% of loans). A stronger algorithm on the same features does not close
that gap, which is consistent with the gap being about the information in the features and not about the model
class; it does not prove it.

Points to keep in mind:

- LightGBM's weighted F1 is *lower* than the Random Forest's. That is a consequence of the balanced class
  weights, not of a worse model: the forest reaches its weighted F1 by predicting "Paid Off" for almost every
  loan (it finds about 6% of the defaults), while LightGBM finds about two thirds of them. Against the logistic
  regression, which uses the same weighting, LightGBM is only marginally better (weighted F1 0.71 against 0.70,
  with practically the same recall on defaults).
- At the paper's threshold M = 0.132 LightGBM selects only a few dozen loans, so the return at that exact
  threshold still rests on a small number of loans. The fixed-share table and the figure are the more reliable
  comparison.
- The chosen configuration lies at the edge of the small grid (lowest learning rate, largest
  `min_child_samples`), so a wider search might add a little. All results come from one train/test split, without
  confidence intervals.
"""


def main():
    args = parse_args(__doc__.split("\n")[0], tasks=TASKS)
    dirs = ExtDirs(args.sample)
    tasks = TASKS if args.task == "all" else (args.task,)
    require(dirs.base.processed / "X_train.npy", "02_preprocess.py", dirs)
    runners = {"clf": task_clf, "reg": task_reg, "select": task_select, "shap": task_shap}
    with step(f"stage 06 total ({', '.join(tasks)})"):
        for t in tasks:
            print(f"\n================ extension task: {t}")
            with step(f"task {t}"):
                runners[t](dirs)
        write_extension_md(dirs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
