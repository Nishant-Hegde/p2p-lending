"""Stage 3 (paper Section V): Logistic Regression, Neural Network and Random Forest classifiers."""
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier

import config
from utils import Dirs, load_xy, parse_args, record_runtime, rel, save_figure, save_table, step

STAGE = "03_classification"
MODELS = ("logreg", "mlp", "rf")


def make_model(name):
    """Return (estimator, dtype used for fitting)."""
    rs = config.RANDOM_STATE
    if name == "logreg":
        return (LogisticRegression(penalty="l2", class_weight="balanced", max_iter=1000, random_state=rs),
                config.LINEAR_DTYPE)
    if name == "mlp":
        return (MLPClassifier(hidden_layer_sizes=(10, 10, 5, 3), activation="logistic", solver="adam",
                              batch_size=200, learning_rate_init=0.001, alpha=0.0001, random_state=rs),
                config.X_DTYPE)
    if name == "rf":
        return (RandomForestClassifier(n_estimators=200, max_features=50, criterion="gini", n_jobs=-1,
                                       random_state=rs), config.X_DTYPE)
    raise ValueError(name)


def evaluate(model, X, y, split):
    """Confusion matrix (rows = true, cols = predicted) and classification report as tidy tables."""
    pred = model.predict(X)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    cm_df = pd.DataFrame({"split": split, "true_class": [f"True {c}" for c in config.CLASS_NAMES],
                          f"Predicted {config.CLASS_NAMES[0]}": cm[:, 0],
                          f"Predicted {config.CLASS_NAMES[1]}": cm[:, 1]})
    rep = classification_report(y, pred, labels=[0, 1], target_names=config.CLASS_NAMES, output_dict=True,
                                zero_division=0)
    rows = [{"split": split, "class": k, "precision": rep[k]["precision"], "recall": rep[k]["recall"],
             "f1": rep[k]["f1-score"], "support": int(rep[k]["support"])}
            for k in config.CLASS_NAMES + ["weighted avg"]]
    print(f"\n     --- {split} set ---")
    print(cm_df.drop(columns="split").to_string(index=False))
    print(classification_report(y, pred, labels=[0, 1], target_names=config.CLASS_NAMES, digits=2, zero_division=0))
    return cm_df, pd.DataFrame(rows)


def plot_roc(model, X_test, y_test, dirs):
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, lw=2, label=f"Random Forest (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC curve, Random Forest classifier (test set)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    save_figure(fig, "roc_random_forest_test", dirs)
    print(f"\n     Random Forest test AUC = {auc:.4f}   (paper: ~0.94)")
    return auc


def run_model(name, dirs):
    model, dtype = make_model(name)
    print(f"\n================ {name}: {model}")
    X_train, y_train = load_xy(dirs, "train", dtype)
    with step(f"fit {name} on {X_train.shape[0]:,} x {X_train.shape[1]:,} ({dtype})") as fit:
        model.fit(X_train, y_train)
    info = {"model": name}          # fit time is recorded only in runtimes.csv
    if getattr(model, "n_iter_", None) is not None:
        info["n_iter"] = int(np.max(model.n_iter_))
        print(f"     iterations / epochs: {info['n_iter']}")
    cm_tr, rep_tr = evaluate(model, X_train, y_train, "train")
    del X_train
    X_test, y_test = load_xy(dirs, "test", dtype)
    cm_te, rep_te = evaluate(model, X_test, y_test, "test")
    if name == "rf":
        info["test_auc"] = plot_roc(model, X_test, y_test, dirs)
    save_table(pd.concat([cm_tr, cm_te], ignore_index=True), f"03_{name}_confusion_matrix", dirs)
    save_table(pd.concat([rep_tr, rep_te], ignore_index=True), f"03_{name}_classification_report", dirs,
               floatfmt="{:.4f}")
    save_table(pd.DataFrame([info]), f"03_{name}_info", dirs, quiet=True)
    path = dirs.models / f"{name}_classifier.joblib"
    with step(f"save model {rel(path)}"):
        joblib.dump(model, path, compress=3)
    record_runtime(dirs, STAGE, f"fit {name}", fit["seconds"])


def main():
    args = parse_args(__doc__, models=MODELS)
    dirs = Dirs(args.sample)
    names = MODELS if args.model == "all" else (args.model,)
    with step(f"stage 03 total ({', '.join(names)})"):
        for name in names:
            run_model(name, dirs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
