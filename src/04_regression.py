"""Stage 4 (paper Sections VI-VII): net annualized return (NAR) label and regression models."""
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

import config
from utils import (Dirs, load_feature_names, load_xy, parse_args, record_runtime, rel, require,
                   save_figure, save_table, step)

STAGE = "04_regression"
MODELS = ("linear", "ridge", "mlp", "rf4", "rf8", "rf10")


# ----------------------------------------------------------------------------- label
def nar_label(reg, split):
    """NAR = (total_pymnt / loan_amnt) ** (365 / D) - 1, D = days from issue_d to last_pymnt_d.

    A loan whose last_pymnt_d is missing or not after issue_d (D <= 0) has no usable D and is EXCLUDED
    from the regression (its NAR is NaN and its `keep` flag is False). For 0 < D < MIN_DAYS the
    MIN_DAYS floor is applied. Returns (nar, keep mask, one-row summary of exclusions for this split)."""
    D = (reg["last_pymnt_d"] - reg["issue_d"]).dt.days
    missing, nonpos = D.isna().to_numpy(), (D <= 0).to_numpy()
    floored = ((D > 0) & (D < config.MIN_DAYS)).to_numpy()
    keep = ~(missing | nonpos)
    nar = ((reg["total_pymnt"] / reg["loan_amnt"]) ** (365.0 / D.clip(lower=config.MIN_DAYS)) - 1)
    nar = nar.to_numpy(dtype=np.float64)
    nar[~keep] = np.nan
    label = reg["label"].to_numpy()
    info = {"split": split, "loans": len(reg),
            "excluded: last_pymnt_d missing": int(missing.sum()),
            "excluded: D <= 0": int(nonpos.sum()),
            "excluded total": int((~keep).sum()),
            "excluded Charged Off / Default": int((~keep & (label == 0)).sum()),
            "excluded Fully Paid": int((~keep & (label == 1)).sum()),
            "kept for regression": int(keep.sum()),
            f"kept with 0 < D < {config.MIN_DAYS} (D set to {config.MIN_DAYS})": int(floored.sum())}
    return nar, keep, info


def describe_nar(nar, label, dirs):
    """Print and save the NAR distribution (kept rows of this run: train + test)."""
    s, co, fp = pd.Series(nar), pd.Series(nar[label == 0]), pd.Series(nar[label == 1])
    qs = [0, .01, .05, .25, .5, .75, .95, .99, 1]
    dist = pd.DataFrame({"statistic": ["count", "mean", "std", "variance"] + [f"quantile {q:.0%}" for q in qs]})
    for nm, x in (("all loans", s), ("Charged Off / Default", co), ("Fully Paid", fp)):
        dist[nm] = [len(x), x.mean(), x.std(), x.var()] + [x.quantile(q) for q in qs]
    print("\n     NAR distribution:")
    print(dist.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    shares = pd.DataFrame({
        "quantity": [f"share of Charged Off / Default loans with NAR < {t}" for t in (-0.25, -0.5, -0.75, -0.9)]
                    + ["share of Fully Paid loans with NAR > 0", "loans with NAR > 1", "largest NAR"],
        "value": [float((co < t).mean()) for t in (-0.25, -0.5, -0.75, -0.9)]
                 + [float((fp > 0).mean()), int((s > 1).sum()), float(s.max())]})
    below = float((co < -0.5).mean())
    print(f"\n     Charged Off / Default loans with NAR < -0.5: {below:.1%} (median NAR {co.median():.3f}). "
          f"Paper: 'most ... are well below -0.5' -> {'CONFIRMED' if below > 0.5 else 'NOT confirmed on this data'}")
    save_table(dist, "04_nar_distribution", dirs)
    save_table(shares, "04_nar_checks", dirs)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    clipped = np.clip(nar, -1.0, 1.0)
    axs[0].hist([clipped[label == 1], clipped[label == 0]], bins=80, stacked=True,
                label=["Fully Paid", "Charged Off / Default"], color=["#2a9d8f", "#e76f51"])
    axs[0].set(title="NAR distribution (values above 1 clipped to 1)", xlabel="NAR", ylabel="loans")
    axs[0].legend()
    axs[1].hist(nar[label == 0], bins=60, color="#e76f51")
    axs[1].axvline(-0.5, color="k", ls="--", lw=1)
    axs[1].set(title="NAR of Charged Off / Default loans", xlabel="NAR", ylabel="loans")
    save_figure(fig, "nar_distribution", dirs)


# ----------------------------------------------------------------------------- models
def make_model(name):
    """Return (estimator, dtype used for fitting)."""
    rs = config.RANDOM_STATE
    if name == "linear":
        return LinearRegression(n_jobs=-1), config.LINEAR_DTYPE
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=rs), config.LINEAR_DTYPE
    if name == "mlp":
        return (MLPRegressor(hidden_layer_sizes=(20, 10, 5, 3), activation="relu", solver="adam", batch_size=200,
                             alpha=0.0001, random_state=rs), config.X_DTYPE)
    if name.startswith("rf"):
        return (RandomForestRegressor(n_estimators=config.RF_REG_TREES, max_depth=int(name[2:]),
                                      max_features=config.RF_REG_MAX_FEATURES, n_jobs=-1, random_state=rs),
                config.X_DTYPE)
    raise ValueError(name)


def linear_diagnostics(model, X_test, y_test, pred, dirs):
    """Numbers needed to explain whether / why LinearRegression blows up on the test set."""
    names = load_feature_names(dirs)
    coef = model.coef_
    worst = int(np.argmax(np.abs(pred - y_test)))
    contrib = X_test[worst].astype(np.float64) * coef
    j = int(np.argmax(np.abs(contrib)))
    top = np.argsort(-np.abs(coef))[:5]
    rows = [("design-matrix columns", len(coef)), ("rank found by least squares", int(model.rank_)),
            ("largest |coefficient|", float(np.abs(coef).max())),
            ("median |coefficient|", float(np.median(np.abs(coef)))),
            ("largest |standardized feature value| in test", float(np.abs(X_test).max())),
            ("test prediction min", float(pred.min())), ("test prediction max", float(pred.max())),
            ("test rows with |prediction| > 10", int((np.abs(pred) > 10).sum())),
            ("largest absolute test error", float(np.abs(pred - y_test).max())),
            ("... its prediction", float(pred[worst])), ("... its true NAR", float(y_test[worst])),
            (f"... largest single term: {names[j]} (x={X_test[worst, j]:.4g}, coef={coef[j]:.4g})", float(contrib[j]))]
    rows += [(f"top-5 |coef|: {names[k]}", float(coef[k])) for k in top]
    diag = pd.DataFrame(rows, columns=["quantity", "value"])
    print("\n     LinearRegression diagnostics:")
    print(diag.to_string(index=False))
    save_table(diag, "04_linear_diagnostics", dirs)


def run_model(name, dirs, y, keep):
    """Fit and evaluate one regressor on the kept rows (y / keep: dicts with 'train' and 'test' entries)."""
    model, dtype = make_model(name)
    print(f"\n================ {name}: {model}")
    X_train, _ = load_xy(dirs, "train", dtype, rows=keep["train"])
    with step(f"fit {name} on {X_train.shape[0]:,} x {X_train.shape[1]:,} ({dtype})") as fit:
        model.fit(X_train, y["train"])
    pred_train = model.predict(X_train)
    del X_train
    X_test, _ = load_xy(dirs, "test", dtype, rows=keep["test"])
    pred_test = model.predict(X_test)
    metrics = pd.DataFrame({"model": name, "split": ["train", "test"],
                            "n_loans": [len(pred_train), len(pred_test)],
                            "mse": [mean_squared_error(y["train"], pred_train), mean_squared_error(y["test"], pred_test)],
                            "r2": [r2_score(y["train"], pred_train), r2_score(y["test"], pred_test)]})
    if getattr(model, "n_iter_", None) is not None:
        metrics["n_iter"] = int(np.max(model.n_iter_))
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    save_table(metrics, f"04_{name}_metrics", dirs, floatfmt="{:.5g}")
    if name == "linear":
        linear_diagnostics(model, X_test, y["test"], pred_test, dirs)
    if name == "rf10":       # predictions used by the loan-selection stage (kept rows only)
        for split, pred in (("train", pred_train), ("test", pred_test)):
            np.savez(dirs.processed / f"rf10_predictions_{split}.npz", predicted_nar=pred, actual_nar=y[split])
        print(f"     saved RF depth-10 predictions to {rel(dirs.processed)}/rf10_predictions_*.npz")
    path = dirs.models / f"{name}_regressor.joblib"
    joblib.dump(model, path, compress=3)
    print(f"     saved model  {rel(path)}")
    record_runtime(dirs, STAGE, f"fit {name}", fit["seconds"])   # timings live only in runtimes.csv


def main():
    args = parse_args(__doc__, models=MODELS)
    dirs = Dirs(args.sample)
    names = MODELS if args.model == "all" else (args.model,)
    with step(f"stage 04 total ({', '.join(names)})"):
        with step("build NAR label"):
            y, keep, infos, labels = {}, {}, [], []
            for split in ("train", "test"):
                reg = pd.read_parquet(require(dirs.processed / f"regression_cols_{split}.parquet",
                                              "02_preprocess.py", dirs))
                nar, keep[split], info = nar_label(reg, split)
                y[split] = nar[keep[split]]
                labels.append(reg["label"].to_numpy()[keep[split]])
                infos.append(info)
            excl = pd.DataFrame(infos)
            print("\n     Loans excluded from the regression (no usable last-payment date) and kept, per split:")
            print(excl.T.to_string(header=False))
            save_table(excl, "04_nar_exclusions", dirs)
            describe_nar(np.concatenate([y["train"], y["test"]]), np.concatenate(labels), dirs)
        for name in names:
            run_model(name, dirs, y, keep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
