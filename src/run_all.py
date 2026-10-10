"""Run stages 01 -> 05 in order, then write results/RESULTS.md (my results vs the paper).

    python src/run_all.py                  full run
    python src/run_all.py --sample 0.01    smoke test on 1% of the loans
    python src/run_all.py --from 3         resume from stage 3
    python src/run_all.py --report-only    only rebuild RESULTS.md from the saved tables
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
from utils import Dirs, fmt_time, rel

SRC = Path(__file__).resolve().parent
STAGES = ["01_load_filter.py", "02_preprocess.py", "03_classification.py", "04_regression.py",
          "05_loan_selection.py"]

# ---------------------------------------------------------------- numbers printed in the paper
PAPER_CLS = {  # test set: class -> (precision, recall, f1)
    "logreg": {"Default": (0.63, 0.88, 0.73), "Paid Off": (0.97, 0.88, 0.92), "weighted avg": (0.90, 0.88, 0.88)},
    "mlp": {"Default": (0.72, 0.71, 0.72), "Paid Off": (0.93, 0.93, 0.93), "weighted avg": (0.89, 0.89, 0.89)},
    "rf": {"Default": (0.76, 0.65, 0.70), "Paid Off": (0.92, 0.95, 0.94), "weighted avg": (0.89, 0.89, 0.89)},
}
PAPER_CLS_TRAIN_WEIGHTED = {"mlp": (0.91, 0.91, 0.91), "rf": (0.97, 0.96, 0.96)}   # logreg train not reported
PAPER_CM_TEST = {"logreg": [[37632, 5135], [22448, 158444]], "mlp": [[30545, 12222], [11864, 169028]],
                 "rf": [[27760, 15007], [8750, 172142]]}
PAPER_AUC = 0.94
PAPER_REG = {  # (train MSE, train R2, test MSE, test R2)
    "linear": (0.040, 0.243, 5.014, -9.494e22), "ridge": (0.040, 0.243, 0.040, 0.238),
    "mlp": (0.036, 0.324, 0.037, 0.306), "rf4": (0.037, 0.298, 0.037, 0.295),
    "rf8": (0.035, 0.329, 0.036, 0.312), "rf10": (0.034, 0.356, 0.036, 0.315),
}
CLS_NAME = {"logreg": "Logistic Regression", "mlp": "Neural Network (10,10,5,3)", "rf": "Random Forest (200 trees)"}
REG_NAME = {"linear": "Linear Regression", "ridge": "Ridge (alpha=1)", "mlp": "Neural Network (20,10,5,3)",
            "rf4": "Random Forest, depth 4", "rf8": "Random Forest, depth 8", "rf10": "Random Forest, depth 10"}


def table(dirs, name):
    path = dirs.tables / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def g(x, fmt="{:.2f}"):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


def sci(x):
    return "n/a" if x is None or np.isnan(x) else (f"{x:.3e}" if abs(x) >= 1e4 else f"{x:.3f}")


# ---------------------------------------------------------------- RESULTS.md
def write_results(dirs):
    L, add = [], None
    add = L.append
    sampled = dirs.sample is not None
    title = "# Results: replication vs paper"
    add(title + (f" (SAMPLE RUN, {dirs.sample:g} of the loans - debug only)" if sampled else ""))
    add("")
    add("Paper: Li & Han, *LendingClub Loan Default and Profitability Prediction* (Stanford CS229), `paper.pdf`. "
        "\"Mine\" = this repository, produced by `python src/run_all.py"
        + (f" --sample {dirs.sample:g}" if sampled else "") + "`. All metrics are on the **test set** unless "
        "stated. Underlying tables: `results/tables/`; decisions where the paper is vague: `ASSUMPTIONS.md`.")
    add("")
    gaps = []          # (short description) collected for the discussion section

    # ---- 1. data
    add("## 1. Dataset and preprocessing (Section III)\n")
    d1, d2 = table(dirs, "01_dataset_summary"), table(dirs, "02_preprocess_summary")
    if d1 is None or d2 is None:
        add("_Stages 01 / 02 have not been run._\n")
    else:
        q1 = dict(zip(d1["quantity"], d1["mine"]))
        q2 = dict(zip(d2["quantity"], d2["mine"]))
        add("| Quantity | Mine | Paper |\n|---|---:|---:|")
        add(f"| Loans (2012-2015, final status) | {int(q1['rows']):,} | 745,529 |")
        add(f"| Positive (Fully Paid) | {q1['pct_positive']:.1f}% | ~81% |")
        add(f"| Negative (Charged Off + Default) | {q1['pct_negative']:.1f}% | ~19% |")
        add(f"| Train / test rows | {int(q2['train rows']):,} / {int(q2['test rows']):,} | 521,870 / 223,659 |")
        add(f"| Features after preprocessing | {int(q2['features']):,} | 1,097 |")
        add(f"| PCA components for 95% variance | {int(q2['PCA components for 95% variance']):,} "
            f"({q2['PCA components / features (%)']:.0f}% of features) | ~900 (82% of features) |")
        add("\nCorrelation heatmap (paper Fig. 1):\n\n![Fig. 1](figures/fig1_correlation_heatmap.png)\n")

    # ---- 2. classification
    add("## 2. Classification (Section V)\n")
    add("Precision / recall / F1 on the test set.\n")
    add("| Model | Class | Mine P / R / F1 | Paper P / R / F1 |\n|---|---|---|---|")
    cls_missing = []
    for m in CLS_NAME:
        rep = table(dirs, f"03_{m}_classification_report")
        if rep is None:
            cls_missing.append(m)
            continue
        te = rep[rep["split"] == "test"].set_index("class")
        for c in ("Default", "Paid Off", "weighted avg"):
            p = PAPER_CLS[m][c]
            add(f"| {CLS_NAME[m]} | {c} | {te.loc[c, 'precision']:.2f} / {te.loc[c, 'recall']:.2f} / "
                f"{te.loc[c, 'f1']:.2f} | {p[0]:.2f} / {p[1]:.2f} / {p[2]:.2f} |")
        d = te.loc["weighted avg", "f1"] - PAPER_CLS[m]["weighted avg"][2]
        if abs(d) >= 0.05:
            gaps.append(f"{CLS_NAME[m]}: weighted F1 {te.loc['weighted avg', 'f1']:.2f} vs {PAPER_CLS[m]['weighted avg'][2]:.2f}")
    add("")
    add("| Model | Train weighted P / R / F1 (mine) | Train weighted (paper) | Test AUC (mine) | Test AUC (paper) |\n|---|---|---|---:|---:|")
    for m in CLS_NAME:
        rep, info = table(dirs, f"03_{m}_classification_report"), table(dirs, f"03_{m}_info")
        if rep is None:
            continue
        tr = rep[(rep["split"] == "train") & (rep["class"] == "weighted avg")].iloc[0]
        pt = PAPER_CLS_TRAIN_WEIGHTED.get(m)
        auc = info["test_auc"].iloc[0] if info is not None and "test_auc" in info else None
        add(f"| {CLS_NAME[m]} | {tr['precision']:.2f} / {tr['recall']:.2f} / {tr['f1']:.2f} | "
            f"{' / '.join(f'{v:.2f}' for v in pt) if pt else 'not reported'} | "
            f"{g(auc, '{:.3f}') if m == 'rf' else ''} | {PAPER_AUC if m == 'rf' else ''} |")
        if m == "rf" and auc is not None and abs(auc - PAPER_AUC) >= 0.03:
            gaps.append(f"Random Forest AUC {auc:.3f} vs {PAPER_AUC}")
    add("")
    add("Test confusion matrices (rows = true class, columns = predicted class):\n")
    add("| Model | True class | Mine: pred. Default | Mine: pred. Paid Off | Paper: pred. Default | Paper: pred. Paid Off |\n|---|---|---:|---:|---:|---:|")
    for m in CLS_NAME:
        cm = table(dirs, f"03_{m}_confusion_matrix")
        if cm is None:
            continue
        te = cm[cm["split"] == "test"].reset_index(drop=True)
        for i in range(2):
            add(f"| {CLS_NAME[m]} | {te.loc[i, 'true_class']} | {int(te.iloc[i, 2]):,} | {int(te.iloc[i, 3]):,} | "
                f"{PAPER_CM_TEST[m][i][0]:,} | {PAPER_CM_TEST[m][i][1]:,} |")
    add("\nTrain confusion matrices and full reports: `results/tables/03_*`. "
        "ROC curve: ![ROC](figures/roc_random_forest_test.png)\n")
    if cls_missing:
        add(f"_Not run yet: {', '.join(cls_missing)}._\n")

    # ---- 3. regression
    add("## 3. Regression on net annualized return (Sections VI-VII)\n")
    chk = table(dirs, "04_nar_checks")
    below = None
    if chk is not None:
        c = dict(zip(chk["quantity"], chk["value"]))
        below = c["share of Charged Off / Default loans with NAR < -0.5"]
        adj = {k.replace("rows adjusted: ", ""): int(v) for k, v in c.items() if k.startswith("rows adjusted")}
        add("NAR = (total_pymnt / loan_amnt)^(365/D) - 1. Rows where D had to be set to the 30-day minimum: "
            + "; ".join(f"{k}: {v:,}" for k, v in adj.items()) + ".\n")
        add(f"Paper claim \"most NARs of defaulted loans are well below -0.5\": in this data **{below:.1%}** of "
            f"Charged Off / Default loans have NAR < -0.5 ({c['share of Charged Off / Default loans with NAR < -0.25']:.1%} "
            f"are below -0.25). Distribution: `results/tables/04_nar_distribution.md`, "
            "figure `figures/nar_distribution.png`.\n")
    add("| Model | Train MSE mine / paper | Train R2 mine / paper | Test MSE mine / paper | Test R2 mine / paper |\n|---|---|---|---|---|")
    reg = {}
    for m in REG_NAME:
        t = table(dirs, f"04_{m}_metrics")
        if t is None:
            add(f"| {REG_NAME[m]} | not run | | | |")
            continue
        tr, te = t[t["split"] == "train"].iloc[0], t[t["split"] == "test"].iloc[0]
        reg[m] = (tr, te)
        p = PAPER_REG[m]
        add(f"| {REG_NAME[m]} | {sci(tr['mse'])} / {p[0]:.3f} | {sci(tr['r2'])} / {p[1]:.3f} | "
            f"{sci(te['mse'])} / {p[2]:.3f} | {sci(te['r2'])} / {sci(p[3])} |")
        if m != "linear" and abs(te["r2"] - p[3]) >= 0.05:
            gaps.append(f"{REG_NAME[m]}: test R2 {te['r2']:.3f} vs {p[3]:.3f}")
    add("")

    # ---- 4. selection
    add("## 4. Loan selection strategy (Section VIII), RF depth-10 predictions\n")
    sel, sweep = table(dirs, "05_selection_at_paper_threshold"), table(dirs, "05_threshold_sweep")
    if sel is None:
        add("_Stage 05 has not been run._\n")
    else:
        add(f"| Split | Loans with predicted NAR > {config.SELECTION_M} (mine) | Mean actual NAR of those loans (mine) | "
            "Mean NAR of all loans (mine) | Paper |\n|---|---:|---:|---:|---|")
        for _, r in sel.iterrows():
            add(f"| {r['split']} | {r['pct_selected']:.2f}% ({int(r['n_selected']):,} loans) | "
                f"{g(r['mean_actual_nar'], '{:.1%}')} | {r['mean_nar_all_loans']:.1%} | ~1.76% of loans, ~15% NAR |")
        te = sel[sel["split"] == "test"].iloc[0]
        if abs(te["pct_selected"] - 1.76) > 0.5 or np.isnan(te["mean_actual_nar"]) or abs(te["mean_actual_nar"] - 0.15) > 0.03:
            gaps.append(f"selection at M={config.SELECTION_M}: {te['pct_selected']:.2f}% of test loans, "
                        f"NAR {g(te['mean_actual_nar'], '{:.1%}')} vs 1.76% / 15%")
        add("\nSweep over the threshold M (share of loans selected and their mean actual NAR):\n")
        add("| M | Train % selected | Train NAR | Test % selected | Test NAR |\n|---:|---:|---:|---:|---:|")
        for _, r in sweep.iterrows():
            add(f"| {r['M']:.2f} | {r['train_pct_selected']:.2f}% | {g(r['train_mean_actual_nar'], '{:.1%}')} | "
                f"{r['test_pct_selected']:.2f}% | {g(r['test_mean_actual_nar'], '{:.1%}')} |")
        add("\nAnnualized return vs share of loans invested (paper Fig. 2):\n\n![Fig. 2](figures/fig2_return_vs_pct_invested.png)\n")

    # ---- 5. discussion
    add("## 5. Where the results differ from the paper, and why\n")
    if sampled:
        add("_This is a sampled debug run; model quality on a small sample is not comparable with the paper. "
            "The discussion below is only meaningful for the full run._\n")
    if gaps:
        add("Large gaps found in this run:\n")
        for s in gaps:
            add(f"- {s}")
        add("")
    add(DISCUSSION.format(
        below="n/a" if below is None else f"{below:.0%}",
        lin_r2=sci(reg["linear"][1]["r2"]) if "linear" in reg else "n/a",
        lin_blew="did" if "linear" in reg and reg["linear"][1]["r2"] < -1e3 else "did **not**"))
    rt = table(dirs, "runtimes")
    if rt is not None:
        add("## 6. Runtime of this run\n")
        add("| Stage | Step | Time | Peak RAM of the process (GB) |\n|---|---|---:|---:|")
        for _, r in rt.iterrows():
            add(f"| {r['stage']} | {r['step']} | {fmt_time(r['seconds'])} | {g(r['peak_ram_gb'])} |")
        add("")
    out = dirs.results / "RESULTS.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {rel(out)}")


DISCUSSION = """**What matches.** The dataset definition (years, final statuses, class balance), the size of the
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

**Linear regression {lin_blew} blow up here** (test R2 = {lin_r2}; paper: about -1e23). In this pipeline
the scaler is fit on the training rows only, so a one-hot column that is constant in train stays exactly zero
after centring and receives a zero coefficient, and scikit-learn's least-squares solver returns the minimum-norm
solution for the collinear one-hot columns. A blow-up of the paper's size needs a column that is numerically
almost, but not exactly, constant in train and takes a large value in a few test rows (for example a rare zip
code after scaling with statistics from the whole dataset); such a column gets an enormous coefficient that only
shows up on the test set. That mechanism is an explanation consistent with the paper's description
("high-variance problem"), not one that was reproduced here. Note also that the paper's own pair of numbers
(test MSE 5.014 with R2 = -9.5e22) cannot both be right for a label with variance ~0.05.

**NAR of defaulted loans.** The paper says most defaulted loans have NAR well below -0.5; here {below} do.
Many charged-off borrowers pay for a year or more before stopping, which gives a moderate negative annualized
return rather than a near-total loss.

**Loan selection.** Because the regressors explain little of the NAR variance, predicted NARs are compressed
towards the mean and very few loans exceed the paper's threshold M = 0.132, so the selected share and its
return differ from the paper's 1.76% / 15% and are noisy; the sweep table shows how the selection behaves at
lower thresholds.
The very high *train* returns in the table come from a handful of loans repaid within a month or two with
fees (NAR above 1, up to 22), which the forest partly memorizes; they do not carry over to the test set.
"""


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=float, default=None, metavar="FRACTION",
                    help="run every stage on a random fraction of the loans, e.g. 0.01")
    ap.add_argument("--from", dest="start", type=int, default=1, choices=range(1, 6),
                    help="first stage to run (1-5); earlier stages must already be cached")
    ap.add_argument("--report-only", action="store_true", help="only rebuild RESULTS.md")
    a = ap.parse_args()
    if a.sample is not None and not 0 < a.sample < 1:
        ap.error("--sample must be a fraction strictly between 0 and 1")
    extra = [] if a.sample is None else ["--sample", f"{a.sample:g}"]
    t_all = time.perf_counter()
    if not a.report_only:
        for i, script in enumerate(STAGES, 1):
            if i < a.start:
                continue
            print(f"\n{'#' * 20} {script} {' '.join(extra)} {'#' * 20}", flush=True)
            t0 = time.perf_counter()
            code = subprocess.run([sys.executable, "-u", str(SRC / script)] + extra).returncode
            if code != 0:
                sys.exit(f"\n{script} failed (exit code {code}); stopping. Fix the problem and resume with "
                         f"python src/run_all.py --from {i}{' ' + ' '.join(extra) if extra else ''}")
            print(f"{'#' * 20} {script} finished in {fmt_time(time.perf_counter() - t0)}", flush=True)
    write_results(Dirs(a.sample))
    if not a.report_only:
        print(f"\nALL DONE in {fmt_time(time.perf_counter() - t_all)}. Setup works." if a.sample
              else f"\nALL DONE in {fmt_time(time.perf_counter() - t_all)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
