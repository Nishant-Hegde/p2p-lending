"""Stage 5 (paper Section VIII): loan selection strategy based on the RF depth-10 NAR predictions."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from utils import Dirs, parse_args, record_runtime, require, save_figure, save_table, step

STAGE = "05_loan_selection"
M_GRID = np.round(np.arange(0.05, 0.2001, 0.01), 2)


def top_k_curve(pred, actual):
    """Mean ACTUAL NAR of the k loans with the highest predicted NAR, for k = min_loans .. N.
    Equal stakes ($25 per loan) make the portfolio return the plain mean of the selected loans' NAR."""
    order = np.argsort(-pred, kind="stable")
    k = np.arange(1, len(pred) + 1)
    mean_nar = np.cumsum(actual[order]) / k
    keep = k >= min(config.SELECTION_MIN_LOANS, len(pred))
    return 100.0 * k[keep] / len(pred), mean_nar[keep]


def at_threshold(pred, actual, M):
    sel = pred > M
    k = int(sel.sum())
    return {"n_selected": k, "pct_selected": 100.0 * k / len(pred),
            "mean_actual_nar": float(actual[sel].mean()) if k else np.nan,
            "invested_usd": k * config.STAKE_USD}


def main():
    args = parse_args(__doc__)
    dirs = Dirs(args.sample)
    with step("stage 05 total") as total:
        data = {}
        for split in ("train", "test"):
            z = np.load(require(dirs.processed / f"rf10_predictions_{split}.npz",
                                "04_regression.py --model rf10", dirs))
            data[split] = (z["predicted_nar"], z["actual_nar"])
            p, a = data[split]
            print(f"     {split}: {len(p):,} loans | predicted NAR in [{p.min():.3f}, {p.max():.3f}] | "
                  f"mean actual NAR if investing in ALL loans: {a.mean():.4f}")

        # ---- Fig. 2
        fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        for ax, (split, (p, a)) in zip(axs, data.items()):
            x, yv = top_k_curve(p, a)
            ax.plot(x, yv, lw=1.8, color="#1f77b4")
            ax.axhline(0, color="k", lw=0.6)
            r = at_threshold(p, a, config.SELECTION_M)
            if r["n_selected"]:
                ax.axvline(r["pct_selected"], color="#d62728", ls="--", lw=1,
                           label=f"M = {config.SELECTION_M}: {r['pct_selected']:.2f}% of loans, "
                                 f"NAR {r['mean_actual_nar']:.3f}")
                ax.legend(loc="upper right")
            ax.set(ylabel="mean actual NAR of selected loans",
                   title=f"{split.capitalize()} set: annualized return vs % of loans invested")
            ax.grid(alpha=0.3)
        axs[1].set_xlabel(f"% of loans invested (ranked by predicted NAR, high to low; "
                          f"from {config.SELECTION_MIN_LOANS} loans, ${config.STAKE_USD} each)")
        save_figure(fig, "fig2_return_vs_pct_invested", dirs)

        # ---- paper threshold
        rows = []
        for split, (p, a) in data.items():
            rows.append({"split": split, "M": config.SELECTION_M, **at_threshold(p, a, config.SELECTION_M),
                         "mean_nar_all_loans": float(a.mean())})
        paper = pd.DataFrame(rows)
        print(f"\n     At M = {config.SELECTION_M}   (paper: ~1.76% of loans selected, ~15% return)")
        print(paper.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        save_table(paper, "05_selection_at_paper_threshold", dirs)

        # ---- sweep
        rows = []
        for M in M_GRID:
            row = {"M": M}
            for split, (p, a) in data.items():
                r = at_threshold(p, a, M)
                row.update({f"{split}_n_selected": r["n_selected"], f"{split}_pct_selected": r["pct_selected"],
                            f"{split}_mean_actual_nar": r["mean_actual_nar"]})
            rows.append(row)
        sweep = pd.DataFrame(rows)
        print("\n     Sweep over M:")
        print(sweep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        save_table(sweep, "05_threshold_sweep", dirs)
    record_runtime(dirs, STAGE, "total", total["seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
