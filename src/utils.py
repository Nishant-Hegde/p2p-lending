"""Shared helpers: CLI arguments, per-run folders, cache checks, timing, saving tables and figures."""
import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

import config


# ----------------------------------------------------------------------------- CLI
def parse_args(description, models=None, tasks=None):
    """Common CLI: --sample <fraction> and, for the model stages, --model <name>
    (or --task <name> for the extension script)."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--sample", type=float, default=None, metavar="FRACTION",
                   help="debug run on a random fraction of the filtered loans, e.g. 0.05 "
                        "(written to separate sample_<fraction> folders; never overwrites full-data outputs)")
    if models:
        p.add_argument("--model", choices=list(models) + ["all"], default="all",
                       help="train only this model (default: all)")
    if tasks:
        p.add_argument("--task", choices=list(tasks) + ["all"], default="all",
                       help="run only this part (default: all)")
    args = p.parse_args()
    if args.sample is not None and not 0 < args.sample < 1:
        p.error("--sample must be a fraction strictly between 0 and 1, e.g. 0.05")
    return args


def sample_tag(sample):
    return None if sample is None else f"sample_{sample:g}"


class Dirs:
    """Output folders of one run. Full run: data/processed, models, results.
    Sampled run: the same folders with a sample_<fraction> sub-folder appended."""

    def __init__(self, sample=None):
        self.sample = sample
        tag = sample_tag(sample)
        sub = (lambda base: base / tag) if tag else (lambda base: base)
        self.processed = sub(config.DATA_PROCESSED)
        self.models = sub(config.MODELS)
        self.results = sub(config.RESULTS)
        self.figures = self.results / "figures"
        self.tables = self.results / "tables"
        for d in (self.processed, self.models, self.figures, self.tables):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def flag(self):
        """CLI suffix that reproduces this run, used in hints printed to the user."""
        return "" if self.sample is None else f" --sample {self.sample:g}"


def rel(path):
    """Path relative to the project root, with forward slashes (for messages)."""
    try:
        return Path(path).resolve().relative_to(config.ROOT).as_posix()
    except ValueError:
        return str(path)


def require(path, script, dirs=None):
    """Stop with a clear message if a previous stage's cached output is missing."""
    if not Path(path).exists():
        flag = dirs.flag if dirs is not None and not script.startswith("00") else ""
        sys.exit(f"\nERROR: required input not found: {rel(path)}\n"
                 f"       Run this first:  python src/{script}{flag}\n")
    return Path(path)


# ----------------------------------------------------------------------------- timing / memory
@contextmanager
def step(label):
    """Print progress and elapsed time for one step. Yields a dict that receives 'seconds'."""
    info = {}
    print(f"[..] {label}", flush=True)
    t0 = time.perf_counter()
    yield info
    info["seconds"] = time.perf_counter() - t0
    print(f"[ok] {label}  ({fmt_time(info['seconds'])})", flush=True)


def fmt_time(s):
    return f"{s:.1f}s" if s < 90 else f"{s / 60:.1f} min"


def peak_memory_gb():
    """Peak resident memory of this process in GB (None if psutil is unavailable)."""
    try:
        import psutil
        m = psutil.Process().memory_info()
        peak = getattr(m, "peak_wset", None)            # Windows
        if peak is None:
            import resource                              # Mac / Linux
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak *= 1 if sys.platform == "darwin" else 1024
        return peak / 1024 ** 3
    except Exception:
        return None


def record_runtime(dirs, stage, step_name, seconds):
    """Insert/update one row of results/tables/runtimes.csv."""
    path = dirs.tables / "runtimes.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    rows = [r for r in rows if not (r["stage"] == stage and r["step"] == step_name)]
    mem = peak_memory_gb()
    rows.append({"stage": stage, "step": step_name, "seconds": round(seconds, 1),
                 "peak_ram_gb": round(mem, 2) if mem else np.nan})
    df = pd.DataFrame(rows).sort_values(["stage", "step"])
    save_table(df, "runtimes", dirs, quiet=True)


# ----------------------------------------------------------------------------- tables / figures / json
def to_markdown(df, floatfmt="{:.4g}"):
    """Minimal GitHub-markdown table writer (avoids the optional 'tabulate' dependency)."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            return f"{v:,}"
        return str(v)
    numeric = [pd.api.types.is_numeric_dtype(df[c]) for c in df.columns]
    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |",
             "|" + "|".join("---:" if n else "---" for n in numeric) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def save_table(df, name, dirs, floatfmt="{:.4g}", quiet=False):
    """Save a table to results/tables/<name>.csv and <name>.md."""
    df.to_csv(dirs.tables / f"{name}.csv", index=False)
    (dirs.tables / f"{name}.md").write_text(to_markdown(df, floatfmt), encoding="utf-8")
    if not quiet:
        print(f"     saved table  {rel(dirs.tables / name)}.csv / .md")


def load_table(name, dirs, script=None):
    path = dirs.tables / f"{name}.csv"
    if script:
        require(path, script, dirs)
    return pd.read_csv(path)


def save_figure(fig, name, dirs):
    import matplotlib.pyplot as plt
    path = dirs.figures / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved figure {rel(path)}")


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                          encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- cached data
def load_xy(dirs, split, dtype=None, rows=None):
    """Load the cached design matrix and classification label of one split ('train' or 'test').
    `rows` (boolean mask) keeps only those rows; the file is memory-mapped so the full matrix is never copied."""
    x_path = require(dirs.processed / f"X_{split}.npy", "02_preprocess.py", dirs)
    y = np.load(require(dirs.processed / f"y_{split}.npy", "02_preprocess.py", dirs))
    if rows is None:
        X = np.load(x_path)
    else:
        X, y = np.load(x_path, mmap_mode="r")[rows], y[rows]
    if dtype is not None and X.dtype != np.dtype(dtype):
        X = X.astype(dtype)
    return X, y


def load_feature_names(dirs):
    return load_json(require(dirs.processed / "feature_names.json", "02_preprocess.py", dirs))
