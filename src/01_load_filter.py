"""Stage 1 (paper Section III-A): keep 2012-2015 loans with a final status and attach the class label."""
import sys

import pandas as pd

import config
from utils import Dirs, parse_args, record_runtime, rel, require, save_table, step

STAGE = "01_load_filter"


def read_and_filter(path):
    """Stream the raw file in chunks and keep only matching rows, so memory stays low."""
    parts, n_raw, n_kept = [], 0, 0
    reader = pd.read_csv(path, chunksize=250_000, low_memory=False, dtype={"id": str, "member_id": str})
    for chunk in reader:
        n_raw += len(chunk)
        issued = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")   # footer rows -> NaT
        keep = issued.dt.year.isin(config.ISSUE_YEARS) & chunk["loan_status"].isin(list(config.LABEL_MAP))
        chunk = chunk[keep].copy()
        for c in chunk.columns[chunk.dtypes == object]:     # keep dtypes consistent across chunks
            if c not in ("id", "member_id"):
                try:
                    chunk[c] = pd.to_numeric(chunk[c])
                except (ValueError, TypeError):
                    pass
        parts.append(chunk)
        n_kept += len(chunk)
        print(f"     read {n_raw:>9,} raw rows | kept {n_kept:>7,}", flush=True)
    return pd.concat(parts, ignore_index=True), n_raw


def main():
    args = parse_args(__doc__)
    dirs = Dirs(args.sample)
    full_cache = Dirs(None).processed / "filtered.parquet"

    with step("stage 01 total") as total:
        if args.sample is not None and full_cache.exists():
            # same rows as filtering the raw file again, just faster
            with step(f"load cached full filtered data ({rel(full_cache)})"):
                df = pd.read_parquet(full_cache)
        else:
            require(config.RAW_FILE, "00_download_data.py")
            with step(f"read + filter {rel(config.RAW_FILE)}"):
                df, n_raw = read_and_filter(config.RAW_FILE)
                print(f"     raw rows in file: {n_raw:,}")
            df["label"] = df["loan_status"].map(config.LABEL_MAP).astype("int8")

        print(f"\nFinalized 2012-2015 loans: {len(df):,}")
        print(df["loan_status"].value_counts().to_string())

        if args.sample is not None:
            df = (df.sample(frac=args.sample, random_state=config.RANDOM_STATE)
                    .sort_index().reset_index(drop=True))
            print(f"\n--sample {args.sample:g}: keeping a random {len(df):,} of those rows")

        n, pos = len(df), int(df["label"].sum())
        print(f"\nRows: {n:,}   (paper: 745,529)")
        print(f"Positive / Fully Paid:            {pos:>8,} = {pos / n:.2%}   (paper: ~81%)")
        print(f"Negative / Charged Off + Default: {n - pos:>8,} = {1 - pos / n:.2%}   (paper: ~19%)")
        years = df["issue_d"].str[-4:].value_counts().sort_index()
        print("Rows per issue year: " + ", ".join(f"{y}: {c:,}" for y, c in years.items()))

        out = dirs.processed / "filtered.parquet"
        with step(f"save {rel(out)}"):
            df.to_parquet(out, index=False)

        summary = pd.DataFrame({
            "quantity": ["rows", "n_positive (Fully Paid)", "n_negative (Charged Off + Default)",
                         "pct_positive", "pct_negative", "raw columns"]
                        + [f"rows issued {y}" for y in years.index],
            # dtype=object keeps counts as integers next to the two percentages
            "mine": pd.Series([n, pos, n - pos, round(100 * pos / n, 2), round(100 * (1 - pos / n), 2),
                               df.shape[1] - 1] + [int(c) for c in years.values], dtype=object),
            "paper": ["745,529", "", "", "~81", "~19", ""] + [""] * len(years)})
        save_table(summary, "01_dataset_summary", dirs, floatfmt="{:.2f}")
    record_runtime(dirs, STAGE, "total", total["seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
