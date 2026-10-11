# LendingClub loan default and profitability prediction: a replication

This project replicates the paper **"LendingClub Loan Default and Profitability Prediction"** by Peiqian Li and
Gao Han (Stanford CS229), included here as [`paper.pdf`](paper.pdf), using the public LendingClub accepted-loans
data from Kaggle. It reproduces what the paper does and nothing more:

1. select 2012-2015 loans that have a final status,
2. clean and encode the features,
3. train three classifiers that predict whether a loan is fully paid,
4. train regressors that predict a loan's net annualized return (NAR),
5. simulate the paper's loan-selection strategy.

Results next to the paper's numbers: [`results/RESULTS.md`](results/RESULTS.md).
Every judgement call where the paper is unclear: [`ASSUMPTIONS.md`](ASSUMPTIONS.md).

## Setup for teammates

You need Git, **Python 3.13** (the version the committed results were produced with) and about 6 GB of free disk
for a full run (raw data, cached matrices, models and the virtual environment).

**1. Clone**

```bash
git clone https://github.com/Nishant-Hegde/p2p-lending.git
cd p2p-lending
```

**2. Create and activate a virtual environment**

Windows (PowerShell):

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
```

Mac / Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

If PowerShell refuses to run the activation script, use `.venv\Scripts\activate.bat` from `cmd` instead.

**3. Install the pinned packages**

```bash
pip install -r requirements.txt
```

**4. Get the data** (375 MB, not stored in git)

Option A, Kaggle API:

1. Create a free account at <https://www.kaggle.com> and sign in.
2. Authenticate the Kaggle client (installed in step 3). Either run `kaggle auth login`, or create a token at
   <https://www.kaggle.com/settings/api> and save it as `~/.kaggle/access_token`
   (Windows: `C:\Users\<you>\.kaggle\access_token`). Keep the token outside this folder and never commit it.
3. Download:

```bash
python src/00_download_data.py
```

Option B, manual: open <https://www.kaggle.com/datasets/wordsforthewise/lending-club>, download
`accepted_2007_to_2018Q4.csv.gz`, and put it, still compressed, at
`data/raw/accepted_2007_to_2018Q4.csv.gz`.

**5. Smoke test, then the full run**

```bash
python src/run_all.py --sample 0.01
```

This runs every stage on 1% of the loans and should end with `ALL DONE ... Setup works.` after about 2 minutes
(up to about 7 on a busy or slower machine). Its outputs go to `sample_0.01` folders and do not touch the
committed results. A `ConvergenceWarning` from the neural network during `--sample` runs is harmless: on a
small sample the network hits its 200-epoch limit; it does not happen on the full data. Then:

```bash
python src/run_all.py
```

**6. Expected runtime and memory for the full run**

Measured for the committed run on a 22-thread laptop CPU with 31 GB RAM: **about 3 hours in total**
(per-step times are also at the bottom of `results/RESULTS.md`). Forest training uses all CPU cores, so expect
several times longer on a 4-core machine. Times also vary a lot with what else the machine is doing: across
three runs on the same laptop the depth-4 regression forest took 16, 38 and 46 minutes.

| Stage | Time | Peak RAM | Notes |
|---|---|---|---|
| 01 load + filter | 2 min | 2.5 GB | reads the 2.26 M-row raw file in chunks |
| 02 preprocess | 3 min | 8.7 GB | writes about 3.6 GB to `data/processed/` |
| 03 classification | 39 min | 7.1 GB | logistic regression 3 min, neural network 8.5 min, random forest 22 min |
| 04 regression | 137 min | 14.1 GB | linear 0.8 min, ridge 0.1 min, neural network 5 min, forests depth 4 / 8 / 10: 46 / 25 / 60 min |
| 05 loan selection | seconds | 0.2 GB | |

RAM: the full run peaked at about **14 GB** for one process (the linear models in stage 04, which fit on float64
copies of the 550,774 x 1,126 training matrix). A 16 GB machine is therefore too tight with default settings;
use 24 GB or more, or set `LINEAR_DTYPE = "float32"` in `src/config.py`, which roughly halves that peak
(the numbers of logistic, linear and ridge regression may then differ slightly from the committed ones; this
setting has not been run on the full data). Stage 02 needs about 9 GB either way. Close other applications.

Both teammates should get the same numbers on the same data file, because every random choice uses
`RANDOM_STATE = 42` and the package versions are pinned. A fresh clone on the build machine reproduced the smoke
test exactly; on a different CPU the linear models and neural networks may differ in the last decimals.

## Project structure

```
p2p-lending/
├── data/
│   ├── raw/                  the .csv.gz goes here (not in git)
│   └── processed/            cached outputs of each stage (not in git)
├── src/
│   ├── config.py             paths, RANDOM_STATE, TEST_SIZE and every tunable constant
│   ├── utils.py              command-line options, cache checks, timing, saving tables and figures
│   ├── 00_download_data.py
│   ├── 01_load_filter.py
│   ├── 02_preprocess.py
│   ├── 03_classification.py
│   ├── 04_regression.py
│   ├── 05_loan_selection.py
│   ├── 06_lightgbm_extension.py   extension beyond the paper (optional)
│   └── run_all.py            runs 01 -> 05, then writes results/RESULTS.md
├── models/                   trained models as joblib files (not in git)
├── results/
│   ├── figures/              heatmap, ROC curve, NAR histogram, return curve
│   ├── tables/               every metric table as .csv and .md
│   ├── extension/            LightGBM extension: EXTENSION.md, tables/, figures/
│   └── RESULTS.md            paper vs this replication
├── paper.pdf
├── ASSUMPTIONS.md
├── requirements.txt
└── README.md
```

| Script | Paper section | What it does | Main outputs |
|---|---|---|---|
| `00_download_data.py` | | Downloads the raw file with the Kaggle API; skips if present | `data/raw/accepted_2007_to_2018Q4.csv.gz` |
| `01_load_filter.py` | III-A | Keeps loans issued 2012-2015 with status Fully Paid (label 1) or Charged Off / Default (label 0) | `data/processed/filtered.parquet`, `tables/01_*` |
| `02_preprocess.py` | III-B | Drops empty, constant, free-text and leaking columns; fills missing values (mean / max / zero sets); one-hot encodes; converts dates; 70/30 stratified split; standardizes; PCA check; correlation heatmap | `X_train.npy`, `X_test.npy`, `y_*.npy`, `feature_names.json`, `regression_cols_*.parquet`, `tables/02_*`, `figures/fig1_correlation_heatmap.png` |
| `03_classification.py` | V | Logistic regression, neural network, random forest; confusion matrices and reports for train and test; ROC curve of the forest | `tables/03_*`, `figures/roc_random_forest_test.png`, `models/*_classifier.joblib` |
| `04_regression.py` | VI-VII | Builds the NAR label (loans without a usable last-payment date are excluded); linear, ridge, neural network and three random-forest regressors; MSE and R2 | `tables/04_*`, `figures/nar_distribution.png`, `models/*_regressor.joblib`, `rf10_predictions_*.npz` |
| `05_loan_selection.py` | VIII | Ranks loans by predicted NAR; return curve; results at M = 0.132; sweep of M | `tables/05_*`, `figures/fig2_return_vs_pct_invested.png` |
| `run_all.py` | | Runs 01 to 05 and builds the comparison report; `--with-extension` also runs 06 | `results/RESULTS.md` |
| `06_lightgbm_extension.py` | not in the paper | LightGBM classifier and NAR regressor, loan selection on its predictions, SHAP explanations | `results/extension/` |

## Rerunning one stage or one model

Each script runs on its own, from the project root or from `src/`. It reads the previous stage's cached output
and, if that is missing, stops with a message naming the script to run first.

```bash
python src/02_preprocess.py
```

Train a single model with `--model` (stage 03: `logreg`, `mlp`, `rf`; stage 04: `linear`, `ridge`, `mlp`,
`rf4`, `rf8`, `rf10`):

```bash
python src/03_classification.py --model rf
```

```bash
python src/04_regression.py --model rf10
```

Resume the pipeline from a stage, or only rebuild the report from the saved tables:

```bash
python src/run_all.py --from 4
```

```bash
python src/run_all.py --report-only
```

Add `--sample <fraction>` to any of these for a quick debug run, for example `--sample 0.05`. A sampled run reads
and writes only `data/processed/sample_<fraction>/`, `models/sample_<fraction>/` and
`results/sample_<fraction>/`, so stage N with `--sample 0.05` needs stage N-1 run with the same value.

## Extension: LightGBM

`src/06_lightgbm_extension.py` goes beyond the paper. It adds LightGBM (gradient boosting, which the paper does
not use) on top of the replication: a classifier, a NAR regressor, the paper's loan selection on the LightGBM
predictions, and SHAP explanations of the classifier. It only reads the cached outputs of stages 02-05 and writes
to `results/extension/`, `models/extension/` and `data/processed/extension/`, so the replication results are
never touched. Results and discussion: [`results/extension/EXTENSION.md`](results/extension/EXTENSION.md).

Run it after the replication stages (it needs their cached data, the Random Forest classifier from stage 03
and the depth-10 forest predictions from stage 04, and says which script to run if something is missing):

```bash
python src/06_lightgbm_extension.py
```

Or one part at a time with `--task` (`clf`, `reg`, `select`, `shap`); `select` needs `reg` first and `shap`
needs `clf` first:

```bash
python src/06_lightgbm_extension.py --task clf
```

To run it at the end of the whole pipeline (off by default):

```bash
python src/run_all.py --with-extension
```

`--sample <fraction>` works as for the other scripts. Runtime on the full data: about 7 minutes on the 22-thread
build laptop (tuning the classifier 3 minutes, the regressor 2 minutes), with a peak of about 8 GB RAM.

## Where to find results

* [`results/RESULTS.md`](results/RESULTS.md): side-by-side comparison with the paper and a discussion of the gaps.
* `results/tables/`: one `.csv` and one `.md` per table, prefixed with the stage number.
* `results/figures/`: `fig1_correlation_heatmap.png`, `roc_random_forest_test.png`, `nar_distribution.png`,
  `fig2_return_vs_pct_invested.png`.
* `results/tables/runtimes.md`: time and peak memory of each step of the committed run.
* [`results/extension/EXTENSION.md`](results/extension/EXTENSION.md): the LightGBM extension (not part of the
  paper), with its own `tables/` and `figures/`.
