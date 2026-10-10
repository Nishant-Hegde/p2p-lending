"""Shared settings for every stage. All paths are built relative to the project root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- folders
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

# ---- raw data
KAGGLE_DATASET = "wordsforthewise/lending-club"
RAW_FILE_NAME = "accepted_2007_to_2018Q4.csv.gz"
RAW_FILE = DATA_RAW / RAW_FILE_NAME

# ---- reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.3

# ---- Section III-A: dataset
ISSUE_YEARS = (2012, 2013, 2014, 2015)
LABEL_MAP = {"Fully Paid": 1, "Charged Off": 0, "Default": 0}
CLASS_NAMES = ["Default", "Paid Off"]          # label 0, label 1 (names used in the paper's tables)

# ---- Section III-B: preprocessing (see ASSUMPTIONS.md)
MISSING_THRESHOLD = 0.90      # drop columns with more than 90% missing values
MAX_SET_FACTOR = 2.0          # max-set fill value = MAX_SET_FACTOR * column max (train rows)
PCA_VARIANCE = 0.95
X_DTYPE = "float32"           # dtype of the stored design matrix
LINEAR_DTYPE = "float64"      # dtype used to fit LogisticRegression / LinearRegression / Ridge
                              # ("float32" roughly halves peak RAM at a small cost in numerical precision)

# ---- Sections VI-VIII
MIN_DAYS = 30                 # minimum D in the NAR formula (dates are month-level)
RF_REG_TREES = 100            # paper gives no tree count for the regression forests
RF_REG_MAX_FEATURES = 1.0     # sklearn default for regressors; paper gives none
SELECTION_M = 0.132           # paper's threshold
SELECTION_MIN_LOANS = 100
STAKE_USD = 25
