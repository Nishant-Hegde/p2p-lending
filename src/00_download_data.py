"""Stage 0: download the LendingClub accepted-loans file from Kaggle into data/raw/.

Credentials are never stored in this project: the Kaggle client reads them from your own
user profile (kaggle auth login, ~/.kaggle/access_token, ~/.kaggle/kaggle.json or environment variables).
"""
import sys
import zipfile

import config
from utils import rel, step

INSTRUCTIONS = f"""
The dataset could not be downloaded automatically. Use ONE of the two options below.

OPTION A - Kaggle API
  1. Create a free account at https://www.kaggle.com and sign in.
  2. Install the client (already in requirements.txt):   pip install kaggle
  3. Authenticate, either
       kaggle auth login
     or create a token at https://www.kaggle.com/settings/api ("Generate New Token") and
       - save the token text as  ~/.kaggle/access_token   (Windows: C:\\Users\\<you>\\.kaggle\\access_token), or
       - set the environment variable KAGGLE_API_TOKEN, or
       - (legacy) put the downloaded kaggle.json in ~/.kaggle/
     Keep these files OUTSIDE the project folder; never commit them.
  4. Run again:   python src/00_download_data.py

OPTION B - manual download
  1. Open https://www.kaggle.com/datasets/{config.KAGGLE_DATASET} in a browser (sign in).
  2. Download {config.RAW_FILE_NAME} (about 375 MB), or "Download all" and unzip.
  3. Put the file here (do not decompress the .gz):
       {rel(config.RAW_FILE)}
"""


def main():
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    if config.RAW_FILE.exists():
        print(f"Already present, skipping download: {rel(config.RAW_FILE)} "
              f"({config.RAW_FILE.stat().st_size / 1e6:.0f} MB)")
        return 0
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi   # exits/raises if not configured
        api = KaggleApi()
        api.authenticate()
    except BaseException as e:                                   # ImportError, OSError, SystemExit, ...
        if isinstance(e, KeyboardInterrupt):
            raise
        print(f"\nKaggle API is not installed or not configured ({type(e).__name__}).")
        print(INSTRUCTIONS)
        return 1
    try:
        with step(f"downloading {config.RAW_FILE_NAME} from kaggle.com/datasets/{config.KAGGLE_DATASET}"):
            api.dataset_download_file(config.KAGGLE_DATASET, config.RAW_FILE_NAME,
                                      path=str(config.DATA_RAW), quiet=False)
        zipped = config.DATA_RAW / (config.RAW_FILE_NAME + ".zip")
        if zipped.exists():                                      # Kaggle may wrap large files in a zip
            with zipfile.ZipFile(zipped) as z:
                z.extractall(config.DATA_RAW)
            zipped.unlink()
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        print(f"\nDownload failed: {type(e).__name__}: {e}")
        print(INSTRUCTIONS)
        return 1
    if not config.RAW_FILE.exists():
        print(f"\nDownload finished but {rel(config.RAW_FILE)} was not found.")
        print(INSTRUCTIONS)
        return 1
    print(f"Saved {rel(config.RAW_FILE)} ({config.RAW_FILE.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
