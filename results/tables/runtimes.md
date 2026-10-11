| stage | step | seconds | peak_ram_gb |
|---|---|---:|---:|
| 01_load_filter | total | 139.3 | 2.54 |
| 02_preprocess | total | 166.5 | 8.66 |
| 03_classification | fit logreg | 180.5 | 7.1 |
| 03_classification | fit mlp | 508.9 | 7.1 |
| 03_classification | fit rf | 1306 | 7.1 |
| 04_regression | fit linear | 46.8 | 14.1 |
| 04_regression | fit mlp | 313 | 14.1 |
| 04_regression | fit rf10 | 3597 | 4.88 |
| 04_regression | fit rf4 | 2757 | 4.88 |
| 04_regression | fit rf8 | 1501 | 4.88 |
| 04_regression | fit ridge | 8.4 | 14.1 |
| 05_loan_selection | total | 0.5 | 0.18 |
