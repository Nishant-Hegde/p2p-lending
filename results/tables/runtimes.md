| stage | step | seconds | peak_ram_gb |
|---|---|---:|---:|
| 01_load_filter | total | 139.3 | 2.54 |
| 02_preprocess | total | 166.5 | 8.66 |
| 03_classification | fit logreg | 180.5 | 7.1 |
| 03_classification | fit mlp | 508.9 | 7.1 |
| 03_classification | fit rf | 1306 | 7.1 |
| 04_regression | fit linear | 98.1 | 14.16 |
| 04_regression | fit mlp | 244.4 | 14.16 |
| 04_regression | fit rf10 | 1796 | 14.16 |
| 04_regression | fit rf4 | 2289 | 14.16 |
| 04_regression | fit rf8 | 3516 | 14.16 |
| 04_regression | fit ridge | 45.5 | 14.16 |
| 05_loan_selection | total | 0.7 | 0.18 |
