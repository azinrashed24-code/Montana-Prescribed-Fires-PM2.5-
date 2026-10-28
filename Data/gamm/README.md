# Hourly GAMM (M4)

Primary sample: **379,433 hours / 761 fires / 21 monitors**.
The n-check is sample counts only. Coefficients, fitted smooths,
and AIC were not refit by that report and remain outside the
primary-target verification. AIC is in `derived/gamm_aic_model_comparison.csv`.
Figure S4 is this GAMM.

M4 uses univariate smooths and random intercepts for fire and monitor. It does **not** include two-dimensional `ti()` terms.

## Hourly table (gzipped)

```
data/gamm/csv/step05_gamm_model_data.csv.gz
```

The uncompressed CSV is about 62 MB. GitHub warns above 50 MB (hard limit 100 MB), so only the gzip is shipped (~6 MB). `read.csv(gzfile(...))` in R and `pandas.read_csv` both read `.gz` directly. Do not add the uncompressed file to git.

Optional decompress:

```
python code/gamm/check_gamm_n.py
python -c "import gzip,shutil; from pathlib import Path; p=Path('data/gamm/csv'); shutil.copyfileobj(gzip.open(p/'step05_gamm_model_data.csv.gz','rb'), open(p/'step05_gamm_model_data.csv','wb'))"
```

## n-check without bam()

```
python code/gamm/check_gamm_n.py
```

## Refit M4 (needs R and mgcv)

```
Rscript code/gamm/step05_fit_gamm_main.R
```

The script looks for `data/gamm/csv/step05_gamm_model_data.csv.gz` relative to the repository root. Writes fit tables next to the CSV and `results/figures/Figure_S4_gamm_smooth_terms.png`.
