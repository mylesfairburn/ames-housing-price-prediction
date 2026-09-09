# Ames Housing — House Price Prediction

A regression model for Kaggle's *House Prices: Advanced Regression Techniques*, predicting residential sale prices in Ames, Iowa from 79 features.

**Result:** 326 / ~3,200 — top ~10% · 5-fold CV RMSE ≈ 0.110 (log scale)

![CV RMSE by modelling step](assets/progression.png)

## Problem
Predict `SalePrice` for each home. Submissions are scored on RMSE between the *logarithm* of the predicted and actual price, so a proportional error on a cheap house counts the same as on an expensive one.

## Approach
- **Cleaning** — domain-aware imputation: an `NA` that means "feature absent" (no pool, no basement) becomes `"None"` or `0`; genuinely missing categoricals take the column mode; `LotFrontage` takes the median for its neighbourhood.
- **Encoding** — ordinal integer mapping for quality grades (`Po` < `Fa` < `TA` < `Gd` < `Ex`); one-hot encoding for nominal fields.
- **Feature engineering** — `TotalSF`, total bathrooms, total porch area, house and remodel age, and binary flags for a pool, garage, fireplace, and second floor.
- **Outliers** — removed the known oversized, underpriced Ames partial sales (`GrLivArea > 4000` with a low price).
- **Skew** — `log1p` applied to the target and to heavily skewed area features.
- **Models** — a blend of Lasso, Ridge and `HistGradientBoostingRegressor` (Lasso 0.45 / GBR 0.30 / Ridge 0.25), with blend weights chosen on out-of-fold predictions to avoid overfitting. The chart above tracks CV RMSE from a RandomForest baseline through to the final blend.

## Repo contents
- `ames-housing-price-prediction.ipynb` — the full analysis
- `requirements.txt` — dependencies
- `assets/progression.png` — the results chart above
- `LICENSE` — MIT
- `.gitignore`

The `data/` folder (which you create — see below) and the generated `submission.csv` are gitignored, so they aren't part of the committed repo.

## Data
The competition CSVs are **not** included — Kaggle competition data shouldn't be redistributed. Download them from the [competition data page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data), or via the Kaggle CLI:

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques
```

Create a `data/` folder at the repository root and place `train.csv` and `test.csv` inside it. The notebook reads them from `data/`.

## Running it
```bash
pip install -r requirements.txt
jupyter notebook ames-housing-price-prediction.ipynb
```

Running the notebook writes predictions to `submission.csv` in the repository root.
