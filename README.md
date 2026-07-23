# Stock Price Predictions 📈

Forecasts NVIDIA's 21-trading-day forward return and volatility from technical indicators using XGBoost, then tests whether either forecast is actually worth trading on.

## Results

![AI-Predicted Probability Cone](assets/heatmap.png)
![Monte Carlo Price Paths](assets/monte_carlo.png)
![Feature Correlation Heatmap](assets/correlation_matrix.png)

Example output from a real run, evaluated on the embargoed held-out test set:

| Metric | Model | Baseline | Result |
|---|---|---|---|
| Volatility MAE | 0.0077 | 0.0082 (20-day persistence) | **Beats baseline**, ~6% lower error |
| Return MAE | 0.1315 | 0.1155 (zero return) | **Loses to baseline**, ~14% higher error |
| Return direction accuracy | 54.3% | 50% (coin flip) | **Beat Baseline** |
| Sharpe ratio (66 non-overlapping 21-day periods, 0% risk-free, long-only) | 0.92 | 0.98 (buy-and-hold) | **Loses to buy-and-hold** |

## What it does

- Pulls NVDA's full daily price history via `yfinance`
- Engineers technical and calendar features from OHLCV data: MACD (normalized), RSI(14), ATR% (14), Bollinger %B (20), distance from the 20-day SMA, distance from a rolling 20-day VWAP, 20-day realized volatility, 3 lagged daily log returns, and day-of-week/month/Friday flags
- Drops features that are highly correlated (|corr| > 0.85) with another feature already kept, so the final feature set isn't padded with near-duplicates
- Defines two forward-looking targets over a 21-trading-day horizon — realized volatility and cumulative log return — using `shift(-21)` so each label is built strictly from the days *after* its row
- Trains two independent `XGBRegressor` models (one per target) on a chronological 80/20 split, embargoing the 21 rows before the test cut so no label overlaps the test period
- Evaluates both models against naive baselines on the held-out test set: MAE for both targets, directional accuracy, and a Sharpe ratio for a simple long/flat strategy vs. buy-and-hold, computed on non-overlapping holding periods
- Feeds the most recent day's features into both models to get a single forecast, then builds an analytical probability cone and a Monte Carlo GBM simulation from it
- Saves three plots to `assets/`: probability cone, Monte Carlo fan chart, and a feature correlation heatmap

## Tech stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![yfinance](https://img.shields.io/badge/yfinance-%2343B02A.svg?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-%2317a2b8.svg?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## Model

```python
XGBRegressor(n_estimators=100, max_depth=5, random_state=42)  # one instance per target
```

**Candidate features:** `MACD_Norm`, `RSI`, `ATR_Pct`, `BB_PctB`, `SMA_20_dist`, `VWAP_Dist`, `Daily_return`, `Volatility_20d`, `Return_Lag_1`, `Return_Lag_2`, `Return_Lag_3`, `Day_of_week`, `Month`, `Friday` — pruned to whichever pass the |corr| > 0.85 redundancy check on each run (`Dropped col:` is printed to stdout).

**Targets:** `Target_Future_Vol` (21-day forward realized volatility), `Target_Future_Return` (21-day forward log return)

## Running it

```bash
pip install -r requirements.txt
python main.py
```

Prints the dropped-feature list, the held-out evaluation (MAE, direction accuracy, Sharpe vs. baseline), the current price, and the predicted 21-day return/volatility to stdout, then writes the three plots above to `assets/`.

## Methodology notes

- Every indicator is built with trailing rolling windows, `.ewm()`, or `.shift()`, so no feature uses information from a day after the one it describes.
- Both targets are built in the opposite direction on purpose — `shift(-21)` — so a row's label is defined only by the 21 trading days that follow it, never by data at or before that row.
- **Embargoed split, not a plain chronological one.** Each label depends on the 21 days *after* its row, so the 21 rows immediately before the test cut are dropped from training rather than left in, since their targets would otherwise overlap the test period.
- `Target_Future_Return` and `Daily_return` are both log returns, and the Monte Carlo/probability-cone drift is applied via `exp()` throughout — one consistent return convention end to end.
- `Target_Future_Vol` is the standard deviation of *daily* returns over the next 21 days — a daily-vol estimate — so it's annualized with `sqrt(252)`.
- The Sharpe ratio is computed on non-overlapping 21-day holding periods (sampling every 21st test row), not every row — adjacent rows share 20 of 21 days and aren't independent trades, so scoring every row would understate the true variance of a trade-by-trade return series.
- The probability cone's plotted price range is derived from the model's own predicted drift and volatility at the 60-day horizon (±4 std devs), not a fixed band, so it can't silently clip a highly volatile forecast the way a fixed window would.

## Limitations

- **Return forecasting does not currently work**, confirmed across three separate real-data runs: MAE consistently ~14% worse than guessing zero, and a strategy built on its direction calls loses to buy-and-hold. This is a result, not an open TODO — the honest conclusion right now is that daily technical features don't carry learnable 21-day return signal for NVDA.
- **Direction accuracy (53-55% across runs) still isn't statistically validated.**
- Sharpe ratio is long-only, no transaction costs or slippage, and based on only ~60-70 non-overlapping periods, a small sample for a Sharpe estimate.
- Volatility forecasting is the one result with real support, a consistent MAE edge over the persistence baseline across three separate runs, including one where the baseline's own feature was pruned from the model.
- No hyperparameter search; both `XGBRegressor` configs are hand-picked.
