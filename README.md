# Semiconductor Production Forecaster

Forecasts next month's value of the US semiconductor production index (FRED series IPG3344S)
using an XGBoost model, and serves the forecast through a small FastAPI backend with a web
dashboard on top.

The data is monthly and goes back to 1972 (about 650 points). The model doesn't predict the
raw index value directly. It predicts the month-over-month change and then adds that back onto
the last known value. The index trends upward over the decades, so predicting the level
directly makes the model chase a moving target. The changes are much more stable, which is why
this works better.

## Results

Tested on 2023 onward (the model only trains on data before 2023, no shuffling):

- MAE 0.64, down from 2.66 for a rolling-average baseline (~76% lower error)
- RMSE 1.22 vs 3.44 for the baseline

The most useful feature by far is the 3-month rolling mean.

## How it works

`features.py` builds the inputs from the raw series: lagged values (1, 3, 12 months back),
rolling means and a rolling std, the calendar month, and month-over-month / year-over-year
percent changes. Every rolling/lag feature is shifted by one month so the current month never
leaks into its own prediction.

The training and evaluation live in `notebooks/01_eda.ipynb`. It loads the data, builds the
features, splits by date (train before 2023, test after), compares against a baseline, trains
the regressor, and saves it to `models/xgb_model.pkl`. The saved file contains both the model
and the list of feature columns so the API knows exactly what to feed it.

`api/main.py` loads that model once at startup and exposes three endpoints:

- `GET /` — health check
- `GET /predict` — next month's forecast
- `GET /history` — the last 60 months of actual values, for the chart

`frontend_web/index.html` calls those endpoints and shows the forecast as three summary cards
plus a line chart of recent history with the predicted point at the end.

## Running it

Install the dependencies:

```
pip install -r requirements.txt
```

Start the API from the project root:

```
uvicorn api.main:app --reload
```

It runs on http://127.0.0.1:8000. With that running, open `frontend_web/index.html` in a
browser to see the dashboard. If you want to retrain, run the notebook top to bottom and it
will overwrite the saved model.

## Layout

```
data/IPG3344S.csv         raw monthly index
features.py               feature engineering, shared by the notebook and the API
notebooks/01_eda.ipynb    exploration, training, evaluation
notebooks/02_disruption.py  experimental disruption classifier (see below)
models/xgb_model.pkl      trained model + feature list
api/main.py               FastAPI server
frontend_web/index.html   dashboard
```

## The disruption experiment

`02_disruption.py` is a side experiment that tries to flag rare "disruption" months (the worst
~10% of monthly drops) with a classifier instead of forecasting the value. It didn't really
work. The first version scored a perfect F1 but only because the label is derived from one of
its own input features, which is leakage. A leak-free version that tries to predict next
month's disruption ahead of time does poorly (F1 around 0.11), which makes sense given there
are only a handful of disruption months in the test window. It's kept in the repo as a record
of what was tried, not as a working model.

## Notes

The paths in `api/main.py` are currently absolute Windows paths, so it expects the project at
`D:\Supply_chain_project`. The evaluation also rests on a single test window from 2023 onward,
so the 0.64 number is best read as one snapshot rather than a fully stress-tested score. A
rolling backtest would make it more trustworthy.

Data source: Federal Reserve Economic Data (FRED), series IPG3344S.
