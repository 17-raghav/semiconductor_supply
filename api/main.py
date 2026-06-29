from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import build_features

app = FastAPI(title="Semiconductor Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

with open(r'D:\Supply_chain_project\models\xgb_model.pkl', 'rb') as f:
    saved = pickle.load(f)
model = saved['model']
feature_cols = saved['feature_cols']

CSV_PATH = r'D:\Supply_chain_project\data\IPG3344S.csv'

@app.get("/")
def home():
    return {"status": "API is running"}

@app.get("/predict")
def predict():
    df = pd.read_csv(CSV_PATH)
    df['observation_date'] = pd.to_datetime(df['observation_date'], format='%d-%m-%Y')
    df = build_features(df)

    last_row = df.iloc[[-1]]
    X = last_row[feature_cols]
    pred_change = float(model.predict(X)[0])

    last_value = float(last_row['IPG3344S'].iloc[0])
    last_date = last_row['observation_date'].iloc[0]
    next_date = last_date + pd.DateOffset(months=1)
    predicted_value = last_value + pred_change

    return {
        "last_known_date": last_date.strftime('%Y-%m-%d'),
        "last_known_value": round(last_value, 4),
        "predicted_date": next_date.strftime('%Y-%m-%d'),
        "predicted_value": round(predicted_value, 4),
        "predicted_change": round(pred_change, 4),
    }

@app.get("/history")
def history():
    df = pd.read_csv(CSV_PATH)
    df['observation_date'] = pd.to_datetime(df['observation_date'], format='%d-%m-%Y')
    df = df.sort_values('observation_date').tail(60) 
    return {
        "dates": df['observation_date'].dt.strftime('%Y-%m-%d').tolist(),
        "values": df['IPG3344S'].round(4).tolist(),
    }