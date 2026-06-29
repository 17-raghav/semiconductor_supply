import pandas as pd

FEATURE_COLS = ['lag_1', 'lag_3', 'lag_12',
                'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
                'rolling_std_3', 'month', 'mom_change', 'yoy_change']


def build_features(df):
    df = df.sort_values('observation_date').reset_index(drop=True)

    df['lag_1'] = df['IPG3344S'].shift(1)
    df['lag_3'] = df['IPG3344S'].shift(3)
    df['lag_12'] = df['IPG3344S'].shift(12)

    df['rolling_mean_3'] = df['IPG3344S'].shift(1).rolling(window=3).mean()
    df['rolling_mean_6'] = df['IPG3344S'].shift(1).rolling(window=6).mean()
    df['rolling_mean_12'] = df['IPG3344S'].shift(1).rolling(window=12).mean()
    df['rolling_std_3'] = df['IPG3344S'].shift(1).rolling(window=3).std()

    df['month'] = df['observation_date'].dt.month

    df['mom_change'] = df['IPG3344S'].pct_change(periods=1) * 100
    df['yoy_change'] = df['IPG3344S'].pct_change(periods=12) * 100

    return df
