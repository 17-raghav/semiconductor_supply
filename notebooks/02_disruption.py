import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

CSV_PATH = r'D:\Supply_chain_project\data\IPG3344S.csv'

def build_features(df):
    df = df.sort_values('observation_date').reset_index(drop=True)
    df['lag_1'] = df['IPG3344S'].shift(1)
    df['lag_3'] = df['IPG3344S'].shift(3)
    df['lag_12'] = df['IPG3344S'].shift(12)
    df['rolling_mean_3'] = df['IPG3344S'].shift(1).rolling(3).mean()
    df['rolling_mean_6'] = df['IPG3344S'].shift(1).rolling(6).mean()
    df['rolling_mean_12'] = df['IPG3344S'].shift(1).rolling(12).mean()
    df['rolling_std_3'] = df['IPG3344S'].shift(1).rolling(3).std()
    df['month'] = df['observation_date'].dt.month
    df['mom_change'] = df['IPG3344S'].pct_change(1) * 100
    df['yoy_change'] = df['IPG3344S'].pct_change(12) * 100
    return df

df = pd.read_csv(CSV_PATH)
df['observation_date'] = pd.to_datetime(df['observation_date'], format='%d-%m-%Y')
df = build_features(df)
data = df.dropna().reset_index(drop=True)

feature_cols = ['lag_1', 'lag_3', 'lag_12',
                'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
                'rolling_std_3', 'month', 'mom_change', 'yoy_change']

train = data[data['observation_date'] < '2023-01-01'].copy()
test = data[data['observation_date'] >= '2023-01-01'].copy()

thresh = train['mom_change'].quantile(0.10) 
data['is_disruption'] = (data['mom_change'] <= thresh).astype(int)
train['is_disruption'] = (train['mom_change'] <= thresh).astype(int)
test['is_disruption'] = (test['mom_change'] <= thresh).astype(int)

print(f"Disruption threshold (bottom 10% MoM, train-based): {thresh:.3f}%")
print(f"Disruption months  -> train: {train['is_disruption'].sum()}/{len(train)}"
      f"  test: {test['is_disruption'].sum()}/{len(test)}")

def evaluate(name, y_true, y_pred):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"Precision (disruption): {p:.3f}")
    print(f"Recall    (disruption): {r:.3f}")
    print(f"F1        (disruption): {f:.3f}")
    print("Confusion matrix [rows=actual 0/1, cols=pred 0/1]:")
    print(cm)
    return p, r, f

clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                    random_state=42, eval_metric='logloss')
clf.fit(train[feature_cols], train['is_disruption'])
pred_a = clf.predict(test[feature_cols])
evaluate("Model A: same features (includes mom_change -> leaks the label)",
         test['is_disruption'], pred_a)

imp_a = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nModel A feature importances:")
print(imp_a)

predict_cols = ['lag_1', 'lag_3', 'lag_12',
                'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
                'rolling_std_3', 'month']

data['target_next_disruption'] = data['is_disruption'].shift(-1)
data_b = data.dropna(subset=['target_next_disruption']).copy()
data_b['target_next_disruption'] = data_b['target_next_disruption'].astype(int)

train_b = data_b[data_b['observation_date'] < '2023-01-01']
test_b = data_b[data_b['observation_date'] >= '2023-01-01']

pos = train_b['target_next_disruption'].sum()
neg = len(train_b) - pos
spw = neg / max(pos, 1)

clf_b = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                      random_state=42, eval_metric='logloss',
                      scale_pos_weight=spw)
clf_b.fit(train_b[predict_cols], train_b['target_next_disruption'])
pred_b = clf_b.predict(test_b[predict_cols])

print(f"\nNext-month disruptions -> train: {train_b['target_next_disruption'].sum()}/{len(train_b)}"
      f"  test: {test_b['target_next_disruption'].sum()}/{len(test_b)}")
evaluate("Model B: forecast NEXT month's disruption (leak-free, the honest test)",
         test_b['target_next_disruption'], pred_b)

imp_b = pd.Series(clf_b.feature_importances_, index=predict_cols).sort_values(ascending=False)
print("\nModel B feature importances:")
print(imp_b)