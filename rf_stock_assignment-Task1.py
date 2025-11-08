# rf_stock_assignment.py
# End-to-end: generate dataset → engineer features → train RF → evaluate → save plots/CSV

import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)

# ---------- 1) Create synthetic OHLCV dataset (2018–2025 business days)
np.random.seed(42)
dates = pd.bdate_range("2018-01-01", "2025-10-31", freq="B")
n = len(dates)
S0, mu, sigma, dt = 2500.0, 0.08, 0.18, 1/252
close = [S0]
for _ in range(1, n):
    z = np.random.normal()
    close.append(close[-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z))
close = np.array(close)

rng = np.random.default_rng(123)
spread = close * (rng.normal(0.0005, 0.002, size=n).clip(-0.01, 0.01))
open_ = np.roll(close, 1); open_[0] = close[0]
high = np.maximum(open_, close) + abs(spread)
low  = np.minimum(open_, close) - abs(spread)
volume = (rng.lognormal(mean=14.2, sigma=0.35, size=n)).astype(int)

df = pd.DataFrame({
    "Date": dates, "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume
}).sort_values("Date")

# ---------- 2) Feature engineering helpers
def SMA(s,w): return s.rolling(w).mean()
def EMA(s,e): return s.ewm(span=e, adjust=False).mean()
def RSI(s, w=14):
    d = s.diff()
    up = d.clip(lower=0); down = -d.clip(upper=0)
    rs = up.rolling(w).mean() / (down.rolling(w).mean() + 1e-9)
    return 100 - (100/(1+rs))
def MACD(s, f=12, sl=26, sig=9):
    ef, es = EMA(s,f), EMA(s,sl)
    m = ef - es
    return m, EMA(m, sig), m - EMA(m, sig)
def Bollinger(s, w=20, k=2):
    mid = SMA(s,w); std = s.rolling(w).std()
    return mid, mid + k*std, mid - k*std

# ---------- 3) Compute features
df["Return_1d"] = df["Close"].pct_change()
df["Ret_5d"] = df["Close"].pct_change(5)
df["SMA_10"] = SMA(df["Close"], 10); df["SMA_20"] = SMA(df["Close"], 20)
df["EMA_10"] = EMA(df["Close"], 10); df["EMA_20"] = EMA(df["Close"], 20)
df["RSI_14"] = RSI(df["Close"], 14)
macd, sig, hist = MACD(df["Close"])
df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd, sig, hist
mid, up, lo = Bollinger(df["Close"], 20, 2)
df["BB_mid"], df["BB_upper"], df["BB_lower"] = mid, up, lo
df["High_Low_Spread"] = (df["High"] - df["Low"]) / df["Close"]
df["Open_Close_Change"] = (df["Close"] - df["Open"]) / df["Open"]
df["Volume_Change"] = df["Volume"].pct_change()
df["Close_lag1"] = df["Close"].shift(1)
df["Volume_lag1"] = df["Volume"].shift(1)

# target
df["Close_next"] = df["Close"].shift(-1)
df["UpTomorrow"] = (df["Close_next"] > df["Close"]).astype(int)

df = df.dropna().reset_index(drop=True)

features = ["Return_1d","Ret_5d","SMA_10","SMA_20","EMA_10","EMA_20","RSI_14",
            "MACD","MACD_signal","MACD_hist","BB_mid","BB_upper","BB_lower",
            "High_Low_Spread","Open_Close_Change","Volume_Change","Close_lag1","Volume_lag1"]

# ---------- 4) Chronological split
split_date = pd.Timestamp("2024-01-01")
train = df[df["Date"] < split_date].copy()
test  = df[df["Date"] >= split_date].copy()

Xtr, ytr = train[features].values, train["UpTomorrow"].values
Xte, yte = test[features].values,  test["UpTomorrow"].values

# ---------- 5) Scale and fit RF
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)

rf = RandomForestClassifier(
    n_estimators=400,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=123,
    n_jobs=-1
)
rf.fit(Xtr_s, ytr)

# ---------- 6) Evaluate
proba = rf.predict_proba(Xte_s)[:,1]
pred  = (proba >= 0.5).astype(int)

acc  = accuracy_score(yte, pred)
prec = precision_score(yte, pred)
rec  = recall_score(yte, pred)
f1   = f1_score(yte, pred)
try:
    auc = roc_auc_score(yte, proba)
except:
    auc = float("nan")

print("=== Test Metrics (from 2024-01-01) ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC AUC  : {auc:.4f}")
print("Confusion Matrix:\n", confusion_matrix(yte, pred))

# ---------- 7) Outputs
os.makedirs("outputs", exist_ok=True)
# a) predictions CSV
out = test[["Date","Close","UpTomorrow"]].copy()
out["Pred"] = pred
out["Prob_Up"] = proba
out.to_csv("outputs/test_predictions.csv", index=False)

# b) ROC curve
fpr, tpr, thr = roc_curve(yte, proba)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/roc_curve.png", dpi=150)
plt.close()

# c) Feature importances
imp = rf.feature_importances_
idx = np.argsort(imp)[::-1]
names = np.array(features)[idx]; vals = imp[idx]
plt.figure(figsize=(9,6))
plt.bar(range(min(15,len(vals))), vals[:15])
plt.xticks(range(min(15,len(vals))), names[:15], rotation=45, ha="right")
plt.title("Top Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.close()

# d) Close vs P(Up)
plt.figure(figsize=(11,5))
plt.plot(test["Date"], test["Close"], label="Close")
plt.twinx()
plt.plot(test["Date"], proba, label="P(Up)", alpha=0.6)
plt.title("Close vs Predicted Up Probability (Test)")
plt.tight_layout()
plt.savefig("outputs/price_vs_prob.png", dpi=150)
plt.close()