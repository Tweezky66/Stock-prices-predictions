import os
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

nvda = yf.Ticker("NVDA")
df = nvda.history(period="max")

# Collect the data and define features & patterns to feed the model
TRADING_DAYS = 252
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD_Norm'] = (ema_12 - ema_26) / df["Close"]

# RSI (14)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
df['RSI'] = 100 - (100 / (1 + (gain / loss)))

# ATR (14)
high_low = df['High'] - df['Low']
high_close = np.abs(df['High'] - df['Close'].shift())
low_close = np.abs(df['Low'] - df['Close'].shift())
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['ATR_Pct'] = true_range.rolling(14).mean() / df["Close"]

# Bollinger Bands (20)
sma_20 = df['Close'].rolling(window=20).mean()
std_20 = df['Close'].rolling(window=20).std()
bb_upper = sma_20 + (std_20 * 2)
bb_lower = sma_20 - (std_20 * 2)
df["BB_PctB"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower)

df["SMA_20_dist"] = (df["Close"] / sma_20) - 1

# 6. VWAP (Distance from a rolling 20-day VWAP, as a %)
typical_price = (df['High'] + df['Low'] + df['Close']) / 3
tp_vol = typical_price * df['Volume']
vwap = tp_vol.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
df['VWAP_Dist'] = (df['Close'] / vwap) - 1

# 7. Targets & Lags (Already stationary daily returns)
df["Daily_return"] = np.log(df["Close"] / df["Close"].shift(1))
df["Volatility_20d"] = df["Daily_return"].rolling(window=20).std()

for i in range(1, 4):
    df[f"Return_Lag_{i}"] = df["Daily_return"].shift(i)

df['Target_Future_Vol'] = df['Daily_return'].rolling(window=21).std().shift(-21)

df['Target_Future_Return'] = np.log(df['Close'].shift(-21) / df['Close'])

# Drop NANs from all sliding window(20 days) features
FEATURE_COLS_WITH_WARMUP = [
    'MACD_Norm', 'RSI', 'ATR_Pct', 'BB_PctB', 'SMA_20_dist',
    'VWAP_Dist', 'Volatility_20d', 'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3',
]

df.dropna(subset=FEATURE_COLS_WITH_WARMUP, inplace=True)

df["Day_of_week"] = df.index.day_of_week
df["Month"] = df.index.month
df["Friday"] = (df.index.day_of_week == 4).astype(int)

# Train phase
X_full = df.drop(columns=[
    'Target_Future_Vol',
    'Target_Future_Return', 
    'Open', 
    'High', 
    'Low', 
    'Close', 
    'Volume'
])

latest_features = X_full.iloc[[-1]]

df_train = df.dropna(subset=['Target_Future_Vol', 'Target_Future_Return'])

X = df_train.drop(columns=['Target_Future_Vol', 'Target_Future_Return', 'Open', 'High', 'Low', 'Close', 'Volume'])
y_vol = df_train['Target_Future_Vol']
y_ret = df_train['Target_Future_Return']


HORIZON = 21

n = len(X)
test_start = n - int(n * 0.2)
embargo_start = test_start - HORIZON

X_train, y_vol_train, y_ret_train = X.iloc[:embargo_start], y_vol.iloc[:embargo_start], y_ret.iloc[:embargo_start]
X_test, y_vol_test, y_ret_test = X.iloc[test_start:], y_vol.iloc[test_start:], y_ret.iloc[test_start:]

model_vol = XGBRegressor(
    n_estimators=100, 
    max_depth=5, 
    random_state=42
)

model_vol.fit(X_train, y_vol_train) 

model_ret = XGBRegressor(
    n_estimators=100, 
    max_depth=5, 
    random_state=42
)

model_ret.fit(X_train, y_ret_train)

# Evaluation: model vs  baseline, on the embargoed test set
vol_pred_test = model_vol.predict(X_test)
ret_pred_test = model_ret.predict(X_test)

vol_mae = mean_absolute_error(y_vol_test, vol_pred_test)
ret_mae = mean_absolute_error(y_ret_test, ret_pred_test)


baseline_vol_pred = X_test['Volatility_20d'].values
baseline_ret_pred = np.zeros(len(y_ret_test))

baseline_vol_mae = mean_absolute_error(y_vol_test, baseline_vol_pred)
baseline_ret_mae = mean_absolute_error(y_ret_test, baseline_ret_pred)

direction_acc = float((np.sign(y_ret_test) == np.sign(ret_pred_test)).mean())

print(f"Volatility MAE  | model: {vol_mae:.4f}  baseline (20d persistence): {baseline_vol_mae:.4f}  beats baseline: {vol_mae < baseline_vol_mae}")
print(f"Return MAE      | model: {ret_mae:.4f}  baseline (zero return):    {baseline_ret_mae:.4f}  beats baseline: {ret_mae < baseline_ret_mae}")
print(f"Return direction accuracy (model): {direction_acc * 100:.1f}%\n")

predicted_vol = model_vol.predict(latest_features)[0]
predicted_ret = model_ret.predict(latest_features)[0]

annualized_predicted_vol = predicted_vol * np.sqrt(TRADING_DAYS)
predicted_daily_drift = predicted_ret / 21
current_price = df['Close'].iloc[-1]
predicted_simple_ret = np.exp(predicted_ret) - 1  # convert log return back to a plain % for display

print(f"Current Price: ${current_price:.2f}")
print(f"AI predicted annual volatility: {annualized_predicted_vol * 100:.2f}%")
print(f"AI predicted 21-Day return: {predicted_simple_ret * 100:.2f}%")

# Visualization time
os.makedirs("assets", exist_ok=True)

future_days = 60 
price_lower = current_price * 0.7
price_higher = current_price * 1.3
prices = np.linspace(price_lower, price_higher, 300)
days = np.arange(1, future_days + 1)

daily_vol = annualized_predicted_vol / np.sqrt(TRADING_DAYS) 
prob_matrix = np.zeros((len(prices), len(days))) 

for i, day in enumerate(days):
    expected_price = current_price * np.exp(predicted_daily_drift * day)
    std_dev = current_price * daily_vol * np.sqrt(day)
    density = norm.pdf(prices, loc=expected_price, scale=std_dev)
    prob_matrix[:, i] = density / np.max(density)

num_simulation = 10000
mc_paths = np.zeros((future_days, num_simulation))
mc_paths[0] = current_price

for t in range(1, future_days):
    random_shock = np.random.normal(0, daily_vol, num_simulation)
    mc_paths[t] = mc_paths[t-1] * np.exp((predicted_daily_drift - 0.5 * daily_vol**2) + random_shock)

hist_days = 100
hist_x = np.arange(-hist_days + 1, 1)
hist_y = df["Close"].tail(hist_days).values
future_x = np.arange(1, future_days + 1)

fig1, ax1 = plt.subplots(figsize=(14, 7)) 
fig1.patch.set_facecolor('#0d1117')
ax1.set_facecolor('#0d1117')

ax1.plot(hist_x, hist_y, color='white', linewidth=2, label="Historical Price")

X_grid, Y_grid = np.meshgrid(future_x, prices)
ax1.pcolormesh(X_grid, Y_grid, prob_matrix, cmap='viridis', shading='auto', alpha=0.95)

ax1.axvline(0, color='white', linestyle='--', alpha=0.5)
ax1.set_title("AI-Predicted Institutional Probability Cone", color='white', fontsize=18, fontweight='bold')
ax1.tick_params(colors='white', labelsize=10)
ax1.spines['bottom'].set_color('white')
ax1.spines['left'].set_color('white')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

plt.savefig("assets/heatmap.png", dpi=300, bbox_inches='tight', facecolor='#0d1117')
plt.show() 

fig2, ax2 = plt.subplots(figsize=(14, 7)) 
fig2.patch.set_facecolor('#0d1117')
ax2.set_facecolor("#0d1117")

ax2.plot(hist_x, hist_y, color='white', linewidth=2, label="Historical Price")

ax2.plot(future_x, mc_paths, color='#00FFFF', alpha=0.04, linewidth=1.5)

ax2.axvline(0, color='white', linestyle='--', alpha=0.5)
ax2.set_title("AI-Predicted Monte Carlo Price Paths", color='white', fontsize=18, fontweight='bold')
ax2.tick_params(colors='white', labelsize=10)
ax2.spines['bottom'].set_color('white')
ax2.spines['left'].set_color('white')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.savefig("assets/monte_carlo.png", dpi=300, bbox_inches='tight', facecolor='#0d1117')
plt.show() 

# Check corralation between features
corr_matrix = df.corr()

fig, ax = plt.subplots(figsize=(14, 12))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

cax = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

ticks = np.arange(len(corr_matrix.columns))
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', color='white', fontsize=10)
ax.set_yticklabels(corr_matrix.columns, color='white', fontsize=10)

for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        val = corr_matrix.iloc[i, j]
        text_color = "black" if abs(val) > 0.5 else "white"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9)

ax.set_title("AI Feature Correlation Heatmap", color='white', fontsize=18, fontweight='bold')
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("assets/correlation_matrix.png", dpi=300, bbox_inches='tight', facecolor='#0d1117')
plt.show()