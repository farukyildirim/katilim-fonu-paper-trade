import pandas as pd
import numpy as np
import vectorbt as vbt
from pathlib import Path

df_close = pd.read_parquet("cache/bist30_5y_1d.parquet")
df_close.index = pd.to_datetime(df_close.index)

FEES, SLIPPAGE, INIT_CASH = 0.002, 0.002, 100_000
FREQ = "1d"

# ------------------------------------------------------------------------------
# 1. TÜM İNDİKATÖRLER FULL DATA'DA (bir kez)
# ------------------------------------------------------------------------------
rsi = vbt.RSI.run(df_close, 14).rsi

entries_raw = rsi.vbt.crossed_above(30)
exits_raw   = rsi.vbt.crossed_below(60)

# Volatilite filtresi — full data'da
vol = df_close.pct_change().rolling(20).std()
vol_ok = vol < vol.rolling(252).quantile(0.8)

# Top-5 rank — full data'da
rsi_rank = rsi.where(entries_raw).rank(axis=1, method="first")
top5_mask = rsi_rank <= 5

# Filtreleri birleştir
entries = entries_raw & vol_ok & top5_mask
exits   = exits_raw

# ------------------------------------------------------------------------------
# 2. WALK-FORWARD: sadece sinyalleri pencereye dilimle
# ------------------------------------------------------------------------------
def backtest_window(c_df, ent_df, exi_df):
    if len(c_df) < 20:
        return None

    pf = vbt.Portfolio.from_signals(
        c_df, ent_df, exi_df,
        init_cash=INIT_CASH, fees=FEES, slippage=SLIPPAGE, freq=FREQ,
        sl_stop=0.05, cash_sharing=True, group_by=True,
    )
    b = c_df.pct_change().mean(axis=1).fillna(0)
    bench_cum = (1+b).cumprod()
    strat_ret = float(pf.total_return() * 100)
    bench_ret = float((bench_cum.iloc[-1] - 1) * 100)

    try:
        trades = int(pf.trades.count())
    except Exception:
        trades = 0

    return {
        'strat': strat_ret,
        'bench': bench_ret,
        'alpha': strat_ret - bench_ret,
        'sharpe': float(pf.sharpe_ratio()) if pf.sharpe_ratio() is not None else 0.0,
        'dd':    float(pf.max_drawdown() * 100),
        'trades': trades,
    }

TRAIN, TEST, STEP = 252*2, 63, 63
n = len(df_close)
results = []
start = 0
while start + TRAIN + TEST <= n:
    ts, te = start + TRAIN, start + TRAIN + TEST
    r = backtest_window(
        df_close.iloc[ts:te],
        entries.iloc[ts:te],
        exits.iloc[ts:te],
    )
    if r:
        r['start'] = df_close.index[ts].date()
        r['end']   = df_close.index[te-1].date()
        results.append(r)
    start += STEP

df_res = pd.DataFrame(results)

# ------------------------------------------------------------------------------
# 3. RAPOR
# ------------------------------------------------------------------------------
print("=" * 108)
print(f"{'Pencere':<26} {'Strateji':>10} {'Bench':>10} {'Alpha':>10} {'Sharpe':>8} {'DD':>8} {'Isl':>5}")
print("-" * 108)
for _, r in df_res.iterrows():
    print(f"{str(r['start'])+' -> '+str(r['end']):<26} "
          f"%{r['strat']:>8.2f} %{r['bench']:>8.2f} %{r['alpha']:>+8.2f} "
          f"{r['sharpe']:>8.2f} %{r['dd']:>7.2f} {r['trades']:>5}")
print("=" * 108)

n_win = len(df_res)
pos_a = (df_res['alpha'] > 0).sum()
shp1  = (df_res['sharpe'] > 1).sum()
tot_trades = df_res['trades'].sum()

print(f"\nToplam pencere : {n_win}")
print(f"Pozitif alpha  : {pos_a}/{n_win} ({pos_a/n_win*100:.1f}%)")
print(f"Sharpe > 1     : {shp1}/{n_win}")
print(f"Ort. Sharpe    : {df_res['sharpe'].mean():.2f}")
print(f"Ort. DD        : %{df_res['dd'].mean():.2f}")
print(f"Toplam işlem   : {tot_trades}")

print("\nYORUM:")
if tot_trades < 10:
    print("  [ŞÜPHELİ] İşlem sayısı çok az. Sinyal üretimi veya filtreleri kontrol et.")
elif pos_a/n_win >= 0.6 and shp1/n_win >= 0.5:
    print("  [GÜÇLÜ] Tutarlı; paper-trading'e geçilebilir.")
elif pos_a/n_win >= 0.4:
    print("  [KARARSIZ] Bazı pencereler iyi, bazıları kötü.")
else:
    print("  [ZAYIF] Çoğu pencerede benchmark'ın altında.")

# ------------------------------------------------------------------------------
# 4. FULL DATA'DA TEK SEFERDE BACKTEST (referans)
# ------------------------------------------------------------------------------
print("\n--- Referans: Full data tek backtest ---")
pf_full = vbt.Portfolio.from_signals(
    df_close, entries, exits,
    init_cash=INIT_CASH, fees=FEES, slippage=SLIPPAGE, freq=FREQ,
    sl_stop=0.05, cash_sharing=True, group_by=True,
)
print(f"Getiri : %{pf_full.total_return()*100:.2f}")
print(f"Sharpe : {pf_full.sharpe_ratio():.2f}")
print(f"Max DD : %{pf_full.max_drawdown()*100:.2f}")
print(f"İşlem  : {pf_full.trades.count()}")