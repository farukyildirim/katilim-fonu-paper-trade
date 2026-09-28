import time
from pathlib import Path
import numpy as np
import pandas as pd
import vectorbt as vbt

# ------------------------------------------------------------------------------
# CACHE'DEN YÜKLE (bir daha Yahoo'ya gitmiyoruz)
# ------------------------------------------------------------------------------
CACHE_FILE = Path("cache/bist30_5y_1d.parquet")
if not CACHE_FILE.exists():
    raise RuntimeError("Önce bir kez cache oluşturmalısın.")

df_close = pd.read_parquet(CACHE_FILE)
if not isinstance(df_close.index, pd.DatetimeIndex):
    df_close.index = pd.to_datetime(df_close.index)

print(f"Veri yüklendi: {len(df_close)} bar, {len(df_close.columns)} hisse")
print(f"Aralık: {df_close.index[0].date()} -> {df_close.index[-1].date()}")

FEES     = 0.002
INIT_CASH = 100_000
FREQ     = "1d"

# ------------------------------------------------------------------------------
# RS SKORU (tek seferlik)
# ------------------------------------------------------------------------------
perf_3m = df_close.pct_change(63)
perf_6m = df_close.pct_change(126)
composite_rs = perf_3m * 0.6 + perf_6m * 0.4

monthly_mask = df_close.index.to_series().dt.month.diff() != 0
monthly_mask.iloc[0] = True

def build_rs_mask(max_pos):
    mask = pd.DataFrame(False, index=df_close.index, columns=df_close.columns)
    current_top = []
    for idx in df_close.index:
        if monthly_mask.loc[idx]:
            row = composite_rs.loc[idx].dropna()
            if len(row) >= max_pos:
                current_top = row.nlargest(max_pos).index.tolist()
        if current_top:
            mask.loc[idx, current_top] = True
    return mask

# ------------------------------------------------------------------------------
# BACKTEST
# ------------------------------------------------------------------------------
def backtest_window(c_df, rs_msk, sl_trail, slippage, max_pos):
    if len(c_df) < 60:
        return None

    sma_fast = c_df.rolling(20).mean()
    sma_slow = c_df.rolling(50).mean()
    raw_entries = (sma_fast > sma_slow) & rs_msk
    entries = raw_entries & (raw_entries.shift(1) == False)
    exits = (sma_fast < sma_slow)

    pos_size = INIT_CASH / max_pos * 0.98

    pf = vbt.Portfolio.from_signals(
        c_df, entries, exits,
        size=pos_size, size_type='value',
        sl_stop=None, sl_trail=sl_trail,
        init_cash=INIT_CASH,
        fees=FEES, slippage=slippage, freq=FREQ,
        cash_sharing=True, group_by=True,
    )

    b = c_df.pct_change().mean(axis=1).fillna(0)
    bench_cum = (1 + b).cumprod()
    bench_ret = (bench_cum.iloc[-1] - 1) * 100
    strat_ret = pf.total_return() * 100

    return {
        "strat_ret": float(strat_ret),
        "bench_ret": float(bench_ret),
        "alpha":     float(strat_ret - bench_ret),
        "sharpe":    float(pf.sharpe_ratio()),
        "max_dd":    float(pf.max_drawdown() * 100),
        "trades":    int(pf.trades.count()) if hasattr(pf.trades, 'count') else 0,
    }

def walk_forward(df, rs_mask, train_bars, test_bars, step_bars,
                 sl_trail, slippage, max_pos):
    results = []
    n = len(df)
    start = 0
    while start + train_bars + test_bars <= n:
        ts = start + train_bars
        te = ts + test_bars
        res = backtest_window(
            df.iloc[ts:te], rs_mask.iloc[ts:te],
            sl_trail, slippage, max_pos,
        )
        if res is not None:
            res["start"] = df.index[ts].date()
            res["end"]   = df.index[te-1].date()
            results.append(res)
        start += step_bars
    return pd.DataFrame(results)

# ------------------------------------------------------------------------------
# TEST MATRİSİ
# ------------------------------------------------------------------------------
configs = [
    # (etiket, train, test, step, sl_trail, slippage, max_pos)
    ("Baseline (3y/6a, sl0.15, slip0.001, 5p)", 252*3, 126, 126, 0.15, 0.001, 5),
    ("A) Kısa pencere (2y/3a)",                 252*2, 63,  63,  0.15, 0.001, 5),
    ("B) Yüksek slipaj (%0.3)",                 252*3, 126, 126, 0.15, 0.003, 5),
    ("C) 8 pozisyon",                           252*3, 126, 126, 0.15, 0.001, 8),
    ("D) Hepsi (kısa + slip + 8p)",             252*2, 63,  63,  0.15, 0.003, 8),
    ("E) Daha sıkı trailing (%10)",             252*3, 126, 126, 0.10, 0.001, 5),
    ("F) Daha gevşek trailing (%20)",           252*3, 126, 126, 0.20, 0.001, 5),
]

print("\n" + "=" * 118)
print(f"{'Konfigürasyon':<42} {'Pencere':>8} {'Ort.α':>9} {'Pozα':>7} {'Ort.Shp':>9} {'Shp>1':>7} {'Ort.DD':>9}")
print("-" * 118)

rows = []
for label, tr, te, st, sl, slip, mp in configs:
    rs_mask = build_rs_mask(mp)
    df_res = walk_forward(df_close, rs_mask, tr, te, st, sl, slip, mp)
    if df_res.empty:
        print(f"{label:<42} {'YOK':>8}")
        continue
    pos_a = (df_res['alpha'] > 0).sum()
    shp_gt1 = (df_res['sharpe'] > 1).sum()
    n = len(df_res)
    print(f"{label:<42} "
          f"{n:>8} "
          f"%{df_res['alpha'].mean():>+7.2f} "
          f"{pos_a:>3}/{n:<3} "
          f"{df_res['sharpe'].mean():>9.2f} "
          f"{shp_gt1:>3}/{n:<3} "
          f"%{df_res['max_dd'].mean():>7.2f}")
    rows.append({
        "config":   label,
        "windows":  n,
        "alpha":    df_res['alpha'].mean(),
        "pos_alpha_ratio": pos_a / n,
        "sharpe":   df_res['sharpe'].mean(),
        "sharpe_ratio": shp_gt1 / n,
        "dd":       df_res['max_dd'].mean(),
    })

print("=" * 118)

# ------------------------------------------------------------------------------
# KARAR TABLOSU
# ------------------------------------------------------------------------------
print("\nROBUSTLUK DEĞERLENDİRMESİ")
print("=" * 118)
df_summary = pd.DataFrame(rows)
df_summary["robust_score"] = (
    df_summary["pos_alpha_ratio"] * 40 +
    df_summary["sharpe_ratio"] * 40 +
    (df_summary["sharpe"].clip(0, 3) / 3) * 20
)
df_summary = df_summary.sort_values("robust_score", ascending=False)

for _, r in df_summary.iterrows():
    flag = "✅" if r["robust_score"] >= 70 else ("⚠️" if r["robust_score"] >= 50 else "❌")
    print(f"{flag} {r['config']:<42} "
          f"skor={r['robust_score']:>5.1f}  "
          f"poz_alpha={r['pos_alpha_ratio']*100:>5.1f}%  "
          f"sharpe>1={r['sharpe_ratio']*100:>5.1f}%")

print("\nYORUM:")
top = df_summary.iloc[0]
if top["robust_score"] >= 70 and top["pos_alpha_ratio"] >= 0.6:
    print("  ✅ Konfigürasyon canlı öncesi doğrulamayı geçti.")
    print("     -> Paper-trading'e geçilebilir.")
elif top["robust_score"] >= 50:
    print("  ⚠️ Kararsız; parametre ayarlarına duyarlı.")
    print("     -> Daha güvenli tarafta kal, daha uzun veriyle test et.")
else:
    print("  ❌ Hiçbir konfigürasyon yeterince sağlam değil.")
    print("     -> Farklı faktör dene (mean-reversion, düşük vol).")