"""
FINAL KARSILASTIRMA: Sade sistem vs Hysteresis-rejimli sistem
====================================================================
Onceki diagnostikten cikan sonuc:
  - MR sleeve tek basina -17.66% alpha -> SISTEMDEN TAMAMEN CIKARILDI.
  - Ham rejim filtresi (SMA200 crossover) whipsaw uretiyor (28 blogun
    14'u < 5 gun) ve tek pozitif bileseni (cok faktorlu skor) bozuyor.
  - En iyi tek konfigurasyon: cok faktorlu skor + esit agirlik,
    rejim YOK, MR YOK (+1.26% fee-free, +0.04% gercek alpha).

Bu script iki adayi karsilastirir:
  (A) CEKIRDEK  : cok faktorlu skor + esit agirlik, rejim yok.
  (B) CEKIRDEK+H: ayni sistem + HYSTERESIS'li rejim filtresi
                  (rejim degisimi ancak N GUN USTUSTE teyit edilirse
                  uygulanir -> whipsaw'i azaltmayi hedefler).

Ikisi de ayni ~14-20 pencerelik walk-forward'da, ayni fee/slippage
ile test edilir. Hangisi net olarak daha iyiyse (alpha + sharpe +
whipsaw/trade sayisi birlikte degerlendirilerek) o aday tutulur.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from pathlib import Path

CACHE_FILE = Path("cache/bist30_5y_1d.parquet")
if not CACHE_FILE.exists():
    raise RuntimeError("Once bir kez cache olusturmalisin.")

df_close = pd.read_parquet(CACHE_FILE)
if not isinstance(df_close.index, pd.DatetimeIndex):
    df_close.index = pd.to_datetime(df_close.index)

FEES, SLIPPAGE = 0.002, 0.0015
INIT_CASH = 100_000
FREQ = "1d"
MAX_POS = 6
WARMUP = 260
TEST_BARS = 126
STEP_BARS = 21          # onceki 63'ten kuculttuk -> daha fazla pencere
CONFIRM_DAYS = 10        # hysteresis: rejim degisimi icin gereken ardisik gun

ret = df_close.pct_change()

# ---------- Cok faktorlu skor ----------
mom_3m, mom_6m, mom_12m = df_close.pct_change(63), df_close.pct_change(126), df_close.pct_change(252)
momentum = mom_3m * 0.5 + mom_6m * 0.3 + mom_12m * 0.2
vol_63 = ret.rolling(63).std()
composite = momentum.rank(axis=1, pct=True) * 0.7 + (-vol_63).rank(axis=1, pct=True) * 0.3

monthly_mask = df_close.index.to_series().dt.month.diff() != 0
monthly_mask.iloc[0] = True

def build_selection():
    mask = pd.DataFrame(False, index=df_close.index, columns=df_close.columns)
    current_top = []
    for idx in df_close.index:
        if monthly_mask.loc[idx]:
            row = composite.loc[idx].dropna()
            if len(row) >= MAX_POS:
                current_top = row.nlargest(MAX_POS).index.tolist()
        if current_top:
            mask.loc[idx, current_top] = True
    return mask

rs_mask = build_selection()
equal_w = rs_mask.astype(float).div(rs_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

sma_fast = df_close.rolling(20).mean()
sma_slow = df_close.rolling(50).mean()

# ---------- Ham rejim + hysteresis'li rejim ----------
eq_weight_index = (1 + ret.mean(axis=1)).cumprod()
regime_sma = eq_weight_index.rolling(200).mean()
raw_regime = eq_weight_index > regime_sma

def apply_hysteresis(raw, confirm_days):
    confirmed = raw.copy()
    state = raw.iloc[0]
    run_len = 0
    prev_raw = raw.iloc[0]
    out = []
    for val in raw:
        if val == prev_raw:
            run_len += 1
        else:
            run_len = 1
        prev_raw = val
        if run_len >= confirm_days:
            state = val
        out.append(state)
    return pd.Series(out, index=raw.index)

hyst_regime = apply_hysteresis(raw_regime, CONFIRM_DAYS)

def to_df(series):
    return pd.DataFrame(
        np.tile(series.values.reshape(-1, 1), (1, len(df_close.columns))),
        index=df_close.index, columns=df_close.columns,
    )

raw_regime_df = to_df(raw_regime)
hyst_regime_df = to_df(hyst_regime)

def regime_stats(series, label):
    blocks = (series != series.shift()).cumsum()
    lens = series.groupby(blocks).size()
    n_short = (lens < 5).sum()
    print(f"{label:<25} degisim={int((series.astype(int).diff().abs().fillna(0)).sum()):>4}  "
          f"ort.sure={lens.mean():>6.1f}g  kisa(<5g) blok={n_short}/{len(lens)}")

print("REJIM WHIPSAW KARSILASTIRMASI")
print("=" * 80)
regime_stats(raw_regime, "Ham rejim (SMA200)")
regime_stats(hyst_regime, f"Hysteresis ({CONFIRM_DAYS}g teyit)")

# ---------- Sinyal insasi ----------
def build_entries_exits(regime_df=None):
    raw = (sma_fast > sma_slow) & rs_mask
    if regime_df is not None:
        raw = raw & regime_df
    entries = raw & (raw.shift(1) == False)
    exits = (sma_fast < sma_slow)
    if regime_df is not None:
        exits = exits | (~regime_df)
    return entries, exits

entries_core, exits_core = build_entries_exits(regime_df=None)
entries_hyst, exits_hyst = build_entries_exits(regime_df=hyst_regime_df)
size_all = equal_w * INIT_CASH

# ---------- Walk-forward degerlendirme ----------
def run_pf(entries, exits, size, ts, te):
    e = entries.iloc[ts:te]
    if e.values.sum() == 0:
        return None
    return vbt.Portfolio.from_signals(
        df_close.iloc[ts:te], e, exits.iloc[ts:te],
        size=size.iloc[ts:te], size_type='value',
        init_cash=INIT_CASH, fees=FEES, slippage=SLIPPAGE,
        freq=FREQ, cash_sharing=True, group_by=True,
    )

def walk_windows():
    ws, start = [], WARMUP
    while start + TEST_BARS <= len(df_close):
        ws.append((start, start + TEST_BARS))
        start += STEP_BARS
    return ws

WINDOWS = walk_windows()

def evaluate(entries, exits, size, label):
    rows = []
    for ts, te in WINDOWS:
        pf = run_pf(entries, exits, size, ts, te)
        if pf is None:
            continue
        val = pf.value()
        strat_ret = (val.iloc[-1] / val.iloc[0] - 1) * 100
        bench = ret.iloc[ts:te].mean(axis=1).fillna(0)
        bench_ret = ((1 + bench).cumprod().iloc[-1] - 1) * 100
        rets = val.pct_change().dropna()
        sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0.0
        rows.append({
            "start": df_close.index[ts].date(), "end": df_close.index[te - 1].date(),
            "alpha": strat_ret - bench_ret, "sharpe": sharpe,
            "trades": int(pf.trades.count()),
        })
    d = pd.DataFrame(rows)
    n = len(d)
    print(f"\n{label}")
    print("-" * 80)
    print(f"  Pencere sayisi     : {n}")
    print(f"  Ort. alpha         : {d['alpha'].mean():+.2f}%  (pozitif: {(d['alpha']>0).sum()}/{n})")
    print(f"  Ort. sharpe        : {d['sharpe'].mean():.2f}   (>1: {(d['sharpe']>1).sum()}/{n})")
    print(f"  Toplam trade       : {int(d['trades'].sum())}")
    rng = np.random.default_rng(42)
    boot = [rng.choice(d["alpha"].values, size=n, replace=True).mean() for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  Alpha %95 CI       : [{lo:+.2f}%, {hi:+.2f}%]")
    return d

print("\n\nA/B KARSILASTIRMA (STEP={} gun, TEST={} gun, {} pencere)".format(STEP_BARS, TEST_BARS, len(WINDOWS)))
print("=" * 80)
d_core = evaluate(entries_core, exits_core, size_all, "(A) CEKIRDEK: cok faktor + esit agirlik, rejim YOK")
d_hyst = evaluate(entries_hyst, exits_hyst, size_all, f"(B) CEKIRDEK + hysteresis-rejim ({CONFIRM_DAYS}g teyit)")

print("\n\nSONUC")
print("=" * 80)
a_mean, h_mean = d_core["alpha"].mean(), d_hyst["alpha"].mean()
if h_mean > a_mean and (d_hyst["sharpe"] > 1).mean() >= (d_core["sharpe"] > 1).mean():
    print("  (B) hysteresis-rejim hem alpha'da hem sharpe'da (A)'yi geciyor -> tutulabilir aday.")
elif a_mean > h_mean:
    print("  (A) cekirdek sistem rejim filtresi olmadan daha iyi -> rejim fikri bu haliyle terk edilmeli.")
else:
    print("  Sonuclar karisik; tek basina bu kiyas karar vermeye yetmiyor, daha uzun veri gerekiyor.")
print(f"  (A) ort.alpha={a_mean:+.2f}%   (B) ort.alpha={h_mean:+.2f}%")