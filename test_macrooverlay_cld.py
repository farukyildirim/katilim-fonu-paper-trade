"""
YENI EDGE ADAYI: USDTRY volatilite-spike exposure overlay
====================================================================
Hipotez: TL'de kisa vadeli gerceklesen volatilite ani yukselince
(FX stres donemi), BIST hisse secimi stratejisi bagimsiz olarak
maruziyetini dusurmeli -- cunku FX stresi genelde genel bir
risk-off/satis dalgasinin oncusudur.

Disiplin: onceki derste ogrendigimiz gibi TEK bir yeni bilesen
ekleniyor, cekirdek sisteme (cok faktor + esit agirlik, rejim yok,
MR yok) ustune. Marjinal katki AYRI olculuyor.

Veri: USDTRY=X gunluk kapanis, yfinance uzerinden cekilip
cache/usdtry_5y_1d.parquet olarak saklanir (BIST cache'i ile ayni
mantik).
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from pathlib import Path

# ------------------------------------------------------------------
# VERI: BIST30 (mevcut cache) + USDTRY (yeni, cache'lenir)
# ------------------------------------------------------------------
CACHE_FILE = Path("cache/bist30_5y_1d.parquet")
if not CACHE_FILE.exists():
    raise RuntimeError("Once bir kez BIST cache olusturmalisin.")

df_close = pd.read_parquet(CACHE_FILE)
if not isinstance(df_close.index, pd.DatetimeIndex):
    df_close.index = pd.to_datetime(df_close.index)

FX_CACHE = Path("cache/usdtry_5y_1d.parquet")
if FX_CACHE.exists():
    usdtry = pd.read_parquet(FX_CACHE).iloc[:, 0]
    if not isinstance(usdtry.index, pd.DatetimeIndex):
        usdtry.index = pd.to_datetime(usdtry.index)
else:
    import yfinance as yf
    print("USDTRY verisi cekiliyor (ilk seferlik)...")
    fx = yf.download("USDTRY=X", start=df_close.index[0] - pd.Timedelta(days=30),
                      end=df_close.index[-1] + pd.Timedelta(days=1), progress=False)
    if fx.empty:
        raise RuntimeError("USDTRY verisi cekilemedi -- internet baglantisini kontrol et.")
    usdtry = fx["Close"] if "Close" in fx.columns else fx.iloc[:, 0]
    if isinstance(usdtry, pd.DataFrame):
        usdtry = usdtry.iloc[:, 0]
    usdtry = usdtry.squeeze()
    usdtry.name = "USDTRY"
    FX_CACHE.parent.mkdir(exist_ok=True, parents=True)
    usdtry.to_frame().to_parquet(FX_CACHE)
    print(f"Cache'lendi: {FX_CACHE}")

# tz-aware ise tz bilgisini at (BIST cache'i tz-naive)
if usdtry.index.tz is not None:
    usdtry.index = usdtry.index.tz_localize(None)
if df_close.index.tz is not None:
    df_close.index = df_close.index.tz_localize(None)

# BIST takvimine hizala (FX 7 gun islem gorur, hisse 5 gun -> ffill)
usdtry = usdtry.reindex(df_close.index.union(usdtry.index)).sort_index().ffill()
usdtry = usdtry.reindex(df_close.index)

FEES, SLIPPAGE = 0.002, 0.0015
INIT_CASH = 100_000
FREQ = "1d"
MAX_POS = 6
WARMUP = 260
TEST_BARS = 126
STEP_BARS = 21
VOL_WINDOW = 20
Z_WINDOW = 252
Z_THRESHOLD = 1.5      # bu esigin ustunde exposure dusurulur
STRESS_EXPOSURE = 0.3  # stres doneminde maruziyet orani

ret = df_close.pct_change()

# ------------------------------------------------------------------
# CEKIRDEK SISTEM (onceki scriptteki (A) ile ayni)
# ------------------------------------------------------------------
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
raw = (sma_fast > sma_slow) & rs_mask
entries_core = raw & (raw.shift(1) == False)
exits_core = (sma_fast < sma_slow)

# ------------------------------------------------------------------
# USDTRY VOLATILITE-SPIKE OVERLAY
# ------------------------------------------------------------------
fx_ret = usdtry.pct_change()
fx_vol = fx_ret.rolling(VOL_WINDOW).std()
fx_vol_z = (fx_vol - fx_vol.rolling(Z_WINDOW).mean()) / fx_vol.rolling(Z_WINDOW).std()

stress = fx_vol_z > Z_THRESHOLD
n_stress_days = int(stress.sum())
print(f"\nFX stres gunu sayisi (z>{Z_THRESHOLD}): {n_stress_days} / {len(stress)} "
      f"({n_stress_days/len(stress)*100:.1f}%)")

exposure = pd.Series(np.where(stress, STRESS_EXPOSURE, 1.0), index=df_close.index)
exposure_df = pd.DataFrame(
    np.tile(exposure.values.reshape(-1, 1), (1, len(df_close.columns))),
    index=df_close.index, columns=df_close.columns,
)

size_core = equal_w * INIT_CASH
size_overlay = equal_w * exposure_df * INIT_CASH

# ------------------------------------------------------------------
# WALK-FORWARD KARSILASTIRMA
# ------------------------------------------------------------------
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
        rows.append({"alpha": strat_ret - bench_ret, "sharpe": sharpe, "trades": int(pf.trades.count())})
    d = pd.DataFrame(rows)
    n = len(d)
    print(f"\n{label}")
    print("-" * 80)
    print(f"  Pencere sayisi   : {n}")
    print(f"  Ort. alpha       : {d['alpha'].mean():+.2f}%  (pozitif: {(d['alpha']>0).sum()}/{n})")
    print(f"  Ort. sharpe      : {d['sharpe'].mean():.2f}   (>1: {(d['sharpe']>1).sum()}/{n})")
    print(f"  Toplam trade     : {int(d['trades'].sum())}")
    rng = np.random.default_rng(42)
    boot = [rng.choice(d["alpha"].values, size=n, replace=True).mean() for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  Alpha %95 CI     : [{lo:+.2f}%, {hi:+.2f}%]")
    return d

print("\nUSDTRY VOLATILITE-SPIKE OVERLAY: MARJINAL KATKI TESTI")
print("=" * 80)
d_core = evaluate(entries_core, exits_core, size_core, "(A) CEKIRDEK (rejim/overlay yok)")
d_overlay = evaluate(entries_core, exits_core, size_overlay, "(C) CEKIRDEK + USDTRY vol-spike exposure kesme")

print("\n\nSONUC")
print("=" * 80)
a_mean, c_mean = d_core["alpha"].mean(), d_overlay["alpha"].mean()
print(f"  (A) ort.alpha={a_mean:+.2f}%   (C) ort.alpha={c_mean:+.2f}%   fark={c_mean-a_mean:+.2f} puan")
if c_mean > a_mean:
    print("  -> USDTRY vol-spike overlay pozitif marjinal katki sagliyor.")
    print("     Ancak tek basina yeterli mi, DD/sharpe'a etkisini de kontrol edin;")
    print("     esik (Z_THRESHOLD) ve STRESS_EXPOSURE degerlerine asiri duyarli")
    print("     olup olmadigini (parametre robustlugu) ayrica test etmeden")
    print("     sonuca guvenmeyin.")
else:
    print("  -> USDTRY vol-spike overlay katki saglamadi/kotulestirdi.")
    print("     Bu hipotez bu haliyle terk edilmeli; farkli bir esik/pencere")
    print("     denemeden once neden basarisiz oldugunu (belki de FX stresi")
    print("     BIST'ten cok GEC yansiyor, belki de coincident degil lagged")
    print("     olmali) dusunun.")