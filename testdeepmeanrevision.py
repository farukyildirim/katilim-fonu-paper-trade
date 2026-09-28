import pandas as pd
import numpy as np
import vectorbt as vbt
from pathlib import Path

# ------------------------------------------------------------------------------
# VERİ
# ------------------------------------------------------------------------------
df_close = pd.read_parquet("cache/bist30_5y_1d.parquet")
df_close.index = pd.to_datetime(df_close.index)

# XU100 (piyasa filtresi için)
try:
    xu100 = pd.read_parquet("cache/xu100_5y_1d.parquet").iloc[:, 0]
except FileNotFoundError:
    import yfinance as yf
    xu100 = yf.Ticker("XU100.IS").history(period="5y", interval="1d")["Close"]
    xu100 = xu100.reindex(df_close.index).ffill()
    xu100.to_frame("XU100").to_parquet("cache/xu100_5y_1d.parquet")

# ------------------------------------------------------------------------------
# ORTAK PARAMETRELER
# ------------------------------------------------------------------------------
FEES      = 0.002
SLIPPAGE  = 0.002
INIT_CASH = 100_000
FREQ      = "1d"

# ------------------------------------------------------------------------------
# BACKTEST FONKSİYONU
# ------------------------------------------------------------------------------
def run(rsi_low, rsi_exit, use_mkt, vol_filter, sl_stop, max_pos, label):
    rsi = vbt.RSI.run(df_close, 14).rsi

    entries = rsi.vbt.crossed_above(rsi_low)
    exits   = rsi.vbt.crossed_below(rsi_exit)

    # KATMAN 1: Piyasa filtresi (XU100 > SMA200)
    if use_mkt:
        mkt_ok = (xu100 > xu100.rolling(200).mean())
        mkt_ok = pd.DataFrame({c: mkt_ok for c in df_close.columns},
                              index=df_close.index)
        entries = entries & mkt_ok

    # KATMAN 2: Volatilite filtresi (aşırı oynak hisseleri alma)
    if vol_filter:
        vol = df_close.pct_change().rolling(20).std()
        vol_ok = vol < vol.rolling(252).quantile(0.8)
        entries = entries & vol_ok

    # KATMAN 3: Aynı anda maksimum N pozisyon
    # (basit yaklaşım: sinyalleri rank'la, en güçlü N'i al)
    if max_pos is not None:
        # Aynı gün birden fazla sinyal varsa RSI'n en düşük olanı seç
        rsi_rank = rsi.where(entries).rank(axis=1, method="first")
        entries = entries & (rsi_rank <= max_pos)

    pf = vbt.Portfolio.from_signals(
        df_close, entries, exits,
        init_cash=INIT_CASH,
        fees=FEES, slippage=SLIPPAGE, freq=FREQ,
        sl_stop=sl_stop,
        cash_sharing=True, group_by=True,
    )

    ret = pf.total_return() * 100
    sharpe = pf.sharpe_ratio()
    dd = pf.max_drawdown() * 100
    calmar = (ret / abs(dd)) if dd != 0 else float('nan')

    print(f"{label:<45} "
          f"getiri=%{ret:>7.2f}  "
          f"sharpe={sharpe:>5.2f}  "
          f"DD=%{dd:>7.2f}  "
          f"calmar={calmar:>5.2f}")
    return {"label": label, "ret": ret, "sharpe": sharpe, "dd": dd, "calmar": calmar}


# ------------------------------------------------------------------------------
# KADEMELİ TEST
# ------------------------------------------------------------------------------
print("=" * 100)
print(f"{'Konfigürasyon':<45} {'Getiri':>14} {'Sharpe':>11} {'Max DD':>11} {'Calmar':>10}")
print("-" * 100)

# Referans: orijinal
run(30, 60, use_mkt=False, vol_filter=False, sl_stop=0.05, max_pos=None,
    label="0) Orijinal (referans)")

# Katman 1: Piyasa filtresi
run(30, 60, use_mkt=True,  vol_filter=False, sl_stop=0.05, max_pos=None,
    label="1) + XU100 > SMA200 filtresi")

# Katman 2: + volatilite filtresi
run(30, 60, use_mkt=True,  vol_filter=True,  sl_stop=0.05, max_pos=None,
    label="2) + volatilite filtresi")

# Katman 3: + maksimum pozisyon limiti
run(30, 60, use_mkt=True,  vol_filter=True,  sl_stop=0.05, max_pos=5,
    label="3) + max 5 pozisyon")

# Katman 4: + daha sıkı stop
run(30, 60, use_mkt=True,  vol_filter=True,  sl_stop=0.03, max_pos=5,
    label="4) + %3 sıkı stop")

# Alternatif RSI parametreleri
run(25, 55, use_mkt=True,  vol_filter=True,  sl_stop=0.05, max_pos=5,
    label="5) RSI 25/55 + tüm filtreler")
run(20, 50, use_mkt=True,  vol_filter=True,  sl_stop=0.05, max_pos=5,
    label="6) RSI 20/50 + tüm filtreler")

print("=" * 100)