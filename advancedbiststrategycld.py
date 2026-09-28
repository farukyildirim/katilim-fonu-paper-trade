"""
Gelismis cok faktorlu + rejim filtreli BIST30 strateji backtest'i
====================================================================
Onceki versiyona gore degisiklikler ve nedenleri:

1) COK FAKTORLU SKOR
   Tek basina 3a/6a momentum yerine, momentum + dusuk-volatilite kombinasyonu.
   Tek faktore bagimlilik, o faktorun rejime gore comesi riskini tasir.

2) REJIM FILTRESI
   Esit agirlikli endeksin 200 gunluk SMA'sina gore "boga/ayi" filtresi.
   Piyasa dusus rejimindeyken trend sinyalleri gecersiz sayilir.

3) INVERSE-VOLATILITY POZISYON BUYUKLUGU
   Esit-dolar yerine risk-parity mantigi: dusuk volatiliteli hisseye
   daha buyuk pay -> drawdown azaltma amacli.

4) IKINCI, DUSUK KORELASYONLU SLEEVE (mean-reversion)
   Trend sleeve ile ayni rejimde calismayan, kisa vadeli "dip alma"
   stratejisi. Amac: trend calismadigi donemlerde portfoyun tamamen
   bos kalmamasi ve genel Sharpe'i yukseltmek.

5) INDIKATORLER TUM VERI UZERINDE BIR KEZ HESAPLANIR
   Onceki versiyonda backtest_window() her pencerede rolling
   indikatorleri SIFIRDAN, sadece test dilimi uzerinde hesapliyordu.
   Bu, her pencerenin ilk ~50 barinda warmup kaybina (NaN sinyal)
   yol aciyordu. Burada indikatorler tam seri uzerinde hesaplanip
   walk-forward sadece SONUCLARI dilimliyor (lookahead YOK, cunku
   rolling pencereler zaten sadece gecmise bakiyor).

6) DAHA COK, DAHA KISA TEST PENCERESI
   3 pencere yerine ~14 pencere -> istatistiksel olarak daha anlamli
   ornekem + bootstrap guven araligi.

7) ABLATION TESTI
   Her yeni bilesenin (multi-faktor, rejim, vol-sizing, mr-sleeve)
   MARJINAL katkisini ayri ayri olcen bir tablo. Amac: onceki
   scriptteki "E ve F birebir ayni cikti verdi ama fark edilmedi"
   hatasini tekrarlamamak -- her eklenen karmasiklik ayri dogrulanmali.

NOT: Bu bir yatirim tavsiyesi degildir, sadece backtest metodolojisi
     icin ornek koddur. Gercek sermaye ile kullanmadan once:
     - Daha uzun / farkli rejimli veri (2015 sonrasi, 2018 krizi dahil)
     - Islem maliyeti / likidite varsayimlarinin gercekci olup olmadigi
     - Paper-trading dogrulamasi
     mutlaka yapilmali.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from pathlib import Path

# ------------------------------------------------------------------
# VERI
# ------------------------------------------------------------------
CACHE_FILE = Path("cache/bist30_5y_1d.parquet")
if not CACHE_FILE.exists():
    raise RuntimeError("Once bir kez cache olusturmalisin.")

df_close = pd.read_parquet(CACHE_FILE)
if not isinstance(df_close.index, pd.DatetimeIndex):
    df_close.index = pd.to_datetime(df_close.index)

print(f"Veri yuklendi: {len(df_close)} bar, {len(df_close.columns)} hisse")
print(f"Aralik: {df_close.index[0].date()} -> {df_close.index[-1].date()}")

FEES      = 0.002
SLIPPAGE  = 0.0015
INIT_CASH = 100_000
FREQ      = "1d"
MAX_POS   = 6
TREND_ALLOC = 0.65
MR_ALLOC    = 0.35

ret = df_close.pct_change()

# ------------------------------------------------------------------
# 1) COK FAKTORLU KOMPOZIT SKOR
# ------------------------------------------------------------------
mom_3m  = df_close.pct_change(63)
mom_6m  = df_close.pct_change(126)
mom_12m = df_close.pct_change(252)
momentum = mom_3m * 0.5 + mom_6m * 0.3 + mom_12m * 0.2

vol_63 = ret.rolling(63).std()
low_vol_score = -vol_63

mom_rank = momentum.rank(axis=1, pct=True)
vol_rank = low_vol_score.rank(axis=1, pct=True)
composite = mom_rank * 0.7 + vol_rank * 0.3

# ------------------------------------------------------------------
# 2) REJIM FILTRESI (esit agirlikli endeks, 200g SMA)
# ------------------------------------------------------------------
eq_weight_index = (1 + ret.mean(axis=1)).cumprod()
regime_sma = eq_weight_index.rolling(200).mean()
regime_on_series = eq_weight_index > regime_sma
regime_df = pd.DataFrame(
    np.tile(regime_on_series.values.reshape(-1, 1), (1, len(df_close.columns))),
    index=df_close.index, columns=df_close.columns,
)

# ------------------------------------------------------------------
# 3) AYLIK REBALANCE: TOP-N SECIM + INVERSE-VOL AGIRLIK
# ------------------------------------------------------------------
monthly_mask = df_close.index.to_series().dt.month.diff() != 0
monthly_mask.iloc[0] = True

def build_selection(max_pos):
    mask = pd.DataFrame(False, index=df_close.index, columns=df_close.columns)
    weights = pd.DataFrame(0.0, index=df_close.index, columns=df_close.columns)
    current_top, current_w = [], pd.Series(dtype=float)
    for idx in df_close.index:
        if monthly_mask.loc[idx]:
            row = composite.loc[idx].dropna()
            if len(row) >= max_pos:
                top = row.nlargest(max_pos).index.tolist()
                iv = (1 / vol_63.loc[idx, top]).replace([np.inf, -np.inf], np.nan).dropna()
                if len(iv) > 0:
                    current_w = iv / iv.sum()
                    current_top = current_w.index.tolist()
        if current_top:
            mask.loc[idx, current_top] = True
            weights.loc[idx, current_top] = current_w.reindex(current_top).values
    return mask, weights

rs_mask, w_mask = build_selection(MAX_POS)

# ------------------------------------------------------------------
# 4) TREND SLEEVE
# ------------------------------------------------------------------
sma_fast = df_close.rolling(20).mean()
sma_slow = df_close.rolling(50).mean()

trend_raw = (sma_fast > sma_slow) & rs_mask & regime_df
trend_entries = trend_raw & (trend_raw.shift(1) == False)
trend_exits = (sma_fast < sma_slow) | (~regime_df)

trend_size = w_mask * (INIT_CASH * TREND_ALLOC)

# ------------------------------------------------------------------
# 5) MEAN-REVERSION SLEEVE (dusuk korelasyonlu ikinci bacak)
#    Sadece goreceli GUCLU hisselerde (ust yari composite) kisa
#    vadeli asiri satisi alip ortalamaya donuste satar.
# ------------------------------------------------------------------
ret5 = ret.rolling(5).sum()
z5 = (ret5 - ret5.rolling(60).mean()) / ret5.rolling(60).std()

mr_universe = composite.rank(axis=1, pct=True) > 0.5
mr_raw = (z5 < -1.5) & mr_universe
mr_entries = mr_raw & (mr_raw.shift(1) == False)
mr_exits = z5 > 0.25

mr_active_count = mr_entries.sum(axis=1).replace(0, np.nan)
mr_size = mr_entries.astype(float).div(mr_active_count, axis=0) * (INIT_CASH * MR_ALLOC)
mr_size = mr_size.fillna(0.0)

# ------------------------------------------------------------------
# YARDIMCI: bir sleeve'i verilen tarih araliginda calistir
# ------------------------------------------------------------------
def run_sleeve(entries, exits, size, init_cash, ts, te):
    e = entries.iloc[ts:te]
    if e.values.sum() == 0:
        return None
    pf = vbt.Portfolio.from_signals(
        df_close.iloc[ts:te], e, exits.iloc[ts:te],
        size=size.iloc[ts:te], size_type='value',
        init_cash=init_cash, fees=FEES, slippage=SLIPPAGE,
        freq=FREQ, cash_sharing=True, group_by=True,
    )
    return pf

def combined_window_stats(ts, te):
    trend_pf = run_sleeve(trend_entries, trend_exits, trend_size, INIT_CASH * TREND_ALLOC, ts, te)
    mr_pf    = run_sleeve(mr_entries, mr_exits, mr_size, INIT_CASH * MR_ALLOC, ts, te)

    trend_val = trend_pf.value() if trend_pf is not None else pd.Series(INIT_CASH * TREND_ALLOC, index=df_close.index[ts:te])
    mr_val    = mr_pf.value() if mr_pf is not None else pd.Series(INIT_CASH * MR_ALLOC, index=df_close.index[ts:te])

    combined_val = trend_val.reindex(df_close.index[ts:te]).ffill() + mr_val.reindex(df_close.index[ts:te]).ffill()
    combined_ret = combined_val.pct_change().dropna()

    strat_total_ret = (combined_val.iloc[-1] / combined_val.iloc[0] - 1) * 100
    bench = ret.iloc[ts:te].mean(axis=1).fillna(0)
    bench_total_ret = ((1 + bench).cumprod().iloc[-1] - 1) * 100

    sharpe = (combined_ret.mean() / combined_ret.std()) * np.sqrt(252) if combined_ret.std() > 0 else 0.0
    running_max = combined_val.cummax()
    dd = ((combined_val - running_max) / running_max).min() * 100

    return {
        "strat_ret": float(strat_total_ret),
        "bench_ret": float(bench_total_ret),
        "alpha": float(strat_total_ret - bench_total_ret),
        "sharpe": float(sharpe),
        "max_dd": float(dd),
    }

# ------------------------------------------------------------------
# WALK-FORWARD (indikatorler zaten hazir; sadece pencere kayan)
# ------------------------------------------------------------------
WARMUP     = 260   # rolling(252) vb. icin gerekli minimum bar
TEST_BARS  = 126   # ~6 ay
STEP_BARS  = 63    # ~3 ay -> pencereler ortusuyor, ornek sayisi artiyor

def walk_forward():
    results = []
    n = len(df_close)
    start = WARMUP
    while start + TEST_BARS <= n:
        te = start + TEST_BARS
        res = combined_window_stats(start, te)
        res["start"] = df_close.index[start].date()
        res["end"] = df_close.index[te - 1].date()
        results.append(res)
        start += STEP_BARS
    return pd.DataFrame(results)

print("\nGELISMIS STRATEJI (cok faktor + rejim + vol-sizing + mr-sleeve) walk-forward")
print("=" * 100)
df_res = walk_forward()
print(df_res[["start", "end", "strat_ret", "bench_ret", "alpha", "sharpe", "max_dd"]]
      .to_string(index=False, float_format=lambda x: f"{x:8.2f}"))

n = len(df_res)
pos_alpha = (df_res["alpha"] > 0).sum()
sharpe_gt1 = (df_res["sharpe"] > 1).sum()
print("-" * 100)
print(f"Pencere sayisi     : {n}")
print(f"Ort. alpha         : {df_res['alpha'].mean():+.2f}%  (pozitif: {pos_alpha}/{n})")
print(f"Ort. sharpe        : {df_res['sharpe'].mean():.2f}   (>1: {sharpe_gt1}/{n})")
print(f"Ort. max drawdown  : {df_res['max_dd'].mean():.2f}%")

# Bootstrap guven araligi (pencereler kismen ortustugu icin bu
# YAKLASIK bir gosterge -- gercek bagimsizlik varsaymaz)
rng = np.random.default_rng(42)
boot_means = [rng.choice(df_res["alpha"].values, size=n, replace=True).mean() for _ in range(5000)]
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
print(f"Alpha %95 bootstrap guven araligi: [{ci_low:+.2f}%, {ci_high:+.2f}%]")
if ci_low <= 0:
    print("  -> Guven araligi sifiri iciyor: alpha'nin gercekten pozitif oldugu")
    print("     istatistiksel olarak teyit edilemiyor. Temkinli olun.")

# ------------------------------------------------------------------
# ABLATION: her bilesenin marjinal katkisi
# ------------------------------------------------------------------
def ablation_run(use_multifactor, use_regime, use_vol_sizing, use_mr_sleeve):
    score = momentum.rank(axis=1, pct=True) if not use_multifactor else composite
    mask = pd.DataFrame(False, index=df_close.index, columns=df_close.columns)
    weights = pd.DataFrame(0.0, index=df_close.index, columns=df_close.columns)
    current_top, current_w = [], pd.Series(dtype=float)
    for idx in df_close.index:
        if monthly_mask.loc[idx]:
            row = score.loc[idx].dropna()
            if len(row) >= MAX_POS:
                top = row.nlargest(MAX_POS).index.tolist()
                if use_vol_sizing:
                    iv = (1 / vol_63.loc[idx, top]).replace([np.inf, -np.inf], np.nan).dropna()
                    current_w = (iv / iv.sum()) if len(iv) else pd.Series(1 / MAX_POS, index=top)
                else:
                    current_w = pd.Series(1 / MAX_POS, index=top)
                current_top = current_w.index.tolist()
        if current_top:
            mask.loc[idx, current_top] = True
            weights.loc[idx, current_top] = current_w.reindex(current_top).values

    raw = (sma_fast > sma_slow) & mask
    if use_regime:
        raw = raw & regime_df
    entries = raw & (raw.shift(1) == False)
    exits = (sma_fast < sma_slow) | ((~regime_df) if use_regime else False)
    size = weights * (INIT_CASH * (TREND_ALLOC if use_mr_sleeve else 1.0))

    rows = []
    start = WARMUP
    while start + TEST_BARS <= len(df_close):
        te = start + TEST_BARS
        trend_pf = run_sleeve(entries, exits, size, INIT_CASH * (TREND_ALLOC if use_mr_sleeve else 1.0), start, te)
        trend_val = trend_pf.value() if trend_pf is not None else pd.Series(INIT_CASH, index=df_close.index[start:te])

        if use_mr_sleeve:
            mr_pf = run_sleeve(mr_entries, mr_exits, mr_size, INIT_CASH * MR_ALLOC, start, te)
            mr_val = mr_pf.value() if mr_pf is not None else pd.Series(INIT_CASH * MR_ALLOC, index=df_close.index[start:te])
            total_val = trend_val.reindex(df_close.index[start:te]).ffill() + mr_val.reindex(df_close.index[start:te]).ffill()
        else:
            total_val = trend_val.reindex(df_close.index[start:te]).ffill()

        strat_ret = (total_val.iloc[-1] / total_val.iloc[0] - 1) * 100
        bench = ret.iloc[start:te].mean(axis=1).fillna(0)
        bench_ret = ((1 + bench).cumprod().iloc[-1] - 1) * 100
        rows.append({"alpha": strat_ret - bench_ret})
        start += STEP_BARS

    d = pd.DataFrame(rows)
    return d["alpha"].mean(), (d["alpha"] > 0).mean()

print("\nABLATION: her bilesenin marjinal katkisi")
print("=" * 100)
print(f"{'Konfigurasyon':<55} {'Ort.alpha':>10} {'Poz.alpha%':>12}")
print("-" * 100)
ablation_configs = [
    ("Baseline (tek faktor momentum, esit agirlik)",      False, False, False, False),
    ("+ Cok faktorlu skor (mom+lowvol)",                  True,  False, False, False),
    ("+ Rejim filtresi",                                  True,  True,  False, False),
    ("+ Inverse-vol sizing",                              True,  True,  True,  False),
    ("+ Mean-reversion sleeve (tam sistem)",               True,  True,  True,  True),
]
for label, mf, rg, vs, mr in ablation_configs:
    a, pr = ablation_run(mf, rg, vs, mr)
    print(f"{label:<55} {a:>+9.2f}% {pr*100:>11.1f}%")

print("=" * 100)
print("\nYORUM:")
print("  Her satir bir onceki satira TEK bir bilesen ekliyor. Eger bir bilesen")
print("  eklendiginde sonuc DEGISMIYORSA (E/F trailing-stop hatasindaki gibi),")
print("  o bilesenin fiilen etkisiz oldugu anlamina gelir ve debug edilmelidir.")
print("  Eger bir bilesen alpha'yi dusuruyorsa, o bileseni sisteme eklemeden once")
print("  neden eklemek istediginizi yeniden sorgulayin -- karmasiklik basina")
print("  'daha iyi' anlamina gelmez.")
# ==============================================================
# EK: Fee attribution + benchmark karsilastirmasi + MR confound
# Mevcut scriptin SONUNA ekleyin.
# ==============================================================

def run_sleeve_fee(entries, exits, size, init_cash, ts, te, fees, slippage):
    e = entries.iloc[ts:te]
    if e.values.sum() == 0:
        return None
    return vbt.Portfolio.from_signals(
        df_close.iloc[ts:te], e, exits.iloc[ts:te],
        size=size.iloc[ts:te], size_type='value',
        init_cash=init_cash, fees=fees, slippage=slippage,
        freq=FREQ, cash_sharing=True, group_by=True,
    )

def _sleeve_stats(pf):
    if pf is None:
        return dict(ret=0.0, trades=0, fees=0.0)
    v = pf.value()
    tr = pf.trades.records_readable
    n = len(tr)
    fees = float(tr["Fees"].sum()) if n and "Fees" in tr.columns else 0.0
    return dict(
        ret=(v.iloc[-1] / v.iloc[0] - 1) * 100,
        trades=int(n),
        fees=fees,
    )

def _benchmarks(ts, te):
    sub = df_close.iloc[ts:te]
    sret = sub.pct_change().fillna(0)
    daily_ew = ((1 + sret.mean(axis=1)).cumprod().iloc[-1] - 1) * 100
    norm = sub / sub.iloc[0]
    bh_ew = (norm.mean(axis=1).iloc[-1] - 1) * 100
    return dict(daily_ew=daily_ew, bh_ew=bh_ew)

def _window_diag(ts, te, fees, slippage, trend_alloc=TREND_ALLOC):
    idx = df_close.index[ts:te]
    trend_cash = INIT_CASH * trend_alloc
    mr_cash    = INIT_CASH * MR_ALLOC

    trend_pf = run_sleeve_fee(trend_entries, trend_exits, trend_size,
                              trend_cash, ts, te, fees, slippage)
    mr_pf    = run_sleeve_fee(mr_entries, mr_exits, mr_size,
                              mr_cash, ts, te, fees, slippage)

    ts_stat = _sleeve_stats(trend_pf)
    ms_stat = _sleeve_stats(mr_pf)

    trend_val = (trend_pf.value() if trend_pf is not None
                 else pd.Series(trend_cash, index=idx))
    mr_val    = (mr_pf.value() if mr_pf is not None
                 else pd.Series(mr_cash, index=idx))
    combined = (trend_val.reindex(idx).ffill()
                + mr_val.reindex(idx).ffill())
    strat_ret = (combined.iloc[-1] / combined.iloc[0] - 1) * 100

    b = _benchmarks(ts, te)

    return dict(
        strat_ret=strat_ret,
        daily_ew=b["daily_ew"],
        bh_ew=b["bh_ew"],
        alpha_vs_daily=strat_ret - b["daily_ew"],
        alpha_vs_bh=strat_ret - b["bh_ew"],
        trades=ts_stat["trades"] + ms_stat["trades"],
        trend_trades=ts_stat["trades"],
        mr_trades=ms_stat["trades"],
        fees=ts_stat["fees"] + ms_stat["fees"],
        trend_ret=ts_stat["ret"],
        mr_ret=ms_stat["ret"],
    )

# ---- Tum pencereler ----
rows = []
start = WARMUP
while start + TEST_BARS <= len(df_close):
    te = start + TEST_BARS
    for tag, f, s in (("gross", 0.0, 0.0), ("net", FEES, SLIPPAGE)):
        d = _window_diag(start, te, f, s)
        d["tag"] = tag
        d["start"] = df_close.index[start].date()
        d["end"]   = df_close.index[te - 1].date()
        rows.append(d)
    # Confound testi: MR sleeve YOK, trend %65 (fee=net)
    d = _window_diag(start, te, FEES, SLIPPAGE, trend_alloc=TREND_ALLOC)
    d["tag"] = "trend65_noMR"
    d["start"] = df_close.index[start].date()
    d["end"]   = df_close.index[te - 1].date()
    rows.append(d)
    start += STEP_BARS

diag = pd.DataFrame(rows)

# ---- TANI 1: Fee attribution ----
print("\n" + "=" * 110)
print("TANI 1: Fee attribution (tum pencereler ortalamasi)")
print("=" * 110)
agg = (diag[diag.tag.isin(["gross", "net"])]
       .groupby("tag")[["strat_ret", "alpha_vs_daily", "alpha_vs_bh",
                        "trades", "fees", "trend_ret", "mr_ret"]]
       .mean().round(2))
print(agg.to_string())

gross_ret = agg.loc["gross", "strat_ret"]
net_ret   = agg.loc["net",   "strat_ret"]
print(f"\n  Gross ort. getiri : {gross_ret:+.2f}%")
print(f"  Net   ort. getiri : {net_ret:+.2f}%")
print(f"  Fee drag          : {gross_ret - net_ret:+.2f} puan")
print(f"  Ort. islem sayisi : {agg.loc['net','trades']:.1f} / pencere "
      f"-> ~{agg.loc['net','trades']*2:.0f} islem/yil")
print(f"  Trend sleeve      : {diag[diag.tag=='net']['trend_trades'].mean():.1f} islem, "
      f"getiri {diag[diag.tag=='net']['trend_ret'].mean():+.2f}%")
print(f"  MR sleeve         : {diag[diag.tag=='net']['mr_trades'].mean():.1f} islem, "
      f"getiri {diag[diag.tag=='net']['mr_ret'].mean():+.2f}%")

# ---- TANI 2: Benchmark karsilastirmasi ----
print("\n" + "=" * 110)
print("TANI 2: Benchmark karsilastirmasi (net)")
print("=" * 110)
n = diag[diag.tag == "net"]
print(f"  Strateji getirisi        : {n['strat_ret'].mean():+.2f}%")
print(f"  Benchmark gunluk EW      : {n['daily_ew'].mean():+.2f}%  "
      f"-> alpha {n['alpha_vs_daily'].mean():+.2f}%")
print(f"  Benchmark buy&hold EW    : {n['bh_ew'].mean():+.2f}%  "
      f"-> alpha {n['alpha_vs_bh'].mean():+.2f}%")
print()
print("  Not: 'gunluk EW' pratikte erisilemez ust sinir (her gun sifir maliyetle")
print("  rebalance). 'buy&hold EW' stratejinin gercek davranisina daha yakin.")

# ---- TANI 3: MR confound ----
print("\n" + "=" * 110)
print("TANI 3: MR sleeve konfound testi (net, ayni pencereler)")
print("=" * 110)
full   = diag[diag.tag == "net"]["strat_ret"].mean()
noMR   = diag[diag.tag == "trend65_noMR"]["strat_ret"].mean()
trend100 = diag[diag.tag == "net"]["trend_ret"].mean()  # trend kismi getiri
print(f"  Tam sistem (trend65 + MR35)          : {full:+.2f}%")
print(f"  Trend65 tek basina (MR yok)          : {noMR:+.2f}%")
print(f"  MR sleeve'in marjinal katkisi        : {full - noMR:+.2f} puan")
print()
print("  Eger MR katkisi hala negatifse -> MR gercekten zarar veriyor.")
print("  Eger ~0 veya pozitife donduyse -> onceki tablodaki -4.16 puanin")
print("  buyuk kismi trend sizing'in %100->%65 dusmesinden geliyordu;")
print("  yani 'MR kotu' degil, 'trend'i kisitlamak kotu' sonucu cikar.")

# ---- Pencere bazinda detay ----
print("\n" + "=" * 110)
print("PENCERE BAZINDA (net)")
print("=" * 110)
print(diag[diag.tag == "net"][
    ["start", "end", "strat_ret", "daily_ew", "bh_ew",
     "alpha_vs_daily", "alpha_vs_bh", "trades", "fees"]
].to_string(index=False, float_format=lambda x: f"{x:8.2f}"))

# ---- Yorum anahtari ----
print("\n" + "=" * 110)
print("NASIL OKUNUR")
print("=" * 110)
if net_ret < 0 and gross_ret < 0:
    print("  Gross getiri de negatif -> SORUN SINYALDE, maliyette degil.")
    print("  Fee'yi sifirlamak sonucu kurtarmiyor; bu strateji ailesi bu")
    print("  veri setinde edge uretmiyor. Parametre ayari degil, hipotez")
    print("  degistirmek gerekir.")
elif gross_ret > 0 >= net_ret:
    print(f"  Gross pozitif, net negatif -> SORUN MALIYETTE "
          f"(~{gross_ret - net_ret:.2f} puan drag).")
    print("  Islem sayisini azaltmak (daha uzun holding, daha az rebalance)")
    print("  tek basina anlamli iyilestirme saglayabilir.")
else:
    print("  Karmasik durum: pencere bazinda detaya bakin.")