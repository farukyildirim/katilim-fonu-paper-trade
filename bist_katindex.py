# app.py
# ============================================================
# BIST Katılım Fon Yönetim Sistemi — Tek Dosya Streamlit Uygulaması
# Sürüm: 4.1 (yfinance + KAP + CSV odaklı, gerçek kaynak etiketi)
# Çalıştırma:  streamlit run app.py
# Gerekli:     pip install streamlit plotly pandas numpy yfinance
#              statsmodels PyPortfolioOpt scipy kap_sdk "websockets<14.0" requests
# ============================================================

import warnings
import hashlib
import asyncio
import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# SABİTLER
# ─────────────────────────────────────────────────────────────
BENCHMARK_TICKERS = {
    "XK100": "XK100.IS", "XK050": "XK050.IS",
    "XK030": "XK030.IS", "XU100": "XU100.IS",
}

DEFAULT_UNIVERSE = [
    "ALFAS.IS", "ALKLC.IS", "CVKMD.IS", "EFOR.IS", "QUAGR.IS",
    "RALYH.IS", "TKFEN.IS", "TUKAS.IS", "DOFRB.IS", "NETCD.IS",
    "MERKO.IS", "GUNDG.IS", "IHLGM.IS", "CEMZY.IS", "ARENA.IS",
    "BAYRK.IS", "BANVT.IS", "BRISA.IS", "BUCIM.IS", "ALCTL.IS",
]

RED_FLAG_THRESHOLDS = {
    "debt_to_ebitda_max": 5.0, "ebitda_to_interest_min": 1.5,
    "equity_to_assets_min": 0.25, "current_ratio_min": 1.0,
    "net_margin_min": 0.0, "min_market_cap_tl": 100_000_000,
    "max_flags_for_inclusion": 2,
}

BACKTEST_CONFIG = {
    "initial_capital": 1_000_000, "commission": 0.001,
    "slippage": 0.0005, "risk_free_rate": 0.0, "max_weight": 0.15,
    "stop_loss_pct": -0.15, "stop_loss_check_freq": 5,
}

SYNTHETIC_MODES = {
    "low_corr": {
        "common_vol": 0.015, "idio_vol": 0.020, "beta_range": (0.7, 1.3),
        "label": "Düşük Korelasyon",
    },
    "realistic": {
        "common_vol": 0.018, "idio_vol": 0.008, "beta_range": (0.8, 1.2),
        "label": "Gerçekçi BIST",
    },
}

# ─────────────────────────────────────────────────────────────
# SAYFA AYARI
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BIST Katılım Fon Yönetimi",
    page_icon="📊", layout="wide",
)

# ─────────────────────────────────────────────────────────────
# YARDIMCILAR
# ─────────────────────────────────────────────────────────────
def ensure_datetime_index(df, fallback_periods=None):
    if df is None or df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    try:
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        n = fallback_periods or len(df)
        df = df.copy()
        df.index = pd.bdate_range(end=datetime.today(), periods=n)
        return df


def robust_std(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 3:
        return float(x.std()) if len(x) > 1 else 0.0
    mad = np.median(np.abs(x - np.median(x)))
    return float(1.4826 * mad)


def deterministic_seed(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def label_regime(mean_ret, vol, median_ret, median_vol):
    high_return = mean_ret > median_ret
    high_vol = vol > median_vol
    if high_return and not high_vol:
        return "Boğa"
    elif not high_return and high_vol:
        return "Ayı"
    elif high_return and high_vol:
        return "Volatil"
    else:
        return "Sıkışma"


def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def avg_correlation(prices_df: pd.DataFrame, window: int = 252) -> float:
    if prices_df is None or prices_df.empty or prices_df.shape[1] < 2:
        return np.nan
    log_ret = np.log(prices_df / prices_df.shift(1)).dropna()
    if len(log_ret) < 30:
        return np.nan
    corr = log_ret.tail(window).corr()
    if corr.empty:
        return np.nan
    vals = corr.values[np.triu_indices_from(corr.values, 1)]
    return float(np.nanmean(vals))


# ─────────────────────────────────────────────────────────────
# CSV PARSER
# ─────────────────────────────────────────────────────────────
def parse_csv_prices(file_bytes, tickers_hint=None):
    text = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1254"):
        try:
            text = file_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("CSV okunamadı (encoding).")

    first_lines = text.split("\n")[:5]
    sep = ","
    for cand in (";", "\t", "|"):
        if cand in first_lines[0]:
            sep = cand
            break

    df = pd.read_csv(io.StringIO(text), sep=sep)
    df.columns = [str(c).strip() for c in df.columns]

    date_col = None
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["date", "tarih", "time", "zaman"]):
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    ticker_col = None
    for c in df.columns:
        if c.lower() in ("ticker", "symbol", "sembol", "kod", "hisse"):
            ticker_col = c
            break

    price_col = None
    for c in df.columns:
        if c.lower() in ("close", "kapanis", "kapanış", "fiyat", "price",
                         "adjusted_close", "adj close"):
            price_col = c
            break

    if ticker_col and price_col:
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        wide = df.pivot_table(index=df.index, columns=ticker_col,
                              values=price_col, aggfunc="last")
    else:
        wide = df.copy()
        for c in wide.columns:
            wide[c] = pd.to_numeric(wide[c], errors="coerce")

    wide = wide.dropna(how="all").sort_index()
    wide = ensure_datetime_index(wide)
    return wide


# ─────────────────────────────────────────────────────────────
# EODHD API (opsiyonel)
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_eodhd_prices(tickers, start, end, api_key):
    import requests
    if not api_key:
        return pd.DataFrame()

    base = "https://eodhd.com/api/eod"
    out = {}
    errors = []
    successful = 0

    progress = st.progress(0, text="EODHD verileri çekiliyor...")
    total = len(tickers)

    for i, tkr in enumerate(tickers):
        progress.progress((i + 1) / total,
                          text=f"EODHD: {tkr} ({i+1}/{total})")
        try:
            params = {"api_token": api_key, "fmt": "json",
                      "from": start, "to": end, "period": "d"}
            r = requests.get(f"{base}/{tkr}", params=params, timeout=15)

            if r.status_code == 200:
                data = r.json()
                if data and isinstance(data, list) and len(data) > 0:
                    s = pd.DataFrame(data)
                    price_col = ("adjusted_close"
                                 if "adjusted_close" in s.columns else "close")
                    if "date" in s.columns and price_col in s.columns:
                        s["date"] = pd.to_datetime(s["date"])
                        out[tkr] = s.set_index("date")[price_col]
                        successful += 1
                else:
                    errors.append(f"{tkr}: boş veri")
            elif r.status_code == 401:
                errors.append(f"{tkr}: API anahtarı geçersiz")
                break
            elif r.status_code == 429:
                errors.append(f"{tkr}: Rate limit aşıldı")
                break
            elif r.status_code == 404:
                errors.append(f"{tkr}: sembol bulunamadı (BIST Free planda yok)")
            else:
                errors.append(f"{tkr}: HTTP {r.status_code}")
        except Exception as e:
            errors.append(f"{tkr}: {str(e)[:50]}")

    progress.empty()

    if successful == 0:
        if errors:
            st.error("❌ EODHD başarısız:\n" +
                     "\n".join(f"- {e}" for e in errors[:5]))
        return pd.DataFrame()

    df = pd.DataFrame(out)
    df = ensure_datetime_index(df).sort_index()
    if errors:
        st.warning(f"⚠️ {len(errors)} hisse alınamadı.")
    st.success(f"✅ EODHD: {successful}/{total} hisse, {len(df)} gün")
    return df


# ─────────────────────────────────────────────────────────────
# VERİ KATMANI
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def download_prices(tickers, start, end):
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(tickers=tickers, start=start, end=end,
                      auto_adjust=True, progress=False, group_by="ticker")
    if raw.empty:
        return pd.DataFrame()
    if len(tickers) == 1:
        raw.columns = pd.MultiIndex.from_product([[tickers[0]], raw.columns])
    return raw


def extract_close(raw):
    if raw.empty:
        return pd.DataFrame()
    out = {}
    for tkr in raw.columns.get_level_values(0).unique():
        try:
            out[tkr] = raw[(tkr, "Close")]
        except KeyError:
            continue
    df = pd.DataFrame(out)
    return ensure_datetime_index(df).sort_index()


def compute_returns(prices):
    if prices.empty:
        return pd.DataFrame()
    return np.log(prices / prices.shift(1)).dropna(how="all")


def generate_sample_data(tickers, periods=800, seed=42, mode="low_corr"):
    cfg = SYNTHETIC_MODES.get(mode, SYNTHETIC_MODES["low_corr"])
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=datetime.today(), periods=periods)
    common = rng.normal(0, cfg["common_vol"], periods)
    beta_lo, beta_hi = cfg["beta_range"]
    data = {}
    for tkr in tickers:
        beta = rng.uniform(beta_lo, beta_hi)
        idio = rng.normal(0, cfg["idio_vol"], periods)
        vol_mult = np.where(np.arange(periods) > periods - 150, 1.8, 1.0)
        data[tkr] = 100 * np.exp(np.cumsum(beta * common + idio * vol_mult))
    return pd.DataFrame(data, index=dates)


# ─────────────────────────────────────────────────────────────
# KAP SDK (temel veri)
# ─────────────────────────────────────────────────────────────
def _to_float(val):
    if val is None or val == "" or val == "-":
        return np.nan
    try:
        if isinstance(val, str):
            val = val.replace(".", "").replace(",", ".").strip()
        return float(val)
    except (ValueError, TypeError):
        return np.nan


async def _fetch_kap_data_async(tickers):
    from kap_sdk.kap_client import KapClient
    client = KapClient(cache_expiry=3600, company_cache_expiry=86400)
    rows = []
    clean_tickers = [t.replace(".IS", "") for t in tickers]

    for tkr_clean, tkr_full in zip(clean_tickers, tickers):
        row = {"ticker": tkr_full, "_kap_source": True}
        try:
            company = await client.get_company(tkr_clean)
            if company is None:
                row["_kap_error"] = "Şirket bulunamadı"
                rows.append(row)
                continue
            row["sector"] = getattr(company, "city", "Bilinmiyor")
            current_year = str(datetime.today().year - 1)
            report = await client.get_financial_report(company, current_year)
            if report is not None:
                def _get(obj, *keys, default=np.nan):
                    for k in keys:
                        v = obj.get(k) if isinstance(obj, dict) \
                            else getattr(obj, k, None)
                        if v is not None:
                            return _to_float(v)
                    return default
                row["market_cap"] = _get(report, "market_cap", "marketCap")
                row["debt_to_ebitda"] = _get(report, "debt_to_ebitda",
                                             "debtToEbitda")
                row["current_ratio"] = _get(report, "current_ratio",
                                            "currentRatio")
                row["profit_margin"] = _get(report, "profit_margin",
                                            "profitMargin")
                row["total_debt"] = _get(report, "total_debt", "totalDebt")
                row["total_assets"] = _get(report, "total_assets",
                                           "totalAssets")
                row["ebitda"] = _get(report, "ebitda")
                row["interest_expense"] = _get(report, "interest_expense",
                                               "interestExpense")
        except Exception as e:
            row["_kap_error"] = str(e)[:100]
        rows.append(row)

    try:
        client.clear_cache()
    except Exception:
        pass
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamentals_kap(tickers):
    try:
        from kap_sdk.kap_client import KapClient  # noqa: F401
    except ImportError:
        st.warning("⚠️ `kap_sdk` kurulu değil. `pip install kap_sdk`")
        return pd.DataFrame()
    try:
        with st.spinner("KAP'tan finansal veriler çekiliyor..."):
            df = run_async(_fetch_kap_data_async(tickers))
        return df
    except Exception as e:
        st.warning(f"⚠️ KAP API hatası: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamentals(tickers, mode="synthetic"):
    rows = []
    any_synthetic = False
    for tkr in tickers:
        row = {"ticker": tkr}
        if mode == "yfinance":
            try:
                info = yf.Ticker(tkr).info
                row.update({
                    "market_cap": info.get("marketCap", np.nan),
                    "debt_to_ebitda": info.get("debtToEbitda", np.nan),
                    "current_ratio": info.get("currentRatio", np.nan),
                    "profit_margin": info.get("profitMargins", np.nan),
                    "total_debt": info.get("totalDebt", np.nan),
                    "total_assets": info.get("totalAssets", np.nan),
                    "ebitda": info.get("ebitda", np.nan),
                    "interest_expense": info.get("interestExpense", np.nan),
                    "sector": info.get("sector", "Bilinmiyor"),
                })
            except Exception:
                pass
        if pd.isna(row.get("market_cap")):
            rng = np.random.default_rng(deterministic_seed(tkr))
            row.update({
                "market_cap": rng.uniform(5e8, 5e10),
                "debt_to_ebitda": rng.uniform(0.5, 8.0),
                "current_ratio": rng.uniform(0.5, 3.0),
                "profit_margin": rng.uniform(-0.15, 0.35),
                "total_debt": rng.uniform(1e8, 5e9),
                "total_assets": rng.uniform(5e8, 2e10),
                "ebitda": rng.uniform(1e8, 3e9),
                "interest_expense": rng.uniform(1e7, 5e8),
                "sector": "Bilinmiyor (demo)", "_synthetic": True,
            })
            any_synthetic = True
        else:
            row["_synthetic"] = False
        rows.append(row)
    return pd.DataFrame(rows), any_synthetic


# ─────────────────────────────────────────────────────────────
# REJİM TESPİTİ
# ─────────────────────────────────────────────────────────────
def detect_regimes(returns, n_regimes=2, lookback=504):
    s = returns.dropna().copy()
    if len(s) < 120:
        return None, None, None
    effective_lookback = min(lookback, len(s))
    train = s.iloc[-effective_lookback:]

    try:
        from statsmodels.tsa.regime_switching.markov_regression import (
            MarkovRegression,
        )
        model = MarkovRegression(train, k_regimes=n_regimes, trend="c",
                                 switching_variance=True)
        res = model.fit(disp=False)
        probs = res.smoothed_marginal_probabilities
        probs.columns = [f"regime_{c}" for c in probs.columns]
        min_share = 0.10
        counts = (probs > 0.5).sum()
        shares = counts / len(probs)
        if shares.min() < min_share:
            st.warning("⚠️ Markov anlamlı rejim ayıramadı → kural bazlı.")
            return _rule_based_regimes(train, n_regimes)
        return _build_regime_summary(probs, train, n_regimes)
    except Exception:
        return _rule_based_regimes(train, n_regimes)


def _build_regime_summary(probs, train, n_regimes):
    overall_vol = robust_std(train)
    MAX_VOL_MULT = 3.0
    means, vols, counts, shares = {}, {}, {}, {}
    for r in range(n_regimes):
        col = f"regime_{r}"
        mask = probs[col] > 0.5
        sub = train[mask].dropna()
        counts[r] = len(sub)
        shares[r] = len(sub) / len(train) if len(train) > 0 else 0.0
        if len(sub) >= 20:
            means[r] = float(np.median(sub))
            v = robust_std(sub)
        else:
            means[r] = float(np.median(train))
            v = overall_vol
        if overall_vol > 0 and v > MAX_VOL_MULT * overall_vol:
            v = MAX_VOL_MULT * overall_vol
        vols[r] = v

    med_ret = np.median(list(means.values()))
    med_vol = np.median(list(vols.values()))
    labels = {r: label_regime(means[r], vols[r], med_ret, med_vol)
              for r in range(n_regimes)}
    bull = min(vols, key=vols.get)

    summary = pd.DataFrame([{
        "Rejim": r, "Etiket": labels[r],
        "Ort. Günlük Getiri": means[r],
        "Yıllık Oynaklık": vols[r] * np.sqrt(252),
        "Gözlem Sayısı": counts[r], "Gözlem Payı": shares[r],
    } for r in range(n_regimes)])
    return probs, summary, bull


def _rule_based_regimes(train, n_regimes=2):
    roll_vol = train.rolling(20).std().bfill()
    threshold = roll_vol.median() * 1.3
    ratio = (roll_vol / threshold).clip(0.3, 3.0)
    bear_prob = 1 / (1 + np.exp(-3 * (ratio - 1)))
    bull_prob = 1 - bear_prob
    probs = pd.DataFrame({
        "regime_0": bull_prob, "regime_1": bear_prob,
    }, index=train.index)

    overall_vol = robust_std(train)
    MAX_VOL_MULT = 3.0
    means, vols, counts, shares = {}, {}, {}, {}
    for r in range(2):
        col = f"regime_{r}"
        mask = probs[col] > 0.5
        sub = train[mask].dropna()
        counts[r] = len(sub)
        shares[r] = len(sub) / len(train) if len(train) > 0 else 0.0
        if len(sub) >= 20:
            means[r] = float(np.median(sub))
            v = robust_std(sub)
        else:
            means[r] = float(np.median(train))
            v = overall_vol
        if overall_vol > 0 and v > MAX_VOL_MULT * overall_vol:
            v = MAX_VOL_MULT * overall_vol
        vols[r] = v

    med_ret = np.median(list(means.values()))
    med_vol = np.median(list(vols.values()))
    labels = {r: label_regime(means[r], vols[r], med_ret, med_vol)
              for r in range(2)}
    bull = min(vols, key=vols.get)
    summary = pd.DataFrame([{
        "Rejim": r, "Etiket": labels[r],
        "Ort. Günlük Getiri": means[r],
        "Yıllık Oynaklık": vols[r] * np.sqrt(252),
        "Gözlem Sayısı": counts[r], "Gözlem Payı": shares[r],
    } for r in range(2)])

    st.info("ℹ️ Kural bazlı rejim tespiti (rolling vol eşiği).")
    return probs, summary, bull


# ─────────────────────────────────────────────────────────────
# KIRMIZI BAYRAK
# ─────────────────────────────────────────────────────────────
def evaluate_red_flags(fund, th):
    df = fund.copy()
    for c in ["market_cap", "debt_to_ebitda", "current_ratio",
              "profit_margin", "total_debt", "total_assets",
              "ebitda", "interest_expense"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["flag_debt"] = (df["debt_to_ebitda"] >
                       th["debt_to_ebitda_max"]).astype(int)
    df["flag_interest"] = (
        (df["ebitda"] / df["interest_expense"].replace(0, np.nan))
        < th["ebitda_to_interest_min"]).astype(int)
    equity_ratio = 1 - (df["total_debt"] /
                        df["total_assets"].replace(0, np.nan))
    df["flag_equity"] = (equity_ratio <
                         th["equity_to_assets_min"]).astype(int)
    df["flag_current"] = (df["current_ratio"] <
                          th["current_ratio_min"]).astype(int)
    df["flag_margin"] = (df["profit_margin"] <
                         th["net_margin_min"]).astype(int)
    df["flag_mktcap"] = (df["market_cap"] <
                         th["min_market_cap_tl"]).astype(int)

    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    df["total_flags"] = df[flag_cols].sum(axis=1)
    df["exclude_candidate"] = df["total_flags"] > th["max_flags_for_inclusion"]
    return df


# ─────────────────────────────────────────────────────────────
# OPTİMİZASYON
# ─────────────────────────────────────────────────────────────
def optimize_portfolio(prices, regime_label, regime_prob,
                       max_weight=0.15, rf=0.0,
                       prev_weights=None, turnover_penalty=0.001,
                       l2_reg=0.0, mu_shrink=0.0):
    from pypfopt import EfficientFrontier, risk_models
    from pypfopt.exceptions import OptimizationError

    if prices.empty or prices.shape[1] < 2:
        return {"weights": {}, "method": "yetersiz_veri", "performance": {}}

    window = prices.tail(504) if len(prices) >= 504 else prices
    valid = window.dropna(axis=1, thresh=int(len(window) * 0.8))
    if valid.shape[1] < 2:
        valid = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
    if valid.shape[1] < 2:
        return {"weights": {}, "method": "yetersiz_veri", "performance": {}}

    try:
        log_ret = np.log(valid / valid.shift(1)).dropna()
        mu = np.expm1(log_ret.mean() * 252)
        if mu_shrink > 0 and len(mu) > 1:
            mu_mean = mu.mean()
            mu = (1 - mu_shrink) * mu + mu_shrink * mu_mean
        if len(mu) >= 5:
            lo, hi = mu.quantile(0.10), mu.quantile(0.90)
            mu = mu.clip(lower=lo, upper=hi)
        try:
            from pypfopt import CovarianceShrinkage
            S = CovarianceShrinkage(valid).ledoit_wolf()
        except Exception:
            S = risk_models.sample_cov(valid)
    except Exception as e:
        return {"weights": {}, "method": f"hata:{e}", "performance": {}}

    max_mu = float(mu.max()) if len(mu) > 0 else 0.0
    force_min_vol = (max_mu <= rf)

    if regime_label in ("Boğa", "Volatil") and regime_prob > 0.6 \
            and not force_min_vol:
        method = "max_sharpe"
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
        try:
            ef.max_sharpe(risk_free_rate=rf)
        except (OptimizationError, ValueError):
            ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
            ef.min_volatility()
            method = "min_vol_fallback"
    elif regime_label in ("Ayı", "Sıkışma") or regime_prob < 0.4 \
            or force_min_vol:
        method = "min_volatility"
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
        ef.min_volatility()
    else:
        n = valid.shape[1]
        return {"weights": {c: 1.0 / n for c in valid.columns},
                "method": "equal_weight",
                "performance": {"note": "Geçiş dönemi eşit ağırlık"}}

    cleaned = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False, risk_free_rate=rf)

    # L2 post-processing
    if l2_reg > 0:
        n = len(cleaned)
        target = 1.0 / n
        blended = {
            k: (1 - l2_reg) * cleaned.get(k, 0) + l2_reg * target
            for k in cleaned
        }
        total = sum(blended.values())
        if total > 0:
            cleaned = {k: v / total for k, v in blended.items()}
        method = method + f"_L2={l2_reg:.2f}"

    if prev_weights:
        keys = set(cleaned) | set(prev_weights)
        adj = {k: max(cleaned.get(k, 0) - turnover_penalty *
                      abs(cleaned.get(k, 0) - prev_weights.get(k, 0)), 0)
               for k in keys}
        total = sum(adj.values())
        if total > 0:
            cleaned = {k: v / total for k, v in adj.items()}

    # Performansı blend sonrası yeniden hesapla
    try:
        w_arr = np.array([cleaned.get(c, 0) for c in valid.columns])
        mu_arr = mu.values if hasattr(mu, "values") else np.asarray(mu)
        S_arr = S.values if hasattr(S, "values") else np.asarray(S)
        ret = float(w_arr @ mu_arr)
        if S_arr.ndim == 2:
            var = float(w_arr @ S_arr @ w_arr)
            vol = float(np.sqrt(max(var, 0.0)))
        else:
            vol = float(perf[1])
        sharpe = (ret - rf) / vol if vol > 0 else 0
        perf = (ret, vol, sharpe)
    except Exception:
        pass

    return {"weights": dict(cleaned), "method": method,
            "performance": {"expected_return": perf[0],
                            "volatility": perf[1], "sharpe": perf[2]}}


def efficient_frontier_points(prices, max_weight, rf, n_points=25):
    from pypfopt import EfficientFrontier, risk_models
    valid = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
    if valid.shape[1] < 2:
        return pd.DataFrame()
    try:
        log_ret = np.log(valid / valid.shift(1)).dropna()
        mu = np.expm1(log_ret.mean() * 252)
        if len(mu) >= 5:
            lo, hi = mu.quantile(0.10), mu.quantile(0.90)
            mu = mu.clip(lower=lo, upper=hi)
        S = risk_models.sample_cov(valid)
    except Exception:
        return pd.DataFrame()
    points = []
    for tr in np.linspace(mu.min(), mu.max() * 0.9, n_points):
        try:
            ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
            ef.efficient_return(tr)
            r, v, _ = ef.portfolio_performance(risk_free_rate=rf)
            points.append({"return": r, "volatility": v})
        except Exception:
            continue
    return pd.DataFrame(points)


# ─────────────────────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────────────────────
class BacktestEngine:
    def __init__(self, prices, benchmark=None, config=None):
        self.prices = prices.sort_index()
        self.benchmark = benchmark
        self.cfg = {**BACKTEST_CONFIG, **(config or {})}
        self.portfolio_values = None
        self.weights_history = []
        self.stop_loss_events = []

    def run(self, weight_schedule):
        dates = self.prices.index
        capital = self.cfg["initial_capital"]
        current_w = {}
        pv = pd.Series(index=dates, dtype=float)
        if len(dates) == 0:
            return pv
        pv.iloc[0] = capital
        rb_dates = sorted(weight_schedule.keys())
        rb_idx = 0
        sl_freq = self.cfg.get("stop_loss_check_freq", 5)
        sl_threshold = self.cfg.get("stop_loss_pct", -0.15)

        for i in range(1, len(dates)):
            date, prev = dates[i], dates[i - 1]
            day_ret = 0.0
            for tkr, w in current_w.items():
                if tkr in self.prices.columns:
                    p0, p1 = (self.prices.loc[prev, tkr],
                              self.prices.loc[date, tkr])
                    if p0 and not np.isnan(p0) and not np.isnan(p1):
                        day_ret += w * (p1 / p0 - 1)
            capital *= (1 + day_ret)

            if current_w and i % sl_freq == 0:
                new_w, removed = {}, []
                for tkr, w in current_w.items():
                    if tkr in self.prices.columns:
                        hist = self.prices[tkr].loc[:date].tail(252).dropna()
                        if len(hist) > 20:
                            peak, cur = hist.max(), hist.iloc[-1]
                            if peak > 0 and (cur / peak - 1) < sl_threshold:
                                removed.append(tkr)
                                continue
                    new_w[tkr] = w
                if removed:
                    total = sum(new_w.values())
                    if total > 0:
                        current_w = {k: v / total for k, v in new_w.items()}
                        capital *= (1 - 0.001)
                        self.stop_loss_events.append({
                            "date": date.strftime("%Y-%m-%d"),
                            "removed": removed})

            ds = date.strftime("%Y-%m-%d")
            if rb_idx < len(rb_dates) and ds >= rb_dates[rb_idx]:
                new_w = weight_schedule[rb_dates[rb_idx]]
                turnover = self._turnover(current_w, new_w)
                cost = turnover * (self.cfg["commission"] +
                                   self.cfg["slippage"])
                capital *= (1 - cost)
                current_w = new_w
                self.weights_history.append({
                    "date": ds, "turnover": turnover, "cost": cost})
                rb_idx += 1
            pv.iloc[i] = capital
        self.portfolio_values = pv.dropna()
        return self.portfolio_values

    @staticmethod
    def _turnover(old, new):
        keys = set(old) | set(new)
        return sum(abs(new.get(k, 0) - old.get(k, 0)) for k in keys) / 2

    def metrics(self):
        if self.portfolio_values is None or len(self.portfolio_values) < 2:
            return {}
        pv = self.portfolio_values
        rets = pv.pct_change().dropna()
        n = len(rets)
        if n == 0:
            return {}
        total = pv.iloc[-1] / pv.iloc[0] - 1
        cagr = (1 + total) ** (252 / n) - 1
        vol = rets.std() * np.sqrt(252)
        rf_daily = self.cfg["risk_free_rate"] / 252
        excess = rets - rf_daily
        sharpe = (excess.mean() / rets.std() * np.sqrt(252)
                  if rets.std() > 0 else 0)
        downside = rets[rets < 0]
        if len(downside) > 1:
            dd_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(252)
            sortino = (excess.mean() * 252 / dd_dev) if dd_dev > 0 else 0
        else:
            sortino = 0
        cummax = pv.cummax()
        max_dd = ((pv - cummax) / cummax).min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        return {"Toplam Getiri": total, "CAGR": cagr, "Yıllık Oynaklık": vol,
                "Sharpe": sharpe, "Sortino": sortino, "Max Drawdown": max_dd,
                "Calmar": calmar, "Kazanma Oranı": (rets > 0).mean(),
                "VaR (95%)": rets.quantile(0.05), "İşlem Günü": n}

    def drawdown_series(self):
        if self.portfolio_values is None:
            return pd.Series()
        pv = self.portfolio_values
        return (pv - pv.cummax()) / pv.cummax()

    def monthly_returns(self):
        if self.portfolio_values is None:
            return pd.DataFrame()
        pv = self.portfolio_values
        if not isinstance(pv.index, pd.DatetimeIndex):
            pv = pv.copy()
            pv.index = pd.to_datetime(pv.index, errors="coerce")
            pv = pv[~pv.index.isna()]
        if pv.empty:
            return pd.DataFrame()
        monthly = pv.resample("ME").last().pct_change()
        df = monthly.to_frame("ret")
        df["year"] = df.index.year
        df["month"] = df.index.month
        return df.pivot_table(index="year", columns="month",
                              values="ret", aggfunc="first")


def build_weight_schedule(prices, universe, regime_probs=None,
                          bull_regime=None, rebalance_freq="M",
                          max_weight=0.15, l2_reg=0.0, mu_shrink=0.0):
    if prices is None or prices.empty:
        return {}
    if not isinstance(prices.index, pd.DatetimeIndex):
        try:
            prices = prices.copy()
            prices.index = pd.to_datetime(prices.index)
        except Exception:
            return {}
    dates = prices.index
    if len(dates) == 0:
        return {}
    try:
        series = pd.Series(dates, index=dates)
        if rebalance_freq == "M":
            rb = series.resample("MS").first().dropna()
        elif rebalance_freq == "Q":
            rb = series.resample("QS").first().dropna()
        else:
            rb = series.resample("MS").first().dropna()
    except Exception:
        rb = pd.Series(dates, index=dates).iloc[::21]

    schedule, prev_w = {}, None
    for d in rb:
        d_ts = pd.Timestamp(d)
        if d_ts not in prices.index:
            available = prices.index[prices.index <= d_ts]
            if len(available) == 0:
                continue
            d_ts = available[-1]
        hist = prices.loc[:d_ts]
        if len(hist) < 60:
            continue
        label, prob = "Geçiş", 0.5
        if regime_probs is not None and bull_regime is not None:
            avail = regime_probs.loc[:d_ts]
            if not avail.empty:
                last = avail.iloc[-1]
                regime = int(last.idxmax().split("_")[1])
                prob = float(last.max())
                label = "Boğa" if regime == bull_regime else "Ayı"
        opt = optimize_portfolio(
            hist, label, prob,
            max_weight=max_weight, prev_weights=prev_w,
            l2_reg=l2_reg, mu_shrink=mu_shrink)
        w = {k: v for k, v in opt["weights"].items() if v > 0.001}
        if rebalance_freq == "R" and label == "Ayı":
            w = {k: v * 0.5 for k, v in w.items()}
        if w:
            schedule[d_ts.strftime("%Y-%m-%d")] = w
            prev_w = w
    return schedule


# ─────────────────────────────────────────────────────────────
# WALK-FORWARD
# ─────────────────────────────────────────────────────────────
def walk_forward_backtest(prices, universe, regime_probs=None,
                           bull_regime=None, train_days=252,
                           test_days=126, step_days=126,
                           rebalance_freq="M", max_weight=0.15,
                           l2_reg=0.0, mu_shrink=0.0,
                           commission=0.001, slippage=0.0005,
                           stop_loss_pct=-0.15):
    if prices is None or prices.empty:
        return None, []

    prices = ensure_datetime_index(prices.copy())
    n = len(prices)
    if n < train_days + test_days:
        return None, []

    all_oos_values = []
    window_results = []

    start = 0
    while start + train_days + test_days <= n:
        train_end = start + train_days
        test_end = min(train_end + test_days, n)

        train_prices = prices.iloc[start:train_end]
        test_prices = prices.iloc[train_end:test_end]

        if len(train_prices) < 126 or len(test_prices) < 5:
            start += step_days
            continue

        train_returns = compute_returns(train_prices)
        if train_returns.empty:
            start += step_days
            continue
        bm_ret = train_returns.mean(axis=1).dropna()
        if len(bm_ret) < 60:
            start += step_days
            continue

        wf_regime_probs, wf_regime_summary, wf_bull = detect_regimes(
            bm_ret, n_regimes=2, lookback=min(504, len(bm_ret)))

        wf_schedule = build_weight_schedule(
            test_prices, universe,
            regime_probs=wf_regime_probs if wf_regime_probs is not None
            else None,
            bull_regime=wf_bull,
            rebalance_freq=rebalance_freq,
            max_weight=max_weight,
            l2_reg=l2_reg, mu_shrink=mu_shrink)

        if not wf_schedule:
            start += step_days
            continue

        engine = BacktestEngine(
            test_prices, config={
                "commission": commission, "slippage": slippage,
                "stop_loss_pct": stop_loss_pct,
                "stop_loss_check_freq": 5})
        oos_pv = engine.run(wf_schedule)
        oos_m = engine.metrics()

        if oos_pv is not None and len(oos_pv) > 1:
            all_oos_values.append(oos_pv)
            window_results.append({
                "train_start": train_prices.index[0].date(),
                "train_end": train_prices.index[-1].date(),
                "test_start": test_prices.index[0].date(),
                "test_end": test_prices.index[-1].date(),
                "test_days": len(test_prices),
                "test_return": oos_m.get("Toplam Getiri", 0),
                "test_sharpe": oos_m.get("Sharpe", 0),
                "test_max_dd": oos_m.get("Max Drawdown", 0),
                "test_vol": oos_m.get("Yıllık Oynaklık", 0),
            })

        start += step_days

    if not all_oos_values:
        return None, []

    combined = []
    for pv in all_oos_values:
        if len(pv) > 0:
            combined.append(pv / pv.iloc[0])

    oos_series = pd.concat(combined)
    oos_series = oos_series[~oos_series.index.duplicated(keep="first")]
    oos_series = oos_series.sort_index()

    return oos_series, window_results


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Fon Parametreleri")

start_date = st.sidebar.date_input(
    "Başlangıç", value=datetime.today() - timedelta(days=365 * 3))
end_date = st.sidebar.date_input("Bitiş", value=datetime.today())

# ─── Veri Kaynağı (CSV öne çıkarıldı) ───
data_source = st.sidebar.selectbox(
    "Fiyat Verisi Kaynağı",
    ["yfinance", "csv_yukle",
     "eodhd", "sentetik_low_corr", "sentetik_realistic"],
    format_func=lambda x: {
        "yfinance": "🟢 yfinance (ücretsiz)",
        "csv_yukle": "📁 CSV Yükle (Matriks/İdeal) ⭐",
        "eodhd": "🔷 EODHD (All World €19.99/ay gerekir)",
        "sentetik_low_corr": "🔵 Sentetik - Düşük Korelasyon",
        "sentetik_realistic": "🟣 Sentetik - Gerçekçi BIST",
    }[x],
    index=0,
)

# EODHD BIST kapsam uyarısı
if data_source == "eodhd":
    st.sidebar.warning(
        "⚠️ **EODHD Free plan BIST'i kapsamaz.** "
        "BIST verisi için **All World (€19.99/ay)** planı gerekir. "
        "Ücretsiz alternatif: 📁 CSV Yükle veya 🟢 yfinance."
    )

eodhd_api_key = ""
if data_source == "eodhd":
    eodhd_api_key = st.sidebar.text_input(
        "EODHD API Anahtarı", type="password",
        help="eodhd.com → All World plan gerekir")

uploaded_csv = None
if data_source == "csv_yukle":
    uploaded_csv = st.sidebar.file_uploader(
        "Fiyat CSV dosyası", type=["csv", "txt"],
        help="Matriks: BİST Analizler → Tarihsel Veri → CSV. "
             "Format: date + ticker sütunları veya date,ticker,close")
    st.sidebar.caption(
        "📌 **Beklenen formatlar:**\n"
        "- Geniş: `date,ALFAS.IS,ALKLC.IS,...`\n"
        "- Uzun: `date,ticker,close`\n"
        "- Matriks: `Tarih;Kapanış;...` (noktalı virgül)"
    )

rebalance_freq = st.sidebar.selectbox(
    "Rebalance Sıklığı", ["M", "Q", "R"],
    format_func=lambda x: {"M": "Aylık", "Q": "Çeyreklik",
                           "R": "Rejim Bazlı"}[x])
n_regimes = st.sidebar.selectbox("Rejim Sayısı", [2, 3])
regime_lookback = st.sidebar.slider("Rejim Penceresi (gün)",
                                    252, 756, 504, step=63)
red_flag_threshold = st.sidebar.slider("Kırmızı Bayrak Eşiği", 1, 5, 2)
benchmark_key = st.sidebar.selectbox("Benchmark",
                                     list(BENCHMARK_TICKERS.keys()))
commission = st.sidebar.number_input("Komisyon", 0.0, 0.01, 0.001,
                                     step=0.0001, format="%.4f")
slippage = st.sidebar.number_input("Slippage", 0.0, 0.01, 0.0005,
                                   step=0.0001, format="%.4f")
max_weight = st.sidebar.slider("Maks. Tek Hisse Ağırlığı",
                               0.05, 0.30, 0.15, step=0.01)
turnover_penalty = st.sidebar.slider("Turnover Cezası", 0.0, 0.01, 0.001,
                                     step=0.0005, format="%.4f")
stop_loss_pct = st.sidebar.slider("Stop-Loss Eşiği (%)",
                                  -0.40, -0.05, -0.15, step=0.05)
annual_inflation = st.sidebar.slider(
    "Yıllık Enflasyon (reel getiri için, %)", 0, 100, 50, step=5)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚖️ Optimizer Regularization")
l2_reg = st.sidebar.slider("L2 Cezası (eşit ağırlığa doğru)",
                           0.0, 1.0, 0.0, step=0.05)
mu_shrink = st.sidebar.slider("Beklenen Getiri Shrinkage",
                              0.0, 1.0, 0.0, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Walk-Forward Backtest")
use_wf = st.sidebar.checkbox("Walk-Forward modu", value=False,
                              help="Rolling window ile overfitting testi")
wf_train = st.sidebar.slider("Eğitim penceresi (gün)",
                              126, 504, 252, step=21)
wf_test = st.sidebar.slider("Test penceresi (gün)",
                            21, 252, 126, step=21)
wf_step = st.sidebar.slider("Kaydırma adımı (gün)",
                            21, 252, 126, step=21)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔌 KAP Veri Kaynağı")
use_kap = st.sidebar.checkbox("KAP SDK kullan (temel veri)", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ Yatırım tavsiyesi değildir. Demo amaçlıdır.")

# ─────────────────────────────────────────────────────────────
# VERİ YÜKLE
# ─────────────────────────────────────────────────────────────
bm_ticker = BENCHMARK_TICKERS[benchmark_key]
all_tickers = DEFAULT_UNIVERSE + [bm_ticker]

prices = pd.DataFrame()
csv_loaded = False

# 1. CSV
if data_source == "csv_yukle" and uploaded_csv is not None:
    try:
        raw_bytes = uploaded_csv.read()
        prices = parse_csv_prices(raw_bytes, DEFAULT_UNIVERSE)
        if not prices.empty:
            csv_loaded = True
            st.success(
                f"✅ CSV: **{prices.shape[1]}** hisse, "
                f"**{len(prices)}** gün "
                f"({prices.index.min().date()} → "
                f"{prices.index.max().date()})"
            )
            if bm_ticker not in prices.columns:
                hisse_cols = [c for c in prices.columns
                              if c in DEFAULT_UNIVERSE] or list(prices.columns)
                norm = prices[hisse_cols] / prices[hisse_cols].iloc[0]
                prices[bm_ticker] = norm.mean(axis=1) * 100
        else:
            st.error("⚠️ CSV boş veya parse edilemedi.")
    except Exception as e:
        st.error(f"⚠️ CSV parse hatası: {e}")

# 2. EODHD
if prices.empty and data_source == "eodhd" and eodhd_api_key:
    with st.spinner("EODHD'den veri çekiliyor..."):
        prices = fetch_eodhd_prices(
            all_tickers,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            eodhd_api_key)
    if not prices.empty:
        if bm_ticker not in prices.columns:
            hisse_cols = [c for c in prices.columns
                          if c in DEFAULT_UNIVERSE]
            if hisse_cols:
                norm = prices[hisse_cols] / prices[hisse_cols].iloc[0]
                prices[bm_ticker] = norm.mean(axis=1) * 100
    else:
        st.warning("⚠️ EODHD'den veri alınamadı, yfinance'e geçiliyor.")

# 3. yfinance
if prices.empty and data_source == "yfinance":
    with st.spinner("yfinance'ten veri yükleniyor..."):
        raw = download_prices(all_tickers,
                              start_date.strftime("%Y-%m-%d"),
                              end_date.strftime("%Y-%m-%d"))
        prices = extract_close(raw)
        hisse_cols = [c for c in DEFAULT_UNIVERSE if c in prices.columns]
        if hisse_cols:
            if bm_ticker not in prices.columns or \
               prices[bm_ticker].dropna().empty:
                st.info(f"⚠️ {bm_ticker} verisi yok. Benchmark = evren ort.")
                norm = prices[hisse_cols] / prices[hisse_cols].iloc[0]
                prices[bm_ticker] = norm.mean(axis=1) * 100

# 4. Sentetik
if prices.empty and data_source.startswith("sentetik"):
    syn_mode = "low_corr" if data_source == "sentetik_low_corr" \
        else "realistic"
    cfg = SYNTHETIC_MODES[syn_mode]
    st.info(f"🧪 Sentetik: **{cfg['label']}**")
    prices = generate_sample_data(DEFAULT_UNIVERSE, 800, 42, mode=syn_mode)
    prices[bm_ticker] = generate_sample_data(
        [bm_ticker], 800, 99, mode=syn_mode).iloc[:, 0]

# 5. Fallback
if prices.empty:
    st.warning("⚠️ Hiçbir kaynaktan veri alınamadı. Sentetik fallback.")
    prices = generate_sample_data(DEFAULT_UNIVERSE, 800, 42, mode="realistic")
    prices[bm_ticker] = generate_sample_data(
        [bm_ticker], 800, 99, mode="realistic").iloc[:, 0]

prices = ensure_datetime_index(prices)
prices = prices[~prices.index.isna()].sort_index()
prices = prices[~prices.index.duplicated(keep="last")]
returns = compute_returns(prices)
benchmark_series = prices[bm_ticker] if bm_ticker in prices.columns else None

# 🔑 GERÇEK KAYNAK TESPİTİ
actual_source_label = "❓ Bilinmiyor"
if csv_loaded:
    actual_source_label = "📁 CSV (kullanıcı yükledi)"
elif data_source == "eodhd" and not prices.empty:
    hisse_count = sum(1 for t in DEFAULT_UNIVERSE if t in prices.columns)
    if hisse_count >= 15 and len(prices) >= 200:
        actual_source_label = "🔷 EODHD (gerçek)"
    else:
        actual_source_label = "⚠️ Sentetik (EODHD fallback)"
elif data_source == "yfinance" and not prices.empty:
    hisse_count = sum(1 for t in DEFAULT_UNIVERSE if t in prices.columns)
    if hisse_count >= 15:
        actual_source_label = "🟢 yfinance"
    else:
        actual_source_label = "⚠️ Sentetik (yfinance fallback)"
elif data_source.startswith("sentetik"):
    actual_source_label = "🔵 Sentetik (manuel)"
else:
    actual_source_label = "⚠️ Sentetik (fallback)"

avg_corr = avg_correlation(
    prices[[c for c in DEFAULT_UNIVERSE if c in prices.columns]])

# Temel veri: KAP → yfinance → sentetik
fundamentals = None
fund_synthetic = False
kap_ok = False

if use_kap:
    kap_df = load_fundamentals_kap(DEFAULT_UNIVERSE)
    if not kap_df.empty:
        kap_df["ticker"] = kap_df["ticker"].astype(str)
        kap_df = kap_df[kap_df["ticker"].isin(DEFAULT_UNIVERSE)]
        if not kap_df.empty:
            sent_df, _ = load_fundamentals(DEFAULT_UNIVERSE, mode="synthetic")
            merged = kap_df.merge(
                sent_df[["ticker", "market_cap", "debt_to_ebitda",
                         "current_ratio", "profit_margin", "total_debt",
                         "total_assets", "ebitda", "interest_expense"]],
                on="ticker", how="left", suffixes=("", "_syn"))
            for col in ["market_cap", "debt_to_ebitda", "current_ratio",
                        "profit_margin", "total_debt", "total_assets",
                        "ebitda", "interest_expense"]:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(
                        merged.get(f"{col}_syn", np.nan))
                    if f"{col}_syn" in merged.columns:
                        merged.drop(columns=[f"{col}_syn"], inplace=True)
            fundamentals = merged
            fundamentals["_synthetic"] = False
            kap_ok = True
            st.success(f"✅ KAP SDK: {len(kap_df)} hisse temel veri.")

if fundamentals is None:
    fundamentals, fund_synthetic = load_fundamentals(
        DEFAULT_UNIVERSE, mode="yfinance")

# ─────────────────────────────────────────────────────────────
# BAŞLIK
# ─────────────────────────────────────────────────────────────
st.title("📊 BIST Katılım Fon Yönetim Sistemi")
st.caption(
    f"Benchmark: **{benchmark_key}** | "
    f"Rebalance: **{rebalance_freq}** | "
    f"Rejim: **{n_regimes}** | "
    f"Evren: **{len(DEFAULT_UNIVERSE)}** hisse | "
    f"Veri: **{prices.index.min().date()} → {prices.index.max().date()}**"
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info(f"Fiyat: **{actual_source_label}**")
with col2:
    if kap_ok:
        st.success("🔌 KAP SDK")
    elif fund_synthetic:
        st.info("ℹ️ Temel: sentetik")
    else:
        st.success("✅ Temel: yfinance")
with col3:
    if not np.isnan(avg_corr):
        sev = "🟢" if avg_corr < 0.4 else "🟡" if avg_corr < 0.6 else "🔴"
        st.info(f"{sev} Korelasyon: **{avg_corr:.2f}**")
with col4:
    if l2_reg > 0 or mu_shrink > 0:
        st.warning(f"⚙️ L2={l2_reg:.2f}, μ={mu_shrink:.2f}")
    else:
        st.info("⚙️ Reg: kapalı")

# ─────────────────────────────────────────────────────────────
# SEKMELER
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔄 Rejim Analizi", "🚩 Kırmızı Bayraklar",
    "📈 Backtest", "⚖️ Optimizasyon",
])

# ═══════════════ SEKME 1 — REJİM ═══════════════
with tab1:
    st.header("Piyasa Rejimi Analizi")
    if bm_ticker in returns.columns:
        bm_ret = returns[bm_ticker].dropna()
    else:
        bm_ret = returns.mean(axis=1).dropna()

    if len(bm_ret) < 120:
        st.warning(f"Yetersiz veri ({len(bm_ret)} gün).")
        regime_probs, regime_summary, bull_regime = None, None, None
    else:
        with st.spinner("Rejim modeli eğitiliyor..."):
            regime_probs, regime_summary, bull_regime = detect_regimes(
                bm_ret, n_regimes=n_regimes, lookback=regime_lookback)

        if regime_probs is None:
            st.error("Rejim modeli eğitilemedi.")
        else:
            last = regime_probs.iloc[-1]
            cur_reg = int(last.idxmax().split("_")[1])
            cur_prob = float(last.max())
            cur_label = regime_summary.loc[
                regime_summary["Rejim"] == cur_reg, "Etiket"
            ].values
            cur_label = cur_label[0] if len(cur_label) > 0 else "Bilinmiyor"

            c1, c2, c3 = st.columns(3)
            c1.metric("Güncel Rejim", cur_label)
            c2.metric("Olasılık", f"{cur_prob:.1%}")
            if len(regime_summary) > 0:
                min_vol_row = regime_summary.loc[
                    regime_summary["Yıllık Oynaklık"].idxmin()]
                c3.metric("En Sakin Rejim Vol'ü",
                          f"{min_vol_row['Yıllık Oynaklık']:.1%}")

            st.subheader("Rejim Olasılık Grafiği")
            fig = go.Figure()
            for col in regime_probs.columns:
                r_num = int(col.split("_")[1])
                lbl = regime_summary.loc[
                    regime_summary["Rejim"] == r_num, "Etiket"
                ].values
                lbl = lbl[0] if len(lbl) > 0 else col
                fig.add_trace(go.Scatter(
                    x=regime_probs.index, y=regime_probs[col],
                    name=f"{col} ({lbl})", stackgroup="one", mode="lines"))
            fig.update_layout(height=380, yaxis_range=[0, 1],
                              xaxis_title="Tarih", yaxis_title="Olasılık")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Rejim İstatistikleri")
            disp = regime_summary.copy()
            disp["Ort. Günlük Getiri"] = disp["Ort. Günlük Getiri"].apply(
                lambda x: f"{x:.4%}" if pd.notna(x) else "—")
            disp["Yıllık Oynaklık"] = disp["Yıllık Oynaklık"].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "—")
            if "Gözlem Sayısı" in disp.columns:
                disp["Gözlem Sayısı"] = disp["Gözlem Sayısı"].apply(
                    lambda x: f"{int(x)}" if pd.notna(x) else "—")
            if "Gözlem Payı" in disp.columns:
                disp["Gözlem Payı"] = disp["Gözlem Payı"].apply(
                    lambda x: f"{x:.1%}" if pd.notna(x) else "—")
            st.dataframe(disp, use_container_width=True, hide_index=True)

            st.caption("ℹ️ Etiketler: Boğa / Ayı / Volatil / Sıkışma.")

            st.subheader("Fiyat + Rejim Gölgelendirmesi")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=prices.index, y=prices[bm_ticker],
                name=benchmark_key, line=dict(color="royalblue")))
            if len(regime_summary) > 0:
                high_vol_reg = regime_summary.loc[
                    regime_summary["Yıllık Oynaklık"].idxmax(), "Rejim"]
                col = f"regime_{int(high_vol_reg)}"
                if col in regime_probs.columns:
                    in_high = regime_probs[col] > 0.5
                    trans = in_high.astype(int).diff().fillna(0)
                    starts = regime_probs.index[trans == 1]
                    ends = regime_probs.index[trans == -1]
                    for s, e in zip(starts, ends):
                        fig2.add_vrect(x0=s, x1=e, fillcolor="red",
                                       opacity=0.12, layer="below",
                                       line_width=0)
            fig2.update_layout(height=380, xaxis_title="Tarih",
                               yaxis_title="Fiyat")
            st.plotly_chart(fig2, use_container_width=True)

# ═══════════════ SEKME 2 — KIRMIZI BAYRAKLAR ═══════════════
with tab2:
    st.header("Kırmızı Bayrak Analizi")

    if kap_ok:
        st.success("🔌 Kaynak: **KAP SDK** (gerçek finansal veri)")
    elif fund_synthetic:
        st.info("ℹ️ Kaynak: **Sentetik demo verisi**.")
    else:
        st.success("✅ Kaynak: **yfinance**")

    th = dict(RED_FLAG_THRESHOLDS)
    th["max_flags_for_inclusion"] = red_flag_threshold
    flags_df = evaluate_red_flags(fundamentals, th)

    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam Hisse", len(flags_df))
    c2.metric("Kırmızı Bayraklı", int((flags_df["total_flags"] > 0).sum()))
    c3.metric("Portföyden Çıkarılacak",
              int(flags_df["exclude_candidate"].sum()))

    flag_cols = [c for c in flags_df.columns if c.startswith("flag_")]
    if flag_cols:
        mat = flags_df.set_index("ticker")[flag_cols].copy()
        mat.columns = [c.replace("flag_", "").upper() for c in mat.columns]
        fig = px.imshow(mat, aspect="auto",
                        color_continuous_scale=["#2ecc71", "#e74c3c"],
                        labels=dict(color="Bayrak"))
        fig.update_layout(height=max(300, len(mat) * 22),
                          title="Kırmızı Bayrak Matrisi")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detaylı Tablo")
    show_cols = [c for c in ["ticker", "sector", "market_cap",
                             "debt_to_ebitda", "current_ratio",
                             "profit_margin", "total_flags",
                             "exclude_candidate"] if c in flags_df.columns]
    disp_flags = flags_df[show_cols].copy()
    for col, fmt in [("market_cap", "{:,.0f}"),
                     ("debt_to_ebitda", "{:.2f}"),
                     ("current_ratio", "{:.2f}"),
                     ("profit_margin", "{:.2%}")]:
        if col in disp_flags.columns:
            disp_flags[col] = disp_flags[col].apply(
                lambda x, f=fmt: f.format(x) if pd.notna(x) else "—")
    st.dataframe(disp_flags, use_container_width=True, hide_index=True)

# ═══════════════ SEKME 3 — BACKTEST ═══════════════
with tab3:
    st.header("Backtest & Portföy Performansı")

    if use_wf:
        st.info(f"🔄 **Walk-Forward aktif** — "
                f"Eğitim: {wf_train}g, Test: {wf_test}g, Adım: {wf_step}g")

    try:
        th_bt = dict(RED_FLAG_THRESHOLDS)
        th_bt["max_flags_for_inclusion"] = red_flag_threshold
        fdf = evaluate_red_flags(fundamentals, th_bt)
        excluded = set(fdf.loc[fdf["exclude_candidate"], "ticker"].tolist())
    except Exception:
        excluded = set()

    universe = [t for t in DEFAULT_UNIVERSE
                if t in prices.columns and t not in excluded]
    if len(universe) < 2:
        universe = [t for t in DEFAULT_UNIVERSE if t in prices.columns]
    price_subset = ensure_datetime_index(prices[universe].copy())

    if use_wf:
        with st.spinner("Walk-Forward çalışıyor..."):
            oos_series, wf_windows = walk_forward_backtest(
                price_subset, universe,
                regime_probs=regime_probs, bull_regime=bull_regime,
                train_days=wf_train, test_days=wf_test, step_days=wf_step,
                rebalance_freq=rebalance_freq, max_weight=max_weight,
                l2_reg=l2_reg, mu_shrink=mu_shrink,
                commission=commission, slippage=slippage,
                stop_loss_pct=stop_loss_pct)

        if oos_series is None or len(oos_series) < 2:
            st.error("Walk-Forward sonuç üretilemedi.")
        else:
            pv = oos_series
            rets = pv.pct_change().dropna()
            n = len(rets)
            total = pv.iloc[-1] / pv.iloc[0] - 1
            cagr = (1 + total) ** (252 / n) - 1 if n > 0 else 0
            vol = rets.std() * np.sqrt(252)
            sharpe = (rets.mean() / rets.std() * np.sqrt(252)
                      if rets.std() > 0 else 0)
            downside = rets[rets < 0]
            dd_dev = (np.sqrt((downside ** 2).mean()) * np.sqrt(252)
                      if len(downside) > 1 else 0)
            sortino = rets.mean() * 252 / dd_dev if dd_dev > 0 else 0
            cummax = pv.cummax()
            max_dd = ((pv - cummax) / cummax).min()

            wf_df = pd.DataFrame(wf_windows)
            is_sharpe = (wf_df["test_sharpe"].mean()
                         if not wf_df.empty else 0)

            st.subheader("📊 In-Sample vs Out-of-Sample")
            comp1, comp2, comp3 = st.columns(3)
            comp1.metric("IS Sharpe (ort.)", f"{is_sharpe:.2f}")
            comp2.metric("OOS Sharpe", f"{sharpe:.2f}",
                         delta=f"{sharpe - is_sharpe:.2f}")
            ratio = (sharpe / is_sharpe) if is_sharpe != 0 else 0
            comp3.metric("OOS/IS Oranı", f"{ratio:.2f}")

            if ratio < 0.5 and is_sharpe > 0:
                st.error("🚨 Ciddi overfitting şüphesi.")
            elif ratio < 0.7 and is_sharpe > 0:
                st.warning("⚠️ Ilımlı overfitting.")
            else:
                st.success("✅ Tutarlı performans.")

            st.subheader("Out-of-Sample Metrikler")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Toplam Getiri", f"{total:.2%}")
            c2.metric("CAGR", f"{cagr:.2%}")
            c3.metric("Sharpe", f"{sharpe:.2f}")
            c4.metric("Max Drawdown", f"{max_dd:.2%}")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Sortino", f"{sortino:.2f}")
            c6.metric("Oynaklık", f"{vol:.2%}")
            c7.metric("OOS Gün", f"{n}")
            c8.metric("Pencere Sayısı", f"{len(wf_windows)}")

            st.subheader("Walk-Forward OOS Equity Curve")
            fig_wf = go.Figure()
            fig_wf.add_trace(go.Scatter(
                x=pv.index, y=pv / pv.iloc[0],
                name="OOS Portföy", line=dict(color="royalblue", width=2)))
            if benchmark_series is not None and \
               not benchmark_series.dropna().empty:
                bm_aligned = benchmark_series.reindex(
                    pv.index).ffill().dropna()
                if not bm_aligned.empty:
                    bm_norm = bm_aligned / bm_aligned.iloc[0]
                    fig_wf.add_trace(go.Scatter(
                        x=bm_norm.index, y=bm_norm, name=benchmark_key,
                        line=dict(color="orange", width=2, dash="dash")))
            fig_wf.update_layout(height=400, xaxis_title="Tarih",
                                 yaxis_title="Normalize")
            st.plotly_chart(fig_wf, use_container_width=True)

            with st.expander("Walk-Forward Pencere Detayları"):
                if not wf_df.empty:
                    disp_wf = wf_df.copy()
                    disp_wf["test_return"] = disp_wf["test_return"].apply(
                        lambda x: f"{x:.2%}")
                    disp_wf["test_sharpe"] = disp_wf["test_sharpe"].apply(
                        lambda x: f"{x:.2f}")
                    disp_wf["test_max_dd"] = disp_wf["test_max_dd"].apply(
                        lambda x: f"{x:.2%}")
                    disp_wf["test_vol"] = disp_wf["test_vol"].apply(
                        lambda x: f"{x:.2%}")
                    st.dataframe(disp_wf, use_container_width=True,
                                 hide_index=True)
    else:
        with st.spinner("Backtest çalışıyor..."):
            schedule = build_weight_schedule(
                price_subset, universe, regime_probs=regime_probs,
                bull_regime=bull_regime, rebalance_freq=rebalance_freq,
                max_weight=max_weight, l2_reg=l2_reg, mu_shrink=mu_shrink)
            engine = BacktestEngine(
                price_subset, benchmark=benchmark_series,
                config={"commission": commission, "slippage": slippage,
                        "stop_loss_pct": stop_loss_pct,
                        "stop_loss_check_freq": 5})
            pv = engine.run(schedule)
            m = engine.metrics()

        if not m:
            st.error("Backtest sonucu üretilemedi.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Toplam Getiri", f"{m['Toplam Getiri']:.2%}")
            c2.metric("CAGR", f"{m['CAGR']:.2%}")
            c3.metric("Sharpe", f"{m['Sharpe']:.2f}")
            c4.metric("Max Drawdown", f"{m['Max Drawdown']:.2%}")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Sortino", f"{m['Sortino']:.2f}")
            c6.metric("Calmar", f"{m['Calmar']:.2f}")
            c7.metric("Kazanma Oranı", f"{m['Kazanma Oranı']:.1%}")
            c8.metric("VaR (95%)", f"{m['VaR (95%)']:.2%}")

            infl = annual_inflation / 100.0
            real_cagr = (1 + m["CAGR"]) / (1 + infl) - 1
            st.markdown("### 💰 Reel Getiri")
            rc1, rc2 = st.columns(2)
            rc1.metric("Nominal CAGR", f"{m['CAGR']:.2%}")
            rc2.metric(f"Reel CAGR (Enf %{annual_inflation:.0f})",
                       f"{real_cagr:.2%}",
                       delta=f"{real_cagr - m['CAGR']:.2%}")

            if real_cagr < 0:
                st.error(f"🚨 Reel getiri negatif (%{real_cagr*100:.2f}).")
            elif real_cagr < 0.05:
                st.warning(f"⚠️ Reel getiri düşük.")
            else:
                st.success(f"✅ Reel getiri makul.")

            st.subheader("Kümülatif Getiri")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pv.index, y=pv / pv.iloc[0],
                                     name="Portföy",
                                     line=dict(color="royalblue", width=2)))
            if benchmark_series is not None and \
               not benchmark_series.dropna().empty:
                bm_aligned = benchmark_series.reindex(
                    pv.index).ffill().dropna()
                if not bm_aligned.empty:
                    bm_norm = bm_aligned / bm_aligned.iloc[0]
                    fig.add_trace(go.Scatter(
                        x=bm_norm.index, y=bm_norm, name=benchmark_key,
                        line=dict(color="orange", width=2, dash="dash")))
            fig.update_layout(height=400, xaxis_title="Tarih",
                              yaxis_title="Normalize")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Drawdown")
            dd = engine.drawdown_series()
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy",
                                        name="Drawdown",
                                        line=dict(color="red")))
            fig_dd.update_layout(height=280, xaxis_title="Tarih",
                                 yaxis_title="Drawdown",
                                 yaxis_tickformat=".1%")
            st.plotly_chart(fig_dd, use_container_width=True)

            st.subheader("Aylık Getiri Isı Haritası")
            mr = engine.monthly_returns()
            if not mr.empty:
                fig_m = px.imshow(mr, aspect="auto",
                                  color_continuous_scale="RdYlGn",
                                  labels=dict(x="Ay", y="Yıl",
                                              color="Getiri"))
                fig_m.update_layout(height=max(250, len(mr) * 28))
                st.plotly_chart(fig_m, use_container_width=True)

            with st.expander("Rebalance Geçmişi"):
                if engine.weights_history:
                    wh = pd.DataFrame(engine.weights_history)
                    wh["turnover"] = wh["turnover"].apply(
                        lambda x: f"{x:.2%}")
                    wh["cost"] = wh["cost"].apply(lambda x: f"{x:.4%}")
                    st.dataframe(wh, use_container_width=True,
                                 hide_index=True)
            with st.expander("Stop-Loss Olayları"):
                if engine.stop_loss_events:
                    sl_df = pd.DataFrame(engine.stop_loss_events)
                    sl_df["removed"] = sl_df["removed"].apply(
                        lambda x: ", ".join(x))
                    st.dataframe(sl_df, use_container_width=True,
                                 hide_index=True)
                else:
                    st.write("Stop-loss tetiklenmedi.")

# ═══════════════ SEKME 4 — OPTİMİZASYON ═══════════════
with tab4:
    st.header("Portföy Optimizasyonu")
    label, prob = "Geçiş", 0.5
    if regime_probs is not None and bull_regime is not None:
        last = regime_probs.iloc[-1]
        reg = int(last.idxmax().split("_")[1])
        prob = float(last.max())
        lbl = regime_summary.loc[
            regime_summary["Rejim"] == reg, "Etiket"
        ].values
        label = lbl[0] if len(lbl) > 0 else "Bilinmiyor"
    st.markdown(f"**Güncel Rejim:** {label} (olasılık {prob:.1%})")

    if l2_reg > 0 or mu_shrink > 0:
        st.info(f"⚙️ Regularization: L2={l2_reg:.2f}, μ={mu_shrink:.2f}")

    try:
        th_op = dict(RED_FLAG_THRESHOLDS)
        th_op["max_flags_for_inclusion"] = red_flag_threshold
        fdf = evaluate_red_flags(fundamentals, th_op)
        excluded_op = set(
            fdf.loc[fdf["exclude_candidate"], "ticker"].tolist())
    except Exception:
        excluded_op = set()
    universe_op = [t for t in DEFAULT_UNIVERSE
                   if t in prices.columns and t not in excluded_op]
    if len(universe_op) < 2:
        universe_op = [t for t in DEFAULT_UNIVERSE if t in prices.columns]
    price_subset_op = ensure_datetime_index(prices[universe_op].copy())

    with st.spinner("Optimizasyon çalışıyor..."):
        opt = optimize_portfolio(
            price_subset_op, label, prob,
            max_weight=max_weight, rf=0.0,
            l2_reg=l2_reg, mu_shrink=mu_shrink)
        ef_pts = efficient_frontier_points(price_subset_op, max_weight, 0.0)

    if not opt["weights"]:
        st.error("Optimizasyon başarısız.")
    else:
        st.success(f"Yöntem: **{opt['method']}**")
        perf = opt["performance"]
        if perf and "expected_return" in perf:
            c1, c2, c3 = st.columns(3)
            c1.metric("Beklenen Getiri", f"{perf['expected_return']:.2%}")
            c2.metric("Oynaklık", f"{perf['volatility']:.2%}")
            c3.metric("Sharpe", f"{perf['sharpe']:.2f}")

            if perf["sharpe"] > 2.5:
                reasons = []
                if l2_reg < 0.2:
                    reasons.append("L2 cezası düşük (0.3 önerilir)")
                if mu_shrink < 0.2:
                    reasons.append("μ shrinkage yok (0.3 önerilir)")
                if not np.isnan(avg_corr) and avg_corr < 0.5:
                    reasons.append(
                        f"Korelasyon düşük ({avg_corr:.2f})")
                st.warning(
                    f"⚠️ **Sharpe = {perf['sharpe']:.2f}** yüksek.\n\n" +
                    "\n".join([f"- {r}" for r in reasons])
                )

            s = perf["sharpe"]
            if s > 3.0:
                rec_l2, rec_mu = 0.5, 0.5
            elif s > 2.5:
                rec_l2, rec_mu = 0.3, 0.3
            elif s > 2.0:
                rec_l2, rec_mu = 0.15, 0.15
            else:
                rec_l2, rec_mu = 0.0, 0.0
            if rec_l2 > 0 and (l2_reg < rec_l2 or mu_shrink < rec_mu):
                st.info(
                    f"💡 **Öneri:** L2 = {rec_l2:.2f}, "
                    f"μ shrink = {rec_mu:.2f} deneyin."
                )

        st.caption("Geometrik ortalama + Ledoit-Wolf shrinkage.")

        w_df = pd.DataFrame(
            [(k, v) for k, v in opt["weights"].items() if v > 0],
            columns=["Hisse", "Ağırlık"]
        ).sort_values("Ağırlık", ascending=False)
        if not w_df.empty:
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.subheader("Optimal Ağırlıklar")
                disp_w = w_df.copy()
                disp_w["Ağırlık"] = disp_w["Ağırlık"].apply(
                    lambda x: f"{x:.2%}")
                st.dataframe(disp_w, use_container_width=True,
                             hide_index=True)
                nakit = max(0.0, 1 - w_df["Ağırlık"].sum())
                st.metric("Nakit/Ağırlık Dışı", f"{nakit:.2%}")
            with col_b:
                fig = px.pie(w_df, names="Hisse", values="Ağırlık",
                             title="Portföy Dağılımı")
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)

        if not ef_pts.empty:
            st.subheader("Efficient Frontier")
            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(
                x=ef_pts["volatility"], y=ef_pts["return"],
                mode="markers+lines", name="Frontier",
                line=dict(color="royalblue")))
            if perf and "expected_return" in perf:
                fig_ef.add_trace(go.Scatter(
                    x=[perf["volatility"]], y=[perf["expected_return"]],
                    mode="markers", name="Seçilen Portföy",
                    marker=dict(color="red", size=14, symbol="star")))
            fig_ef.update_layout(height=420, xaxis_title="Oynaklık",
                                 yaxis_title="Beklenen Getiri",
                                 xaxis_tickformat=".1%",
                                 yaxis_tickformat=".1%")
            st.plotly_chart(fig_ef, use_container_width=True)

        with st.expander("📈 Hisse Korelasyon Matrisi (tanı)"):
            log_ret = np.log(
                price_subset_op / price_subset_op.shift(1)).dropna()
            if len(log_ret) >= 30:
                corr = log_ret.tail(252).corr()
                if not corr.empty:
                    vals = corr.values[np.triu_indices_from(
                        corr.values, 1)]
                    mean_corr = float(np.nanmean(vals))
                    fig_corr = px.imshow(
                        corr, aspect="auto",
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1,
                        labels=dict(color="Korelasyon"))
                    fig_corr.update_layout(
                        height=max(400, len(corr) * 20),
                        title=f"Korelasyon — Ortalama: {mean_corr:.2f}")
                    st.plotly_chart(fig_corr, use_container_width=True)

                    if mean_corr < 0.3:
                        st.error(f"🚨 Korelasyon **{mean_corr:.2f}** çok "
                                 f"düşük. CSV veya farklı veri deneyin.")
                    elif mean_corr < 0.5:
                        st.warning(f"⚠️ Korelasyon **{mean_corr:.2f}** — "
                                   f"BIST'ten düşük.")
                    else:
                        st.success(f"✅ Korelasyon **{mean_corr:.2f}** "
                                   f"— gerçekçi.")

st.markdown("---")
st.caption("Bu uygulama eğitim/demo amaçlıdır.")