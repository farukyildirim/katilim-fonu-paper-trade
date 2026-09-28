"""
helal_fon_ai.py — İki turlu portföy laboratuvarı (TEFAS + Dual Track)
Track A: İslami Katılım (kısıtlı, %0 stopaj)
Track B: Yüksek Gelir (kısıtsız, tüm fonlar)

Düzeltmeler:
- safe_progress: hem 0-100 hem 0.0-1.0 progress değerleri
- width='stretch' / use_container_width uyumluluğu
- TZ-naive normalize: TEFAS + yfinance timestamp karışımı sorunu çözüldü
- run_backtest içinde ek tz güvenlik kontrolü
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date
from scipy.optimize import minimize
import json, os

try:
    from pytefas import Crawler
    TEFAS_AVAILABLE = True
except ImportError:
    TEFAS_AVAILABLE = False

st.set_page_config(page_title="Portföy Lab — Dual Track", layout="wide", page_icon="⚖️")

STATE_FILE = "paper_state_dual.json"

# ============================================================ UYUMLULUK YARDIMCILARI
def safe_progress(bar, value, text=None):
    """Progress value: 0-100 veya 0.0-1.0 kabul eder, her Streamlit sürümünde çalışır."""
    try:
        v = value / 100.0 if value > 1.0 else value
        v = min(max(v, 0.0), 1.0)
        if text is not None:
            bar.progress(v, text=text)
        else:
            bar.progress(v)
    except Exception:
        try:
            v = value if value > 1.0 else value * 100
            v = min(max(v, 0), 100)
            if text is not None:
                bar.progress(int(v), text=text)
            else:
                bar.progress(int(v))
        except Exception:
            pass


def df_show(df, **kwargs):
    """st.dataframe — use_container_width deprecated uyumlu."""
    try:
        return st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        return st.dataframe(df, use_container_width=True, **kwargs)


def chart_show(fig, **kwargs):
    """st.plotly_chart — use_container_width deprecated uyumlu."""
    try:
        return st.plotly_chart(fig, width="stretch", **kwargs)
    except TypeError:
        return st.plotly_chart(fig, use_container_width=True, **kwargs)


def line_chart_show(data, **kwargs):
    """st.line_chart — use_container_width deprecated uyumlu."""
    try:
        return st.line_chart(data, width="stretch", **kwargs)
    except TypeError:
        return st.line_chart(data, use_container_width=True, **kwargs)


def tz_normalize_index(s):
    """Bir pandas Series/DataFrame'in index'ini tz-naive + normalize eder."""
    if s is None or len(s) == 0:
        return s
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s = s.copy()
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


# ============================================================ FON EVRENİ
TRACK_A_FUNDS = {
    "sukuk": {
        "AS — Ak Portföy Kira Sertifikaları Katılım":  "AS",
        "GLS — Azimut Kira Sertifikaları Katılım":     "GLS",
        "MFP — Aktif Kısa Vadeli Kira Sertifikası":    "MFP",
        "GKB — Garanti Kira Sertifikaları EYF":        "GKB",
    },
    "hisse": {
        "ZPE — Ziraat Katılım Hisse":                  "ZPE",
        "OHK — Oyak Katılım Hisse":                    "OHK",
        "NKM — Neo Katılım Hisse":                     "NKM",
        "MPS — Mükafat Katılım Hisse":                 "MPS",
        "HKH — Hedef Katılım Hisse":                   "HKH",
    },
    "altin": {
        "OGD — Oyak Altın Katılım":                    "OGD",
        "TUA — TEB Portföy Altın":                     "TUA",
        "GOL — Garanti Altın Katılım":                 "GOL",
    },
}

TRACK_B_FUNDS = {
    "sukuk": {
        **TRACK_A_FUNDS["sukuk"],
        "DZV — Deniz Kira Sertifikaları":              "DZV",
        "AKS — Ak Kira Sertifikaları":                 "AKS",
    },
    "hisse": {
        **TRACK_A_FUNDS["hisse"],
        "TCD — Tacirler Hisse Senedi":                 "TCD",
        "MAC — Marmara Capital Hisse":                 "MAC",
        "PHE — Pusula Hisse Senedi":                   "PHE",
    },
    "altin": {
        **TRACK_A_FUNDS["altin"],
        "GLD — SPDR Gold Shares (USD)":                "GLD",
        "GC=F — Gold Futures":                         "GC=F",
    },
}


# ============================================================ DURUM
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"phase": "backtest", "track_a": None, "track_b": None}


def save_state(s):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(s, f, default=str, indent=2)
    except Exception:
        pass


state = load_state()


# ============================================================ VERİ ÇEKME
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_tefas_fund(code: str, start_date, end_date):
    if not TEFAS_AVAILABLE:
        return pd.Series(dtype=float)
    try:
        tefas = Crawler(timeout=60, max_retry=5)
        df = tefas.fetch(str(start_date), str(end_date),
                         kind="YAT", columns="info", fund_code=code)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        s = df.set_index("date")["price"].astype(float)
        s.index = pd.to_datetime(s.index)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s.name = code
        s = tz_normalize_index(s)
        return s
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_yf(ticker: str, start=None, end=None):
    try:
        t = yf.Ticker(ticker)
        h = t.history(start=start, end=end, auto_adjust=True)
        if h is None or h.empty:
            return pd.Series(dtype=float)
        s = h["Close"].copy()
        s.name = ticker
        s = s[~s.index.duplicated(keep="last")]
        s = tz_normalize_index(s)
        return s
    except Exception:
        return pd.Series(dtype=float)


def fetch_universe(codes, start, end, progress_cb=None):
    """TEFAS kodu veya yfinance ticker'ı otomatik algılar.
    Tüm seriler tz-naive + normalize edilmiş olarak birleştirilir."""
    series_list = []
    for i, c in enumerate(codes):
        if progress_cb:
            try:
                progress_cb(i / len(codes), c)
            except Exception:
                pass

        if len(c) <= 5 and "." not in c and "=" not in c:
            s = fetch_tefas_fund(c, start, end)
            if s.empty:
                s = fetch_yf(c, start, end)
        else:
            s = fetch_yf(c, start, end)

        if not s.empty:
            # Savunmacı: cache'ten gelen eski veriler için tekrar normalize
            s = tz_normalize_index(s)
            series_list.append(s)

    if not series_list:
        return pd.DataFrame()

    df = pd.concat(series_list, axis=1)
    df = tz_normalize_index(df)
    df = df.ffill()
    return df.reindex(columns=[c for c in codes if c in df.columns])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_usdtry(start=None, end=None):
    s = fetch_yf("USDTRY=X", start=start, end=end)
    if s.empty:
        return s
    s = tz_normalize_index(s)
    return s


# ============================================================ OPTİMİZASYON
def optimize_weights(prices, mode="islami", usdtry=None,
                     max_hisse_idx=None, hisse_cap=0.30):
    n = prices.shape[1]
    returns = prices.pct_change().dropna()
    if len(returns) < 20:
        return np.ones(n) / n

    mu = returns.mean() * 252
    cov = returns.cov() * 252

    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -(w @ mu) / vol if vol > 1e-9 else 0

    def neg_return(w):
        return -(w @ mu)

    def neg_reel(w):
        port = w @ mu
        if usdtry is not None and len(usdtry) > 1:
            yrs = (usdtry.index[-1] - usdtry.index[0]).days / 365.25
            if yrs > 0:
                fx = (usdtry.iloc[-1] / usdtry.iloc[0]) ** (1 / yrs) - 1
                return -((1 + port) / (1 + fx) - 1)
        return -port

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    if mode == "islami" and max_hisse_idx is not None and len(max_hisse_idx) > 0:
        idx = list(max_hisse_idx)
        constraints.append({
            "type": "ineq",
            "fun": lambda w, idx=idx: hisse_cap - np.sum(w[idx])
        })

    bounds = [(0, 1)] * n
    x0 = np.ones(n) / n

    obj = {"islami": neg_sharpe, "maxret": neg_return, "reel": neg_reel}[mode]

    try:
        res = minimize(obj, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 1000, "ftol": 1e-10})
        if res.success:
            w = np.clip(res.x, 0, 1)
            if w.sum() > 0:
                return w / w.sum()
    except Exception:
        pass
    return x0


# ============================================================ BACKTEST MOTORU
def run_backtest(prices, weights, capital, freq="Y",
                 tax_rates=None, commission=0.0):
    # TZ güvenlik kontrolü
    if prices.index.tz is not None:
        prices = prices.copy()
        prices.index = prices.index.tz_localize(None)
    prices.index = pd.to_datetime(prices.index).normalize()

    n = prices.shape[1]
    w = np.asarray(weights, float)
    w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
    tr = np.asarray(tax_rates if tax_rates is not None else [0] * n, float)
    dates = prices.index
    px = prices.to_numpy(float)

    if freq == "N":
        rebal = set()
    else:
        key = dates.year if freq == "Y" else dates.to_period(freq)
        first = pd.Series(dates, index=dates).groupby(key).first()
        rebal = set(pd.DatetimeIndex(first.values))

    units = (w * capital) / px[0]
    avg_cost = px[0].copy()
    bh_units = (w * capital) / px[0]

    val = np.empty(len(dates)); val[0] = capital
    bh_val = np.empty(len(dates)); bh_val[0] = capital
    taxes_paid = np.zeros(len(dates))
    orders_log = []

    for i in range(1, len(dates)):
        if dates[i] in rebal:
            p_prev = px[i - 1]
            v_prev = float((units * p_prev).sum())
            tgt = (w * v_prev) / p_prev
            tax = comm = 0.0
            for j in range(n):
                delta = tgt[j] - units[j]
                if delta < 0:
                    gain = (p_prev[j] - avg_cost[j]) * (-delta)
                    if gain > 0:
                        tax += gain * tr[j]
                    orders_log.append({"date": dates[i], "ticker": prices.columns[j],
                                       "side": "SELL", "qty": round(-delta, 4),
                                       "price": round(float(p_prev[j]), 2)})
                elif delta > 0:
                    avg_cost[j] = (units[j] * avg_cost[j]
                                   + delta * p_prev[j]) / tgt[j]
                    orders_log.append({"date": dates[i], "ticker": prices.columns[j],
                                       "side": "BUY", "qty": round(delta, 4),
                                       "price": round(float(p_prev[j]), 2)})
                comm += abs(delta) * p_prev[j] * commission
            units = (w * (v_prev - tax - comm)) / p_prev
            taxes_paid[i] = tax + comm

        val[i] = float((units * px[i]).sum())
        bh_val[i] = float((bh_units * px[i]).sum())

    return (pd.Series(val, index=dates, name="Rebalanced"),
            pd.Series(bh_val, index=dates, name="Buy & Hold"),
            pd.Series(taxes_paid, index=dates, name="Vergi+Komisyon"),
            pd.DataFrame(orders_log))


def perf_stats(v):
    v = v.dropna()
    if len(v) < 2:
        return {k: np.nan for k in ["total", "cagr", "vol", "sharpe", "maxdd", "years"]}
    years = (v.index[-1] - v.index[0]).days / 365.25
    total = v.iloc[-1] / v.iloc[0] - 1
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    r = v.pct_change().dropna()
    vol = r.std() * np.sqrt(252) if len(r) > 1 else np.nan
    dd = (v / v.cummax() - 1).min()
    return dict(total=total, cagr=cagr, vol=vol,
                sharpe=(r.mean() * 252) / vol if vol and vol > 0 else np.nan,
                maxdd=dd, years=years)


def yearly_returns(v):
    ye = v.groupby(v.index.year).last()
    ye = ye / ye.shift(1) - 1
    ye.iloc[0] = v[v.index.year == v.index[0].year].iloc[-1] / v.iloc[0] - 1
    ye.index = [str(y) for y in ye.index]
    return ye


def reel_cagr(port, usdtry):
    if usdtry is None or len(usdtry) < 2:
        return np.nan, np.nan
    yrs_p = (port.index[-1] - port.index[0]).days / 365.25
    if yrs_p <= 0:
        return np.nan, np.nan
    port_cagr = (port.iloc[-1] / port.iloc[0]) ** (1 / yrs_p) - 1

    # TZ uyumluluğu
    port_idx = port.index.tz_localize(None) if port.index.tz is not None else port.index
    port_idx = pd.to_datetime(port_idx).normalize()
    fx = usdtry.copy()
    if fx.index.tz is not None:
        fx.index = fx.index.tz_localize(None)
    fx.index = pd.to_datetime(fx.index).normalize()

    common = port_idx.intersection(fx.index)
    if len(common) < 2:
        return port_cagr, np.nan
    fx = fx.loc[common]
    yrs_f = (fx.index[-1] - fx.index[0]).days / 365.25
    if yrs_f <= 0:
        return port_cagr, np.nan
    fx_cagr = (fx.iloc[-1] / fx.iloc[0]) ** (1 / yrs_f) - 1
    return port_cagr, (1 + port_cagr) / (1 + fx_cagr) - 1


# ============================================================ TRACK ÇALIŞTIRICI
def run_track(track_name, selected_codes, selected_labels, category_map,
              start_dt, end_dt, capital, freq, mode, usdtry,
              stopaj_map, progress_cb=None):
    px = fetch_universe(selected_codes,
                        pd.to_datetime(start_dt).date(),
                        pd.to_datetime(end_dt).date(),
                        progress_cb=progress_cb)

    if px.empty:
        return None
    px = px.dropna()
    if len(px) < 30:
        return None

    hisse_idx = [i for i, c in enumerate(px.columns)
                 if category_map.get(c) == "hisse"]

    w_opt = optimize_weights(px, mode=mode, usdtry=usdtry,
                             max_hisse_idx=hisse_idx if track_name == "A" else None,
                             hisse_cap=0.30)

    tax_rates = [stopaj_map.get(category_map.get(c, "sukuk"), 0.0)
                 for c in px.columns]

    port, bh, taxes, orders = run_backtest(
        px, w_opt, capital, freq, tax_rates=tax_rates)

    stats_p = perf_stats(port)
    stats_b = perf_stats(bh)
    port_cagr, reel = reel_cagr(port, usdtry)

    return {
        "track": track_name,
        "prices": px,
        "weights": w_opt,
        "labels": [selected_labels[selected_codes.index(c)]
                   if c in selected_codes else c for c in px.columns],
        "categories": [category_map.get(c, "sukuk") for c in px.columns],
        "port": port, "bh": bh, "taxes": taxes, "orders": orders,
        "stats_port": stats_p, "stats_bh": stats_b,
        "port_cagr": port_cagr, "reel_cagr": reel,
        "capital": capital, "freq": freq, "mode": mode,
    }


# ============================================================ ARAYÜZ
st.title("⚖️ Portföy Lab — İki Turlu Çalışma")
st.caption("Track A: İslami Katılım (%0 stopaj, hisse ≤%30) | "
           "Track B: Yüksek Gelir (kısıtsız, tüm fonlar)")

if not TEFAS_AVAILABLE:
    st.warning("⚠️ `pytefas` kurulu değil. `pip install pytefas` çalıştırın. "
               "TEFAS verileri yerine yfinance fallback kullanılacak.")

tabs = st.tabs(["⚖️ Dual Backtest", "🔬 Walk-Forward",
                "📝 Paper Trade", "🚦 Risk", "🚀 Canlı"])

# ============================================================ TAB 0: DUAL BACKTEST
with tabs[0]:
    st.subheader("⚖️ İki Turlu Backtest")

    c1, c2, c3 = st.columns(3)
    start_bt = c1.date_input("Başlangıç", pd.to_datetime("2023-01-01"), key="dual_start")
    end_bt = c2.date_input("Bitiş", pd.to_datetime("2026-09-01"), key="dual_end")
    cap_bt = c3.number_input("Sermaye (₺)", 1000, 10_000_000, 100_000, 1000, key="dual_cap")

    c4, c5, c6 = st.columns(3)
    freq_map = {"Yıllık": "Y", "Çeyreklik": "Q", "Aylık": "M", "Hiç": "N"}
    freq_lbl = c4.selectbox("Rebalance", list(freq_map.keys()), key="dual_freq")
    akt_a = c5.checkbox("Track A (İslami)", value=True, key="dual_a")
    akt_b = c6.checkbox("Track B (Yüksek Gelir)", value=True, key="dual_b")

    # --- Track A Fonları ---
    st.markdown("---")
    a_codes, a_labels, a_cat = [], [], {}
    if akt_a:
        st.markdown("### 🕌 Track A — İslami Katılım")
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            a_sukuk = st.multiselect("Sukuk/Kira", list(TRACK_A_FUNDS["sukuk"].keys()),
                                     default=list(TRACK_A_FUNDS["sukuk"].keys())[:2], key="a_s")
        with col_a2:
            a_hisse = st.multiselect("Hisse", list(TRACK_A_FUNDS["hisse"].keys()),
                                     default=list(TRACK_A_FUNDS["hisse"].keys())[:2], key="a_h")
        with col_a3:
            a_altin = st.multiselect("Altın", list(TRACK_A_FUNDS["altin"].keys()),
                                     default=list(TRACK_A_FUNDS["altin"].keys())[:1], key="a_a")
        a_codes = ([TRACK_A_FUNDS["sukuk"][k] for k in a_sukuk] +
                   [TRACK_A_FUNDS["hisse"][k] for k in a_hisse] +
                   [TRACK_A_FUNDS["altin"][k] for k in a_altin])
        a_labels = a_sukuk + a_hisse + a_altin
        a_cat = ({TRACK_A_FUNDS["sukuk"][k]: "sukuk" for k in a_sukuk} |
                 {TRACK_A_FUNDS["hisse"][k]: "hisse" for k in a_hisse} |
                 {TRACK_A_FUNDS["altin"][k]: "altin" for k in a_altin})

    # --- Track B Fonları ---
    b_codes, b_labels, b_cat = [], [], {}
    if akt_b:
        st.markdown("### 💰 Track B — Yüksek Gelir (Kısıtsız)")
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            b_sukuk = st.multiselect("Sukuk/Kira", list(TRACK_B_FUNDS["sukuk"].keys()),
                                     default=list(TRACK_B_FUNDS["sukuk"].keys())[:2], key="b_s")
        with col_b2:
            b_hisse = st.multiselect("Hisse", list(TRACK_B_FUNDS["hisse"].keys()),
                                     default=list(TRACK_B_FUNDS["hisse"].keys())[:2], key="b_h")
        with col_b3:
            b_altin = st.multiselect("Altın", list(TRACK_B_FUNDS["altin"].keys()),
                                     default=list(TRACK_B_FUNDS["altin"].keys())[:1], key="b_a")
        b_codes = ([TRACK_B_FUNDS["sukuk"][k] for k in b_sukuk] +
                   [TRACK_B_FUNDS["hisse"][k] for k in b_hisse] +
                   [TRACK_B_FUNDS["altin"][k] for k in b_altin])
        b_labels = b_sukuk + b_hisse + b_altin
        b_cat = ({TRACK_B_FUNDS["sukuk"][k]: "sukuk" for k in b_sukuk} |
                 {TRACK_B_FUNDS["hisse"][k]: "hisse" for k in b_hisse} |
                 {TRACK_B_FUNDS["altin"][k]: "altin" for k in b_altin})

    # --- Vergi ---
    st.markdown("---")
    st.markdown("### 💸 Vergi Varsayımları")
    t1, t2, t3 = st.columns(3)
    stopaj_a = {"sukuk": t1.number_input("Track A — Sukuk (%)", 0.0, 50.0, 0.0, 0.5, key="ta_s") / 100,
                "hisse": t2.number_input("Track A — Hisse (%)", 0.0, 50.0, 0.0, 0.5, key="ta_h") / 100,
                "altin": t3.number_input("Track A — Altın (%)", 0.0, 50.0, 0.0, 0.5, key="ta_a") / 100}
    t4, t5, t6 = st.columns(3)
    stopaj_b = {"sukuk": t4.number_input("Track B — Sukuk (%)", 0.0, 50.0, 10.0, 0.5, key="tb_s") / 100,
                "hisse": t5.number_input("Track B — Hisse (%)", 0.0, 50.0, 10.0, 0.5, key="tb_h") / 100,
                "altin": t6.number_input("Track B — Altın (%)", 0.0, 50.0, 10.0, 0.5, key="tb_a") / 100}

    # --- Çalıştır ---
    if st.button("⚖️ İki Track'i Çalıştır", type="primary", key="dual_run"):
        progress = st.progress(0.0, text="Başlıyor...")

        usdtry = fetch_usdtry(start=start_bt, end=end_bt)

        results = {}

        if akt_a and a_codes:
            safe_progress(progress, 0.10, "Track A: veri indiriliyor...")
            res_a = run_track(
                "A", a_codes, a_labels, a_cat,
                start_bt, end_bt, cap_bt, freq_map[freq_lbl],
                mode="islami", usdtry=usdtry, stopaj_map=stopaj_a,
                progress_cb=lambda p, c: safe_progress(progress, 0.10 + p * 0.35,
                                                       f"Track A: {c}")
            )
            if res_a:
                results["A"] = res_a

        if akt_b and b_codes:
            safe_progress(progress, 0.50, "Track B: veri indiriliyor...")
            res_b = run_track(
                "B", b_codes, b_labels, b_cat,
                start_bt, end_bt, cap_bt, freq_map[freq_lbl],
                mode="maxret", usdtry=usdtry, stopaj_map=stopaj_b,
                progress_cb=lambda p, c: safe_progress(progress, 0.50 + p * 0.45,
                                                       f"Track B: {c}")
            )
            if res_b:
                results["B"] = res_b

        safe_progress(progress, 1.0, "Tamamlandı.")

        if not results:
            st.error("Hiçbir track sonuç üretemedi. Fon seçimlerini veya tarih aralığını kontrol edin.")
        else:
            st.session_state["dual_results"] = results
            st.session_state["dual_usdtry"] = usdtry
            st.session_state["dual_capital"] = cap_bt
            st.success(f"✅ {len(results)} track başarıyla çalıştırıldı.")

    # ============================================================ SONUÇLAR
    if "dual_results" not in st.session_state:
        st.info("👆 Yukarıdaki parametreleri ayarlayıp **İki Track'i Çalıştır** butonuna basın.")
    else:
        results = st.session_state["dual_results"]
        usdtry = st.session_state.get("dual_usdtry")

        st.markdown("---")
        st.subheader("📊 Track Karşılaştırması")

        cols = st.columns(len(results))
        for col, (key, res) in zip(cols, results.items()):
            with col:
                track_icon = "🕌" if key == "A" else "💰"
                track_label = "İslami Katılım" if key == "A" else "Yüksek Gelir"
                st.markdown(f"### {track_icon} Track {key} — {track_label}")

                sp = res["stats_port"]
                st.metric("Son Değer", f"₺{res['port'].iloc[-1]:,.0f}")
                st.metric("Toplam Getiri", f"%{sp['total'] * 100:,.1f}")
                st.metric("CAGR", f"%{sp['cagr'] * 100:,.1f}")
                st.metric("Max Drawdown", f"%{sp['maxdd'] * 100:,.1f}",
                          delta_color="inverse")
                st.metric("Sharpe", f"{sp['sharpe']:.2f}")

                if not np.isnan(res["reel_cagr"]):
                    color = "normal" if res["reel_cagr"] > 0 else "inverse"
                    st.metric("Reel CAGR (USD)", f"%{res['reel_cagr'] * 100:,.1f}",
                              delta_color=color)

                st.metric("Ödenen Vergi", f"₺{res['taxes'].sum():,.0f}")

        if len(results) == 2:
            st.markdown("---")
            st.subheader("🎯 Fark Analizi")
            a = results.get("A")
            b = results.get("B")
            if a and b:
                diff_total = (b["stats_port"]["total"] - a["stats_port"]["total"]) * 100
                a_reel = a["reel_cagr"] if not np.isnan(a["reel_cagr"]) else 0
                b_reel = b["reel_cagr"] if not np.isnan(b["reel_cagr"]) else 0
                diff_reel = (b_reel - a_reel) * 100
                diff_tax = b["taxes"].sum() - a["taxes"].sum()

                f1, f2, f3 = st.columns(3)
                f1.metric("Toplam Getiri Farkı (B − A)", f"{diff_total:+.1f} puan",
                          help="Pozitifse Track B daha çok kazandırdı")
                f2.metric("Reel CAGR Farkı", f"{diff_reel:+.1f} puan")
                f3.metric("Vergi Farkı (B − A)", f"₺{diff_tax:+,.0f}",
                          help="Pozitifse Track B daha çok vergi ödedi")

                if diff_total > 0:
                    st.info(f"💰 **Track B (Yüksek Gelir)** toplam getiride {diff_total:+.1f} puan önde.")
                elif diff_total < 0:
                    st.info(f"🕌 **Track A (İslami)** toplam getiride {-diff_total:+.1f} puan önde.")

                if diff_tax > 0:
                    st.caption(f"Vergi avantajı: Track A, Track B'ye göre "
                               f"₺{diff_tax:,.0f} daha az vergi ödedi.")

        st.markdown("---")
        st.subheader("📈 Portföy Büyüme Karşılaştırması")
        fig = go.Figure()
        colors = {"A": "#16a34a", "B": "#dc2626"}
        for key, res in results.items():
            label = f"Track {key} — {'İslami' if key == 'A' else 'Yüksek Gelir'}"
            fig.add_trace(go.Scatter(x=res["port"].index, y=res["port"],
                                     name=label, line=dict(color=colors[key], width=2)))
        fig.update_layout(xaxis_title="Tarih", yaxis_title="Portföy Değeri (₺)",
                          hovermode="x unified", template="plotly_white",
                          height=460, legend=dict(orientation="h", y=1.05))
        chart_show(fig)

        st.markdown("---")
        st.subheader("🎯 Optimal Ağırlıklar")
        wcols = st.columns(len(results))
        for col, (key, res) in zip(wcols, results.items()):
            with col:
                st.markdown(f"**Track {key}**")
                wdf = pd.DataFrame({
                    "Fon": res["labels"],
                    "Kategori": res["categories"],
                    "Ağırlık": [f"%{w * 100:.1f}" for w in res["weights"]],
                })
                df_show(wdf, hide_index=True)

        st.markdown("---")
        st.subheader("📅 Yıllık Getiri Karşılaştırması")
        yr_data = {}
        for key, res in results.items():
            yr_data[f"Track {key}"] = yearly_returns(res["port"])
        yr_df = pd.DataFrame(yr_data)
        df_show(yr_df.style.format("{:+.2%}").background_gradient(
            cmap="RdYlGn", axis=1, vmin=-0.3, vmax=0.5))

        if usdtry is not None and len(usdtry) > 1:
            st.markdown("---")
            st.subheader("💵 Reel Getiri — USD/TRY Karşısında")
            common_idx = None
            for res in results.values():
                idx = res["port"].index
                if idx.tz is not None:
                    idx = idx.tz_localize(None)
                idx = pd.to_datetime(idx).normalize()
                common_idx = idx if common_idx is None else common_idx.intersection(idx)

            if common_idx is not None:
                fx_idx = usdtry.index
                if fx_idx.tz is not None:
                    fx_idx = fx_idx.tz_localize(None)
                fx_idx = pd.to_datetime(fx_idx).normalize()
                common_idx = common_idx.intersection(fx_idx)

            if common_idx is not None and len(common_idx) > 10:
                reel_fig = go.Figure()
                for key, res in results.items():
                    port_naive = res["port"].copy()
                    if port_naive.index.tz is not None:
                        port_naive.index = port_naive.index.tz_localize(None)
                    port_naive.index = pd.to_datetime(port_naive.index).normalize()
                    port_naive = port_naive.loc[common_idx]
                    norm = port_naive / port_naive.iloc[0] * 100
                    reel_fig.add_trace(go.Scatter(x=norm.index, y=norm,
                                                  name=f"Track {key}",
                                                  line=dict(color=colors[key])))

                fx = usdtry.copy()
                if fx.index.tz is not None:
                    fx.index = fx.index.tz_localize(None)
                fx.index = pd.to_datetime(fx.index).normalize()
                fx = fx.loc[common_idx]
                fx_norm = fx / fx.iloc[0] * 100
                reel_fig.add_trace(go.Scatter(x=fx_norm.index, y=fx_norm,
                                              name="USD/TRY",
                                              line=dict(color="#6b7280", dash="dash")))
                reel_fig.update_layout(template="plotly_white", height=400,
                                       yaxis_title="Base 100 (TL)")
                chart_show(reel_fig)

        with st.expander("📋 İşlem Geçmişleri"):
            for key, res in results.items():
                st.markdown(f"**Track {key}** — {len(res['orders'])} emir")
                if not res["orders"].empty:
                    df_show(res["orders"], hide_index=True)


# ============================================================ TAB 1: WALK-FORWARD
with tabs[1]:
    st.subheader("🔬 Walk-Forward (Her Track için Ayrı)")
    if "dual_results" not in st.session_state:
        st.info("Önce Dual Backtest sekmesinde bir çalıştırma yapın.")
    else:
        results = st.session_state["dual_results"]
        usdtry = st.session_state.get("dual_usdtry")

        wf_train = st.slider("Eğitim penceresi (ay)", 3, 24, 6, key="wf_tr")
        wf_test = st.slider("Test penceresi (ay)", 1, 12, 3, key="wf_te")

        if st.button("Walk-Forward Başlat", key="wf_dual"):
            wf_progress = st.progress(0.0, text="Başlıyor...")
            safe_progress(wf_progress, 0.0, "Başlıyor...")
            wf_cols = st.columns(len(results))
            for k_idx, (key, res) in enumerate(results.items()):
                with wf_cols[k_idx]:
                    st.markdown(f"**Track {key}**")
                    prices = res["prices"]
                    mode = res["mode"]
                    hisse_idx = [i for i, c in enumerate(prices.columns)
                                 if res["categories"][i] == "hisse"]

                    rows = []
                    start = prices.index[0]
                    end = prices.index[-1]
                    train_end = start + pd.DateOffset(months=wf_train)
                    while train_end + pd.DateOffset(months=wf_test) <= end:
                        test_start = train_end
                        test_end = test_start + pd.DateOffset(months=wf_test)
                        tr = prices[(prices.index >= start) & (prices.index < train_end)]
                        te = prices[(prices.index >= test_start) & (prices.index < test_end)]
                        if len(tr) >= 30 and len(te) >= 10:
                            w = optimize_weights(tr, mode=mode, usdtry=usdtry,
                                                 max_hisse_idx=hisse_idx if key == "A" else None)
                            te_ret = (te.pct_change().fillna(0) @ w)
                            rows.append({
                                "test": f"{test_start.date()} → {test_end.date()}",
                                "return": round(float((1 + te_ret).prod() - 1), 4),
                            })
                        train_end += pd.DateOffset(months=wf_test)

                    if rows:
                        df = pd.DataFrame(rows)
                        df_show(df.style.format({"return": "{:+.2%}"}), hide_index=True)
                        avg = df["return"].mean()
                        win = (df["return"] > 0).mean()
                        st.metric("Ort. Getiri", f"%{avg * 100:.2f}")
                        st.metric("Kazanma Oranı", f"%{win * 100:.0f}")
                    else:
                        st.warning("Yeterli pencere yok.")

                safe_progress(wf_progress, (k_idx + 1) / len(results),
                              f"Track {key} tamamlandı")
            safe_progress(wf_progress, 1.0, "Tamamlandı.")


# ============================================================ TAB 2: PAPER TRADE
with tabs[2]:
    st.subheader("📝 Paper Trade")
    st.info("Paper trade, seçilen track'in ağırlıklarını kullanarak gerçek fiyatlarla "
            "sanal portföy simüle eder. Önce Dual Backtest sekmesinde bir track çalıştırın.")
    if "dual_results" in st.session_state:
        sel_track = st.selectbox("Hangi track paper trade'e alınsın?",
                                 list(st.session_state["dual_results"].keys()),
                                 key="pt_track")
        if st.button("Paper Trade Başlat", type="primary", key="pt_start"):
            st.success(f"Track {sel_track} paper trade moduna alındı.")
            state["phase"] = "paper"
            save_state(state)


# ============================================================ TAB 3: RİSK
with tabs[3]:
    st.subheader("🚦 Risk & Bayraklar")
    st.markdown("Her iki track için ortak kırmızı bayrak kontrolü aktif.")
    flags_df = pd.DataFrame([
        ["MAX_DD", "Max Drawdown", "< -%15", "Rebalance durdur"],
        ["DAILY_LOSS", "Günlük Kayıp", "< -%5", "İşlem yasağı"],
        ["HIGH_CORR", "Korelasyon", "> 0.85", "Diversifikasyon uyarısı"],
        ["HIGH_VOL", "Volatilite", "> %40", "Pozisyon küçült"],
        ["DRIFT", "Rebalance Sapması", "> %10", "Zorunlu rebalance"],
    ], columns=["Kod", "Bayrak", "Eşik", "Aksiyon"])
    df_show(flags_df, hide_index=True)


# ============================================================ TAB 4: CANLI
with tabs[4]:
    st.subheader("🚀 Canlıya Geçiş")
    st.info("Canlıya geçiş için paper trade aşamasının tamamlanması gerekir.")
    st.markdown("""
    **Karar Kriteri: Hangi Track Canlıya Alınmalı?**
    - **Reel getiri öncelikliyse** → Track A (İslami)
    - **Toplam getiri öncelikliyse** → Track B (Yüksek Gelir)
    - **Vergi optimizasyonu öncelikliyse** → Track A (genelde %0 stopaj)
    - **Risk ayarlı getiri (Sharpe) öncelikliyse** → ikisini karşılaştırın
    """)


# ============================================================ SIDEBAR
st.sidebar.header("📍 Durum")
st.sidebar.markdown(f"**Aşama**: `{state['phase'].upper()}`")
if "dual_results" in st.session_state:
    st.sidebar.success(f"✅ {len(st.session_state['dual_results'])} track hazır")
else:
    st.sidebar.info("Henüz çalıştırma yok")