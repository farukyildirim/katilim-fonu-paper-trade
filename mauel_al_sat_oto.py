import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from io import BytesIO
import time
import os
import json
from pathlib import Path

# ============================================================
# SAYFA AYARLARI
# ============================================================
st.set_page_config(
    page_title="Katılım Fonu Paper Trade",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SABİTLER
# ============================================================
SIM_END_DATE = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
USD_RET_ASSUMPTION = 0.20
TOLERANCE_TL = 10.0
STATE_VERSION = 12
MACRO = {"tufe": 0.3151, "usdtry_now": 48.75, "usdtry_12m_exp": 56.0}

RISK_FREE_NOMINAL = 0.45
RISK_FREE_FORWARD = 0.30
RISK_FREE_REEL = MACRO["tufe"]

MC_SIMULATIONS = 1000
MC_DAYS = 252

DATA_DIR = Path("paper_trade_data")
DATA_DIR.mkdir(exist_ok=True)
STATE_FILE = DATA_DIR / "state.json"

RED_FLAG_THRESHOLDS = {
    "max_drawdown_kritik": -0.20, "max_drawdown_uyari": -0.10,
    "volatilite_kritik": 0.25, "volatilite_uyari": 0.18,
    "sharpe_negatif": 0.0,
    "konsantrasyon_kritik": 50.0, "konsantrasyon_uyari": 35.0,
    "sektor_konsantrasyon": 60.0,
    "reel_getiri_kritik": -0.05, "sapma_carpani": 1.5,
}

PORTFOLIO_TEMPLATES = {
    "🟢 Muhafazakar": {"ZPG": 0.35, "KTN": 0.25, "KZL": 0.20, "KPC": 0.10, "KIS": 0.10},
    "🟡 Dengeli": {"ZPG": 0.15, "KTN": 0.15, "CPU": 0.25, "KZL": 0.20, "KPC": 0.15, "KIS": 0.10},
    "🟣 Karma": {"CPU": 0.25, "KTJ": 0.15, "KZL": 0.20, "KPC": 0.10,
                 "RBH": 0.05, "ZPG": 0.15, "KTN": 0.10},
    "🔴 Agresif": {"CPU": 0.35, "KTJ": 0.25, "KZL": 0.20, "KPC": 0.10, "RBH": 0.10},
    "⚡ Teknoloji Odaklı": {"CPU": 0.40, "KTJ": 0.30, "KZL": 0.15, "KPC": 0.10, "ZPG": 0.05},
    "🛡️ Getiri Odaklı": {"ZPG": 0.30, "KTN": 0.25, "KZL": 0.20, "KIS": 0.15, "KPC": 0.10},
}

DEFAULT_FUNDS = {
    "KPC": {"name": "Kuveyt Türk Katılım Hisse Senedi", "type": "Hisse Senedi",
            "tax": 0.0, "vol": 0.38, "price": 20.967209, "r1y": 0.4171},
    "RBH": {"name": "Albaraka Katılım Hisse Senedi", "type": "Hisse Senedi",
            "tax": 0.0, "vol": 0.35, "price": 33.184389, "r1y": 0.3447},
    "KTJ": {"name": "Kuveyt Türk Teknoloji Katılım", "type": "Teknoloji",
            "tax": 0.175, "vol": 0.45, "price": 2.788390, "r1y": 0.6401},
    "KZL": {"name": "Kuveyt Türk Altın Katılım", "type": "Altın",
            "tax": 0.175, "vol": 0.28, "price": 28.316445, "r1y": 0.3246},
    "CPU": {"name": "Aktif Portföy Teknoloji Katılım", "type": "Teknoloji",
            "tax": 0.0, "vol": 0.42, "price": 3.952677, "r1y": 0.7873},
    "KIS": {"name": "Astra Portföy Kira Sertifikası Döviz", "type": "Kira Sert. (Döviz)",
            "tax": 0.175, "vol": 0.22, "price": 0.223354, "r1y": 0.1941},
    "ZPG": {"name": "Ziraat Portföy Kira Sertifikaları Sukuk", "type": "Kira Sert. (TL)",
            "tax": 0.0, "vol": 0.12, "price": 11.018887, "r1y": 0.4132},
    "KTN": {"name": "Kuveyt Türk Kira Sertifikaları TL", "type": "Kira Sert. (TL)",
            "tax": 0.0, "vol": 0.10, "price": 8.240707, "r1y": 0.3603},
}

# ============================================================
# TEFAS
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_tefas_price(code: str):
    try:
        from tefas import Crawler
        tefas = Crawler()
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
        data = tefas.fetch(start=start, end=end, name=code)
        if data is None or data.empty:
            return {"error": f"{code}: Boş veri"}
        data = data.sort_values("date").copy()
        data["date"] = pd.to_datetime(data["date"])
        data = data.set_index("date")
        series = data["price"].astype(float)
        price = float(series.iloc[-1])
        first_price = float(series.iloc[0])
        r1y = (price / first_price - 1) if first_price > 0 else None
        return {"price": price, "r1y": r1y,
                "date": str(series.index[-1].date()),
                "series": series, "rows": len(series)}
    except Exception as e:
        return {"error": f"{code}: {type(e).__name__}: {e}"}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_tefas(codes: tuple):
    out = {}
    for i, c in enumerate(codes):
        out[c] = fetch_tefas_price(c)
        if i < len(codes) - 1:
            time.sleep(2)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def search_tefas_fund(query: str):
    KNOWN_FUNDS = [
        ("KPC", "Kuveyt Türk Katılım Hisse Senedi"),
        ("RBH", "Albaraka Katılım Hisse Senedi"),
        ("KTJ", "Kuveyt Türk Teknoloji Katılım"),
        ("KZL", "Kuveyt Türk Altın Katılım"),
        ("CPU", "Aktif Portföy Teknoloji Katılım"),
        ("KIS", "Astra Portföy Kira Sertifikası Döviz"),
        ("ZPG", "Ziraat Portföy Kira Sertifikaları Sukuk"),
        ("KTN", "Kuveyt Türk Kira Sertifikaları TL"),
        ("TCD", "TEB Katılım Değişken Fon"),
        ("TGE", "TEB Katılım Emeklilik"),
        ("AFA", "Ak Portföy Katılım Hisse Senedi"),
        ("MAC", "Marmara Capital Katılım Hisse"),
    ]
    if not query or len(query) < 2:
        return pd.DataFrame(KNOWN_FUNDS, columns=["code", "title"])
    q = query.upper()
    filtered = [(code, title) for code, title in KNOWN_FUNDS
                if q in code.upper() or q in title.upper()]
    try:
        from pytefas import Crawler
        tefas = Crawler()
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        df = tefas.fetch(start, end, kind="YAT")
        if df is not None and not df.empty and "code" in df.columns and "title" in df.columns:
            funds = df[["code", "title"]].drop_duplicates("code")
            match = funds[
                funds["code"].str.upper().str.contains(q, na=False) |
                funds["title"].str.upper().str.contains(q, na=False)
            ]
            existing = set(c for c, _ in filtered)
            for _, row in match.iterrows():
                if row["code"] not in existing:
                    filtered.append((row["code"], row["title"]))
    except Exception:
        pass
    return pd.DataFrame(filtered, columns=["code", "title"])


# ============================================================
# FİYAT SİMÜLASYONU
# ============================================================
def generate_price_history(funds_dict: dict, days: int = 365) -> pd.DataFrame:
    end_date = SIM_END_DATE
    dates = pd.date_range(end=end_date, periods=days, freq="D")
    out = {"date": dates}
    for code, meta in funds_dict.items():
        price = float(meta.get("price", 10.0))
        r1y = float(meta.get("r1y", 0.30))
        vol = float(meta.get("vol", 0.30))
        start_price = price / (1 + r1y) if r1y else price
        n = days
        t = np.linspace(0, 1, n)
        np.random.seed(abs(hash(code)) % 2**32)
        dW = np.random.normal(0, np.sqrt(1 / n), n)
        W = np.cumsum(dW)
        W = W - t * W[-1]
        mu = np.log(price / start_price) if start_price > 0 and price > 0 else 0
        path = start_price * np.exp(mu * t + vol * W)
        path[-1] = price
        out[code] = path
    return pd.DataFrame(out).set_index("date")


# ============================================================
# KALICI DEPOLAMA
# ============================================================
def serialize_datetime(obj):
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def save_state():
    try:
        data = {
            "state_version": STATE_VERSION,
            "start_date": st.session_state.start_date.isoformat(),
            "initial_capital": st.session_state.initial_capital,
            "funds": st.session_state.funds,
            "target_weights": st.session_state.target_weights,
            "transactions": st.session_state.transactions,
            "manual_positions": st.session_state.manual_positions,
            "template_history": st.session_state.get("template_history", []),
            "rebalance_log": st.session_state.rebalance_log,
            "saved_at": datetime.now().isoformat(),
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, default=serialize_datetime,
                      ensure_ascii=False, indent=2)
        return True, f"💾 Kaydedildi ({datetime.now().strftime('%H:%M:%S')})"
    except Exception as e:
        return False, f"❌ Kayıt hatası: {e}"


def load_state():
    if not STATE_FILE.exists():
        return False, "Kayıt dosyası yok"
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        st.session_state.start_date = datetime.fromisoformat(data["start_date"])
        st.session_state.initial_capital = data["initial_capital"]
        st.session_state.funds = data["funds"]
        st.session_state.target_weights = data["target_weights"]
        st.session_state.manual_positions = data.get("manual_positions", {})
        st.session_state.template_history = data.get("template_history", [])
        st.session_state.rebalance_log = data.get("rebalance_log", [])
        tx = []
        for t in data.get("transactions", []):
            if "tarih" in t and isinstance(t["tarih"], str):
                try: t["tarih"] = datetime.fromisoformat(t["tarih"])
                except Exception: t["tarih"] = datetime.now()
            if "created_at" in t and isinstance(t["created_at"], str):
                try: t["created_at"] = datetime.fromisoformat(t["created_at"])
                except Exception: pass
            tx.append(t)
        st.session_state.transactions = tx
        st.session_state.price_history = generate_price_history(st.session_state.funds)
        for code in st.session_state.funds.keys():
            w = st.session_state.target_weights.get(code, 0.0)
            st.session_state[f"w_{code}"] = int(round(w * 100))
        saved_at = data.get("saved_at", "")
        return True, f"✅ {len(tx)} işlem yüklendi ({saved_at[:16]})"
    except Exception as e:
        return False, f"❌ Yükleme hatası: {e}"


def calculate_group_pnl(group_txs):
    total_cost = 0.0
    current_value = 0.0
    rows = []
    for t in group_txs:
        code = t.get("fon")
        units = float(t.get("birim", 0))
        entry_price = float(t.get("fiyat", 0))
        cost = float(t.get("tutar", 0))
        current_price = get_price(code, SIM_END_DATE) if code in st.session_state.funds else entry_price

        # ✅ Floating-point temizliği
        value = round(units * current_price, 4)
        pnl = round(value - cost, 4)
        pnl_pct = round((value / cost - 1) * 100, 4) if cost > 0 else 0

        total_cost += cost
        current_value += value

        rows.append({
            "Fon": code,
            "Birim": round(units, 4),
            "Giriş Fiyatı": round(entry_price, 4),
            "Güncel Fiyat": round(current_price, 4),
            "Maliyet": round(cost, 2),
            "Güncel Değer": value,
            "K/Z (TL)": pnl,
            "K/Z (%)": pnl_pct,
        })

    total_pnl = round(current_value - total_cost, 4)
    total_pnl_pct = round((current_value / total_cost - 1) * 100, 4) if total_cost > 0 else 0

    return {
        "rows": rows,
        "total_cost": total_cost,
        "current_value": current_value,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
    }

# ============================================================
# FON YÖNETİMİ
# ============================================================
def add_fund(code, name, ftype, tax, vol, price=10.0, r1y=0.30):
    if code in st.session_state.funds:
        return False, f"{code} zaten mevcut."
    st.session_state.funds[code] = {
        "name": name, "type": ftype, "tax": tax, "vol": vol,
        "price": price, "r1y": r1y,
    }
    st.session_state.price_history = generate_price_history(st.session_state.funds)
    if st.session_state.get("auto_save", True):
        save_state()
    return True, f"{code} eklendi."


def remove_fund(code):
    if code not in st.session_state.funds:
        return False, f"{code} bulunamadı."
    del st.session_state.funds[code]
    st.session_state.target_weights.pop(code, None)
    st.session_state.manual_positions.pop(code, None)
    st.session_state.price_history = generate_price_history(st.session_state.funds)
    if st.session_state.get("auto_save", True):
        save_state()
    return True, f"{code} çıkarıldı."


def apply_template(template_name: str):
    if template_name not in PORTFOLIO_TEMPLATES:
        return False, "Şablon bulunamadı."
    template = PORTFOLIO_TEMPLATES[template_name]
    new_weights = {code: w for code, w in template.items() if code in st.session_state.funds}
    if not new_weights:
        return False, "Şablondaki fonlar portföyünüzde yok."
    st.session_state.target_weights = new_weights
    for code in st.session_state.funds.keys():
        w = new_weights.get(code, 0.0)
        st.session_state[f"w_{code}"] = int(round(w * 100))
    if st.session_state.get("auto_save", True):
        save_state()
    return True, f"{template_name} uygulandı ({len(new_weights)} fon)"


def auto_generate_paper_trades(template_name: str, mode: str = "append"):
    if template_name not in PORTFOLIO_TEMPLATES:
        return False, "Şablon bulunamadı."
    template = PORTFOLIO_TEMPLATES[template_name]
    cap = st.session_state.initial_capital
    start = st.session_state.start_date
    for code in template.keys():
        if code not in st.session_state.funds and code in DEFAULT_FUNDS:
            st.session_state.funds[code] = dict(DEFAULT_FUNDS[code])
    new_tx = []
    for code, w in template.items():
        if code not in st.session_state.funds:
            continue
        price = get_price(code, start)
        if price <= 0:
            continue
        units = (cap * w) / price
        amount = cap * w
        new_tx.append({
            "tarih": start, "tip": "OTOMATİK ALIM", "fon": code,
            "birim": units, "fiyat": price, "tutar": amount,
            "not": f"Şablon: {template_name}", "template": template_name,
            "created_at": datetime.now(),
        })
    if not new_tx:
        return False, "İşlem oluşturulamadı."
    if mode == "replace_same":
        st.session_state.transactions = [
            t for t in st.session_state.transactions
            if not (t.get("tip") == "OTOMATİK ALIM" and t.get("template") == template_name)
        ]
    elif mode == "replace_all":
        st.session_state.transactions = [
            t for t in st.session_state.transactions
            if t.get("tip") != "OTOMATİK ALIM"
        ]
    st.session_state.transactions.extend(new_tx)
    if "template_history" not in st.session_state:
        st.session_state.template_history = []
    st.session_state.template_history.append({
        "template": template_name, "tarih": datetime.now(),
        "fon_sayisi": len(new_tx),
        "toplam_tutar": sum(t["tutar"] for t in new_tx), "mode": mode,
    })
    if st.session_state.get("auto_save", True):
        save_state()
    return True, f"{len(new_tx)} otomatik işlem oluşturuldu ({template_name}, mod: {mode})"


# ============================================================
# SESSION STATE
# ============================================================
def init_state():
    if st.session_state.get("state_version") != STATE_VERSION:
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.session_state.state_version = STATE_VERSION
        st.session_state.initialized = True
        st.session_state.start_date = SIM_END_DATE
        st.session_state.initial_capital = 100_000.0
        st.session_state.funds = dict(DEFAULT_FUNDS)
        st.session_state.target_weights = dict(PORTFOLIO_TEMPLATES["🟡 Dengeli"])
        st.session_state.transactions = []
        st.session_state.rebalance_log = []
        st.session_state.manual_positions = {}
        st.session_state.use_tefas = False
        st.session_state.tefas_cache = {}
        st.session_state.price_history = generate_price_history(st.session_state.funds)
        st.session_state.tefas_series = {}
        st.session_state.red_flag_dismissed = set()
        st.session_state.template_history = []
        st.session_state.auto_save = True
        if STATE_FILE.exists():
            ok, msg = load_state()
            if ok:
                st.session_state["_load_msg"] = msg


init_state()


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================
def get_price(code: str, when: datetime) -> float:
    when = pd.Timestamp(when).normalize()
    if st.session_state.get("use_tefas", False):
        series = st.session_state.get("tefas_series", {}).get(code)
        if series is not None and len(series) > 0:
            if when <= series.index[0]: return float(series.iloc[0])
            if when >= series.index[-1]: return float(series.iloc[-1])
            return float(series.asof(when))
    if "price_history" not in st.session_state:
        st.session_state.price_history = generate_price_history(st.session_state.funds)
    df = st.session_state.price_history
    if code not in df.columns: return 10.0
    if when <= df.index[0]: return float(df[code].iloc[0])
    if when >= df.index[-1]: return float(df[code].iloc[-1])
    return float(df[code].asof(when))


def get_fund_meta(code: str) -> dict:
    meta = dict(st.session_state.funds.get(code, {}))
    if st.session_state.get("use_tefas", False):
        tf = st.session_state.get("tefas_cache", {}).get(code)
        if tf and "price" in tf:
            meta["price"] = tf["price"]
            if tf.get("r1y") is not None:
                meta["r1y"] = tf["r1y"]
    return meta


def get_tax_rate(code: str) -> float:
    return st.session_state.funds.get(code, {}).get("tax", 0.0)


def build_portfolio():
    cap = st.session_state.initial_capital
    start = st.session_state.start_date
    positions = {}
    for code, w in st.session_state.target_weights.items():
        if code not in st.session_state.funds or w <= 0:
            continue
        price = get_price(code, start)
        positions[code] = {"units": (cap * w) / price, "cost": cap * w, "buy_price": price}
    for code, mp in st.session_state.manual_positions.items():
        if code in positions:
            tc = positions[code]["cost"] + mp["cost"]
            tu = positions[code]["units"] + mp["units"]
            positions[code] = {"units": tu, "cost": tc,
                               "buy_price": tc / tu if tu else 0}
        else:
            positions[code] = dict(mp)
    return positions


def net_portfolio_on(date, positions):
    total = 0.0
    for code, p in positions.items():
        val = p["units"] * get_price(code, date)
        gross_pl = val - p["cost"]
        tax = max(0.0, gross_pl) * get_tax_rate(code)
        total += val - tax
    return total


def get_portfolio_history(positions):
    df_hist = st.session_state.price_history
    if st.session_state.get("use_tefas", False) and st.session_state.get("tefas_series"):
        all_s = [s for s in st.session_state.tefas_series.values() if s is not None and len(s) > 0]
        if all_s:
            idx = max(all_s, key=len).index
            df_hist = pd.DataFrame(index=idx)
            for code, series in st.session_state.tefas_series.items():
                if series is not None and len(series) > 0:
                    df_hist[code] = series.reindex(idx).ffill()
            df_hist = df_hist.dropna(how="all")
    if df_hist.empty:
        return None, None
    port_values = pd.Series(
        [net_portfolio_on(d, positions) for d in df_hist.index],
        index=df_hist.index,
    )
    daily_rets = port_values.pct_change().dropna()
    return port_values, daily_rets


# ============================================================
# SHARPE — 30 GÜN KONTROLÜ EKLENDİ
# ============================================================
def calculate_all_sharpes(positions):
    port_values, daily_rets = get_portfolio_history(positions)
    if daily_rets is None or len(daily_rets) < 30:
        return None

    days_held_check = (SIM_END_DATE - st.session_state.start_date).days
    if days_held_check < 30:
        return None

    start_ts = pd.Timestamp(st.session_state.start_date)
    end_ts = port_values.index[-1]
    try:
        start_val = float(port_values.asof(start_ts))
    except Exception:
        start_val = float(port_values.iloc[0])
    end_val = float(port_values.iloc[-1])
    if start_val <= 0 or end_val <= 0:
        return None
    days_elapsed = (end_ts - start_ts).days
    if days_elapsed < 1:
        days_elapsed = max((port_values.index[-1] - port_values.index[0]).days, 1)
    total_growth = end_val / start_val
    annual_ret = total_growth ** (365 / days_elapsed) - 1
    annual_vol = daily_rets.std() * np.sqrt(252)
    if annual_vol == 0:
        return None
    downside = daily_rets[daily_rets < 0].std() * np.sqrt(252)
    sortino_reel = (annual_ret - RISK_FREE_REEL) / downside if downside > 0 else None
    var_95 = np.percentile(daily_rets, 5)
    cvar_95 = daily_rets[daily_rets <= var_95].mean()
    roll_max = port_values.cummax()
    dd = (port_values / roll_max - 1)
    max_dd = dd.min()
    return {
        "annual_return": annual_ret, "annual_vol": annual_vol,
        "sharpe_nominal": (annual_ret - RISK_FREE_NOMINAL) / annual_vol,
        "sharpe_forward": (annual_ret - RISK_FREE_FORWARD) / annual_vol,
        "sharpe_reel": (annual_ret - RISK_FREE_REEL) / annual_vol,
        "sortino_reel": sortino_reel, "var_95": var_95, "cvar_95": cvar_95,
        "max_dd": max_dd, "n_days": len(daily_rets),
    }


# ============================================================
# KIRMIZI BAYRAK — 30 GÜN ERKEN ÇIKIŞ EKLENDİ
# ============================================================
def check_red_flags(df_pos, totals, sharpe_data, days_held):
    flags = []

    if days_held < 30:
        if not df_pos.empty:
            max_drift = df_pos["Sapma (pp)"].abs().max()
            threshold = st.session_state.get("drift_threshold", 5)
            if max_drift > threshold:
                flags.append({"id": "sapma_uyari", "seviye": "uyari",
                              "baslik": "⚠️ Rebalans Gerekli",
                              "mesaj": f"Maks. sapma **{max_drift:.2f} pp** > {threshold}%",
                              "aksiyon": "Rebalans sekmesinden uygulayın"})
        return flags

    if sharpe_data and sharpe_data.get("max_dd") is not None:
        dd = sharpe_data["max_dd"]
        if dd < RED_FLAG_THRESHOLDS["max_drawdown_kritik"]:
            flags.append({"id": "dd_kritik", "seviye": "kritik",
                          "baslik": "🚨 Yüksek Drawdown",
                          "mesaj": f"Maks. düşüş **{dd*100:.2f}%**",
                          "aksiyon": "Pozisyonları gözden geçirin"})
        elif dd < RED_FLAG_THRESHOLDS["max_drawdown_uyari"]:
            flags.append({"id": "dd_uyari", "seviye": "uyari",
                          "baslik": "⚠️ Drawdown Artıyor",
                          "mesaj": f"Maks. düşüş **{dd*100:.2f}%**",
                          "aksiyon": "Risk yönetimini gözden geçirin"})

    if sharpe_data:
        vol = sharpe_data.get("annual_vol", 0)
        if vol > RED_FLAG_THRESHOLDS["volatilite_kritik"]:
            flags.append({"id": "vol_kritik", "seviye": "kritik",
                          "baslik": "🚨 Yüksek Volatilite",
                          "mesaj": f"Yıllık volatilite **%{vol*100:.2f}**",
                          "aksiyon": "Düşük volatiliteli fonlara ağırlık verin"})
        elif vol > RED_FLAG_THRESHOLDS["volatilite_uyari"]:
            flags.append({"id": "vol_uyari", "seviye": "uyari",
                          "baslik": "⚠️ Volatilite Yükseldi",
                          "mesaj": f"Yıllık volatilite **%{vol*100:.2f}**",
                          "aksiyon": "Portföy çeşitlendirmesini artırın"})

    if sharpe_data:
        sh = sharpe_data.get("sharpe_forward", 0)
        if sh < RED_FLAG_THRESHOLDS["sharpe_negatif"]:
            flags.append({"id": "sharpe_neg", "seviye": "uyari",
                          "baslik": "⚠️ Sharpe Negatif",
                          "mesaj": f"Forward Sharpe **{sh:+.3f}**",
                          "aksiyon": "Daha yüksek getirili fonlar düşünün"})

    if not df_pos.empty:
        max_w = df_pos["Ağırlık (%)"].max()
        max_code = df_pos.loc[df_pos["Ağırlık (%)"].idxmax(), "Fon"]
        if max_w > RED_FLAG_THRESHOLDS["konsantrasyon_kritik"]:
            flags.append({"id": "kons_kritik", "seviye": "kritik",
                          "baslik": "🚨 Konsantrasyon Riski",
                          "mesaj": f"**{max_code}** fonu **%{max_w:.1f}**",
                          "aksiyon": "Çeşitlendirme yapın"})
        elif max_w > RED_FLAG_THRESHOLDS["konsantrasyon_uyari"]:
            flags.append({"id": "kons_uyari", "seviye": "uyari",
                          "baslik": "⚠️ Yüksek Konsantrasyon",
                          "mesaj": f"**{max_code}** fonu **%{max_w:.1f}**",
                          "aksiyon": "Diğer fonlara ağırlık verin"})

    if not df_pos.empty:
        tech_mask = df_pos["Tür"].str.contains("Teknoloji", na=False)
        tech_w = df_pos.loc[tech_mask, "Ağırlık (%)"].sum()
        if tech_w > RED_FLAG_THRESHOLDS["sektor_konsantrasyon"]:
            flags.append({"id": "tech_kons", "seviye": "kritik",
                          "baslik": "🚨 Sektör Konsantrasyonu",
                          "mesaj": f"Teknoloji fonları toplam **%{tech_w:.1f}**",
                          "aksiyon": "Altın, kira sertifikası veya hisse fonlarla dengeleyin"})

    if days_held >= 30 and totals.get("reel") is not None:
        reel = totals["reel"]
        if reel < RED_FLAG_THRESHOLDS["reel_getiri_kritik"]:
            flags.append({"id": "reel_neg", "seviye": "kritik",
                          "baslik": "🚨 Enflasyon Altında Getiri",
                          "mesaj": f"Reel getiri **{reel*100:+.2f}%**",
                          "aksiyon": "Agresif veya Teknoloji Odaklı şablonu değerlendirin"})

    if not df_pos.empty:
        max_drift = df_pos["Sapma (pp)"].abs().max()
        threshold = st.session_state.get("drift_threshold", 5)
        if max_drift > threshold * RED_FLAG_THRESHOLDS["sapma_carpani"]:
            flags.append({"id": "sapma_kritik", "seviye": "kritik",
                          "baslik": "🚨 Aşırı Sapma",
                          "mesaj": f"Maks. sapma **{max_drift:.2f} pp**",
                          "aksiyon": "Rebalans sekmesinden düzeltin"})
        elif max_drift > threshold:
            flags.append({"id": "sapma_uyari", "seviye": "uyari",
                          "baslik": "⚠️ Rebalans Gerekli",
                          "mesaj": f"Maks. sapma **{max_drift:.2f} pp**",
                          "aksiyon": "Rebalans sekmesinden uygulayın"})

    return flags


# ============================================================
# MONTE CARLO
# ============================================================
def run_monte_carlo(positions, n_sims=MC_SIMULATIONS, n_days=MC_DAYS):
    port_values, daily_rets = get_portfolio_history(positions)
    if daily_rets is None or len(daily_rets) < 30:
        return None
    log_rets = np.log(1 + daily_rets.dropna())
    mu_log_daily = log_rets.mean()
    sigma_daily = log_rets.std()
    if sigma_daily == 0:
        return None
    np.random.seed(42)
    simulations = np.zeros((n_days, n_sims))
    initial = st.session_state.initial_capital
    for i in range(n_sims):
        shocks = np.random.normal(0, 1, n_days)
        log_returns = mu_log_daily + sigma_daily * shocks
        simulations[:, i] = initial * np.exp(np.cumsum(log_returns))
    return simulations


# ============================================================
# JSON EXPORT / IMPORT
# ============================================================
def export_portfolio_json():
    data = {
        "funds": st.session_state.funds,
        "target_weights": st.session_state.target_weights,
        "initial_capital": st.session_state.initial_capital,
        "start_date": st.session_state.start_date.isoformat(),
        "manual_positions": st.session_state.manual_positions,
        "transactions": st.session_state.transactions,
        "state_version": STATE_VERSION,
        "exported_at": datetime.now().isoformat(),
    }
    return json.dumps(data, indent=2, ensure_ascii=False, default=serialize_datetime)


def import_portfolio_json(json_str: str):
    try:
        data = json.loads(json_str)
        if "funds" not in data or "target_weights" not in data:
            return False, "Geçersiz portföy formatı."
        st.session_state.funds = data["funds"]
        st.session_state.target_weights = data["target_weights"]
        st.session_state.initial_capital = data.get("initial_capital", 100_000.0)
        if "start_date" in data:
            try: st.session_state.start_date = datetime.fromisoformat(data["start_date"])
            except Exception: pass
        st.session_state.manual_positions = data.get("manual_positions", {})
        st.session_state.transactions = []
        for t in data.get("transactions", []):
            try: t["tarih"] = datetime.fromisoformat(t["tarih"])
            except Exception: t["tarih"] = datetime.now()
            st.session_state.transactions.append(t)
        st.session_state.price_history = generate_price_history(st.session_state.funds)
        for code in st.session_state.funds.keys():
            w = st.session_state.target_weights.get(code, 0.0)
            st.session_state[f"w_{code}"] = int(round(w * 100))
        if st.session_state.get("auto_save", True):
            save_state()
        return True, f"✅ {len(data['funds'])} fon yüklendi."
    except Exception as e:
        return False, f"❌ Hata: {e}"


# ============================================================
# REBALANS TAKVİMİ
# ============================================================
def get_rebalance_rule(freq: str):
    if freq == "3 Aylık": return pd.DateOffset(months=3)
    if freq == "6 Aylık": return pd.DateOffset(months=6)
    if freq == "Yıllık": return pd.DateOffset(years=1)
    return None


def compute_rebalance_dates(start_date, freq, months_ahead=12):
    rule = get_rebalance_rule(freq)
    if rule is None: return [], [], None
    start = pd.Timestamp(start_date)
    today = pd.Timestamp(SIM_END_DATE)
    all_dates = []
    cur = start - rule
    end = today + pd.DateOffset(months=months_ahead)
    while cur <= end:
        all_dates.append(cur)
        cur = cur + rule
    if start not in all_dates:
        all_dates.append(start)
        all_dates.sort()
    past = [d for d in all_dates if d < today]
    future = [d for d in all_dates if d >= today]
    next_reb = future[0] if future else None
    return past, future, next_reb


def days_until(target):
    if target is None: return None
    return (pd.Timestamp(target) - pd.Timestamp(SIM_END_DATE)).days


# ============================================================
# PDF
# ============================================================
def get_font_path():
    for p in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/Library/Fonts/Arial.ttf"]:
        if os.path.exists(p): return p
    return None


def build_pdf_report(df_pos, totals, positions, days_held, next_reb, sharpe_data=None, red_flags=None):
    try:
        from fpdf import FPDF
    except ImportError:
        return None, "fpdf2 kurulu değil. `pip install fpdf2`"
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    font_path = get_font_path()
    FONT = "Helvetica"; FONT_OK = False
    if font_path:
        try:
            pdf.add_font("Custom", "", font_path)
            pdf.add_font("Custom", "B", font_path)
            pdf.add_font("Custom", "I", font_path)
            pdf.add_font("Custom", "BI", font_path)
            FONT = "Custom"; FONT_OK = True
        except Exception: pass
    def set_font(size, style=""):
        if FONT_OK:
            try: pdf.set_font(FONT, style=style, size=size); return
            except Exception: pass
        safe = style if style in ("", "B", "I", "BI") else ""
        pdf.set_font("Helvetica", style=safe, size=size)
    set_font(18, "B")
    pdf.cell(0, 10, "Katilim Fonu Portfoy Raporu", ln=True, align="C")
    set_font(10)
    pdf.cell(0, 6, f"Rapor: {SIM_END_DATE.strftime('%d.%m.%Y')}  |  "
                   f"Baslangic: {st.session_state.start_date.strftime('%d.%m.%Y')}  |  "
                   f"Gecen: {days_held} gun", ln=True, align="C")
    pdf.ln(4)
    if red_flags:
        set_font(13, "B"); pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 8, "! UYARILAR", ln=True); pdf.set_text_color(0, 0, 0)
        set_font(9)
        for flag in red_flags[:5]:
            prefix = "[KRITIK]" if flag["seviye"] == "kritik" else "[UYARI]"
            pdf.cell(0, 5, f"{prefix} {flag['baslik']}", ln=True)
            pdf.cell(0, 5, f"  {flag['mesaj']}", ln=True)
        pdf.ln(3)
    set_font(13, "B"); pdf.cell(0, 8, "1. Portfoy Ozeti", ln=True)
    set_font(10)
    kpi_rows = [
        ("Baslangic Sermayesi", f"{st.session_state.initial_capital:,.0f} TL"),
        ("Brut Deger", f"{totals['total_value']:,.0f} TL"),
        ("Net Deger", f"{totals['total_net']:,.0f} TL"),
        ("Brut Kar/Zarar", f"{totals['gross_pl']:+,.0f} TL"),
        ("Toplam Stopaj", f"-{totals['total_tax']:,.0f} TL"),
        ("Net Kar/Zarar", f"{totals['net_pl']:+,.0f} TL"),
        ("Nominal Getiri", f"{totals['nominal']*100:+.2f}%"),
    ]
    if days_held >= 1 and totals.get("reel") is not None:
        kpi_rows.append(("Reel Getiri", f"{totals['reel']*100:+.2f}%"))
        kpi_rows.append(("Doviz Ustu Getiri", f"{totals['doviz']*100:+.2f}%"))
    if sharpe_data is not None and sharpe_data.get("n_days", 0) >= 30:
        s = sharpe_data
        kpi_rows.extend([
            ("Nominal Sharpe", f"{s['sharpe_nominal']:+.3f}"),
            ("Forward Sharpe", f"{s['sharpe_forward']:+.3f}"),
            ("Reel Sharpe", f"{s['sharpe_reel']:+.3f}"),
            ("Yillik Volatilite", f"{s['annual_vol']*100:.2f}%"),
            ("Maks Drawdown", f"{s['max_dd']*100:.2f}%"),
        ])
    for label, val in kpi_rows:
        set_font(10); pdf.cell(90, 6, label, border=0)
        set_font(10, "B"); pdf.cell(0, 6, val, border=0, ln=True)
    pdf.ln(5)
    set_font(13, "B"); pdf.cell(0, 8, "2. Pozisyonlar", ln=True)
    headers = ["Fon", "Birim", "A.Fiyat", "G.Fiyat", "Deger", "Stopaj", "Net"]
    widths = [18, 28, 24, 24, 32, 24, 32]
    set_font(9, "B")
    for h, w in zip(headers, widths):
        pdf.cell(w, 7, h, border=1, align="C")
    pdf.ln()
    set_font(8)
    for _, row in df_pos.iterrows():
        pdf.cell(widths[0], 6, str(row["Fon"]), border=1)
        pdf.cell(widths[1], 6, f"{row['Birim']:,.2f}", border=1, align="R")
        pdf.cell(widths[2], 6, f"{row['Alış Fiyatı']:.4f}", border=1, align="R")
        pdf.cell(widths[3], 6, f"{row['Güncel Fiyat']:.4f}", border=1, align="R")
        pdf.cell(widths[4], 6, f"{row['Güncel Değer']:,.0f}", border=1, align="R")
        pdf.cell(widths[5], 6, f"{row['Stopaj (TL)']:,.0f}", border=1, align="R")
        pdf.cell(widths[6], 6, f"{row['Net Değer']:,.0f}", border=1, align="R")
        pdf.ln()
    pdf.ln(4)
    set_font(8, "I")
    pdf.multi_cell(0, 5, "Uyari: Bu rapor bir paper trade simulasyonudur.")
    out = pdf.output()
    return (bytes(out) if not isinstance(out, str) else out.encode("latin-1")), None


# ============================================================
# EXCEL
# ============================================================
def build_excel(df_pos, df_funds, df_tx, df_perf, totals, sharpe_data=None,
                mc_data=None, red_flags=None, paper_trade_groups=None):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_pos.to_excel(writer, sheet_name="Pozisyonlar", index=False)
        def safe_pct(v): return round(v * 100, 4) if v is not None else None
        summary_rows = [
            ("Başlangıç Sermayesi", st.session_state.initial_capital),
            ("Brüt Değer", totals["total_value"]),
            ("Net Değer (Stopaj Sonrası)", totals["total_net"]),
            ("Brüt K/Z", totals["gross_pl"]),
            ("Toplam Stopaj", totals["total_tax"]),
            ("Net K/Z", totals["net_pl"]),
            ("Nominal Getiri (%)", safe_pct(totals.get("nominal"))),
            ("Reel Getiri (%)", safe_pct(totals.get("reel"))),
            ("Döviz Üstü Getiri (%)", safe_pct(totals.get("doviz"))),
        ]
        if sharpe_data is not None and sharpe_data.get("n_days", 0) >= 30:
            s = sharpe_data
            summary_rows.extend([
                ("Nominal Sharpe", round(s["sharpe_nominal"], 4)),
                ("Forward Sharpe", round(s["sharpe_forward"], 4)),
                ("Reel Sharpe", round(s["sharpe_reel"], 4)),
                ("Yıllık Getiri (%)", round(s["annual_return"] * 100, 2)),
                ("Yıllık Volatilite (%)", round(s["annual_vol"] * 100, 2)),
                ("Maks. Drawdown (%)", round(s["max_dd"] * 100, 2)),
            ])
        pd.DataFrame(summary_rows, columns=["Gösterge", "Değer"]).to_excel(
            writer, sheet_name="Özet", index=False)
        if red_flags:
            rf_rows = [{"Seviye": f["seviye"], "Başlık": f["baslik"],
                        "Mesaj": f["mesaj"], "Aksiyon": f["aksiyon"]} for f in red_flags]
            pd.DataFrame(rf_rows).to_excel(writer, sheet_name="Uyarılar", index=False)
        if df_funds is not None and not df_funds.empty:
            df_funds.to_excel(writer, sheet_name="Fonlar", index=False)
        if df_tx is not None and not df_tx.empty:
            df_tx.to_excel(writer, sheet_name="İşlemler", index=False)
        if df_perf is not None and not df_perf.empty:
            df_perf.to_excel(writer, sheet_name="Performans", index=False)
        if paper_trade_groups:
            all_rows = []
            for tpl, rows in paper_trade_groups.items():
                for r in rows:
                    all_rows.append({"Şablon": tpl, **r})
            if all_rows:
                pd.DataFrame(all_rows).to_excel(writer, sheet_name="Paper Trades", index=False)
        pd.DataFrame(list(st.session_state.target_weights.items()),
                     columns=["Fon", "Hedef Ağırlık"]).to_excel(
            writer, sheet_name="Hedef Ağırlıklar", index=False)
    return output.getvalue()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Portföy Ayarları")
    if "_load_msg" in st.session_state:
        st.success(st.session_state.pop("_load_msg"))
    st.session_state.use_tefas = st.toggle("🌐 TEFAS Canlı Fiyat",
                                            value=st.session_state.use_tefas)
    if st.session_state.use_tefas:
        if st.button("🔄 TEFAS Fiyatlarını Güncelle", use_container_width=True):
            progress_bar = st.progress(0.0); status_text = st.empty()
            def update_progress(i, total, code):
                progress_bar.progress((i + 1) / total)
                status_text.text(f"⏳ {code} çekiliyor... ({i+1}/{total})")
            try:
                with st.spinner("TEFAS'tan veri çekiliyor..."):
                    result = fetch_all_tefas(tuple(st.session_state.funds.keys()))
                progress_bar.empty(); status_text.empty()
                st.session_state.tefas_cache = result
                st.session_state.tefas_series = {k: v["series"] for k, v in result.items()
                                                  if v and "series" in v}
                ok = sum(1 for v in result.values() if v and "price" in v)
                errs = {k: v.get("error") for k, v in result.items() if v and "error" in v}
                if ok == 0:
                    st.error("❌ TEFAS'a erişilemedi.")
                    st.session_state.use_tefas = False
                else:
                    st.success(f"✅ {ok}/{len(st.session_state.funds)} fon güncellendi")
                    if errs:
                        with st.expander(f"⚠️ {len(errs)} fon başarısız"):
                            for k, e in errs.items(): st.code(f"{k}: {e}")
            except Exception as e:
                progress_bar.empty(); status_text.empty()
                st.error(f"❌ Hata: {e}")
    st.markdown("---")
    st.markdown("### 💾 Veri Kalıcılığı")
    st.session_state.auto_save = st.toggle("🔄 Otomatik kaydet",
                                            value=st.session_state.get("auto_save", True))
    col_save, col_load = st.columns(2)
    with col_save:
        if st.button("💾 Kaydet", use_container_width=True):
            ok, msg = save_state()
            if ok: st.success(msg)
            else: st.error(msg)
    with col_load:
        if st.button("📂 Yükle", use_container_width=True):
            ok, msg = load_state()
            if ok: st.success(msg); time.sleep(0.6); st.rerun()
            else: st.error(msg)
    if STATE_FILE.exists():
        try:
            last = datetime.fromtimestamp(STATE_FILE.stat().st_mtime)
            st.caption(f"📁 Son kayıt: **{last.strftime('%d.%m.%Y %H:%M')}**")
        except Exception: pass
    st.markdown("---")
    st.markdown("### 🎨 Portföy Şablonları")
    template_names = list(PORTFOLIO_TEMPLATES.keys())
    selected_template = st.selectbox("Şablon Seç", ["— Seç —"] + template_names)
    if selected_template != "— Seç —":
        auto_tx = st.checkbox("🤖 Otomatik paper trade kaydı oluştur",
                               value=False, key="auto_tx_checkbox")
        tx_mode_label = st.radio("Kayıt modu",
                                  ["📌 Ekle (korumalı)", "🔄 Aynı şablonu güncelle", "🗑️ Tümünü yenile"],
                                  index=0)
        tx_mode_map = {"📌 Ekle (korumalı)": "append",
                       "🔄 Aynı şablonu güncelle": "replace_same",
                       "🗑️ Tümünü yenile": "replace_all"}
        tx_mode = tx_mode_map[tx_mode_label]
        if st.button(f"✅ {selected_template} Uygula", use_container_width=True):
            ok, msg = apply_template(selected_template)
            if ok:
                if auto_tx:
                    ok2, msg2 = auto_generate_paper_trades(selected_template, mode=tx_mode)
                    st.success(f"{msg} | {msg2}" if ok2 else f"{msg} | {msg2}")
                else:
                    st.success(msg)
                st.balloons(); time.sleep(0.8); st.rerun()
            else:
                st.error(msg)
    st.markdown("---")
    st.session_state.initial_capital = st.number_input(
        "Başlangıç Sermayesi (TL)", min_value=1000.0,
        value=st.session_state.initial_capital, step=10000.0, format="%.2f")
    new_date = st.date_input("Başlangıç Tarihi", value=st.session_state.start_date.date())
    st.session_state.start_date = datetime.combine(new_date, datetime.min.time())
    st.markdown("### 🎯 Hedef Ağırlıklar (%)")
    total_w = 0; new_weights = {}
    for code in list(st.session_state.funds.keys()):
        if code not in st.session_state.target_weights:
            st.session_state.target_weights[code] = 0.0
        default = int(st.session_state.target_weights.get(code, 0.0) * 100)
        w = st.slider(f"{code} — {st.session_state.funds[code]['name'][:25]}",
                      0, 50, default, 1, key=f"w_{code}")
        new_weights[code] = w / 100; total_w += w
    st.session_state.target_weights = {k: v for k, v in new_weights.items() if v > 0}
    if abs(total_w - 100) > 0.01: st.warning(f"⚠️ Toplam: %{total_w}")
    else: st.success(f"✅ Toplam: %{total_w}")
    st.markdown("### 📅 Rebalans")
    st.session_state.rebalance_freq = st.selectbox(
        "Sıklık", ["3 Aylık", "6 Aylık", "Yıllık", "Eşik Bazlı (%5 sapma)"], index=0)
    st.session_state.drift_threshold = st.slider("Sapma Eşiği (%)", 1, 15, 5)
    _past, _future, _next = compute_rebalance_dates(
        st.session_state.start_date, st.session_state.rebalance_freq)
    if _next is not None:
        _d = days_until(_next)
        if _d == 0: st.error("🔔 **Bugün rebalans günü!**")
        elif _d is not None and _d <= 7: st.warning(f"⏰ **{_d} gün kaldı**")
        elif _d is not None: st.info(f"📅 Sonraki: {_next.strftime('%d.%m.%Y')} ({_d} gün)")
    st.markdown("---")
    st.markdown("### 📦 Portföy Kopyalama")
    with st.expander("📤 Portföyü Dışa Aktar (JSON)"):
        st.download_button("⬇️ JSON İndir",
                           data=export_portfolio_json().encode("utf-8"),
                           file_name=f"portfoy_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                           mime="application/json", use_container_width=True)
    with st.expander("📥 Portföy İçe Aktar (JSON)"):
        uploaded = st.file_uploader("JSON Dosyası Yükle", type=["json"], key="import_file")
        pasted = st.text_area("Veya JSON Yapıştır", height=100, key="import_paste")
        if st.button("📥 İçe Aktar", use_container_width=True):
            json_str = None
            if uploaded is not None: json_str = uploaded.read().decode("utf-8")
            elif pasted: json_str = pasted
            if json_str:
                ok, msg = import_portfolio_json(json_str)
                if ok: st.success(msg); time.sleep(0.6); st.rerun()
                else: st.error(msg)
            else: st.warning("Dosya yükleyin veya JSON yapıştırın.")
    st.markdown("---")
    st.markdown("### ➕ Fon Ekle")
    with st.expander("📥 TEFAS'tan Fon Ekle"):
        search_q = st.text_input("Fon Ara (kod veya ad)", key="fund_search")
        if search_q and len(search_q) >= 2:
            with st.spinner("Aranıyor..."):
                results = search_tefas_fund(search_q)
            if not results.empty:
                st.dataframe(results.head(20), use_container_width=True, hide_index=True)
                selected_code = st.selectbox("Seç", results["code"].tolist(), key="search_select")
                selected_name = results[results["code"] == selected_code]["title"].iloc[0]
                if st.button("➕ Ekle", key="add_searched"):
                    ok, msg = add_fund(selected_code, selected_name, "Hisse Senedi", 0.0, 0.30, 10.0, 0.30)
                    if ok: st.success(msg); st.rerun()
                    else: st.error(msg)
            else: st.warning("Sonuç yok.")
    with st.expander("✏️ Manuel Fon Ekle"):
        new_code = st.text_input("Fon Kodu", max_chars=5, key="m_code").upper()
        new_name = st.text_input("Fon Adı", key="m_name")
        new_type = st.selectbox("Tür", ["Hisse Senedi", "Teknoloji", "Altın", "Döviz",
                                        "Kira Sert. (TL)", "Kira Sert. (Döviz)"], key="m_type")
        new_tax = st.number_input("Stopaj (%)", 0.0, 50.0, 0.0, 0.5, key="m_tax") / 100
        new_vol = st.number_input("Volatilite", 0.05, 1.0, 0.30, 0.05, key="m_vol")
        new_price = st.number_input("Fiyat (TL)", 0.0001, 10000.0, 10.0, 0.01, format="%.4f", key="m_price")
        new_r1y = st.number_input("1 Yıl Getiri (%)", -50.0, 500.0, 30.0, 1.0, key="m_r1y") / 100
        if st.button("➕ Ekle", key="m_add"):
            if new_code and new_name:
                ok, msg = add_fund(new_code, new_name, new_type, new_tax, new_vol, new_price, new_r1y)
                if ok: st.success(msg); st.rerun()
                else: st.error(msg)
            else: st.warning("Kod ve ad gerekli.")
    if st.session_state.funds:
        remove_code = st.selectbox("🗑️ Çıkarılacak Fon",
                                    list(st.session_state.funds.keys()), key="remove_select")
        if st.button("Çıkar"):
            ok, msg = remove_fund(remove_code)
            if ok: st.success(msg); st.rerun()
            else: st.error(msg)
    st.markdown("---")
    tefas_ok = st.session_state.use_tefas and len(st.session_state.get("tefas_series", {})) > 0
    st.caption(f"📌 Fiyat kaynağı: **{'🟢 TEFAS' if tefas_ok else '🟡 Simülasyon'}**")


# ============================================================
# BAŞLIK
# ============================================================
st.title("📈 Katılım Fonu Paper Trade & Rebalans Paneli")
st.caption(f"Başlangıç: **{st.session_state.start_date.strftime('%d.%m.%Y')}** | "
           f"Sermaye: **{st.session_state.initial_capital:,.0f} TL** | "
           f"Referans: **{SIM_END_DATE.strftime('%d.%m.%Y')}**")

# ============================================================
# ANA HESAPLAR
# ============================================================
positions = build_portfolio()
today = SIM_END_DATE

rows = []
for code, p in positions.items():
    cur_price = get_price(code, today)
    value = p["units"] * cur_price
    meta = get_fund_meta(code)
    gross_pl = value - p["cost"]
    tax_rate = get_tax_rate(code)
    tax_amount = max(0.0, gross_pl) * tax_rate
    net_value = value - tax_amount
    net_pl = gross_pl - tax_amount
    rows.append({
        "Fon": code, "Ad": meta.get("name", code), "Tür": meta.get("type", "—"),
        "Birim": p["units"], "Alış Fiyatı": p["buy_price"], "Güncel Fiyat": cur_price,
        "Maliyet": p["cost"], "Güncel Değer": value,
        "Brüt K/Z (TL)": gross_pl,
        "Brüt K/Z (%)": (gross_pl / p["cost"] * 100) if p["cost"] else 0,
        "Stopaj (%)": tax_rate * 100, "Stopaj (TL)": tax_amount,
        "Net Değer": net_value, "Net K/Z (TL)": net_pl,
        "Net K/Z (%)": (net_pl / p["cost"] * 100) if p["cost"] else 0,
    })

df_pos = pd.DataFrame(rows)
if not df_pos.empty:
    for col in ["Birim", "Alış Fiyatı", "Güncel Fiyat", "Maliyet", "Güncel Değer",
                "Brüt K/Z (TL)", "Brüt K/Z (%)", "Stopaj (TL)", "Net Değer",
                "Net K/Z (TL)", "Net K/Z (%)"]:
        if col in df_pos.columns:
            df_pos[col] = df_pos[col].round(4)

total_value = df_pos["Güncel Değer"].sum() if not df_pos.empty else 0
total_net_value = df_pos["Net Değer"].sum() if not df_pos.empty else 0
total_gross_pl = df_pos["Brüt K/Z (TL)"].sum() if not df_pos.empty else 0
total_tax = df_pos["Stopaj (TL)"].sum() if not df_pos.empty else 0
total_net_pl = df_pos["Net K/Z (TL)"].sum() if not df_pos.empty else 0

if not df_pos.empty:
    df_pos["Ağırlık (%)"] = (df_pos["Güncel Değer"] / total_value * 100).round(4) if total_value else 0
    df_pos["Hedef (%)"] = df_pos["Fon"].map(st.session_state.target_weights).fillna(0) * 100
    df_pos["Sapma (pp)"] = (df_pos["Ağırlık (%)"] - df_pos["Hedef (%)"]).round(4)

days_held = max((today - st.session_state.start_date).days, 0)
nominal_net = total_net_value / st.session_state.initial_capital - 1 if st.session_state.initial_capital else 0
if days_held < 1:
    reel_val = None; doviz_val = None
else:
    tufe_period = (1 + MACRO["tufe"]) ** (days_held / 365) - 1
    usd_period = (1 + USD_RET_ASSUMPTION) ** (days_held / 365) - 1
    reel_val = nominal_net - tufe_period
    doviz_val = nominal_net - usd_period

totals = {"total_value": total_value, "total_net": total_net_value,
          "gross_pl": total_gross_pl, "total_tax": total_tax,
          "net_pl": total_net_pl, "nominal": nominal_net,
          "reel": reel_val, "doviz": doviz_val}

sharpe_data = calculate_all_sharpes(positions)
_, _, next_reb = compute_rebalance_dates(st.session_state.start_date,
                                          st.session_state.rebalance_freq)
red_flags = check_red_flags(df_pos, totals, sharpe_data, days_held)
kritik_count = sum(1 for f in red_flags if f["seviye"] == "kritik")
uyari_count = sum(1 for f in red_flags if f["seviye"] == "uyari")

# KIRMIZI BAYRAK PANELİ
if red_flags:
    if kritik_count > 0:
        st.error(f"🚨 **{kritik_count} KRİTİK UYARI** ve {uyari_count} uyarı tespit edildi")
    else:
        st.warning(f"⚠️ **{uyari_count} uyarı** tespit edildi")
    with st.expander(f"🔍 Uyarı Detaylarını Göster ({len(red_flags)} adet)",
                     expanded=(kritik_count > 0)):
        for flag in red_flags:
            if flag["seviye"] == "kritik":
                st.error(f"**{flag['baslik']}**\n\n{flag['mesaj']}\n\n💡 *{flag['aksiyon']}*")
            else:
                st.warning(f"**{flag['baslik']}**\n\n{flag['mesaj']}\n\n💡 *{flag['aksiyon']}*")
else:
    st.success("✅ **Tüm risk göstergeleri normal** — kırmızı bayrak yok")

# KPI
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Portföy (Brüt)", f"{total_value:,.0f} TL",
          f"{(total_value/st.session_state.initial_capital - 1)*100:+.2f}%" if st.session_state.initial_capital else "—")
c2.metric("Net (Stopaj Sonrası)", f"{total_net_value:,.0f} TL",
          f"{(total_net_value/st.session_state.initial_capital - 1)*100:+.2f}%" if st.session_state.initial_capital else "—")
c3.metric("Brüt K/Z", f"{total_gross_pl:+,.0f} TL")
c4.metric("Toplam Stopaj", f"-{total_tax:,.0f} TL",
          delta=f"{(total_tax/total_value*100):.2f}%" if total_value else None,
          delta_color="inverse")
c5.metric("Net K/Z", f"{total_net_pl:+,.0f} TL",
          f"{(total_net_pl/st.session_state.initial_capital)*100:+.2f}%" if st.session_state.initial_capital else "—")

c1b, c2b, c3b, c4b, c5b = st.columns(5)
if days_held < 1:
    c1b.metric("Net Reel Getiri", "—", "Bugün başladı")
    c2b.metric("Net Döviz Üstü", "—", "Bugün başladı")
else:
    c1b.metric("Net Reel Getiri", f"{reel_val*100:+.2f}%", f"{days_held} günlük")
    c2b.metric("Net Döviz Üstü", f"{doviz_val*100:+.2f}%", f"{days_held} günlük")
max_drift = df_pos["Sapma (pp)"].abs().max() if not df_pos.empty else 0
c3b.metric("Maks. Sapma", f"{max_drift:.2f} pp",
           delta="Rebalans gerekli" if max_drift > st.session_state.drift_threshold else "OK",
           delta_color="inverse" if max_drift > st.session_state.drift_threshold else "normal")
eff_tax_rate = (total_tax / total_gross_pl * 100) if total_gross_pl > 0 else 0
c4b.metric("Efektif Vergi", f"{eff_tax_rate:.2f}%")

if sharpe_data is not None:
    fwd = sharpe_data.get('sharpe_forward', 0)
    ann_ret = sharpe_data.get('annual_return', 0)
    color = "normal" if fwd > 1.0 else ("off" if fwd > 0.5 else "inverse")
    c5b.metric("Forward Sharpe", f"{fwd:+.3f}",
               f"Yıllık: {ann_ret*100:+.1f}%",
               delta_color=color)
else:
    c5b.metric("Forward Sharpe", "—", f"{days_held} gün (30+ gerekli)")

st.markdown("---")

# ============================================================
# SEKMELER
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs(
    ["📊 Genel Bakış", "📈 Performans", "⚖️ Rebalans",
     "🔍 Fon Analizi", "💰 Vergi Analizi", "📝 İşlem Geçmişi",
     "📤 Rapor & Export", "🎲 Monte Carlo", "📐 Risk Analizi",
     "🚨 Uyarılar", "📋 Paper Trade Sonuçları"])

# ---------------- TAB 1: GENEL BAKIŞ ----------------
with tab1:
    if df_pos.empty:
        st.info("Henüz pozisyon yok.")
    else:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("Pozisyonlar (Brüt & Net)")
            show = df_pos[["Fon", "Ad", "Tür", "Birim", "Alış Fiyatı", "Güncel Fiyat",
                           "Güncel Değer", "Brüt K/Z (%)", "Stopaj (%)", "Stopaj (TL)",
                           "Net Değer", "Net K/Z (%)", "Ağırlık (%)", "Hedef (%)", "Sapma (pp)"]]
            st.dataframe(show.style.format({
                "Birim": "{:,.2f}", "Alış Fiyatı": "{:,.4f}", "Güncel Fiyat": "{:,.4f}",
                "Güncel Değer": "{:,.0f}", "Brüt K/Z (%)": "{:+.2f}%",
                "Stopaj (%)": "{:.2f}%", "Stopaj (TL)": "{:,.0f}",
                "Net Değer": "{:,.0f}", "Net K/Z (%)": "{:+.2f}%",
                "Ağırlık (%)": "{:.2f}%", "Hedef (%)": "{:.2f}%", "Sapma (pp)": "{:+.2f}",
            }).background_gradient(subset=["Brüt K/Z (%)", "Net K/Z (%)"], cmap="RdYlGn")
              .background_gradient(subset=["Stopaj (TL)"], cmap="Reds")
              .background_gradient(subset=["Sapma (pp)"], cmap="RdBu_r"),
              use_container_width=True, hide_index=True)
        with col2:
            st.subheader("Brüt vs Net")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_pos["Fon"], y=df_pos["Güncel Değer"],
                                 name="Brüt", marker_color="#94a3b8"))
            fig.add_trace(go.Bar(x=df_pos["Fon"], y=df_pos["Net Değer"],
                                 name="Net", marker_color="#10b981"))
            fig.update_layout(barmode="group", height=380,
                              margin=dict(l=10, r=10, t=30, b=10),
                              legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Varlık Sınıfı Dağılımı")
        df_pos["Kategori"] = df_pos["Tür"].apply(
            lambda x: "Kira Sertifikası" if "Kira" in x else
                      "Hisse Senedi" if "Hisse" in x else
                      "Teknoloji" if "Teknoloji" in x else
                      "Altın" if "Altın" in x else
                      "Döviz" if "Döviz" in x else "Diğer")
        cat = df_pos.groupby("Kategori")["Güncel Değer"].sum().reset_index()
        fig2 = px.pie(cat, names="Kategori", values="Güncel Değer", hole=0.45,
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_traces(textinfo="percent+label")
        fig2.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig2, use_container_width=True)

# ---------------- TAB 2: PERFORMANS ----------------
with tab2:
    st.subheader("Portföy Değeri vs Benchmarklar (Net)")
    df_hist = st.session_state.price_history.copy()
    if st.session_state.get("use_tefas", False) and st.session_state.get("tefas_series"):
        all_s = [s for s in st.session_state.tefas_series.values() if s is not None and len(s) > 0]
        if all_s:
            idx = max(all_s, key=len).index
            df_hist = pd.DataFrame(index=idx)
            for code, series in st.session_state.tefas_series.items():
                if series is not None and len(series) > 0:
                    df_hist[code] = series.reindex(idx).ffill()
            df_hist = df_hist.dropna(how="all")
    if not df_hist.empty:
        port_series = pd.Series([net_portfolio_on(d, positions) for d in df_hist.index],
                                 index=df_hist.index)
        start_val = port_series.asof(pd.Timestamp(st.session_state.start_date))
        if start_val and start_val > 0:
            port_series = port_series / start_val * st.session_state.initial_capital
        days_elapsed = pd.Series(np.maximum((df_hist.index - st.session_state.start_date).days, 0),
                                  index=df_hist.index)
        tufe_series = st.session_state.initial_capital * (1 + MACRO["tufe"]) ** (days_elapsed / 365)
        usd_series = st.session_state.initial_capital * (1 + USD_RET_ASSUMPTION) ** (days_elapsed / 365)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port_series.index, y=port_series, name="Portföy (Net)",
                                 line=dict(color="#10b981", width=3)))
        fig.add_trace(go.Scatter(x=tufe_series.index, y=tufe_series,
                                 name=f"TÜFE (%{MACRO['tufe']*100:.1f})",
                                 line=dict(color="#f59e0b", width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=usd_series.index, y=usd_series,
                                 name=f"USD/TRY (%{USD_RET_ASSUMPTION*100:.0f})",
                                 line=dict(color="#0ea5e9", width=2, dash="dot")))
        fig.add_vline(x=st.session_state.start_date.timestamp() * 1000,
                      line_dash="dash", line_color="gray",
                      annotation_text="Başlangıç", annotation_position="top")
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10),
                          yaxis_title="TL", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Dönemsel Getiri (Net)")
        periods = {"1 Hafta": 7, "1 Ay": 30, "3 Ay": 90, "6 Ay": 180, "YBB": 365}
        perf_rows = []
        for label, days in periods.items():
            d0 = today - timedelta(days=days)
            v0 = net_portfolio_on(d0, positions); v1 = total_net_value
            if v0 <= 0: continue
            r = (v1 / v0 - 1) * 100
            tufe_r = ((1 + MACRO["tufe"]) ** (days / 365) - 1) * 100
            usd_r = ((1 + USD_RET_ASSUMPTION) ** (days / 365) - 1) * 100
            perf_rows.append({"Dönem": label, "Net Portföy (%)": r, "TÜFE (%)": tufe_r,
                              "USD/TRY (%)": usd_r, "Reel Fark (pp)": r - tufe_r,
                              "Döviz Fark (pp)": r - usd_r})
        df_perf = pd.DataFrame(perf_rows)
        if not df_perf.empty:
            st.dataframe(df_perf.style.format({
                "Net Portföy (%)": "{:+.2f}", "TÜFE (%)": "{:+.2f}", "USD/TRY (%)": "{:+.2f}",
                "Reel Fark (pp)": "{:+.2f}", "Döviz Fark (pp)": "{:+.2f}",
            }).background_gradient(subset=["Reel Fark (pp)", "Döviz Fark (pp)"], cmap="RdYlGn"),
                use_container_width=True, hide_index=True)
        st.subheader("Drawdown")
        roll_max = port_series.cummax()
        dd = (port_series / roll_max - 1) * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy",
                                    line=dict(color="#ef4444"), name="Drawdown"))
        fig_dd.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10),
                             yaxis_title="%", yaxis_ticksuffix="%")
        st.plotly_chart(fig_dd, use_container_width=True)

# ---------------- TAB 3: REBALANS ----------------
with tab3:
    if not df_pos.empty:
        st.subheader("Hedef vs Gerçek — Sapma Analizi")
        drift = df_pos[["Fon", "Ad", "Hedef (%)", "Ağırlık (%)", "Sapma (pp)", "Güncel Değer"]].copy()
        drift["Hedef Değer"] = total_value * drift["Hedef (%)"] / 100
        drift["Fark (TL)"] = (drift["Hedef Değer"] - drift["Güncel Değer"]).round(2)
        drift["İşlem"] = drift["Fark (TL)"].apply(
            lambda x: "🟢 AL" if x > TOLERANCE_TL else ("🔴 SAT" if x < -TOLERANCE_TL else "—"))
        drift["Uyarı"] = drift["Sapma (pp)"].abs() > st.session_state.drift_threshold
        def color_islem(val):
            if "AL" in str(val): return "background-color: #d1fae5"
            if "SAT" in str(val): return "background-color: #fee2e2"
            return ""
        st.dataframe(drift.style.format({
            "Hedef (%)": "{:.2f}%", "Ağırlık (%)": "{:.2f}%", "Sapma (pp)": "{:+.2f}",
            "Güncel Değer": "{:,.0f}", "Hedef Değer": "{:,.0f}", "Fark (TL)": "{:+,.0f}",
        }).map(color_islem, subset=["İşlem"]), use_container_width=True, hide_index=True)
        n_alert = int(drift["Uyarı"].sum())
        if n_alert: st.warning(f"⚠️ **{n_alert} fon** eşiği aştı.")
        else: st.success("✅ Tüm fonlar hedef aralıkta.")
        st.subheader("🔄 Rebalans Simülasyonu")
        if st.button("Rebalansı Uygula (sanal)", type="primary"):
            st.session_state.manual_positions = {}
            st.session_state.transactions.append({
                "tarih": today, "tip": "REBALANS", "fon": "—", "birim": "—",
                "fiyat": "—", "tutar": f"{n_alert} fon düzeltildi", "not": ""})
            st.session_state.rebalance_log.append({
                "tarih": today, "toplam": total_value, "uyarı": n_alert})
            if st.session_state.get("auto_save", True): save_state()
            st.success(f"✅ {today.strftime('%d.%m.%Y')} rebalans uygulandı.")
            st.rerun()
        st.markdown("---")
        st.subheader("📅 Otomatik Rebalans Takvimi")
        past_rebs, future_rebs_tab, next_reb_tab = compute_rebalance_dates(
            st.session_state.start_date, st.session_state.rebalance_freq)
        if get_rebalance_rule(st.session_state.rebalance_freq) is None:
            st.info("ℹ️ Eşik bazlı rebalans seçildi.")
        else:
            if next_reb_tab is not None:
                d_next = days_until(next_reb_tab)
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Sonraki Rebalans", next_reb_tab.strftime("%d.%m.%Y"))
                col_b.metric("Kalan Gün", f"{d_next} gün")
                col_c.metric("Sıklık", st.session_state.rebalance_freq)
            st.markdown("**📜 Geçmiş Rebalans Tarihleri**")
            if past_rebs:
                past_df = pd.DataFrame([{
                    "Tarih": d.strftime("%d.%m.%Y"),
                    "Durum": "✅ Uygulandı" if any(
                        abs((pd.Timestamp(t["tarih"]) - d).days) <= 1
                        for t in st.session_state.transactions if t["tip"] == "REBALANS"
                    ) else "⚠️ Atlandı",
                } for d in past_rebs])
                st.dataframe(past_df, use_container_width=True, hide_index=True)
    else:
        st.info("Pozisyon yok.")

# ---------------- TAB 4: FON ANALİZİ ----------------
with tab4:
    st.subheader("Fon Karşılaştırma Tablosu")
    fund_rows = []
    for code in st.session_state.funds:
        meta = get_fund_meta(code)
        fund_rows.append({
            "Kod": code, "Ad": meta.get("name", code), "Tür": meta.get("type", "—"),
            "Güncel Fiyat": meta.get("price", 0), "1 Yıl (%)": (meta.get("r1y") or 0) * 100,
            "Stopaj (%)": (meta.get("tax") or 0) * 100, "Volatilite": meta.get("vol", 0.30)})
    df_funds = pd.DataFrame(fund_rows)
    st.dataframe(df_funds.style.format({
        "Güncel Fiyat": "{:,.4f}", "1 Yıl (%)": "{:+.2f}",
        "Stopaj (%)": "{:.2f}", "Volatilite": "{:.2f}"}),
        use_container_width=True, hide_index=True)
    st.subheader("Korelasyon Matrisi")
    if len(st.session_state.funds) > 1:
        rets = st.session_state.price_history.pct_change().dropna()
        if not rets.empty:
            fig_c = px.imshow(rets.corr(), text_auto=".2f",
                              color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            fig_c.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_c, use_container_width=True)

# ---------------- TAB 5: VERGİ ANALİZİ ----------------
with tab5:
    if not df_pos.empty:
        st.subheader("💰 Stopaj Sonrası Net Getiri")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Brüt Değer", f"{total_value:,.0f} TL")
        v2.metric("Stopaj", f"-{total_tax:,.0f} TL",
                  delta=f"{(total_tax/total_value*100):.2f}%" if total_value else None,
                  delta_color="inverse")
        v3.metric("Net Değer", f"{total_net_value:,.0f} TL")
        v4.metric("Efektif Vergi", f"{eff_tax_rate:.2f}%")
        tax_tbl = df_pos[["Fon", "Ad", "Tür", "Maliyet", "Güncel Değer",
                          "Brüt K/Z (TL)", "Stopaj (%)", "Stopaj (TL)",
                          "Net Değer", "Net K/Z (TL)", "Net K/Z (%)"]]
        st.dataframe(tax_tbl.style.format({
            "Maliyet": "{:,.0f}", "Güncel Değer": "{:,.0f}", "Brüt K/Z (TL)": "{:+,.0f}",
            "Stopaj (%)": "{:.2f}%", "Stopaj (TL)": "{:,.0f}", "Net Değer": "{:,.0f}",
            "Net K/Z (TL)": "{:+,.0f}", "Net K/Z (%)": "{:+.2f}%",
        }).background_gradient(subset=["Stopaj (TL)"], cmap="Reds")
          .background_gradient(subset=["Net K/Z (%)"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True)

# ---------------- TAB 6: İŞLEM GEÇMİŞİ ----------------
with tab6:
    st.subheader("📝 Manuel İşlem Girişi")
    if st.session_state.funds:
        with st.form("manual_trade", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                trade_date = st.date_input("İşlem Tarihi", value=today.date())
                trade_code = st.selectbox("Fon", list(st.session_state.funds.keys()),
                                          format_func=lambda x: f"{x} — {st.session_state.funds[x]['name'][:30]}")
            with c2:
                trade_type = st.selectbox("Tip", ["AL", "SAT"])
                trade_units = st.number_input("Birim", min_value=0.01, value=100.0, step=10.0)
            with c3:
                default_price = get_price(trade_code, datetime.combine(trade_date, datetime.min.time()))
                trade_price = st.number_input("Fiyat (TL)", min_value=0.0001,
                                              value=float(round(default_price, 4)),
                                              step=0.0001, format="%.4f")
                trade_note = st.text_input("Not", "")
            submitted = st.form_submit_button("✅ Kaydet", type="primary")
            if submitted:
                amount = trade_units * trade_price
                st.session_state.transactions.append({
                    "tarih": datetime.combine(trade_date, datetime.min.time()),
                    "tip": trade_type, "fon": trade_code, "birim": trade_units,
                    "fiyat": trade_price, "tutar": amount, "not": trade_note})
                mp = st.session_state.manual_positions.get(trade_code, {"units": 0.0, "cost": 0.0})
                if trade_type == "AL":
                    mp["units"] += trade_units; mp["cost"] += amount
                else:
                    mp["units"] = max(0.0, mp["units"] - trade_units)
                    mp["cost"] = max(0.0, mp["cost"] - amount)
                if mp["units"] <= 0: st.session_state.manual_positions.pop(trade_code, None)
                else:
                    mp["buy_price"] = mp["cost"] / mp["units"]
                    st.session_state.manual_positions[trade_code] = mp
                if st.session_state.get("auto_save", True): save_state()
                st.success(f"✅ {trade_type}: {trade_code} {trade_units:.2f} @ {trade_price:.4f}")
                time.sleep(0.5); st.rerun()
    st.markdown("---")
    st.subheader("📋 İşlem Geçmişi")
    st.markdown("**🤖 Otomatik İşlem Oluştur**")
    colA, colB, colC = st.columns([2, 2, 1])
    with colA:
        auto_tpl = st.selectbox("Şablon", list(PORTFOLIO_TEMPLATES.keys()), key="auto_tx_template")
    with colB:
        tx_mode_inline = st.selectbox("Mod",
            ["📌 Ekle (korumalı)", "🔄 Aynı şablonu güncelle", "🗑️ Tümünü yenile"],
            key="tx_mode_inline")
        tx_mode_inline_map = {"📌 Ekle (korumalı)": "append",
                              "🔄 Aynı şablonu güncelle": "replace_same",
                              "🗑️ Tümünü yenile": "replace_all"}
    with colC:
        st.write(""); st.write("")
        if st.button("🤖 Oluştur", use_container_width=True):
            ok, msg = auto_generate_paper_trades(auto_tpl,
                mode=tx_mode_inline_map[tx_mode_inline])
            if ok: st.success(msg); time.sleep(0.5); st.rerun()
            else: st.error(msg)
    st.markdown("---")
    all_tx = []
    for t in st.session_state.transactions:
        all_tx.append({
            "Tarih": t["tarih"].strftime("%d.%m.%Y") if hasattr(t["tarih"], "strftime") else str(t["tarih"]),
            "Tip": t["tip"], "Fon": t.get("fon", "—"), "Birim": t.get("birim", "—"),
            "Fiyat": t.get("fiyat", "—"), "Tutar": t.get("tutar", "—"),
            "Not": t.get("not", "")})
    df_tx = pd.DataFrame(all_tx) if all_tx else pd.DataFrame()
    if not df_tx.empty:
        st.dataframe(df_tx.style.format({
            "Birim": "{:,.2f}", "Fiyat": "{:,.4f}", "Tutar": "{:,.0f}"}, na_rep="—"),
            use_container_width=True, hide_index=True)
    else: st.info("Henüz işlem kaydı yok.")

# ---------------- TAB 7: RAPOR & EXPORT ----------------
with tab7:
    st.subheader("📤 Rapor ve Veri Dışa Aktarma")
    auto_txs_for_excel = [t for t in st.session_state.transactions if t.get("tip") == "OTOMATİK ALIM"]
    paper_groups_for_excel = {}
    if auto_txs_for_excel:
        tmp_groups = {}
        for t in auto_txs_for_excel:
            tpl = t.get("template", "Bilinmeyen")
            tmp_groups.setdefault(tpl, []).append(t)
        for tpl, txs in tmp_groups.items():
            pnl_data = calculate_group_pnl(txs)
            paper_groups_for_excel[tpl] = pnl_data["rows"]
    col_pdf, col_xlsx = st.columns(2)
    with col_pdf:
        st.markdown("### 📄 PDF Rapor")
        if st.button("📄 PDF Rapor Oluştur", type="primary", use_container_width=True):
            with st.spinner("PDF üretiliyor..."):
                pdf_bytes, err = build_pdf_report(df_pos, totals, positions,
                                                   days_held, next_reb, sharpe_data, red_flags)
            if err: st.error(f"❌ {err}")
            elif pdf_bytes:
                st.session_state["_pdf_bytes"] = pdf_bytes
                st.success(f"✅ PDF hazır ({len(pdf_bytes):,} byte)")
        if st.session_state.get("_pdf_bytes"):
            st.download_button("⬇️ PDF İndir",
                               data=st.session_state["_pdf_bytes"],
                               file_name=f"portfoy_raporu_{SIM_END_DATE.strftime('%Y%m%d')}.pdf",
                               mime="application/pdf", use_container_width=True)
    with col_xlsx:
        st.markdown("### 📊 Excel Export")
        if st.button("📊 Excel Oluştur", type="primary", use_container_width=True):
            with st.spinner("Excel üretiliyor..."):
                fund_rows = []
                for code in st.session_state.funds:
                    meta = get_fund_meta(code)
                    fund_rows.append({
                        "Kod": code, "Ad": meta.get("name", code),
                        "Tür": meta.get("type", "—"),
                        "Güncel Fiyat": meta.get("price", 0),
                        "1 Yıl (%)": (meta.get("r1y") or 0) * 100,
                        "Stopaj (%)": (meta.get("tax") or 0) * 100,
                        "Volatilite": meta.get("vol", 0.30)})
                df_funds_exp = pd.DataFrame(fund_rows)
                periods = {"1 Hafta": 7, "1 Ay": 30, "3 Ay": 90, "6 Ay": 180, "YBB": 365}
                perf_rows = []
                for label, days in periods.items():
                    d0 = today - timedelta(days=days)
                    v0 = net_portfolio_on(d0, positions); v1 = total_net_value
                    if v0 <= 0: continue
                    r = (v1 / v0 - 1) * 100
                    tufe_r = ((1 + MACRO["tufe"]) ** (days / 365) - 1) * 100
                    usd_r = ((1 + USD_RET_ASSUMPTION) ** (days / 365) - 1) * 100
                    perf_rows.append({"Dönem": label, "Net Portföy (%)": r,
                                      "TÜFE (%)": tufe_r, "USD/TRY (%)": usd_r,
                                      "Reel Fark (pp)": r - tufe_r,
                                      "Döviz Fark (pp)": r - usd_r})
                df_perf_exp = pd.DataFrame(perf_rows)
                xlsx_bytes = build_excel(df_pos, df_funds_exp, df_tx, df_perf_exp,
                                          totals, sharpe_data, mc_data=None,
                                          red_flags=red_flags,
                                          paper_trade_groups=paper_groups_for_excel)
                st.session_state["_xlsx_bytes"] = xlsx_bytes
                st.success(f"✅ Excel hazır ({len(xlsx_bytes):,} byte)")
        if st.session_state.get("_xlsx_bytes"):
            st.download_button("⬇️ Excel İndir",
                               data=st.session_state["_xlsx_bytes"],
                               file_name=f"portfoy_{SIM_END_DATE.strftime('%Y%m%d')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

# ---------------- TAB 8: MONTE CARLO ----------------
with tab8:
    st.subheader("🎲 Monte Carlo Simülasyonu")
    st.markdown(f"**{MC_SIMULATIONS:,}** simülasyon × **{MC_DAYS}** iş günü (~1 yıl)")
    if days_held < 30:
        st.info(f"ℹ️ Monte Carlo için en az 30 günlük veri gerekli. "
                f"Şu an **{days_held} gün** geçti.")
    else:
        if st.button("🎲 Simülasyonu Çalıştır", type="primary", use_container_width=True):
            with st.spinner(f"{MC_SIMULATIONS} senaryo hesaplanıyor..."):
                mc_result = run_monte_carlo(positions, n_sims=MC_SIMULATIONS, n_days=MC_DAYS)
                st.session_state["_mc_result"] = mc_result
        mc_result = st.session_state.get("_mc_result")
        if mc_result is not None:
            final_vals = mc_result[-1, :]
            initial = st.session_state.initial_capital
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Ortalama", f"{np.mean(final_vals):,.0f} TL",
                       f"{(np.mean(final_vals)/initial - 1)*100:+.2f}%")
            mc2.metric("Medyan", f"{np.median(final_vals):,.0f} TL",
                       f"{(np.median(final_vals)/initial - 1)*100:+.2f}%")
            mc3.metric("5. Persentil", f"{np.percentile(final_vals, 5):,.0f} TL",
                       f"{(np.percentile(final_vals, 5)/initial - 1)*100:+.2f}%",
                       delta_color="inverse")
            mc4.metric("95. Persentil", f"{np.percentile(final_vals, 95):,.0f} TL",
                       f"{(np.percentile(final_vals, 95)/initial - 1)*100:+.2f}%")
            prob_profit = (final_vals > initial).mean() * 100
            prob_tufe = (final_vals > initial * (1 + MACRO["tufe"])).mean() * 100
            prob_usd = (final_vals > initial * (1 + USD_RET_ASSUMPTION)).mean() * 100
            p1, p2, p3 = st.columns(3)
            p1.metric("Kâr Olasılığı", f"{prob_profit:.1f}%")
            p2.metric("Enflasyon Üstü Olasılığı", f"{prob_tufe:.1f}%",
                      delta_color="normal" if prob_tufe > 50 else "inverse")
            p3.metric("Döviz Üstü Olasılığı", f"{prob_usd:.1f}%",
                      delta_color="normal" if prob_usd > 50 else "inverse")
            fig_mc = go.Figure()
            n_show = min(100, mc_result.shape[1])
            for i in range(n_show):
                fig_mc.add_trace(go.Scatter(y=mc_result[:, i], mode="lines",
                                            line=dict(width=0.5, color="rgba(14,165,233,0.15)"),
                                            showlegend=False, hoverinfo="skip"))
            fig_mc.add_trace(go.Scatter(y=mc_result.mean(axis=1), mode="lines",
                                        line=dict(color="#10b981", width=3), name="Ortalama"))
            fig_mc.add_hline(y=initial, line_dash="dash", line_color="gray",
                             annotation_text="Başlangıç")
            fig_mc.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10),
                                 yaxis_title="TL", xaxis_title="İş Günü")
            st.plotly_chart(fig_mc, use_container_width=True)
        else:
            st.info("👆 Simülasyonu çalıştırmak için butona basın.")

# ---------------- TAB 9: RİSK ANALİZİ ----------------
with tab9:
    st.subheader("📐 Risk ve Performans Metrikleri")
    if sharpe_data is None:
        st.info(f"ℹ️ **Risk metrikleri için yeterli veri yok.** "
                f"En az **30 günlük** veri gerekli. "
                f"Başlangıç tarihinizden bu yana **{days_held} gün** geçti.")
        if days_held < 30:
            kalan = 30 - days_held
            st.warning(f"⏳ Risk metrikleri **{kalan} gün** sonra aktif olacak.")
    else:
        s = sharpe_data
        st.markdown("### Sharpe Oranı (3 Farklı Referans)")
        ss1, ss2, ss3 = st.columns(3)
        ss1.metric("Nominal Sharpe", f"{s['sharpe_nominal']:+.3f}",
                   "Risksiz %45 (2026)",
                   delta_color="inverse" if s['sharpe_nominal'] < 0.5 else "normal")
        ss2.metric("Forward Sharpe", f"{s['sharpe_forward']:+.3f}",
                   "Risksiz %30 (2027)",
                   delta_color="inverse" if s['sharpe_forward'] < 0.5 else "normal")
        ss3.metric("Reel Sharpe", f"{s['sharpe_reel']:+.3f}",
                   f"Risksiz = TÜFE %{MACRO['tufe']*100:.1f}",
                   delta_color="inverse" if s['sharpe_reel'] < 0.5 else "normal")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Yıllık Getiri", f"{s['annual_return']*100:+.2f}%")
        s2.metric("Yıllık Volatilite", f"{s['annual_vol']*100:.2f}%")
        s3.metric("Maks. Drawdown", f"{s['max_dd']*100:.2f}%", delta_color="inverse")
        s4.metric("Veri Noktası", f"{s['n_days']} gün")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("VaR %95 (Günlük)", f"{s['var_95']*100:.3f}%")
        r2.metric("CVaR %95", f"{s['cvar_95']*100:.3f}%")
        if s.get("sortino_reel") is not None:
            r3.metric("Reel Sortino", f"{s['sortino_reel']:.3f}")
        r4.metric("Maks. Drawdown", f"{s['max_dd']*100:.2f}%", delta_color="inverse")

# ---------------- TAB 10: UYARILAR ----------------
with tab10:
    st.subheader("🚨 Kırmızı Bayrak Sistemi")
    if days_held < 30:
        st.info(f"ℹ️ **Uyarı sistemi için en az 30 gün gerekli.** "
                f"Şu an **{days_held} gün** geçti.")
    if red_flags:
        kritik = [f for f in red_flags if f["seviye"] == "kritik"]
        uyarilar = [f for f in red_flags if f["seviye"] == "uyari"]
        col1, col2 = st.columns(2)
        col1.metric("🚨 Kritik Uyarılar", len(kritik),
                    delta_color="inverse" if kritik else "normal")
        col2.metric("⚠️ Uyarılar", len(uyarilar), delta_color="off")
        st.markdown("---")
        for flag in red_flags:
            if flag["seviye"] == "kritik":
                st.error(f"### 🚨 {flag['baslik']}\n**Durum:** {flag['mesaj']}\n\n**💡 Aksiyon:** {flag['aksiyon']}")
            else:
                st.warning(f"### ⚠️ {flag['baslik']}\n**Durum:** {flag['mesaj']}\n\n**💡 Aksiyon:** {flag['aksiyon']}")
    else:
        st.success("### ✅ Tüm Risk Göstergeleri Normal")
    st.markdown("---")
    st.subheader("📋 Kontrol Edilen Kriterler")
    thresholds = RED_FLAG_THRESHOLDS
    criteria = pd.DataFrame([
        {"Kriter": "Maks. Drawdown",
         "Kritik Eşik": f"{thresholds['max_drawdown_kritik']*100:.0f}%",
         "Uyarı Eşiği": f"{thresholds['max_drawdown_uyari']*100:.0f}%"},
        {"Kriter": "Yıllık Volatilite",
         "Kritik Eşik": f"{thresholds['volatilite_kritik']*100:.0f}%",
         "Uyarı Eşiği": f"{thresholds['volatilite_uyari']*100:.0f}%"},
        {"Kriter": "Forward Sharpe", "Kritik Eşik": "Negatif", "Uyarı Eşiği": "—"},
        {"Kriter": "Tek Fon Konsantrasyonu",
         "Kritik Eşik": f"{thresholds['konsantrasyon_kritik']:.0f}%",
         "Uyarı Eşiği": f"{thresholds['konsantrasyon_uyari']:.0f}%"},
        {"Kriter": "Tek Sektör Konsantrasyonu",
         "Kritik Eşik": f"{thresholds['sektor_konsantrasyon']:.0f}%",
         "Uyarı Eşiği": "—"},
        {"Kriter": "Reel Getiri",
         "Kritik Eşik": f"{thresholds['reel_getiri_kritik']*100:.0f}%",
         "Uyarı Eşiği": "0%"},
    ])
    st.dataframe(criteria, use_container_width=True, hide_index=True)

# ---------------- TAB 11: PAPER TRADE SONUÇLARI ----------------
with tab11:
    st.subheader("📋 Paper Trade Sonuçları")
    st.markdown("Otomatik oluşturulan paper trade gruplarının **güncel K/Z durumu**")
    auto_txs = [t for t in st.session_state.transactions if t.get("tip") == "OTOMATİK ALIM"]
    if not auto_txs:
        st.info("Henüz otomatik paper trade oluşturulmadı. "
                "Sidebar → Portföy Şablonları → 🤖 Otomatik kayıt seçeneğini kullanın.")
    else:
        groups = {}
        for t in auto_txs:
            tpl = t.get("template", "Bilinmeyen")
            groups.setdefault(tpl, []).append(t)
        total_cost_all = 0; total_value_all = 0
        for tpl, txs in groups.items():
            pnl_data = calculate_group_pnl(txs)
            total_cost_all += pnl_data["total_cost"]
            total_value_all += pnl_data["current_value"]
        genel_pnl_pct = (total_value_all / total_cost_all - 1) * 100 if total_cost_all > 0 else 0
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Toplam Grup", len(groups))
        g2.metric("Toplam İşlem", len(auto_txs))
        g3.metric("Toplam Maliyet", f"{total_cost_all:,.0f} TL")
        g4.metric("Güncel Değer", f"{total_value_all:,.0f} TL", f"{genel_pnl_pct:+.2f}%")
        st.markdown("---")
        for tpl_name, txs in groups.items():
            with st.expander(f"🎯 **{tpl_name}** — {len(txs)} işlem", expanded=True):
                pnl_data = calculate_group_pnl(txs)
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Maliyet", f"{pnl_data['total_cost']:,.0f} TL")
                k2.metric("Güncel Değer", f"{pnl_data['current_value']:,.0f} TL")
                k3.metric("K/Z (TL)", f"{pnl_data['total_pnl']:+,.0f} TL",
                          delta_color="normal" if pnl_data['total_pnl'] >= 0 else "inverse")
                k4.metric("K/Z (%)", f"{pnl_data['total_pnl_pct']:+.2f}%",
                          delta_color="normal" if pnl_data['total_pnl_pct'] >= 0 else "inverse")
                df_group = pd.DataFrame(pnl_data["rows"])
                st.dataframe(df_group.style.format({
                    "Birim": "{:,.2f}", "Giriş Fiyatı": "{:,.4f}",
                    "Güncel Fiyat": "{:,.4f}", "Maliyet": "{:,.0f}",
                    "Güncel Değer": "{:,.0f}", "K/Z (TL)": "{:+,.0f}",
                    "K/Z (%)": "{:+.2f}%",
                }).background_gradient(subset=["K/Z (%)"], cmap="RdYlGn"),
                    use_container_width=True, hide_index=True)
                created = txs[0].get("created_at", "—")
                if isinstance(created, datetime):
                    created = created.strftime("%d.%m.%Y %H:%M")
                st.caption(f"📅 Oluşturulma: {created}")
        st.markdown("---")
        st.subheader("📊 Şablon Karşılaştırması")
        comp_rows = []
        for tpl_name, txs in groups.items():
            pnl_data = calculate_group_pnl(txs)
            comp_rows.append({
                "Şablon": tpl_name, "Maliyet": pnl_data["total_cost"],
                "Güncel Değer": pnl_data["current_value"],
                "K/Z (TL)": pnl_data["total_pnl"], "K/Z (%)": pnl_data["total_pnl_pct"]})
        df_comp = pd.DataFrame(comp_rows).sort_values("K/Z (%)", ascending=False)
        fig_comp = go.Figure()
        colors = ["#10b981" if v >= 0 else "#ef4444" for v in df_comp["K/Z (%)"]]
        fig_comp.add_trace(go.Bar(x=df_comp["Şablon"], y=df_comp["K/Z (%)"],
                                  marker_color=colors,
                                  text=df_comp["K/Z (%)"].round(2).astype(str) + "%",
                                  textposition="outside"))
        fig_comp.add_hline(y=0, line_color="gray")
        fig_comp.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10),
                                yaxis_title="K/Z (%)", showlegend=False)
        st.plotly_chart(fig_comp, use_container_width=True)
        st.dataframe(df_comp.style.format({
            "Maliyet": "{:,.0f}", "Güncel Değer": "{:,.0f}",
            "K/Z (TL)": "{:+,.0f}", "K/Z (%)": "{:+.2f}%",
        }).background_gradient(subset=["K/Z (%)"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
tefas_aktif = st.session_state.use_tefas and len(st.session_state.get("tefas_series", {})) > 0
src_note = ("🟢 TEFAS canlı verisi kullanılıyor." if tefas_aktif
            else "🟡 Simülasyon fiyatları kullanılıyor.")
flag_note = ""
if red_flags:
    flag_note = f" | 🚨 **{kritik_count} kritik, {uyari_count} uyarı**"
save_note = ""
if STATE_FILE.exists():
    try:
        last = datetime.fromtimestamp(STATE_FILE.stat().st_mtime)
        save_note = f" | 💾 Son kayıt: {last.strftime('%d.%m %H:%M')}"
    except Exception: pass
st.caption(f"⚠️ **Uyarı:** Bu panel bir *paper trade* simülasyonudur. {src_note}{flag_note}{save_note} "
           "Stopaj oranları 2026 vergi mevzuatına göre varsayılmıştır.")