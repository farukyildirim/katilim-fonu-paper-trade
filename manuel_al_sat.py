import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from io import BytesIO
import time
import os

# ============================================================
# SAYFA AYARLARI
# ============================================================
st.set_page_config(
    page_title="Katılım Fonu Paper Trade & Rebalans",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SABİTLER
# ============================================================
SIM_END_DATE = datetime(2026, 9, 24)
USD_RET_ASSUMPTION = 0.20
TOLERANCE_TL = 10.0
MACRO = {"tufe": 0.3151, "usdtry_now": 48.75, "usdtry_12m_exp": 56.0}

# ============================================================
# FON VERİSİ
# ============================================================
FUNDS_FALLBACK = {
    "KPC": {"name": "Kuveyt Türk Katılım Hisse Senedi", "bank": "Kuveyt Türk", "type": "Hisse Senedi",
            "price": 20.967209, "r1y": 0.4171, "r3y": 1.4815, "r5y": 14.1084,
            "cagr3": 0.3538, "cagr5": 0.7212, "risk": 6, "fee": 0.0240, "tax": 0.0, "vol": 0.38,
            "tefas": "Açık"},
    "RBH": {"name": "Albaraka Katılım Hisse Senedi", "bank": "Albaraka", "type": "Hisse Senedi",
            "price": 33.184389, "r1y": 0.3447, "r3y": 1.2738, "r5y": 11.4557,
            "cagr3": 0.3150, "cagr5": 0.6561, "risk": 5, "fee": 0.0195, "tax": 0.0, "vol": 0.35,
            "tefas": "Açık"},
    "KTJ": {"name": "Kuveyt Türk Teknoloji Katılım", "bank": "Kuveyt Türk", "type": "Teknoloji",
            "price": 2.788390, "r1y": 0.6401, "r3y": None, "r5y": None,
            "cagr3": None, "cagr5": None, "risk": 6, "fee": 0.0240, "tax": 0.175, "vol": 0.45,
            "tefas": "Açık"},
    "KZL": {"name": "Kuveyt Türk Altın Katılım", "bank": "Kuveyt Türk", "type": "Altın",
            "price": 28.316445, "r1y": 0.3246, "r3y": 3.1162, "r5y": 13.2560,
            "cagr3": 0.6026, "cagr5": 0.7014, "risk": 6, "fee": 0.0030, "tax": 0.175, "vol": 0.28,
            "tefas": "Açık"},
    "CPU": {"name": "Aktif Portföy Teknoloji Katılım", "bank": "Aktif", "type": "Teknoloji",
            "price": 3.952677, "r1y": 0.7873, "r3y": None, "r5y": None,
            "cagr3": None, "cagr5": None, "risk": 5, "fee": 0.0200, "tax": 0.0, "vol": 0.42,
            "tefas": "Açık"},
    "OFK": {"name": "OYAK Türkiye Finans Katılım Serbest Döviz-USD", "bank": "OYAK", "type": "Döviz",
            "price": 53.031734, "r1y": 0.2012, "r3y": 0.9218, "r5y": None,
            "cagr3": 0.2433, "cagr5": None, "risk": 4, "fee": 0.0100, "tax": 0.175, "vol": 0.18,
            "tefas": "Kapalı"},
    "KIS": {"name": "Astra Portföy Kira Sertifikası Katılım Döviz", "bank": "Astra", "type": "Kira Sert. (Döviz)",
            "price": 0.223354, "r1y": 0.1941, "r3y": 1.0654, "r5y": 5.5621,
            "cagr3": 0.2735, "cagr5": 0.4568, "risk": 6, "fee": 0.0125, "tax": 0.175, "vol": 0.22,
            "tefas": "Açık"},
    "ZPG": {"name": "Ziraat Portföy Kira Sertifikaları Sukuk Katılım", "bank": "Ziraat", "type": "Kira Sert. (TL)",
            "price": 11.018887, "r1y": 0.4132, "r3y": 2.0908, "r5y": 4.1530,
            "cagr3": 0.4567, "cagr5": 0.3881, "risk": 2, "fee": 0.0150, "tax": 0.0, "vol": 0.12,
            "tefas": "Açık"},
    "KTN": {"name": "Kuveyt Türk Kira Sertifikaları Katılım TL", "bank": "Kuveyt Türk", "type": "Kira Sert. (TL)",
            "price": 8.240707, "r1y": 0.3603, "r3y": 1.6374, "r5y": 3.0945,
            "cagr3": 0.3816, "cagr5": 0.3257, "risk": 2, "fee": 0.0150, "tax": 0.0, "vol": 0.10,
            "tefas": "Açık"},
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
        data = tefas.fetch(start=start, end=end, name=code, columns="info")
        if data is None or data.empty:
            return None
        data = data.sort_values("date")
        last, first = data.iloc[-1], data.iloc[0]
        price = float(last["price"])
        r1y = (price / float(first["price"]) - 1) if float(first["price"]) > 0 else None
        return {"price": price, "r1y": r1y, "date": str(last["date"])}
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_tefas(codes: tuple):
    return {c: fetch_tefas_price(c) for c in codes}


# ============================================================
# FİYAT SİMÜLASYONU
# ============================================================
@st.cache_data(show_spinner=False)
def generate_price_history(days: int = 365) -> pd.DataFrame:
    end_date = SIM_END_DATE
    dates = pd.date_range(end=end_date, periods=days, freq="D")
    out = {"date": dates}
    for code, f in FUNDS_FALLBACK.items():
        n = days
        start_price = f["price"] / (1 + f["r1y"]) if f["r1y"] else f["price"]
        t = np.linspace(0, 1, n)
        np.random.seed(abs(hash(code)) % 2**32)
        dW = np.random.normal(0, np.sqrt(1 / n), n)
        W = np.cumsum(dW)
        W = W - t * W[-1]
        mu = np.log(f["price"] / start_price)
        path = start_price * np.exp(mu * t + f["vol"] * W)
        path[-1] = f["price"]
        out[code] = path
    return pd.DataFrame(out).set_index("date")


# ============================================================
# FİYAT / META
# ============================================================
def get_price(code: str, when: datetime) -> float:
    if st.session_state.get("use_tefas", False):
        tf = st.session_state.get("tefas_cache", {}).get(code)
        if tf and "price" in tf:
            return float(tf["price"])
    df = st.session_state.price_history
    when = pd.Timestamp(when).normalize()
    if when <= df.index[0]:
        return float(df[code].iloc[0])
    if when >= df.index[-1]:
        return float(df[code].iloc[-1])
    return float(df[code].asof(when))


def get_fund_meta(code: str) -> dict:
    meta = dict(FUNDS_FALLBACK.get(code, {}))
    if st.session_state.get("use_tefas", False):
        tf = st.session_state.get("tefas_cache", {}).get(code)
        if tf and "price" in tf:
            meta["price"] = tf["price"]
            if tf.get("r1y") is not None:
                meta["r1y"] = tf["r1y"]
    return meta


def get_tax_rate(code: str) -> float:
    return FUNDS_FALLBACK.get(code, {}).get("tax", 0.0)


# ============================================================
# PORTFÖY
# ============================================================
def build_portfolio():
    cap = st.session_state.initial_capital
    start = st.session_state.start_date
    positions = {}
    for code, w in st.session_state.target_weights.items():
        if w <= 0:
            continue
        price = get_price(code, start)
        positions[code] = {
            "units": (cap * w) / price,
            "cost": cap * w,
            "buy_price": price,
        }
    for code, mp in st.session_state.manual_positions.items():
        if code in positions:
            total_cost = positions[code]["cost"] + mp["cost"]
            total_units = positions[code]["units"] + mp["units"]
            positions[code] = {
                "units": total_units,
                "cost": total_cost,
                "buy_price": total_cost / total_units if total_units else 0,
            }
        else:
            positions[code] = dict(mp)
    return positions


def portfolio_value_on(date, positions):
    return sum(p["units"] * get_price(code, date) for code, p in positions.items())


def net_portfolio_on(date, positions):
    total = 0.0
    for code, p in positions.items():
        val = p["units"] * get_price(code, date)
        gross_pl = val - p["cost"]
        tax = max(0.0, gross_pl) * get_tax_rate(code)
        total += val - tax
    return total


# ============================================================
# REBALANS TAKVİMİ
# ============================================================
def get_rebalance_rule(freq: str):
    if freq == "3 Aylık":
        return pd.DateOffset(months=3)
    if freq == "6 Aylık":
        return pd.DateOffset(months=6)
    if freq == "Yıllık":
        return pd.DateOffset(years=1)
    return None


def compute_rebalance_dates(start_date, freq, months_back=12, months_ahead=12):
    """Geçmiş + gelecek rebalans tarihlerini hesaplar."""
    rule = get_rebalance_rule(freq)
    if rule is None:
        return [], [], None

    start = pd.Timestamp(start_date)
    today = pd.Timestamp(SIM_END_DATE)

    # Tüm tarihler
    all_dates = []
    cur = start
    end = today + pd.DateOffset(months=months_ahead)
    # Başlangıçtan bir önceki periyoda git
    cur = start - rule
    while cur <= end:
        all_dates.append(cur)
        cur = cur + rule
    # start'ı da ekle
    if start not in all_dates:
        all_dates.append(start)
        all_dates.sort()

    past = [d for d in all_dates if d < today]
    future = [d for d in all_dates if d >= today]
    next_reb = future[0] if future else None
    return past, future, next_reb


def days_until(target):
    if target is None:
        return None
    delta = (pd.Timestamp(target) - pd.Timestamp(SIM_END_DATE)).days
    return delta


# ============================================================
# PDF RAPOR
# ============================================================
def get_font_path():
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def build_pdf_report(df_pos, totals, positions, days_held, next_reb):
    """PDF raporu üretir ve bytes olarak döner."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None, "fpdf2 kurulu değil. `pip install fpdf2` komutunu çalıştırın."

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None, "matplotlib kurulu değil. `pip install matplotlib` komutunu çalıştırın."

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # === FONT HAZIRLIĞI ===
    font_path = get_font_path()
    FONT = "Helvetica"
    FONT_OK = False

    if font_path:
        try:
            pdf.add_font("Custom", "", font_path)
            pdf.add_font("Custom", "B", font_path)
            pdf.add_font("Custom", "I", font_path)
            pdf.add_font("Custom", "BI", font_path)
            FONT = "Custom"
            FONT_OK = True
        except Exception:
            FONT = "Helvetica"
            FONT_OK = False

    def set_font(size, style=""):
        if FONT_OK:
            try:
                pdf.set_font(FONT, style=style, size=size)
                return
            except Exception:
                pass
        safe_style = style if style in ("", "B", "I", "BI") else ""
        pdf.set_font("Helvetica", style=safe_style, size=size)

    # ---- BAŞLIK ----
    set_font(18, "B")
    pdf.cell(0, 10, "Katilim Fonu Portfoy Raporu", ln=True, align="C")
    set_font(10)
    pdf.cell(0, 6,
             f"Rapor Tarihi: {SIM_END_DATE.strftime('%d.%m.%Y')}  |  "
             f"Baslangic: {st.session_state.start_date.strftime('%d.%m.%Y')}  |  "
             f"Gecen Gun: {days_held}",
             ln=True, align="C")
    pdf.ln(4)

    # ---- KPI ----
    set_font(13, "B")
    pdf.cell(0, 8, "1. Portfoy Ozeti", ln=True)
    set_font(10)

    kpi_rows = [
        ("Baslangic Sermayesi", f"{st.session_state.initial_capital:,.0f} TL"),
        ("Brut Deger", f"{totals['total_value']:,.0f} TL"),
        ("Net Deger (Stopaj Sonrasi)", f"{totals['total_net']:,.0f} TL"),
        ("Brut Kar/Zarar", f"{totals['gross_pl']:+,.0f} TL"),
        ("Toplam Stopaj", f"-{totals['total_tax']:,.0f} TL"),
        ("Net Kar/Zarar", f"{totals['net_pl']:+,.0f} TL"),
        ("Nominal Getiri", f"{totals['nominal']*100:+.2f}%"),
    ]
    if days_held >= 1:
        kpi_rows.append(("Reel Getiri (TUFE ustu)", f"{totals['reel']*100:+.2f}%"))
        kpi_rows.append(("Doviz Ustu Getiri", f"{totals['doviz']*100:+.2f}%"))
    if next_reb is not None:
        kpi_rows.append(("Sonraki Rebalans",
                         f"{next_reb.strftime('%d.%m.%Y')} ({days_until(next_reb)} gun)"))

    for label, val in kpi_rows:
        set_font(10)
        pdf.cell(90, 6, label, border=0)
        set_font(10, "B")
        pdf.cell(0, 6, val, border=0, ln=True)

    pdf.ln(5)

    # ---- POZİSYON TABLOSU ----
    set_font(13, "B")
    pdf.cell(0, 8, "2. Pozisyonlar", ln=True)
    pdf.ln(1)

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

    # ---- GRAFİK: AĞIRLIKLAR ----
    pdf.add_page()
    set_font(13, "B")
    pdf.cell(0, 8, "3. Hedef vs Gercek Agirliklar", ln=True)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(df_pos))
    w = 0.38
    ax.bar(x - w/2, df_pos["Hedef (%)"], w, label="Hedef", color="#94a3b8")
    ax.bar(x + w/2, df_pos["Ağırlık (%)"], w, label="Gercek", color="#0ea5e9")
    ax.set_xticks(x)
    ax.set_xticklabels(df_pos["Fon"], fontsize=9)
    ax.set_ylabel("Agirlik (%)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    tmp_png = os.path.join(os.path.expanduser("~"), "_st_pdf_chart.png")
    with open(tmp_png, "wb") as f:
        f.write(buf.getvalue())
    pdf.image(tmp_png, x=15, w=180)
    try:
        os.remove(tmp_png)
    except Exception:
        pass

    pdf.ln(4)

    # ---- GRAFİK: VARLIK SINIFI ----
    set_font(13, "B")
    pdf.cell(0, 8, "4. Varlik Sinifi Dagilimi", ln=True)

    df_pos["Kategori"] = df_pos["Tür"].apply(
        lambda x: "Kira Sert." if "Kira" in x else
                  "Hisse Senedi" if "Hisse" in x else
                  "Teknoloji" if "Teknoloji" in x else
                  "Altin" if "Altın" in x else
                  "Doviz" if "Döviz" in x else "Diger"
    )
    cat = df_pos.groupby("Kategori")["Güncel Değer"].sum()

    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    ax2.pie(cat, labels=cat.index, autopct="%1.1f%%", startangle=90,
            colors=plt.cm.Set2.colors[:len(cat)])
    ax2.axis("equal")
    plt.tight_layout()

    buf2 = BytesIO()
    fig2.savefig(buf2, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig2)
    buf2.seek(0)
    tmp_png2 = os.path.join(os.path.expanduser("~"), "_st_pdf_pie.png")
    with open(tmp_png2, "wb") as f:
        f.write(buf2.getvalue())
    pdf.image(tmp_png2, x=40, w=130)
    try:
        os.remove(tmp_png2)
    except Exception:
        pass

    # ---- FOOTER ----
    pdf.ln(4)
    set_font(8, "I")
    pdf.multi_cell(0, 5,
                   "Uyari: Bu rapor bir paper trade simulasyonudur. "
                   "Fiyatlar ve stopaj oranlari varsayima dayalidir. "
                   "Yatirim karari icin TEFAS, KAP ve ilgili bankalarin resmi kaynaklarini kullanin.")

    out = pdf.output()
    if isinstance(out, str):
        return out.encode("latin-1"), None
    return bytes(out), None

# ============================================================
# EXCEL EXPORT
# ============================================================
def build_excel(df_pos, df_funds, df_tx, df_perf, totals):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1 — Pozisyonlar
        df_pos.to_excel(writer, sheet_name="Pozisyonlar", index=False)

        # Sheet 2 — Özet (None güvenli)
        def safe_pct(v):
            return round(v * 100, 4) if v is not None else None

        summary = pd.DataFrame([
            ("Başlangıç Sermayesi", st.session_state.initial_capital),
            ("Brüt Değer", totals["total_value"]),
            ("Net Değer (Stopaj Sonrası)", totals["total_net"]),
            ("Brüt K/Z", totals["gross_pl"]),
            ("Toplam Stopaj", totals["total_tax"]),
            ("Net K/Z", totals["net_pl"]),
            ("Nominal Getiri (%)", safe_pct(totals.get("nominal"))),
            ("Reel Getiri (%)", safe_pct(totals.get("reel"))),
            ("Döviz Üstü Getiri (%)", safe_pct(totals.get("doviz"))),
        ], columns=["Gösterge", "Değer"])
        summary.to_excel(writer, sheet_name="Özet", index=False)

        # Sheet 3 — Fonlar
        if df_funds is not None and not df_funds.empty:
            df_funds.to_excel(writer, sheet_name="Fonlar", index=False)

        # Sheet 4 — İşlem Geçmişi
        if df_tx is not None and not df_tx.empty:
            df_tx.to_excel(writer, sheet_name="İşlemler", index=False)

        # Sheet 5 — Performans
        if df_perf is not None and not df_perf.empty:
            df_perf.to_excel(writer, sheet_name="Performans", index=False)

        # Sheet 6 — Hedef Ağırlıklar
        tw = pd.DataFrame(
            list(st.session_state.target_weights.items()),
            columns=["Fon", "Hedef Ağırlık"]
        )
        tw.to_excel(writer, sheet_name="Hedef Ağırlıklar", index=False)

    return output.getvalue()

# ============================================================
# SESSION STATE
# ============================================================
def init_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.start_date = SIM_END_DATE
        st.session_state.initial_capital = 100_000.0
        st.session_state.target_weights = {
            "ZPG": 0.15, "KTN": 0.15, "CPU": 0.25,
            "KZL": 0.20, "KPC": 0.15, "KIS": 0.10,
        }
        st.session_state.transactions = []
        st.session_state.rebalance_log = []
        st.session_state.manual_positions = {}
        st.session_state.use_tefas = False
        st.session_state.tefas_cache = {}
        st.session_state.price_history = generate_price_history()


init_state()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Portföy Ayarları")

    st.session_state.use_tefas = st.toggle(
        "🌐 TEFAS Canlı Fiyat", value=st.session_state.use_tefas,
        help="Kapatılırsa simülasyon fiyatları kullanılır"
    )
    if st.session_state.use_tefas:
        if st.button("🔄 TEFAS Fiyatlarını Güncelle", use_container_width=True):
            with st.spinner("TEFAS'tan veri çekiliyor..."):
                st.session_state.tefas_cache = fetch_all_tefas(tuple(FUNDS_FALLBACK.keys()))
            ok = sum(1 for v in st.session_state.tefas_cache.values() if v and "price" in v)
            if ok == 0:
                st.error("TEFAS'a erişilemedi.")
                st.session_state.use_tefas = False
            else:
                st.success(f"✅ {ok}/{len(FUNDS_FALLBACK)} fon güncellendi")

    st.session_state.initial_capital = st.number_input(
        "Başlangıç Sermayesi (TL)", min_value=1_000.0,
        value=st.session_state.initial_capital, step=10_000.0, format="%.2f"
    )
    new_date = st.date_input("Başlangıç Tarihi", value=st.session_state.start_date.date())
    st.session_state.start_date = datetime.combine(new_date, datetime.min.time())

    st.markdown("### 🎯 Hedef Ağırlıklar (%)")
    total_w = 0
    new_weights = {}
    for code in FUNDS_FALLBACK:
        default = int(st.session_state.target_weights.get(code, 0.0) * 100)
        w = st.slider(f"{code}", 0, 50, default, 1, key=f"w_{code}",
                      help=FUNDS_FALLBACK[code]["name"])
        new_weights[code] = w / 100
        total_w += w
    st.session_state.target_weights = {k: v for k, v in new_weights.items() if v > 0}

    if abs(total_w - 100) > 0.01:
        st.warning(f"⚠️ Toplam: %{total_w}")
    else:
        st.success(f"✅ Toplam: %{total_w}")

    st.markdown("### 📅 Rebalans")
    st.session_state.rebalance_freq = st.selectbox(
        "Sıklık", ["3 Aylık", "6 Aylık", "Yıllık", "Eşik Bazlı (%5 sapma)"], index=0
    )
    st.session_state.drift_threshold = st.slider("Sapma Eşiği (%)", 1, 15, 5)

    # Sonraki rebalans uyarısı
    _, _, next_reb = compute_rebalance_dates(
        st.session_state.start_date, st.session_state.rebalance_freq
    )
    if next_reb is not None:
        d = days_until(next_reb)
        if d == 0:
            st.error(f"🔔 **Bugün rebalans günü!** ({next_reb.strftime('%d.%m.%Y')})")
        elif d is not None and d <= 7:
            st.warning(f"⏰ **Sonraki rebalans: {d} gün sonra** ({next_reb.strftime('%d.%m.%Y')})")
        else:
            st.info(f"📅 Sonraki rebalans: {next_reb.strftime('%d.%m.%Y')} ({d} gün)")

    st.markdown("---")
    src = "TEFAS" if st.session_state.use_tefas and any(
        v and "price" in v for v in st.session_state.tefas_cache.values()
    ) else "Simülasyon"
    st.caption(f"📌 Fiyat kaynağı: **{src}**")

# ============================================================
# BAŞLIK
# ============================================================
st.title("📈 Katılım Fonu Paper Trade & Rebalans Paneli")
st.caption(
    f"Başlangıç: **{st.session_state.start_date.strftime('%d.%m.%Y')}** | "
    f"Sermaye: **{st.session_state.initial_capital:,.0f} TL** | "
    f"Referans: **{SIM_END_DATE.strftime('%d.%m.%Y')}**"
)

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
        "Fon": code,
        "Ad": meta.get("name", code),
        "Tür": meta.get("type", "—"),
        "Birim": p["units"],
        "Alış Fiyatı": p["buy_price"],
        "Güncel Fiyat": cur_price,
        "Maliyet": p["cost"],
        "Güncel Değer": value,
        "Brüt K/Z (TL)": gross_pl,
        "Brüt K/Z (%)": (gross_pl / p["cost"] * 100) if p["cost"] else 0,
        "Stopaj (%)": tax_rate * 100,
        "Stopaj (TL)": tax_amount,
        "Net Değer": net_value,
        "Net K/Z (TL)": net_pl,
        "Net K/Z (%)": (net_pl / p["cost"] * 100) if p["cost"] else 0,
    })

df_pos = pd.DataFrame(rows)
total_value = df_pos["Güncel Değer"].sum()
total_net_value = df_pos["Net Değer"].sum()
total_gross_pl = df_pos["Brüt K/Z (TL)"].sum()
total_tax = df_pos["Stopaj (TL)"].sum()
total_net_pl = df_pos["Net K/Z (TL)"].sum()

df_pos["Ağırlık (%)"] = df_pos["Güncel Değer"] / total_value * 100
df_pos["Hedef (%)"] = df_pos["Fon"].map(st.session_state.target_weights).fillna(0) * 100
df_pos["Sapma (pp)"] = df_pos["Ağırlık (%)"] - df_pos["Hedef (%)"]

days_held = max((today - st.session_state.start_date).days, 0)
nominal_net = total_net_value / st.session_state.initial_capital - 1
if days_held < 1:
    reel_val = None
    doviz_val = None
else:
    tufe_period = (1 + MACRO["tufe"]) ** (days_held / 365) - 1
    usd_period = (1 + USD_RET_ASSUMPTION) ** (days_held / 365) - 1
    reel_val = nominal_net - tufe_period
    doviz_val = nominal_net - usd_period

totals = {
    "total_value": total_value,
    "total_net": total_net_value,
    "gross_pl": total_gross_pl,
    "total_tax": total_tax,
    "net_pl": total_net_pl,
    "nominal": nominal_net,
    "reel": reel_val,
    "doviz": doviz_val,
}

# Rebalans tarihleri
_, future_rebs, next_reb = compute_rebalance_dates(
    st.session_state.start_date, st.session_state.rebalance_freq
)

# ============================================================
# KPI
# ============================================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Portföy (Brüt)", f"{total_value:,.0f} TL",
          f"{(total_value/st.session_state.initial_capital - 1)*100:+.2f}%")
c2.metric("Net (Stopaj Sonrası)", f"{total_net_value:,.0f} TL",
          f"{(total_net_value/st.session_state.initial_capital - 1)*100:+.2f}%")
c3.metric("Brüt K/Z", f"{total_gross_pl:+,.0f} TL")
c4.metric("Toplam Stopaj", f"-{total_tax:,.0f} TL",
          delta=f"{(total_tax/total_value*100):.2f}%" if total_value else None,
          delta_color="inverse")
c5.metric("Net K/Z", f"{total_net_pl:+,.0f} TL",
          f"{(total_net_pl/st.session_state.initial_capital)*100:+.2f}%")

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
c4b.metric("Efektif Vergi Oranı", f"{eff_tax_rate:.2f}%")
if next_reb is not None:
    d_next = days_until(next_reb)
    c5b.metric("Sonraki Rebalans", next_reb.strftime("%d.%m.%Y"),
               f"{d_next} gün" if d_next is not None else None,
               delta_color="inverse" if d_next is not None and d_next <= 7 else "normal")
else:
    c5b.metric("Sonraki Rebalans", "Eşik bazlı", "Sapmaya göre")

st.markdown("---")

# ============================================================
# SEKMELER
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["📊 Genel Bakış", "📈 Performans", "⚖️ Rebalans", "🔍 Fon Analizi",
     "💰 Vergi Analizi", "📝 İşlem Geçmişi", "📤 Rapor & Export"]
)

# ---------------- TAB 1: GENEL BAKIŞ ----------------
with tab1:
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Pozisyonlar (Brüt & Net)")
        show = df_pos[["Fon", "Ad", "Tür", "Birim", "Alış Fiyatı", "Güncel Fiyat",
                       "Güncel Değer", "Brüt K/Z (%)", "Stopaj (%)", "Stopaj (TL)",
                       "Net Değer", "Net K/Z (%)", "Ağırlık (%)", "Hedef (%)", "Sapma (pp)"]].copy()
        st.dataframe(
            show.style.format({
                "Birim": "{:,.2f}", "Alış Fiyatı": "{:,.4f}", "Güncel Fiyat": "{:,.4f}",
                "Güncel Değer": "{:,.0f}", "Brüt K/Z (%)": "{:+.2f}%",
                "Stopaj (%)": "{:.2f}%", "Stopaj (TL)": "{:,.0f}",
                "Net Değer": "{:,.0f}", "Net K/Z (%)": "{:+.2f}%",
                "Ağırlık (%)": "{:.2f}%", "Hedef (%)": "{:.2f}%", "Sapma (pp)": "{:+.2f}",
            }).background_gradient(subset=["Brüt K/Z (%)", "Net K/Z (%)"], cmap="RdYlGn")
              .background_gradient(subset=["Stopaj (TL)"], cmap="Reds")
              .background_gradient(subset=["Sapma (pp)"], cmap="RdBu_r"),
            use_container_width=True, hide_index=True,
        )
        st.markdown(
            f"**Brüt K/Z:** `{total_gross_pl:,.0f} TL` | "
            f"**Stopaj:** `-{total_tax:,.0f} TL` | "
            f"**Net K/Z:** `{total_net_pl:,.0f} TL`"
        )
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
                  "Döviz" if "Döviz" in x else "Diğer"
    )
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
    port_series = pd.Series(
        [net_portfolio_on(d, positions) for d in df_hist.index],
        index=df_hist.index,
    )
    start_val = port_series.asof(pd.Timestamp(st.session_state.start_date))
    if start_val and start_val > 0:
        port_series = port_series / start_val * st.session_state.initial_capital

    days_elapsed = pd.Series(
        np.maximum((df_hist.index - st.session_state.start_date).days, 0),
        index=df_hist.index,
    )
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
        v0 = net_portfolio_on(d0, positions)
        v1 = total_net_value
        if v0 <= 0:
            continue
        r = (v1 / v0 - 1) * 100
        tufe_r = ((1 + MACRO["tufe"]) ** (days / 365) - 1) * 100
        usd_r = ((1 + USD_RET_ASSUMPTION) ** (days / 365) - 1) * 100
        perf_rows.append({
            "Dönem": label, "Net Portföy (%)": r,
            "TÜFE (%)": tufe_r, "USD/TRY (%)": usd_r,
            "Reel Fark (pp)": r - tufe_r, "Döviz Fark (pp)": r - usd_r,
        })
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
    st.subheader("Hedef vs Gerçek — Sapma Analizi")
    st.markdown(f"**Kural:** `{st.session_state.rebalance_freq}` | "
                f"**Eşik:** `±{st.session_state.drift_threshold}%` | "
                f"**Tolerans:** `{TOLERANCE_TL:.0f} TL`")

    drift = df_pos[["Fon", "Ad", "Hedef (%)", "Ağırlık (%)",
                    "Sapma (pp)", "Güncel Değer"]].copy()
    drift["Hedef Değer"] = total_value * drift["Hedef (%)"] / 100
    drift["Fark (TL)"] = (drift["Hedef Değer"] - drift["Güncel Değer"]).round(2)
    drift["Hedef (%)"] = drift["Hedef (%)"].round(4)
    drift["Ağırlık (%)"] = drift["Ağırlık (%)"].round(4)
    drift["Sapma (pp)"] = (drift["Ağırlık (%)"] - drift["Hedef (%)"]).round(4)
    drift["İşlem"] = drift["Fark (TL)"].apply(
        lambda x: "🟢 AL" if x > TOLERANCE_TL
        else ("🔴 SAT" if x < -TOLERANCE_TL else "—"))
    drift["Uyarı"] = drift["Sapma (pp)"].abs() > st.session_state.drift_threshold

    def color_islem(val):
        if "AL" in str(val):  return "background-color: #d1fae5"
        if "SAT" in str(val): return "background-color: #fee2e2"
        return ""

    st.dataframe(
        drift.style.format({
            "Hedef (%)": "{:.2f}%", "Ağırlık (%)": "{:.2f}%", "Sapma (pp)": "{:+.2f}",
            "Güncel Değer": "{:,.0f}", "Hedef Değer": "{:,.0f}", "Fark (TL)": "{:+,.0f}",
        }).map(color_islem, subset=["İşlem"]),
        use_container_width=True, hide_index=True,
    )

    n_alert = int(drift["Uyarı"].sum())
    if n_alert:
        st.warning(f"⚠️ **{n_alert} fon** eşiği aştı.")
    else:
        st.success("✅ Tüm fonlar hedef aralıkta.")

    fig = go.Figure()
    colors = ["#ef4444" if abs(v) > st.session_state.drift_threshold else "#0ea5e9"
              for v in drift["Sapma (pp)"]]
    fig.add_trace(go.Bar(x=drift["Fon"], y=drift["Sapma (pp)"],
                         marker_color=colors, text=drift["Sapma (pp)"].round(2),
                         textposition="outside"))
    fig.add_hline(y=st.session_state.drift_threshold, line_dash="dash", line_color="red")
    fig.add_hline(y=-st.session_state.drift_threshold, line_dash="dash", line_color="red")
    fig.add_hline(y=0, line_color="gray")
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="Sapma (pp)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔄 Rebalans Simülasyonu")
    if st.button("Rebalansı Uygula (sanal)", type="primary"):
        st.session_state.manual_positions = {}
        st.session_state.transactions.append({
            "tarih": today, "tip": "REBALANS", "fon": "—",
            "birim": "—", "fiyat": "—",
            "tutar": f"{n_alert} fon düzeltildi", "not": "",
        })
        st.session_state.rebalance_log.append({
            "tarih": today, "toplam": total_value, "uyarı": n_alert,
        })
        st.success(f"✅ {today.strftime('%d.%m.%Y')} rebalans uygulandı.")
        st.rerun()

    st.markdown("---")
    st.subheader("📅 Otomatik Rebalans Takvimi")
    past_rebs, future_rebs_tab, next_reb_tab = compute_rebalance_dates(
        st.session_state.start_date, st.session_state.rebalance_freq
    )

    if get_rebalance_rule(st.session_state.rebalance_freq) is None:
        st.info("ℹ️ Eşik bazlı rebalans seçildi — takvim yerine sapma ±%5'i aştığında tetiklenir.")
    else:
        # Sonraki rebalans vurgusu
        if next_reb_tab is not None:
            d_next = days_until(next_reb_tab)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Sonraki Rebalans", next_reb_tab.strftime("%d.%m.%Y"))
            col_b.metric("Kalan Gün", f"{d_next} gün")
            col_c.metric("Sıklık", st.session_state.rebalance_freq)

            if d_next == 0:
                st.error("🔔 **Bugün rebalans günü!**")
            elif d_next <= 7:
                st.warning(f"⏰ **{d_next} gün kaldı** — hazırlık yapın.")
            elif d_next <= 30:
                st.info(f"📅 **{d_next} gün** sonra rebalans.")

        # Geçmiş rebalans
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
            st.info("Henüz geçmiş rebalans tarihi yok.")

        # Gelecek
        st.markdown("**🔮 Gelecek Rebalans Tarihleri (12 Ay)**")
        if future_rebs_tab:
            future_df = pd.DataFrame([{
                "Tarih": d.strftime("%d.%m.%Y"),
                "Gün": d.strftime("%A"),
                "Kalan Gün": (d - pd.Timestamp(SIM_END_DATE)).days,
            } for d in future_rebs_tab])
            st.dataframe(future_df, use_container_width=True, hide_index=True)

        # Timeline görsel
        all_rebs = past_rebs + future_rebs_tab
        if all_rebs:
            fig_tl = go.Figure()
            for d in all_rebs:
                is_future = d >= pd.Timestamp(SIM_END_DATE)
                fig_tl.add_trace(go.Scatter(
                    x=[d], y=[0], mode="markers+text",
                    marker=dict(size=18,
                                color="#f59e0b" if is_future else "#10b981",
                                symbol="diamond"),
                    text=[d.strftime("%d.%m")],
                    textposition="top center",
                    showlegend=False,
                    hovertext=[d.strftime("%d.%m.%Y")],
                    hoverinfo="text",
                ))
            fig_tl.add_vline(x=pd.Timestamp(SIM_END_DATE).timestamp() * 1000,
                             line_dash="dash", line_color="red",
                             annotation_text="Bugün", annotation_position="top")
            fig_tl.update_layout(height=220,
                                 margin=dict(l=10, r=10, t=40, b=10),
                                 yaxis=dict(showticklabels=False, range=[-0.5, 0.5]),
                                 xaxis_title="", showlegend=False)
            st.plotly_chart(fig_tl, use_container_width=True)

        # Log
        if st.session_state.rebalance_log:
            st.markdown("**📋 Rebalans Kayıtları**")
            st.dataframe(pd.DataFrame(st.session_state.rebalance_log),
                         use_container_width=True, hide_index=True)

# ---------------- TAB 4: FON ANALİZİ ----------------
with tab4:
    st.subheader("Fon Karşılaştırma Tablosu")
    fund_rows = []
    for code in FUNDS_FALLBACK:
        meta = get_fund_meta(code)
        fund_rows.append({
            "Kod": code, "Ad": meta.get("name", code),
            "Banka": meta.get("bank", "—"), "Tür": meta.get("type", "—"),
            "Güncel Fiyat": meta.get("price", 0),
            "1 Yıl (%)": (meta.get("r1y") or 0) * 100,
            "3 Yıl CAGR (%)": (meta.get("cagr3") or 0) * 100,
            "5 Yıl CAGR (%)": (meta.get("cagr5") or 0) * 100,
            "Risk": f"{meta.get('risk', '—')}/7" if isinstance(meta.get("risk"), int) else "—",
            "Yön. Üc. (%)": (meta.get("fee") or 0) * 100,
            "Stopaj (%)": (meta.get("tax") or 0) * 100,
            "TEFAS": meta.get("tefas", "—"),
        })
    df_funds = pd.DataFrame(fund_rows)
    st.dataframe(df_funds.style.format({
        "Güncel Fiyat": "{:,.4f}", "1 Yıl (%)": "{:+.2f}",
        "3 Yıl CAGR (%)": "{:+.2f}", "5 Yıl CAGR (%)": "{:+.2f}",
        "Yön. Üc. (%)": "{:.2f}", "Stopaj (%)": "{:.2f}",
    }).background_gradient(subset=["1 Yıl (%)", "3 Yıl CAGR (%)", "5 Yıl CAGR (%)"], cmap="RdYlGn")
      .background_gradient(subset=["Stopaj (%)"], cmap="Reds"),
        use_container_width=True, hide_index=True)

    st.subheader("Risk — Getiri Haritası")
    df_sc = df_funds[df_funds["Risk"] != "—"].copy()
    if not df_sc.empty:
        df_sc["Risk_Num"] = df_sc["Risk"].str.replace("/7", "", regex=False).astype(int)
        fig_sc = px.scatter(df_sc, x="Risk_Num", y="1 Yıl (%)", text="Kod",
                            size="1 Yıl (%)", color="Tür", hover_data=["Ad"],
                            color_discrete_sequence=px.colors.qualitative.Bold)
        fig_sc.update_traces(textposition="top center")
        fig_sc.update_layout(height=420, xaxis_title="Risk (1–7)",
                             yaxis_title="1 Yıl Getiri (%)",
                             margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_sc, use_container_width=True)

    st.subheader("Korelasyon Matrisi")
    rets = st.session_state.price_history.pct_change().dropna()
    fig_c = px.imshow(rets.corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                      zmin=-1, zmax=1, aspect="auto")
    fig_c.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_c, use_container_width=True)

# ---------------- TAB 5: VERGİ ANALİZİ ----------------
with tab5:
    st.subheader("💰 Stopaj Sonrası Net Getiri")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Brüt Değer", f"{total_value:,.0f} TL")
    v2.metric("Stopaj", f"-{total_tax:,.0f} TL",
              delta=f"{(total_tax/total_value*100):.2f}%" if total_value else None,
              delta_color="inverse")
    v3.metric("Net Değer", f"{total_net_value:,.0f} TL")
    v4.metric("Efektif Vergi", f"{eff_tax_rate:.2f}%")

    st.markdown("---")
    st.subheader("Fon Bazında Vergi Detayı")
    tax_tbl = df_pos[[
        "Fon", "Ad", "Tür", "Maliyet", "Güncel Değer",
        "Brüt K/Z (TL)", "Stopaj (%)", "Stopaj (TL)",
        "Net Değer", "Net K/Z (TL)", "Net K/Z (%)"
    ]].copy()
    st.dataframe(
        tax_tbl.style.format({
            "Maliyet": "{:,.0f}", "Güncel Değer": "{:,.0f}",
            "Brüt K/Z (TL)": "{:+,.0f}", "Stopaj (%)": "{:.2f}%",
            "Stopaj (TL)": "{:,.0f}", "Net Değer": "{:,.0f}",
            "Net K/Z (TL)": "{:+,.0f}", "Net K/Z (%)": "{:+.2f}%",
        }).background_gradient(subset=["Stopaj (TL)"], cmap="Reds")
          .background_gradient(subset=["Net K/Z (%)"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Brüt K/Z vs Stopaj vs Net K/Z")
    fig_tax = go.Figure()
    fig_tax.add_trace(go.Bar(x=df_pos["Fon"], y=df_pos["Brüt K/Z (TL)"],
                             name="Brüt K/Z", marker_color="#94a3b8"))
    fig_tax.add_trace(go.Bar(x=df_pos["Fon"], y=-df_pos["Stopaj (TL)"],
                             name="Stopaj", marker_color="#ef4444"))
    fig_tax.add_trace(go.Bar(x=df_pos["Fon"], y=df_pos["Net K/Z (TL)"],
                             name="Net K/Z", marker_color="#10b981"))
    fig_tax.update_layout(barmode="group", height=400,
                          margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", y=1.1), yaxis_title="TL")
    st.plotly_chart(fig_tax, use_container_width=True)

    st.info(
        "**2026 Stopaj Kuralları:**\n\n"
        "- Hisse Senedi Fonları (KPC, RBH, CPU): **%0**\n"
        "- Kira Sertifikası (ZPG, KTN): **%0** (31.12.2026'ya kadar istisna)\n"
        "- Teknoloji (KTJ), Altın (KZL), Döviz (KIS, OFK): **%17,50**\n\n"
        "Stopaj yalnızca **kâr** üzerinden alınır."
    )

# ---------------- TAB 6: İŞLEM GEÇMİŞİ ----------------
with tab6:
    st.subheader("📝 Manuel İşlem Girişi")
    with st.form("manual_trade", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            trade_date = st.date_input("İşlem Tarihi", value=today.date())
            trade_code = st.selectbox(
                "Fon", list(FUNDS_FALLBACK.keys()),
                format_func=lambda x: f"{x} — {FUNDS_FALLBACK[x]['name'][:30]}"
            )
        with c2:
            trade_type = st.selectbox("Tip", ["AL", "SAT"])
            trade_units = st.number_input("Birim", min_value=0.01, value=100.0, step=10.0)
        with c3:
            default_price = get_price(trade_code, datetime.combine(trade_date, datetime.min.time()))
            trade_price = st.number_input(
                "Fiyat (TL)", min_value=0.0001,
                value=float(round(default_price, 4)), step=0.0001, format="%.4f"
            )
            trade_note = st.text_input("Not", "")

        submitted = st.form_submit_button("✅ Kaydet", type="primary")
        if submitted:
            amount = trade_units * trade_price
            st.session_state.transactions.append({
                "tarih": datetime.combine(trade_date, datetime.min.time()),
                "tip": trade_type, "fon": trade_code,
                "birim": trade_units, "fiyat": trade_price,
                "tutar": amount, "not": trade_note,
            })
            mp = st.session_state.manual_positions.get(
                trade_code, {"units": 0.0, "cost": 0.0}
            )
            if trade_type == "AL":
                mp["units"] += trade_units
                mp["cost"] += amount
            else:
                mp["units"] = max(0.0, mp["units"] - trade_units)
                mp["cost"] = max(0.0, mp["cost"] - amount)
            if mp["units"] <= 0:
                st.session_state.manual_positions.pop(trade_code, None)
            else:
                mp["buy_price"] = mp["cost"] / mp["units"]
                st.session_state.manual_positions[trade_code] = mp
            st.success(f"✅ {trade_type}: {trade_code} {trade_units:.2f} @ {trade_price:.4f}")
            time.sleep(0.5)
            st.rerun()

    st.markdown("---")
    st.subheader("📋 İşlem Geçmişi")
    all_tx = []
    for code, p in positions.items():
        if code in st.session_state.target_weights:
            all_tx.append({
                "Tarih": st.session_state.start_date.strftime("%d.%m.%Y"),
                "Tip": "BAŞLANGIÇ", "Fon": code,
                "Birim": p["units"], "Fiyat": p["buy_price"],
                "Tutar": p["cost"], "Not": "",
            })
    for t in st.session_state.transactions:
        all_tx.append({
            "Tarih": t["tarih"].strftime("%d.%m.%Y")
                     if hasattr(t["tarih"], "strftime") else str(t["tarih"]),
            "Tip": t["tip"], "Fon": t.get("fon", "—"),
            "Birim": t.get("birim", "—"), "Fiyat": t.get("fiyat", "—"),
            "Tutar": t.get("tutar", "—"), "Not": t.get("not", ""),
        })
    df_tx = pd.DataFrame(all_tx) if all_tx else pd.DataFrame()
    if not df_tx.empty:
        st.dataframe(
            df_tx.style.format({
                "Birim": "{:,.2f}", "Fiyat": "{:,.4f}", "Tutar": "{:,.0f}",
            }, na_rep="—"),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Henüz işlem kaydı yok.")

# ---------------- TAB 7: RAPOR & EXPORT ----------------
with tab7:
    st.subheader("📤 Rapor ve Veri Dışa Aktarma")

    col_pdf, col_xlsx = st.columns(2)

    # ---- PDF ----
    with col_pdf:
        st.markdown("### 📄 PDF Rapor")
        st.caption("Portföy özeti, pozisyon tablosu ve grafikleri içeren tek sayfalık rapor.")

        if st.button("📄 PDF Rapor Oluştur", type="primary", use_container_width=True):
            with st.spinner("PDF üretiliyor..."):
                pdf_bytes, err = build_pdf_report(
                    df_pos, totals, positions, days_held, next_reb
                )
            if err:
                st.error(f"❌ {err}")
            elif pdf_bytes:
                st.session_state["_pdf_bytes"] = pdf_bytes
                st.success(f"✅ PDF hazır ({len(pdf_bytes):,} byte)")

        if st.session_state.get("_pdf_bytes"):
            st.download_button(
                "⬇️ PDF İndir",
                data=st.session_state["_pdf_bytes"],
                file_name=f"portfoy_raporu_{SIM_END_DATE.strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    # ---- EXCEL ----
    with col_xlsx:
        st.markdown("### 📊 Excel Export")
        st.caption("Pozisyonlar, özet, fonlar, işlemler, performans ve hedef ağırlıklar — 6 sheet.")

        if st.button("📊 Excel Oluştur", type="primary", use_container_width=True):
            with st.spinner("Excel üretiliyor..."):
                fund_rows = []
                for code in FUNDS_FALLBACK:
                    meta = get_fund_meta(code)
                    fund_rows.append({
                        "Kod": code, "Ad": meta.get("name", code),
                        "Banka": meta.get("bank", "—"), "Tür": meta.get("type", "—"),
                        "Güncel Fiyat": meta.get("price", 0),
                        "1 Yıl (%)": (meta.get("r1y") or 0) * 100,
                        "3 Yıl CAGR (%)": (meta.get("cagr3") or 0) * 100,
                        "5 Yıl CAGR (%)": (meta.get("cagr5") or 0) * 100,
                        "Risk": meta.get("risk", "—"),
                        "Yön. Üc. (%)": (meta.get("fee") or 0) * 100,
                        "Stopaj (%)": (meta.get("tax") or 0) * 100,
                    })
                df_funds_exp = pd.DataFrame(fund_rows)

                # Performans tablosu
                periods = {"1 Hafta": 7, "1 Ay": 30, "3 Ay": 90, "6 Ay": 180, "YBB": 365}
                perf_rows = []
                for label, days in periods.items():
                    d0 = today - timedelta(days=days)
                    v0 = net_portfolio_on(d0, positions)
                    v1 = total_net_value
                    if v0 <= 0:
                        continue
                    r = (v1 / v0 - 1) * 100
                    tufe_r = ((1 + MACRO["tufe"]) ** (days / 365) - 1) * 100
                    usd_r = ((1 + USD_RET_ASSUMPTION) ** (days / 365) - 1) * 100
                    perf_rows.append({
                        "Dönem": label, "Net Portföy (%)": r,
                        "TÜFE (%)": tufe_r, "USD/TRY (%)": usd_r,
                        "Reel Fark (pp)": r - tufe_r, "Döviz Fark (pp)": r - usd_r,
                    })
                df_perf_exp = pd.DataFrame(perf_rows)

                xlsx_bytes = build_excel(df_pos, df_funds_exp, df_tx, df_perf_exp, totals)
                st.session_state["_xlsx_bytes"] = xlsx_bytes
                st.success(f"✅ Excel hazır ({len(xlsx_bytes):,} byte)")

        if st.session_state.get("_xlsx_bytes"):
            st.download_button(
                "⬇️ Excel İndir",
                data=st.session_state["_xlsx_bytes"],
                file_name=f"portfoy_{SIM_END_DATE.strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    st.markdown("---")
    st.subheader("📥 Hızlı CSV İndirme")
    col_csv1, col_csv2 = st.columns(2)
    with col_csv1:
        st.download_button(
            "📄 Pozisyonlar (CSV)",
            data=df_pos.to_csv(index=False).encode("utf-8"),
            file_name="pozisyonlar.csv", mime="text/csv",
            use_container_width=True,
        )
    with col_csv2:
        if not df_tx.empty:
            st.download_button(
                "📄 İşlemler (CSV)",
                data=df_tx.to_csv(index=False).encode("utf-8"),
                file_name="islemler.csv", mime="text/csv",
                use_container_width=True,
            )

    st.markdown("---")
    st.subheader("👁️ Rapor Önizlemesi")
    st.markdown(f"""
    **Rapor İçeriği:**
    - ✅ Portföy özeti: {len(positions)} fon, {total_value:,.0f} TL brüt, {total_net_value:,.0f} TL net
    - ✅ Stopaj analizi: {total_tax:,.0f} TL toplam, %{eff_tax_rate:.2f} efektif oran
    - ✅ Pozisyon tablosu: birim, fiyat, değer, stopaj, net
    - ✅ Hedef vs Gerçek ağırlık grafiği
    - ✅ Varlık sınıfı dağılımı grafiği
    - ✅ {len(st.session_state.transactions)} manuel işlem kaydı
    - ✅ Sonraki rebalans: {next_reb.strftime('%d.%m.%Y') if next_reb is not None else "Eşik bazlı"}
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
tefas_aktif = (
    st.session_state.use_tefas
    and any(v and "price" in v for v in st.session_state.tefas_cache.values())
)
src_note = ("TEFAS canlı verisi kullanılıyor." if tefas_aktif
            else "Simülasyon fiyatları kullanılıyor.")
st.caption(
    f"⚠️ **Uyarı:** Bu panel bir *paper trade* simülasyonudur. {src_note} "
    "Stopaj oranları 2026 vergi mevzuatına göre varsayılmıştır; "
    "gerçek vergi yükümlülüğünüz için mali müşavirinize danışın."
)