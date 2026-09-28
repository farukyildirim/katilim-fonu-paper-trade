import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from tefas import Crawler

st.set_page_config(page_title="Çoklu Fonlu Portföy Backtest & Katılım / BYF Takibi", layout="wide")

st.title("🛡️ Portföy Backtest, Katılım / BYF & Otomatik Fon Seçim Modülü")
st.markdown(
    "Geniş fon kategorilerini analiz edebilir, **Katılım Fonları (İslami Finans)** ve **BYF (Borsa Yatırım Fonları)** bazlı özel değerlendirmeler yapabilir, **Sharpe Skoruna göre en iyi fonları otomatik seçerek** backtest çalıştırabilirsiniz."
)

# =========================================================
# GENİŞ FON HAVUZLARI & KATILIM / BYF ETİKETLERİ
# =========================================================
CANDIDATE_POOLS = {
    "Sukuk (Katılım Kira Sertifikaları)": ["KTT", "ZCN", "KIB", "OKT", "TZD", "FKB", "HVT"],
    "Hisse (Geleneksel & Katılım)": ["KHK", "PJF", "AFA", "TTE", "BIO", "NNF", "MAC", "TI3", "IIH", "GMR"],
    "Altın & Kıymetli Madenler": ["KZL", "MKG", "HAM", "GGK", "OTJ", "TCA", "YGG", "FUA"],
    "Borsa Yatırım Fonları (BYF)": ["ZGOLD", "GLDTR", "USDTR", "GMSTR", "ZKPBL", "ZELKR"]
}

ISLAMIC_METADATA = {
    "KTT": {"Tür": "Sukuk / Katılım", "Katılım Uygun": True, "Açıklama": "Kuveyt Türk Portföy Kira Sertifikaları Katılım Fonu"},
    "ZCN": {"Tür": "Sukuk / Katılım", "Katılım Uygun": True, "Açıklama": "Ziraat Portföy Katılım Kira Sertifikaları Fonu"},
    "KIB": {"Tür": "Sukuk / Katılım", "Katılım Uygun": True, "Açıklama": "Kuveyt Türk Portföy İkinci Katılım Fonu"},
    "OKT": {"Tür": "Sukuk / Katılım", "Katılım Uygun": True, "Açıklama": "Albaraka Portföy Kira Sertifikaları Katılım Fonu"},
    "TZD": {"Tür": "Sukuk / Katılım", "Katılım Uygun": True, "Açıklama": "Ziraat Portföy Katılım Fonu"},
    "FKB": {"Tür": "Sukuk / Katılım", "Katılım Uygun": True, "Açıklama": "QNB Finans Portföy Kira Sertifikaları Katılım Fonu"},
    "HVT": {"Tür": "Sukuk / Katılım", "Katılım Uygun": True, "Açıklama": "Hedef Portföy Katılım Fonu"},
    "KHK": {"Tür": "Hisse / Katılım", "Katılım Uygun": True, "Açıklama": "Kuveyt Türk Portföy Katılım Hisse Senedi Fonu"},
    "PJF": {"Tür": "Hisse / Katılım", "Katılım Uygun": True, "Açıklama": "Qinvest Portföy Katılım Hisse Senedi Fonu"},
    "AFA": {"Tür": "Hisse / Geleneksel", "Katılım Uygun": False, "Açıklama": "Ak Portföy Amerika Yabancı Hisse Senedi Fonu"},
    "TTE": {"Tür": "Hisse / Geleneksel", "Katılım Uygun": False, "Açıklama": "İş Portföy Teknoloji Ağırlıklı Hisse Senedi Fonu"},
    "BIO": {"Tür": "Hisse / Geleneksel", "Katılım Uygun": False, "Açıklama": "Azimut Portföy BIST Teknoloji Fonu"},
    "NNF": {"Tür": "Hisse / Geleneksel", "Katılım Uygun": False, "Açıklama": "Hedef Portföy Birinci Hisse Senedi Fonu"},
    "MAC": {"Tür": "Hisse / Geleneksel", "Katılım Uygun": False, "Açıklama": "Marmara Capital Portföy Hisse Senedi Fonu"},
    "KZL": {"Tür": "Altın / Katılım Uyumlu", "Katılım Uygun": True, "Açıklama": "Kuveyt Türk Portföy Altın Katılım Fonu"},
    "MKG": {"Tür": "Altın / Katılım Uyumlu", "Katılım Uygun": True, "Açıklama": "Mükatıs Portföy Altın Katılım Fonu"},
    "HAM": {"Tür": "Altın / Katılım Uyumlu", "Katılım Uygun": True, "Açıklama": "Hedef Portföy Altın Katılım Fonu"},
    "GGK": {"Tür": "Altın / Katılım Uyumlu", "Katılım Uygun": True, "Açıklama": "Garanti Portföy Altın Katılım Fonu"},
    "ZGOLD": {"Tür": "BYF (ETF) / Altın", "Katılım Uygun": True, "Açıklama": "Ziraat Portföy BIST Altın BYF"},
    "GLDTR": {"Tür": "BYF (ETF) / Altın", "Katılım Uygun": True, "Açıklama": "QNB Finans Portföy BIST Altın BYF"},
    "USDTR": {"Tür": "BYF (ETF) / Döviz", "Katılım Uygun": False, "Açıklama": "QNB Finans Portföy ABD Doları BYF"},
    "GMSTR": {"Tür": "BYF (ETF) / Gümüş", "Katılım Uygun": True, "Açıklama": "QNB Finans Portföy Gümüş BYF"},
    "ZKPBL": {"Tür": "BYF (ETF) / Katılım", "Katılım Uygun": True, "Açıklama": "Ziraat Portföy BIST Katılım 30 BYF"},
}

# =========================================================
# TEFAS VERİ ÇEKME FONKSİYONU
# =========================================================
@st.cache_data(ttl=3600)
def download_tefas_data(tickers, start, end):
    try:
        tefas = Crawler()
        start_str = pd.to_datetime(start).strftime("%Y-%m-%d")
        end_str = pd.to_datetime(end).strftime("%Y-%m-%d")

        tickers = list(set([t.strip().upper() for t in tickers if t and isinstance(t, str)]))
        if not tickers:
            return pd.DataFrame()

        df_list = []
        for ticker in tickers:
            try:
                df_single = tefas.fetch(start=start_str, end=end_str, name=ticker)
                if df_single is not None and not df_single.empty:
                    df_list.append(df_single)
            except Exception:
                continue

        if not df_list:
            return pd.DataFrame()

        df_raw = pd.concat(df_list, ignore_index=True)
        rename_dict = {}
        for col in df_raw.columns:
            c_lower = str(col).lower()
            if any(k in c_lower for k in ['tarih', 'date']):
                rename_dict[col] = 'date'
            elif any(k in c_lower for k in ['kod', 'code']):
                rename_dict[col] = 'code'
            elif any(k in c_lower for k in ['fiyat', 'price']):
                rename_dict[col] = 'price'

        df_raw = df_raw.rename(columns=rename_dict)
        df_raw['price'] = pd.to_numeric(df_raw['price'], errors='coerce')

        df_pivot = df_raw.pivot(index='date', columns='code', values='price')
        df_pivot.index = pd.to_datetime(df_pivot.index)
        return df_pivot.sort_index().replace(0, np.nan).ffill().bfill()
    except Exception as e:
        st.error(f"TEFAS Veri Hatası: {e}")
        return pd.DataFrame()

# =========================================================
# OTOMATİK EN İYİ 2 FONU BULMA FONKSİYONU
# =========================================================
def get_top_2_funds_per_category(start_dt, end_dt, is_islamic_only=True):
    top_selection = {}

    all_tickers = []
    for cat, t_list in CANDIDATE_POOLS.items():
        all_tickers.extend(t_list)

    data = download_tefas_data(all_tickers, start_dt, end_dt)
    if data.empty:
        return {}

    for cat_name, t_list in CANDIDATE_POOLS.items():
        scores = []
        for code in t_list:
            if code in data.columns:
                if is_islamic_only and not ISLAMIC_METADATA.get(code, {}).get("Katılım Uygun", True):
                    continue

                s = data[code].dropna()
                if len(s) > 10:
                    yrs = max((s.index[-1] - s.index[0]).days / 365.25, 0.01)
                    cg = (((s.iloc[-1] / s.iloc[0]) ** (1 / yrs)) - 1) * 100
                    vl = s.pct_change().std() * np.sqrt(252) * 100
                    sharpe = cg / vl if vl > 0 else 0
                    scores.append((code, sharpe))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_2 = [item[0] for item in scores[:2]]

        while len(top_2) < 2:
            top_2.append(t_list[len(top_2)] if len(t_list) > len(top_2) else "")

        top_selection[cat_name] = top_2

    return top_selection

# SESSION STATE DEĞERLERİ
if "s1" not in st.session_state: st.session_state["s1"] = "OKT"
if "s2" not in st.session_state: st.session_state["s2"] = "HVT"
if "h1" not in st.session_state: st.session_state["h1"] = "KHK"
if "h2" not in st.session_state: st.session_state["h2"] = "PJF"
if "a1" not in st.session_state: st.session_state["a1"] = "MKG"
if "a2" not in st.session_state: st.session_state["a2"] = "GGK"

if "real_portfolio" not in st.session_state:
    st.session_state.real_portfolio = pd.DataFrame([
        {"Fon Kodu": "KTT", "Giriş Tarihi": "2024-01-15", "Yatırılan Tutar (₺)": 25000.0},
        {"Fon Kodu": "KHK", "Giriş Tarihi": "2024-02-01", "Yatırılan Tutar (₺)": 15000.0},
        {"Fon Kodu": "KZL", "Giriş Tarihi": "2024-01-10", "Yatırılan Tutar (₺)": 10000.0},
    ])

# =========================================================
# SIDEBAR (Sıralama Hataları Düzeltildi)
# =========================================================
st.sidebar.header("1. Ana Varlık Grubu Ağırlıkları (%)")
w_sukuk = st.sidebar.number_input("Katılım Sukuk Grubu Toplam (%)", value=50, step=5)
w_hisse = st.sidebar.number_input("Hisse Grubu Toplam (%)", value=30, step=5)
w_altin = st.sidebar.number_input("Altın Grubu Toplam (%)", value=20, step=5)

if w_sukuk + w_hisse + w_altin != 100:
    st.sidebar.error(f"Ağırlıklar toplamı %100 olmalıdır! Şu an: %{w_sukuk + w_hisse + w_altin}")

# --- TARİH VE PARAMETRELER (Butondan önce tanımlanmalı) ---
st.sidebar.header("2. Backtest Parametreleri")
start_default = pd.to_datetime("2021-01-01")
end_default = pd.to_datetime("2026-01-01")

start_date = st.sidebar.date_input("Başlangıç Tarihi", start_default)
end_date = st.sidebar.date_input("Bitiş Tarihi", end_default)
initial_capital = st.sidebar.number_input("Başlangıç Sermayesi (₺)", value=100000, step=10000)

rebalance_mode = st.sidebar.radio(
    "Dengeleme Stratejisi Seçin:",
    ["Yıllık (Takvim Başı)", "Tolerans Bandı (%10 Sapma)", "Buy & Hold (Hiç Etme)"]
)

# --- FON SEÇİMİ VE OTOMASYON BUTONU ---
st.sidebar.header("3. Fon Seçimi ve Otomasyon")

if st.sidebar.button("⚡ Risk Skoruna Göre En İyi 2 Fonu Otomatik Seç", type="primary"):
    with st.spinner("Katılım havuzundaki en yüksek Sharpe skorlu fonlar belirleniyor..."):
        best_funds = get_top_2_funds_per_category(start_date, end_date, is_islamic_only=True)
        if best_funds:
            st.session_state["s1"] = best_funds["Sukuk (Katılım Kira Sertifikaları)"][0]
            st.session_state["s2"] = best_funds["Sukuk (Katılım Kira Sertifikaları)"][1]
            st.session_state["h1"] = best_funds["Hisse (Geleneksel & Katılım)"][0]
            st.session_state["h2"] = best_funds["Hisse (Geleneksel & Katılım)"][1]
            st.session_state["a1"] = best_funds["Altın & Kıymetli Madenler"][0]
            st.session_state["a2"] = best_funds["Altın & Kıymetli Madenler"][1]
            st.sidebar.success("Seçilen tarih aralığına göre en iyi fonlar otomatik atandı!")

st.sidebar.subheader("A. Sukuk / Katılım Fonları")
s1 = st.sidebar.text_input("Sukuk Fon 1", key="s1").strip().upper()
s2 = st.sidebar.text_input("Sukuk Fon 2", key="s2").strip().upper()

st.sidebar.subheader("B. Hisse Fonları")
h1 = st.sidebar.text_input("Hisse Fon 1", key="h1").strip().upper()
h2 = st.sidebar.text_input("Hisse Fon 2", key="h2").strip().upper()

st.sidebar.subheader("C. Altın Fonları")
a1 = st.sidebar.text_input("Altın Fon 1", key="a1").strip().upper()
a2 = st.sidebar.text_input("Altın Fon 2", key="a2").strip().upper()

# =========================================================
# TAB MİMARİSİ
# =========================================================
tab_backtest, tab_katilim_byf, tab_all_eval, tab_real_portfolio = st.tabs([
    "📈 Backtest Simülasyonu",
    "☪️ Katılım Fonları & BYF Değerlendirmesi",
    "🔍 Tüm Fonların Değerlendirmesi",
    "💼 Gerçek Yatırım & Canlı Rebalance"
])

# ---------------------------------------------------------
# TAB 1: BACKTEST SİMÜLASYONU
# ---------------------------------------------------------
with tab_backtest:
    st.header("🔍 Portföy Yapılandırması ve Backtest")

    group_mapping = [
        ([s1, s2], w_sukuk),
        ([h1, h2], w_hisse),
        ([a1, a2], w_altin)
    ]

    raw_weights = {}
    for tickers, group_weight in group_mapping:
        valid_t = [t for t in tickers if t]
        if valid_t:
            w_per_ticker = (group_weight / len(valid_t)) / 100.0
            for t in valid_t:
                raw_weights[t] = raw_weights.get(t, 0) + w_per_ticker

    active_tickers = list(raw_weights.keys())

    if not active_tickers:
        st.warning("Lütfen sidebar üzerinden en az bir fon kodu giriniz.")
    else:
        data = download_tefas_data(active_tickers, start_date, end_date)

        if not data.empty:
            valid_cols = list(data.columns)
            weights_dict = {col: raw_weights.get(col, 0) for col in valid_cols}
            total_w = sum(weights_dict.values())
            if total_w > 0:
                weights_dict = {k: v / total_w for k, v in weights_dict.items()}

            def run_backtest(data, weights_dict, initial_cap, mode):
                returns = data.pct_change().fillna(0).replace([np.inf, -np.inf], 0)
                dates = data.index
                rebalance_dates = data.groupby(data.index.year).head(1).index
                portfolio_val = pd.Series(index=dates, dtype=float)

                target_weights = np.array([weights_dict.get(col, 0) for col in data.columns])
                current_alloc = target_weights * initial_cap
                rebalance_logs = []

                for date in dates:
                    if date == dates[0]:
                        portfolio_val.loc[date] = initial_cap
                        continue

                    daily_ret = returns.loc[date].values
                    current_alloc = current_alloc * (1 + daily_ret)
                    total_val = current_alloc.sum()

                    should_rebalance = False
                    trigger_reason = ""

                    if mode == "Yıllık (Takvim Başı)" and date in rebalance_dates:
                        should_rebalance = True
                        trigger_reason = "Planlı Yıl Başı Dengelemesi"
                    elif mode == "Tolerans Bandı (%10 Sapma)" and total_val > 0:
                        current_weights = current_alloc / total_val
                        relative_dev = np.abs(current_weights - target_weights) / np.maximum(target_weights, 1e-6)
                        if np.any(relative_dev >= 0.10):
                            should_rebalance = True
                            max_dev_idx = np.argmax(relative_dev)
                            trigger_reason = f"{data.columns[max_dev_idx]} varlığında %{relative_dev[max_dev_idx] * 100:.1f} sapma"

                    if should_rebalance and total_val > 0:
                        old_alloc_str = ", ".join(
                            [f"{col}: ₺{current_alloc[i]:,.0f} (%{(current_alloc[i] / total_val) * 100:.1f})" for i, col
                             in enumerate(data.columns)])
                        rebalance_logs.append({
                            "Tarih": date.strftime('%Y-%m-%d'),
                            "Neden": trigger_reason,
                            "Rebalance Öncesi Değer": f"₺{total_val:,.2f}",
                            "Varlık Dağılımı Detayı": old_alloc_str
                        })
                        current_alloc = target_weights * total_val

                    portfolio_val.loc[date] = total_val

                return portfolio_val, pd.DataFrame(rebalance_logs)

            portfolio_history, log_df = run_backtest(data, weights_dict, initial_capital, rebalance_mode)

            tot_ret_reb = ((portfolio_history.iloc[-1] - initial_capital) / initial_capital) * 100
            years = max((portfolio_history.index[-1] - portfolio_history.index[0]).days / 365.25, 0.01)
            cagr_reb = (((portfolio_history.iloc[-1] / initial_capital) ** (1 / years)) - 1) * 100
            max_dd = ((portfolio_history - portfolio_history.cummax()) / portfolio_history.cummax()).min() * 100

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Son Portföy Değeri (TL)", f"₺{portfolio_history.iloc[-1]:,.2f}")
            col2.metric(f"Getiri ({rebalance_mode})", f"%{tot_ret_reb:.2f}")
            col3.metric("CAGR (Yıllık Büyüme)", f"%{cagr_reb:.2f}")
            col4.metric("Max Drawdown", f"%{max_dd:.2f}")
            col5.metric("Rebalance Sayısı", len(log_df))

            st.subheader(f"📊 Portföy Büyüme Grafiği (TL) — Mod: {rebalance_mode}")
            fig_backtest = px.line(portfolio_history, labels={"value": "Portföy Değeri (₺)", "index": "Tarih"})
            fig_backtest.update_layout(showlegend=False)
            st.plotly_chart(fig_backtest, use_container_width=True)

            st.subheader("📋 Rebalance İşlem Logları")
            st.dataframe(log_df, use_container_width=True)
        else:
            st.error("Seçilen tarihler ve fonlar için TEFAS'tan veri çekilemedi.")

# ---------------------------------------------------------
# TAB 2: KATILIM FONLARI, BYF & İSLAMİ KRİTER ANALİZİ
# ---------------------------------------------------------
with tab_katilim_byf:
    st.header("☪️ Katılım Fonları (İslami Finans) & BYF Analiz Modülü")
    st.markdown(
        "Bu bölümde **Danışma Kurulu / Faizsiz Finans Prensiplerine** uygun Katılım Fonları ile **Borsa Yatırım Fonları (BYF / ETF)** özel olarak filtrelenip kıyaslanmaktadır."
    )

    filter_option = st.radio(
        "Filtreleme Tercihi Seçin:",
        ["Sadece Katılım (İslami Uyumlu) Fonlar", "Sadece Borsa Yatırım Fonları (BYF)", "Tüm Katılım ve BYF Listesi"],
        horizontal=True
    )

    selected_tickers = []
    for code, meta in ISLAMIC_METADATA.items():
        if filter_option == "Sadece Katılım (İslami Uyumlu) Fonlar" and meta["Katılım Uygun"]:
            selected_tickers.append(code)
        elif filter_option == "Sadece Borsa Yatırım Fonları (BYF)" and "BYF" in meta["Tür"]:
            selected_tickers.append(code)
        elif filter_option == "Tüm Katılım ve BYF Listesi":
            selected_tickers.append(code)

    if st.button("☪️ İslami Uyumlu / BYF Performanslarını Analiz Et", type="primary"):
        with st.spinner("Katılım Fonları ve BYF verileri TEFAS'tan indiriliyor..."):
            k_data = download_tefas_data(selected_tickers, start_date, end_date)
            if not k_data.empty:
                k_rows = []
                for col in k_data.columns:
                    s = k_data[col].dropna()
                    if len(s) < 10:
                        continue
                    tot_r = ((s.iloc[-1] / s.iloc[0]) - 1) * 100
                    yrs = max((s.index[-1] - s.index[0]).days / 365.25, 0.01)
                    cg = (((s.iloc[-1] / s.iloc[0]) ** (1 / yrs)) - 1) * 100
                    vl = s.pct_change().std() * np.sqrt(252) * 100
                    dd = ((s - s.cummax()) / s.cummax()).min() * 100
                    sk = cg / vl if vl > 0 else 0

                    meta = ISLAMIC_METADATA.get(col, {"Tür": "Bilinmiyor", "Katılım Uygun": True, "Açıklama": col})

                    k_rows.append({
                        "Fon Kodu": col,
                        "Fon Unvanı / Açıklama": meta["Açıklama"],
                        "Kategori / Tür": meta["Tür"],
                        "İslami Statü": "✅ Katılım Uyumlu" if meta["Katılım Uygun"] else "❌ Geleneksel",
                        "Toplam Getiri (%)": round(tot_r, 2),
                        "CAGR (%)": round(cg, 2),
                        "Yıllık Volatilite (%)": round(vl, 2),
                        "Max Drawdown (%)": round(dd, 2),
                        "Sharpe Skoru": round(sk, 2)
                    })

                k_df = pd.DataFrame(k_rows).sort_values("Sharpe Skoru", ascending=False)
                st.subheader("📊 Analiz ve İslami Kriter Sonuç Tablosu")
                st.dataframe(k_df, use_container_width=True, hide_index=True)

                st.subheader("📈 Karşılaştırmalı Göreli Performans Grafiği (100 Bazlı)")
                norm_k_df = (k_data / k_data.iloc[0]) * 100
                st.plotly_chart(
                    px.line(norm_k_df, labels={"value": "Endekslenmiş Değer (100=Başlangıç)", "date": "Tarih"}),
                    use_container_width=True)

# ---------------------------------------------------------
# TAB 3: TÜM FONLARIN KATEGORİ BAZLI DEĞERLENDİRMESİ
# ---------------------------------------------------------
with tab_all_eval:
    st.header("📊 Kategorilerdeki BÜTÜN Fonların Detaylı Değerlendirmesi")

    sel_cat = st.selectbox("Analiz Edilecek Varlık Kategorisini Seçin:", list(CANDIDATE_POOLS.keys()))
    cat_tickers = CANDIDATE_POOLS[sel_cat]

    if st.button("🚀 Kategorideki Tüm Fonları Getir ve Analiz Et"):
        with st.spinner("Tüm fonların fiyat geçmişi çekiliyor..."):
            cat_data = download_tefas_data(cat_tickers, start_date, end_date)
            if not cat_data.empty:
                eval_rows = []
                for col in cat_data.columns:
                    s = cat_data[col].dropna()
                    if len(s) < 10:
                        continue
                    tot_r = ((s.iloc[-1] / s.iloc[0]) - 1) * 100
                    yrs = max((s.index[-1] - s.index[0]).days / 365.25, 0.01)
                    cg = (((s.iloc[-1] / s.iloc[0]) ** (1 / yrs)) - 1) * 100
                    vl = s.pct_change().std() * np.sqrt(252) * 100
                    dd = ((s - s.cummax()) / s.cummax()).min() * 100
                    sk = cg / vl if vl > 0 else 0

                    eval_rows.append({
                        "Fon Kodu": col,
                        "Başlangıç Fiyatı (₺)": round(s.iloc[0], 4),
                        "Son Fiyat (₺)": round(s.iloc[-1], 4),
                        "Toplam Getiri (%)": round(tot_r, 2),
                        "CAGR (%)": round(cg, 2),
                        "Yıllık Volatilite (%)": round(vl, 2),
                        "Max Drawdown (%)": round(dd, 2),
                        "Risk/Getiri Skoru": round(sk, 2)
                    })

                eval_df = pd.DataFrame(eval_rows).sort_values("Risk/Getiri Skoru", ascending=False)
                st.dataframe(eval_df, use_container_width=True, hide_index=True)

                norm_df = (cat_data / cat_data.iloc[0]) * 100
                st.plotly_chart(px.line(norm_df, labels={"value": "Endekslenmiş Değer", "date": "Tarih"}),
                                use_container_width=True)

# ---------------------------------------------------------
# TAB 4: GERÇEK YATIRIM GİRİŞİ VE CANLI REBALANCE TAKİBİ
# ---------------------------------------------------------
with tab_real_portfolio:
    st.header("💼 Gerçek Yatırım Portföyü ve Canlı Rebalance Takibi")
    st.caption("Gerçekte satın aldığınız fonları, **giriş tarihleri ve harcadığınız miktar** ile girin.")

    with st.expander("➕ Yeni Fon Pozisyonu Ekle", expanded=True):
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            in_code = st.text_input("Fon Kodu", value="KTT").strip().upper()
        with p_col2:
            in_date = st.date_input("Giriş / Satın Alım Tarihi", pd.to_datetime("2024-01-01"))
        with p_col3:
            in_amount = st.number_input("Yatırılan Tutar (₺)", value=20000.0, step=1000.0)

        if st.button("➕ Pozisyonu Portföye Ekle"):
            new_p = pd.DataFrame(
                [{"Fon Kodu": in_code, "Giriş Tarihi": in_date.strftime("%Y-%m-%d"), "Yatırılan Tutar (₺)": in_amount}])
            st.session_state.real_portfolio = pd.concat([st.session_state.real_portfolio, new_p], ignore_index=True)
            st.success(f"{in_code} eklendi!")

    st.subheader("📋 Mevcut Girişleriniz")
    edited_p = st.data_editor(st.session_state.real_portfolio, num_rows="dynamic", use_container_width=True)
    st.session_state.real_portfolio = edited_p

    if st.button("🔄 Canlı Getirileri ve Rebalance Durumunu Hesapla", type="primary"):
        if not edited_p.empty:
            records = edited_p.to_dict('records')
            p_codes = list(set([r["Fon Kodu"] for r in records if r.get("Fon Kodu")]))
            min_dt = min([pd.to_datetime(r["Giriş Tarihi"]) for r in records])

            p_data = download_tefas_data(p_codes, min_dt, pd.to_datetime("today"))

            if not p_data.empty:
                calc_rows = []
                tot_maliyet = 0
                tot_guncel = 0

                def get_group(code):
                    if code in [s1, s2] or code in CANDIDATE_POOLS["Sukuk (Katılım Kira Sertifikaları)"]:
                        return "Katılım Sukuk Grubu"
                    if code in [h1, h2] or code in CANDIDATE_POOLS["Hisse (Geleneksel & Katılım)"]:
                        return "Hisse Grubu"
                    if code in [a1, a2] or code in CANDIDATE_POOLS["Altın & Kıymetli Madenler"]:
                        return "Altın Grubu"
                    return "Diğer"

                for r in records:
                    code = r["Fon Kodu"]
                    e_dt = pd.to_datetime(r["Giriş Tarihi"])
                    inv = float(r["Yatırılan Tutar (₺)"])

                    if code in p_data.columns:
                        s_prices = p_data[code].dropna()
                        v_prices = s_prices[s_prices.index >= e_dt]

                        if not v_prices.empty:
                            buy_p = v_prices.iloc[0]
                            curr_p = s_prices.iloc[-1]
                            lots = inv / buy_p if buy_p > 0 else 0
                            curr_val = lots * curr_p
                            profit = curr_val - inv
                            profit_pct = (profit / inv) * 100 if inv > 0 else 0

                            tot_maliyet += inv
                            tot_guncel += curr_val

                            meta = ISLAMIC_METADATA.get(code, {"Katılım Uygun": True})

                            calc_rows.append({
                                "Fon Kodu": code,
                                "Varlık Grubu": get_group(code),
                                "İslami Statü": "✅ Katılım" if meta.get("Katılım Uygun", True) else "❌ Geleneksel",
                                "Giriş Tarihi": r["Giriş Tarihi"],
                                "Yatırılan (Maliyet) ₺": round(inv, 2),
                                "Alış Fiyatı ₺": round(buy_p, 4),
                                "Adet (Lot)": round(lots, 2),
                                "Güncel Fiyat ₺": round(curr_p, 4),
                                "Güncel Değer ₺": round(curr_val, 2),
                                "Kâr/Zarar ₺": round(profit, 2),
                                "Getiri (%)": round(profit_pct, 2)
                            })

                res_df = pd.DataFrame(calc_rows)

                st.divider()
                st.subheader("💰 Canlı Portföy Özeti")
                k1, k2, k3 = st.columns(3)
                k1.metric("Toplam Maliyet", f"₺{tot_maliyet:,.2f}")
                k2.metric("Güncel Portföy Değeri", f"₺{tot_guncel:,.2f}")
                k3.metric("Toplam Kâr / Zarar", f"₺{(tot_guncel - tot_maliyet):,.2f}",
                          delta=f"%{((tot_guncel - tot_maliyet) / tot_maliyet) * 100:.2f}" if tot_maliyet > 0 else "0%")

                st.dataframe(res_df, use_container_width=True, hide_index=True)

                st.divider()
                st.subheader("⚖️ Portföy Dağılımı & Rebalance Kontrolü")

                cat_summary = res_df.groupby("Varlık Grubu")["Güncel Değer ₺"].sum().reset_index()
                cat_summary["Mevcut Pay (%)"] = (cat_summary["Güncel Değer ₺"] / tot_guncel) * 100 if tot_guncel > 0 else 0

                target_map = {"Katılım Sukuk Grubu": w_sukuk, "Hisse Grubu": w_hisse, "Altın Grubu": w_altin}
                cat_summary["Hedef Pay (%)"] = cat_summary["Varlık Grubu"].map(target_map).fillna(0)
                cat_summary["Sapma (%)"] = cat_summary["Mevcut Pay (%)"] - cat_summary["Hedef Pay (%)"]

                c_col1, c_col2 = st.columns([1, 1])
                with c_col1:
                    fig_p = px.pie(cat_summary, values="Güncel Değer ₺", names="Varlık Grubu",
                                   title="Mevcut Varlık Dağılımı", hole=0.4)
                    st.plotly_chart(fig_p, use_container_width=True)

                with c_col2:
                    st.write("**Hedef ve Mevcut Dağılım Karşılaştırması:**")
                    st.dataframe(cat_summary[["Varlık Grubu", "Mevcut Pay (%)", "Hedef Pay (%)", "Sapma (%)"]],
                                 hide_index=True, use_container_width=True)

                    max_dev = cat_summary["Sapma (%)"].abs().max()
                    if max_dev >= 10.0:
                        st.warning(
                            f"⚠️ **Rebalance Zamanı!** Varlık sınıflarınız hedeflenen oranlardan %{max_dev:.1f} sapmış durumda.")
                    else:
                        st.success("✅ Portföy dağılımınız belirlediğiniz hedef ağırlıklarla uyumlu.")