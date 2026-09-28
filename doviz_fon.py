import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Çoklu Fonlu Katılım Portföy Backtest", layout="wide")

st.title("🛡️ Çoklu Fonlu Katılım & Döviz Portföyü Backtest")
st.markdown(
    "Her varlık grubu için **2 farklı fon/enstrüman** seçilerek oluşturulan dinamik rebalance ve loglama simülasyonu. "
    "Ayrıca her grup için tarih bazlı geçmiş performansa göre **en iyi 2 alternatif** otomatik önerilir."
)

# =========================================================
# ADAY FON HAVUZLARI
# =========================================================
CANDIDATE_POOLS = {
    "Sukuk": ["EMLC", "SJNK", "PGHY", "IGOV", "BWX", "EMB"],
    "Hisse": ["HLAL", "SPUS", "SPWO", "UMMA", "ISWD"],
    "Altın": ["GLD", "IAU", "SGOL", "GLDM"],
}
CANDIDATE_POOLS = {
    "Sukuk": ["EMLC", "SJNK", "PGHY", "IGOV", "BWX", "EMB"],
    "Hisse": ["HLAL", "SPUS", "SPWO", "UMMA", "ISWD"],
    "Altın": ["GLD", "IAU", "SGOL", "GLDM"],
}
# =========================================================
# SESSION STATE INITIALIZATION (Sadece ilk yüklemede atanır)
# =========================================================
defaults = {
    "s1": "EMLC", "s2": "SJNK",
    "h1": "HLAL", "h2": "SPUS",
    "a1": "GLD", "a2": "IAU",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================================================
# GÜVENLİ VERİ ÇEKME FONKSİYONLARI
# =========================================================
@st.cache_data
def download_data(tickers, start, end):
    df_list = []
    for t in tickers:
        try:
            raw = yf.download(t, start=start, end=end, auto_adjust=True)
            if not raw.empty:
                close_series = raw['Close'] if 'Close' in raw.columns else raw.iloc[:, 0]
                if isinstance(close_series, pd.DataFrame):
                    close_series = close_series.iloc[:, 0]
                close_series.name = t
                df_list.append(close_series)
        except Exception:
            continue

    if not df_list:
        return pd.DataFrame()

    df = pd.concat(df_list, axis=1).ffill()
    return df


@st.cache_data
def download_fx(start, end):
    try:
        fx_raw = yf.download("USDTRY=X", start=start, end=end, auto_adjust=True)
        if fx_raw.empty:
            return pd.Series(dtype=float)

        fx = fx_raw['Close'] if 'Close' in fx_raw.columns else fx_raw.iloc[:, 0]
        if isinstance(fx, pd.DataFrame):
            fx = fx.iloc[:, 0]
        return fx.dropna()
    except Exception:
        return pd.Series(dtype=float)


# =========================================================
# TARİH BAZLI ALTERNATİF ÖNERİ MOTORU
# =========================================================
@st.cache_data
def evaluate_candidates(tickers, start, end):
    data = download_data(tickers, start, end)
    if data.empty:
        return pd.DataFrame()

    rows = []
    for col in data.columns:
        series = data[col].dropna()
        if len(series) < 60:
            continue

        ret = series.pct_change().dropna()
        total_ret = (series.iloc[-1] / series.iloc[0] - 1) * 100
        years = (series.index[-1] - series.index[0]).days / 365.25

        if years <= 0:
            continue

        cagr = ((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100
        vol = ret.std() * np.sqrt(252) * 100
        peak = series.cummax()
        dd = ((series - peak) / peak).min() * 100
        sharpe_like = cagr / vol if vol > 0 and not np.isnan(vol) else np.nan

        rows.append({
            "Ticker": col,
            "Toplam Getiri (%)": round(total_ret, 2),
            "CAGR (%)": round(cagr, 2),
            "Yıllık Volatilite (%)": round(vol, 2),
            "Max Drawdown (%)": round(dd, 2),
            "Risk/Getiri Skoru": round(sharpe_like, 2) if not np.isnan(sharpe_like) else np.nan,
        })

    result = pd.DataFrame(rows)
    if not result.empty and "Risk/Getiri Skoru" in result.columns:
        result = result.sort_values("Risk/Getiri Skoru", ascending=False).reset_index(drop=True)

    return result


start_default = pd.to_datetime("2018-01-01")
end_default = pd.to_datetime("2026-01-01")


# =========================================================
# ÖNERİ UYGULAMA CALLBACK FONKSİYONLARI (Kalıcılık Sağlar)
# =========================================================
def apply_recommendation(key1, key2, val1, val2):
    st.session_state[key1] = val1
    st.session_state[key2] = val2


# =========================================================
# TABLO VE ÖNERİ BÖLÜMÜ
# =========================================================
st.header("🔍 Tarih Bazlı Alternatif Fon Önerileri")
st.caption(
    "Seçtiğiniz başlangıç/bitiş tarihleri arasındaki geçmiş performansa göre her varlık grubu için "
    "havuzdaki adaylar Risk/Getiri Skoru (CAGR ÷ Yıllık Volatilite) baz alınarak sıralanır ve en iyi 2 tanesi önerilir."
)

group_labels = {"Sukuk": ("s1", "s2"), "Hisse": ("h1", "h2"), "Altın": ("a1", "a2")}
rec_cols = st.columns(3)

for idx, (group_name, pool) in enumerate(CANDIDATE_POOLS.items()):
    with rec_cols[idx]:
        st.subheader(group_name)
        try:
            ranked = evaluate_candidates(pool, start_default, end_default)
            if ranked.empty:
                st.warning("Veri alınamadı.")
                continue

            top2 = ranked.head(2)
            st.dataframe(ranked, use_container_width=True, hide_index=True)

            st.markdown(
                f"**Önerilen 2 Alternatif:** `{top2.iloc[0]['Ticker']}` ve `{top2.iloc[1]['Ticker']}` "
                f"(seçilen tarih aralığındaki en yüksek risk/getiri skoruna sahip ikili)"
            )

            key1, key2 = group_labels[group_name]
            st.button(
                f"✅ {group_name} için öneriyi uygula",
                key=f"apply_{group_name}",
                on_click=apply_recommendation,
                args=(key1, key2, top2.iloc[0]["Ticker"], top2.iloc[1]["Ticker"])
            )

        except Exception as e:
            st.error(f"{group_name} adayları değerlendirilirken hata oluştu: {e}")

st.divider()

# =========================================================
# SIDEBAR: GİRDİLER VE AĞIRLIKLAR
# =========================================================
st.sidebar.header("1. Ana Varlık Grubu Ağırlıkları (%)")

w_sukuk = st.sidebar.number_input("Döviz Sukuk Grubu Toplam (%)", value=50, step=5)
w_hisse = st.sidebar.number_input("Hisse Grubu Toplam (%)", value=30, step=5)
w_altin = st.sidebar.number_input("Altın Grubu Toplam (%)", value=20, step=5)

if w_sukuk + w_hisse + w_altin != 100:
    st.sidebar.error(f"Ağırlıklar toplamı %100 olmalıdır! Şu an: %{w_sukuk + w_hisse + w_altin}")

st.sidebar.header("2. Fon / Enstrüman Seçimleri (Her Gruba 2 Fon)")

st.sidebar.subheader("A. Sukuk Fonları (Eşit Dağıtılır)")
s1 = st.sidebar.text_input("Sukuk Fon 1", key="s1")
s2 = st.sidebar.text_input("Sukuk Fon 2", key="s2")

st.sidebar.subheader("B. Hisse Fonları (Eşit Dağıtılır)")
h1 = st.sidebar.text_input("Hisse Fon 1", key="h1")
h2 = st.sidebar.text_input("Hisse Fon 2", key="h2")

st.sidebar.subheader("C. Altın Enstrümanları (Eşit Dağıtılır)")
a1 = st.sidebar.text_input("Altın Enstrüman 1", key="a1")
a2 = st.sidebar.text_input("Altın Enstrüman 2", key="a2")

st.sidebar.header("3. Dengeleme (Rebalance) Yöntemi")
rebalance_mode = st.sidebar.radio(
    "Dengeleme Stratejisi Seçin:",
    ["Yıllık (Takvim Başı)", "Tolerans Bandı (%10 Sapma)", "Buy & Hold (Hiç Etme)"]
)

st.sidebar.header("4. Backtest Parametreleri")
start_date = st.sidebar.date_input("Başlangıç Tarihi", start_default)
end_date = st.sidebar.date_input("Bitiş Tarihi", end_default)
initial_capital = st.sidebar.number_input("Başlangıç Sermayesi ($)", value=10000, step=1000)

st.sidebar.header("5. Döviz Bazlı Getiri Ayarları")
show_try = st.sidebar.checkbox("TL Bazlı Getiriyi Göster", value=True)

# =========================================================
# ANA BACKTEST
# =========================================================
tickers = [s1, s2, h1, h2, a1, a2]

weights_dict = {
    s1: (w_sukuk / 2) / 100.0,
    s2: (w_sukuk / 2) / 100.0,
    h1: (w_hisse / 2) / 100.0,
    h2: (w_hisse / 2) / 100.0,
    a1: (w_altin / 2) / 100.0,
    a2: (w_altin / 2) / 100.0
}

try:
    data = download_data(tickers, start_date, end_date)

    if data.empty:
        st.error("Seçilen fonlara ait fiyat verisi indirilemedi.")
        st.stop()


    def run_backtest(data, weights_dict, initial_cap, mode):
        returns = data.pct_change().fillna(0)
        dates = data.index

        rebalance_dates = data.groupby(data.index.year).head(1).index
        portfolio_val = pd.Series(index=dates, dtype=float)

        target_weights = np.array([weights_dict[col] for col in data.columns])
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

            if mode == "Yıllık (Takvim Başı)":
                if date in rebalance_dates:
                    should_rebalance = True
                    trigger_reason = "Planlı Yıl Başı Dengelemesi"

            elif mode == "Tolerans Bandı (%10 Sapma)":
                current_weights = current_alloc / total_val
                relative_dev = np.abs(current_weights - target_weights) / target_weights
                if np.any(relative_dev >= 0.10):
                    should_rebalance = True
                    max_dev_idx = np.argmax(relative_dev)
                    max_dev_ticker = data.columns[max_dev_idx]
                    dev_pct = relative_dev[max_dev_idx] * 100
                    trigger_reason = f"{max_dev_ticker} varlığında %{dev_pct:.1f} oranında sapma tespit edildi"

            elif mode == "Buy & Hold (Hiç Etme)":
                should_rebalance = False

            if should_rebalance:
                old_alloc_str = ", ".join(
                    [f"{col}: ${current_alloc[i]:,.0f} (%{(current_alloc[i] / total_val) * 100:.1f})" for i, col in
                     enumerate(data.columns)])
                rebalance_logs.append({
                    "Tarih": date.strftime('%Y-%m-%d'),
                    "Neden": trigger_reason,
                    "Rebalance Öncesi Portföy Değeri": f"${total_val:,.2f}",
                    "Varlık Dağılımı Detayı": old_alloc_str
                })
                current_alloc = target_weights * total_val

            portfolio_val.loc[date] = total_val

        return portfolio_val, pd.DataFrame(rebalance_logs)


    portfolio_history, log_df = run_backtest(data, weights_dict, initial_capital, rebalance_mode)

    init_weights = np.array([weights_dict[col] for col in data.columns])
    buy_hold_alloc = (data / data.iloc[0]) * (init_weights * initial_capital)
    buy_hold_history = buy_hold_alloc.sum(axis=1)

    tot_ret_reb = ((portfolio_history.iloc[-1] - initial_capital) / initial_capital) * 100
    tot_ret_bh = ((buy_hold_history.iloc[-1] - initial_capital) / initial_capital) * 100

    years = (portfolio_history.index[-1] - portfolio_history.index[0]).days / 365.25
    cagr_reb = (((portfolio_history.iloc[-1] / initial_capital) ** (1 / years)) - 1) * 100

    peak = portfolio_history.cummax()
    dd = (portfolio_history - peak) / peak
    max_dd = dd.min() * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Son Portföy Değeri", f"${portfolio_history.iloc[-1]:,.2f}")
    col2.metric(f"Getiri ({rebalance_mode})", f"%{tot_ret_reb:.2f}")
    col3.metric("CAGR (Yıllık Büyüme)", f"%{cagr_reb:.2f}")
    col4.metric("Max Drawdown", f"%{max_dd:.2f}")
    col5.metric("Rebalance Sayısı", len(log_df))

    st.subheader(f"📊 Portföy Büyüme Grafiği — Mod: {rebalance_mode}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_history.index, y=portfolio_history, mode='lines',
                             name=f'Aktif Strateji ({rebalance_mode})', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=buy_hold_history.index, y=buy_hold_history, mode='lines', name='Buy & Hold (Referans)',
                             line=dict(color='gray', dash='dash')))

    fig.update_layout(xaxis_title="Tarih", yaxis_title="Portföy Değeri ($)", hovermode="x unified",
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Rebalance İşlem Logları")
    if not log_df.empty:
        st.dataframe(log_df, use_container_width=True)
    else:
        st.info("Seçilen periyot ve stratejide herhangi bir rebalance işlemi gerçekleşmedi.")

    st.subheader("📈 Seçilen 6 Fonun Bireysel Performansı (Temettü/Kupon Dahil - Base 100)")
    norm_data = (data / data.iloc[0]) * 100
    st.line_chart(norm_data)

    # =========================================================
    # DÖVİZ (USD/TRY) VERİSİ VE DÖNÜŞÜM
    # =========================================================
    if show_try:
        fx_data = download_fx(start_date, end_date)

        if not fx_data.empty:
            fx_aligned = fx_data.reindex(data.index).ffill().bfill()

            portfolio_try = portfolio_history * fx_aligned
            buy_hold_try = buy_hold_history * fx_aligned

            initial_try = initial_capital * fx_aligned.iloc[0]
            tot_ret_try = ((portfolio_try.iloc[-1] - initial_try) / initial_try) * 100
            cagr_try = (((portfolio_try.iloc[-1] / initial_try) ** (1 / years)) - 1) * 100
            usdtry_change = ((fx_aligned.iloc[-1] - fx_aligned.iloc[0]) / fx_aligned.iloc[0]) * 100

            st.subheader("💱 Döviz (USD/TRY) Etkisi Dahil TL Bazlı Getiri")
            fcol1, fcol2, fcol3, fcol4 = st.columns(4)
            fcol1.metric("USD/TRY Değişimi", f"%{usdtry_change:.2f}",
                         f"{fx_aligned.iloc[0]:.2f} → {fx_aligned.iloc[-1]:.2f}")
            fcol2.metric("Son Portföy Değeri (TL)", f"₺{portfolio_try.iloc[-1]:,.2f}")
            fcol3.metric("TL Bazlı Toplam Getiri", f"%{tot_ret_try:.2f}")
            fcol4.metric("TL Bazlı CAGR", f"%{cagr_try:.2f}")

            fig_try = go.Figure()
            fig_try.add_trace(go.Scatter(x=portfolio_try.index, y=portfolio_try, mode='lines',
                                         name='Aktif Strateji (TL)', line=dict(color='darkgreen', width=2)))
            fig_try.add_trace(go.Scatter(x=buy_hold_try.index, y=buy_hold_try, mode='lines',
                                         name='Buy & Hold (TL, Referans)', line=dict(color='firebrick', dash='dash')))
            fig_try.update_layout(xaxis_title="Tarih", yaxis_title="Portföy Değeri (₺)",
                                  hovermode="x unified", template="plotly_white")
            st.plotly_chart(fig_try, use_container_width=True)

            st.subheader("📉 USD/TRY Kur Grafiği")
            st.line_chart(fx_aligned)
        else:
            st.warning("USD/TRY verisi çekilemediği için TL bazlı grafikler gösterilemiyor.")

except Exception as e:
    st.error(f"Veri yüklenirken veya hesaplama yapılırken bir hata oluştu: {e}")