import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import vectorbt as vbt
import yfinance as yf

# ==========================================
# 1. SAYFA YAPILANDIRMASI & EVREN TANIMI
# ==========================================
st.set_page_config(
    page_title="Histerezis RS Momentum Portföy Yöneticisi",
    page_icon="📈",
    layout="wide"
)

TICKERS = [
    "AKBNK.IS", "ALARK.IS", "ASELS.IS", "BIMAS.IS", "BRSAN.IS",
    "EKGYO.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS",
    "GUBRF.IS", "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KRDMD.IS",
    "ODAS.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS",
    "SASA.IS", "SISE.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS",
    "TUPRS.IS", "YKBNK.IS"
]

FEE_RATE = 0.0015  # Backtest'teki komisyon oranıyla tutarlı olması için sabit


# ==========================================
# 2. VERİ ÇEKME & CACHE FONKSİYONLARI
# ==========================================
@st.cache_data(ttl=3600)
def load_historical_data(selected_period):
    """Backtest için geçmiş fiyat verilerini ve XU100 endeksini çeker."""
    c_dict = {}
    for t in TICKERS:
        try:
            df = yf.Ticker(t).history(period=selected_period, interval="1d", auto_adjust=True)
            if not df.empty and 'Close' in df.columns:
                c_dict[t] = df['Close']
        except Exception:
            pass
    df_c = pd.DataFrame(c_dict).dropna(how='all', axis=1).ffill().dropna()

    xu100 = yf.Ticker("XU100.IS").history(period=selected_period, interval="1d", auto_adjust=True)['Close']
    xu100 = xu100.reindex(df_c.index).ffill()
    return df_c, xu100


@st.cache_data(ttl=900)
def load_live_data():
    """Rebalance ve Sinyal Tablosu için son 1 yıllık verileri çeker."""
    data = {}
    for t in TICKERS:
        try:
            df = yf.Ticker(t).history(period="1y", interval="1d", auto_adjust=True)
            if not df.empty and len(df) >= 200:
                data[t] = df['Close']
        except Exception:
            pass

    df_close = pd.DataFrame(data).ffill().dropna()
    xu100 = yf.Ticker("XU100.IS").history(period="1y", interval="1d", auto_adjust=True)['Close']
    xu100 = xu100.reindex(df_close.index).ffill()

    latest_prices = df_close.iloc[-1]
    stock_sma200 = df_close.rolling(200).mean().iloc[-1]
    rs_scores = df_close.pct_change(126).iloc[-1]

    xu100_sma200 = xu100.rolling(200).mean().iloc[-1]
    mkt_filter_pass = xu100.iloc[-1] > xu100_sma200

    return latest_prices, stock_sma200, rs_scores, mkt_filter_pass, df_close


# ==========================================
# 3. BACKTEST ÇEKİRDEK FONKSİYONU (yeniden kullanılabilir hale getirildi
#    ki hem ana backtest hem de duyarlılık analizi aynı mantığı kullansın)
# ==========================================
def run_backtest(df_close, xu100, max_pos, sl_stop, sl_trail, cash_yield_annual):
    xu100_sma200 = xu100.rolling(200).mean()
    mkt_filter = pd.DataFrame(
        np.tile((xu100 > xu100_sma200).values[:, None], (1, df_close.shape[1])),
        index=df_close.index, columns=df_close.columns
    )

    stock_sma200 = df_close.rolling(200).mean()
    stock_filter = df_close > stock_sma200
    rs_score = df_close.pct_change(126)

    monthly_mask = df_close.index.to_series().dt.month.diff() != 0
    monthly_mask.iloc[0] = True

    active_positions = pd.DataFrame(False, index=df_close.index, columns=df_close.columns)
    current_portfolio = []

    for idx in df_close.index:
        if monthly_mask.loc[idx]:
            scores = rs_score.loc[idx].dropna()
            if len(scores) >= max_pos:
                top_in = scores.nlargest(max_pos).index.tolist()
                top_buffer = scores.nlargest(max_pos * 2).index.tolist()

                new_port = [t for t in current_portfolio if t in top_buffer]
                for t in top_in:
                    if len(new_port) < max_pos and t not in new_port:
                        new_port.append(t)
                current_portfolio = new_port
        if current_portfolio:
            active_positions.loc[idx, current_portfolio] = True

    raw_entries = active_positions & stock_filter & mkt_filter
    entries = raw_entries & (~raw_entries.shift(1).fillna(False))
    exits = (~active_positions) | (~stock_filter)

    pf = vbt.Portfolio.from_signals(
        df_close, entries, exits,
        size=1.0 / max_pos, size_type='percent',
        sl_stop=sl_stop, sl_trail=sl_trail,
        init_cash=100_000, cash_sharing=True,
        fees=FEE_RATE, slippage=0.001, freq="1d"
    )

    # --- DÜZELTME 1: Faiz artık ekrana yazılan rakamla değil,
    # gerçekten equity eğrisine eklenen kümülatif bir seri ile hesaplanıyor.
    strat_value_no_interest = pf.value()
    daily_yield = (1 + cash_yield_annual) ** (1 / 252) - 1
    cash_series = pf.cash()
    # Faiz, bir önceki günün boşta duran nakdi üzerinden o gün işlemiş kabul edilir
    daily_interest = cash_series.shift(1).fillna(cash_series.iloc[0]) * daily_yield
    cumulative_interest = daily_interest.cumsum()
    strat_value_with_interest = strat_value_no_interest + cumulative_interest

    total_ret_no_interest = (strat_value_no_interest.iloc[-1] - 100_000) / 100_000 * 100
    total_ret_with_interest = (strat_value_with_interest.iloc[-1] - 100_000) / 100_000 * 100
    interest_earned = cumulative_interest.iloc[-1]

    sharpe = pf.sharpe_ratio()
    max_dd = pf.max_drawdown() * 100
    win_rate = pf.trades.win_rate() * 100
    total_trades = pf.trades.count()

    return {
        "pf": pf,
        "strat_value_no_interest": strat_value_no_interest,
        "strat_value_with_interest": strat_value_with_interest,
        "total_ret_no_interest": total_ret_no_interest,
        "total_ret_with_interest": total_ret_with_interest,
        "interest_earned": interest_earned,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "total_trades": total_trades,
    }


# ==========================================
# 4. YAN MENÜ (SIDEBAR) PARAMETRELERİ
# ==========================================
st.sidebar.title("⚙️ Strateji Parametreleri")
period = st.sidebar.selectbox("Backtest Veri Periyodu", ["3y", "5y", "10y"], index=1)
max_pos = st.sidebar.slider("Maksimum Pozisyon Sayısı", 3, 10, 5)
sl_stop = st.sidebar.slider("Sabit Stop Loss (%)", 5, 25, 15) / 100.0
sl_trail = st.sidebar.slider("İzleyen Stop (%)", 10, 35, 25) / 100.0
cash_yield_annual = st.sidebar.slider("Nakit Gecelik Faiz/PPF (Yıllık %)", 0, 50, 35) / 100.0

st.title("📈 RS Trend & Histerezis Momentum Portföy Yöneticisi")

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Bu araç eğitim/analiz amaçlıdır, yatırım tavsiyesi değildir. "
    "Backtest sonuçları geleceği garanti etmez."
)

tab1, tab2, tab3 = st.tabs([
    "📈 Strateji Backtest Engine",
    "⚖️ Canlı Rebalance & Lot Hesaplayıcı",
    "📋 Anlık RS Sıralaması & Sinyaller"
])

# ==========================================
# TAB 1: BACKTEST ENGINE
# ==========================================
with tab1:
    st.markdown("### 🚀 Histerezis RS + SMA200 Trend Backtest")

    if st.button("Backtest'i Çalıştır", use_container_width=True):
        with st.spinner("Veriler işleniyor ve simülasyon koşturuluyor..."):
            df_close, xu100 = load_historical_data(period)
            result = run_backtest(df_close, xu100, max_pos, sl_stop, sl_trail, cash_yield_annual)

            st.session_state["last_df_close"] = df_close
            st.session_state["last_xu100"] = xu100
            st.session_state["last_result"] = result

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Toplam Getiri (Faiz Dahil)", f"%{result['total_ret_with_interest']:.2f}")
            c2.metric("Sharpe Oranı", f"{result['sharpe']:.2f}")
            c3.metric("Max Drawdown", f"%{result['max_dd']:.2f}")
            c4.metric("Kazanma Oranı", f"%{result['win_rate']:.2f}")
            c5.metric("İşlem Sayısı", f"{result['total_trades']}")

            st.info(
                f"💡 **Nakit Yönetimi Detayı:** Boş nakde işleyen ve equity eğrisine eklenen toplam faiz: "
                f"**₺{result['interest_earned']:,.0f}** | "
                f"Faiz Öncesi (Sadece Hisse) Getiri: **%{result['total_ret_no_interest']:.2f}** | "
                f"Faiz Dahil Net Getiri: **%{result['total_ret_with_interest']:.2f}**"
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=result["strat_value_with_interest"].index,
                y=result["strat_value_with_interest"].values,
                name="RS Histerezis Stratejisi (Faiz Dahil)",
                line=dict(color="#00CC96", width=2.5)
            ))
            fig.add_trace(go.Scatter(
                x=result["strat_value_no_interest"].index,
                y=result["strat_value_no_interest"].values,
                name="RS Histerezis Stratejisi (Faizsiz)",
                line=dict(color="#00CC96", width=1.2, dash="dot")
            ))

            bench = df_close.pct_change().mean(axis=1).add(1).cumprod().mul(100000)
            fig.add_trace(go.Scatter(
                x=bench.index, y=bench.values,
                name="BIST30 Eşit Ağırlıklı Benchmark",
                line=dict(color="#AB63FA", width=1.5, dash="dash")
            ))

            fig.update_layout(
                title="Sermaye Eğrisi Gelişimi (TL)",
                xaxis_title="Tarih", yaxis_title="Portföy Değeri (TL)",
                template="plotly_dark", height=500, hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- EKLENTİ: Yıllık (alt-dönem) getiri dağılımı ---
            st.markdown("#### 📅 Yıllık Getiri Dağılımı (Alt-Dönem Analizi)")
            st.caption(
                "Tek bir 5-10 yıllık toplam getiri rakamı, stratejinin farklı piyasa "
                "rejimlerindeki (yatay, düşüş, güçlü yükseliş) davranışını gizleyebilir. "
                "Aşağıdaki tablo yıl bazında performansı ayırır."
            )
            yearly_value = result["strat_value_with_interest"].resample("YE").last()
            yearly_value_start = result["strat_value_with_interest"].resample("YE").first()
            yearly_ret = ((yearly_value / yearly_value_start) - 1) * 100
            yearly_df = pd.DataFrame({
                "Yıl": yearly_ret.index.year,
                "Yıllık Getiri (%)": yearly_ret.values.round(2)
            })
            st.dataframe(yearly_df, use_container_width=True, hide_index=True)

    # --- EKLENTİ: Parametre Duyarlılık Analizi (Overfitting kontrolü) ---
    st.markdown("---")
    st.markdown("### 🔬 Parametre Duyarlılık Analizi")
    st.caption(
        "Stop-loss ve maksimum pozisyon sayısını küçük bir ızgara (grid) üzerinde değiştirip "
        "sonucun tek bir parametre setine ne kadar bağımlı (overfit) olduğunu gösterir. "
        "Sharpe oranı parametre değiştikçe çok dalgalanıyorsa, sonuç kırılgan demektir."
    )
    if st.button("Duyarlılık Analizini Çalıştır (birkaç dakika sürebilir)"):
        with st.spinner("Farklı parametre kombinasyonları test ediliyor..."):
            df_close_sens, xu100_sens = load_historical_data(period)
            sl_grid = sorted(set([max(0.05, sl_stop - 0.05), sl_stop, min(0.25, sl_stop + 0.05)]))
            pos_grid = sorted(set([max(3, max_pos - 2), max_pos, min(10, max_pos + 2)]))

            rows = []
            for sp in pos_grid:
                for sl in sl_grid:
                    r = run_backtest(df_close_sens, xu100_sens, sp, sl, sl_trail, cash_yield_annual)
                    rows.append({
                        "Max Pozisyon": sp,
                        "Stop Loss (%)": round(sl * 100, 1),
                        "Toplam Getiri (%)": round(r["total_ret_with_interest"], 1),
                        "Sharpe": round(r["sharpe"], 2),
                        "Max DD (%)": round(r["max_dd"], 1),
                        "İşlem Sayısı": r["total_trades"],
                    })
            sens_df = pd.DataFrame(rows)
            st.dataframe(sens_df, use_container_width=True, hide_index=True)

            sharpe_std = sens_df["Sharpe"].std()
            sharpe_mean = sens_df["Sharpe"].mean()
            if sharpe_mean != 0 and abs(sharpe_std / sharpe_mean) > 0.3:
                st.warning(
                    f"⚠️ Sharpe oranının parametreler arası değişkenliği yüksek "
                    f"(ortalama {sharpe_mean:.2f}, std {sharpe_std:.2f}). Bu, seçilen parametrelerin "
                    f"geçmiş veriye aşırı uyum (overfitting) riski taşıdığına işaret edebilir."
                )
            else:
                st.success(
                    f"✅ Sharpe oranı parametre değişikliklerine göre nispeten kararlı görünüyor "
                    f"(ortalama {sharpe_mean:.2f}, std {sharpe_std:.2f})."
                )

# ==========================================
# TAB 2: CANLI REBALANCE & LOT HESAPLAYICI
# ==========================================
with tab2:
    st.markdown("### ⚖️ Portföy Rebalance ve Alım-Satım Lot Hesaplayıcı")

    latest_prices, stock_sma200, rs_scores, mkt_filter_pass, df_close_live = load_live_data()

    if not mkt_filter_pass:
        st.error(
            "⚠️ **Piyasa Filtresi Uyarısı:** XU100 endeksi SMA200 altında! Yeni pozisyon açılmamalı, nakitte kalınmalıdır.")
    else:
        st.success("✅ **Piyasa Filtresi Uygun:** XU100 endeksi SMA200 üzerinde. Rebalance sinyalleri aktif.")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        cash_balance = st.number_input("Mevcut Boş Nakit (TL)", min_value=0.0, value=25000.0, step=1000.0)
        st.caption(
            "Mevcut Eldeki Hisseler, Lot Sayıları ve Giriş Bilgileri "
            "(Stop-loss/trailing-stop hesaplanabilmesi için Giriş Fiyatı ve Giriş Tarihi gereklidir):"
        )

        min_date = df_close_live.index.min().date()
        max_date = df_close_live.index.max().date()

        initial_df = pd.DataFrame({
            "Hisse": ["THYAO.IS", "GARAN.IS", "ASELS.IS"],
            "Mevcut Lot": [150, 400, 300],
            "Giriş Fiyatı": [0.0, 0.0, 0.0],
            "Giriş Tarihi": [max_date, max_date, max_date],
        })

        edited_portfolio = st.data_editor(
            initial_df, num_rows="dynamic",
            column_config={
                "Hisse": st.column_config.SelectboxColumn("Hisse Kodu", options=TICKERS, required=True),
                "Mevcut Lot": st.column_config.NumberColumn("Mevcut Lot", min_value=0, step=1, required=True),
                "Giriş Fiyatı": st.column_config.NumberColumn(
                    "Giriş Fiyatı (₺)", min_value=0.0, step=0.01,
                    help="0 girilirse stop-loss/trailing-stop bu pozisyon için hesaplanmaz."
                ),
                "Giriş Tarihi": st.column_config.DateColumn(
                    "Giriş Tarihi", min_value=min_date, max_value=max_date,
                    help="Trailing stop, bu tarihten bugüne kadarki en yüksek fiyata göre hesaplanır."
                ),
            },
            use_container_width=True
        )

    # --- DÜZELTME 2: RS/buffer mantığına ek olarak stop-loss / trailing-stop kontrolü ---
    def check_stop_loss(hisse, entry_price, entry_date):
        """Sabit stop-loss ve trailing-stop tetiklenmiş mi kontrol eder."""
        if entry_price <= 0 or hisse not in df_close_live.columns:
            return False, None
        current_price = latest_prices.get(hisse, 0.0)
        fixed_stop_price = entry_price * (1 - sl_stop)

        entry_ts = pd.Timestamp(entry_date)
        price_since_entry = df_close_live[hisse].loc[df_close_live[hisse].index >= entry_ts]
        peak_price = price_since_entry.max() if not price_since_entry.empty else entry_price
        trailing_stop_price = peak_price * (1 - sl_trail)

        effective_stop_price = max(fixed_stop_price, trailing_stop_price)
        stopped_out = current_price < effective_stop_price
        return stopped_out, effective_stop_price

    valid_scores = rs_scores[(latest_prices > stock_sma200)].dropna()
    top_in = valid_scores.nlargest(max_pos).index.tolist()
    top_buffer = valid_scores.nlargest(max_pos * 2).index.tolist()

    current_held = [h for h in edited_portfolio["Hisse"].tolist() if h in latest_prices.index]

    # Stop-loss tetiklenen pozisyonları buffer'dan çıkar (RS'de hâlâ tutunuyor olsalar bile satılmalı)
    stop_status = {}
    for _, row in edited_portfolio.iterrows():
        h = row["Hisse"]
        stopped, stop_price = check_stop_loss(h, row.get("Giriş Fiyatı", 0.0), row.get("Giriş Tarihi", max_date))
        stop_status[h] = {"stopped": stopped, "stop_price": stop_price}

    target_portfolio = [h for h in current_held if h in top_buffer and not stop_status.get(h, {}).get("stopped", False)]
    for h in top_in:
        if len(target_portfolio) < max_pos and h not in target_portfolio:
            target_portfolio.append(h)

    # Toplam Değer Hesaplama
    total_stock_val = 0.0
    holding_dict = {}

    for idx, row in edited_portfolio.iterrows():
        hisse = row["Hisse"]
        lot = row["Mevcut Lot"]
        price = latest_prices.get(hisse, 0.0)
        total_stock_val += lot * price
        holding_dict[hisse] = lot

    net_portfolio_value = total_stock_val + cash_balance
    target_weight_per_stock = 1.0 / max_pos
    target_allocation_tl = net_portfolio_value * target_weight_per_stock

    rebalance_data = []
    all_relevant_tickers = list(set(list(holding_dict.keys()) + target_portfolio))

    for h in all_relevant_tickers:
        price = latest_prices.get(h, 0.0)
        curr_lot = holding_dict.get(h, 0)
        curr_val = curr_lot * price
        is_target = h in target_portfolio
        was_stopped = stop_status.get(h, {}).get("stopped", False)

        # --- DÜZELTME 3: Lot hesaplamasında komisyon payı düşülüyor ---
        effective_buy_price = price * (1 + FEE_RATE) if price > 0 else 0.0
        target_lot = int(target_allocation_tl // effective_buy_price) if (is_target and effective_buy_price > 0) else 0
        diff_lot = target_lot - curr_lot
        diff_val = diff_lot * price
        # satış işleminde komisyon net tahsilatı azaltır
        est_amount = abs(diff_val) * (1 - FEE_RATE) if diff_lot < 0 else abs(diff_val) * (1 + FEE_RATE)

        if diff_lot > 0:
            action = "🟢 AL"
        elif diff_lot < 0:
            action = "🛑 STOP-SAT" if was_stopped else "🔴 SAT"
        else:
            action = "⚪ TUT"

        rebalance_data.append({
            "Hisse": h,
            "Fiyat": f"₺{price:.2f}",
            "Mevcut Lot": curr_lot,
            "Hedef Lot": target_lot,
            "Aksiyon": action,
            "İşlem (Lot)": abs(diff_lot),
            "Tahmini Tutar (Komisyon Dahil)": f"₺{est_amount:,.2f}",
            "Stop Fiyatı": f"₺{stop_status.get(h, {}).get('stop_price'):.2f}" if stop_status.get(h, {}).get("stop_price") else "—",
        })

    reb_df = pd.DataFrame(rebalance_data)

    with col_right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Toplam Varlık", f"₺{net_portfolio_value:,.2f}")
        m2.metric("Hisse Portföy Değeri", f"₺{total_stock_val:,.2f}")
        m3.metric("Hisse Başı Hedef Büyüklük", f"₺{target_allocation_tl:,.2f}")

        any_stopped = any(v["stopped"] for v in stop_status.values())
        if any_stopped:
            stopped_names = [h for h, v in stop_status.items() if v["stopped"]]
            st.error(f"🛑 Stop-loss/trailing-stop tetiklenen pozisyon(lar): {', '.join(stopped_names)}")

        st.dataframe(reb_df, use_container_width=True)
        st.caption(
            "Not: Stop-loss/trailing-stop hesaplaması yalnızca Giriş Fiyatı > 0 girilen satırlar için yapılır. "
            "Giriş Fiyatı 0 bırakılan pozisyonlarda satış kararı yalnızca RS/buffer mantığına dayanır."
        )

# ==========================================
# TAB 3: ANLIK RS SIRALAMASI & SİNYALLER
# ==========================================
with tab3:
    st.markdown("### 📋 BIST30 RS Momentum ve Filtre Durum Tablosu")

    latest_prices, stock_sma200, rs_scores, mkt_filter_pass, _ = load_live_data()

    top_in_live = rs_scores[(latest_prices > stock_sma200)].nlargest(max_pos).index.tolist()
    top_buffer_live = rs_scores[(latest_prices > stock_sma200)].nlargest(max_pos * 2).index.tolist()

    status_data = []
    for t in TICKERS:
        price = latest_prices.get(t, 0.0)
        sma = stock_sma200.get(t, 0.0)
        score = rs_scores.get(t, 0.0)

        pass_sma = price > sma

        if t in top_in_live:
            pos_status = "🔥 İlk Hedef Grubu (Top IN)"
        elif t in top_buffer_live:
            pos_status = "🛡️ Buffer/Tampon Grubu (Hold)"
        else:
            pos_status = "⚪ Listede Yok"

        status_data.append({
            "Hisse": t,
            "Son Fiyat": f"₺{price:.2f}",
            "SMA 200": f"₺{sma:.2f}",
            "SMA200 Üstünde mi?": "✅ Evet" if pass_sma else "❌ Hayır",
            "6 Aylık RS Skoru": f"%{score * 100:.2f}",
            "Histerezis Durumu": pos_status
        })

    status_df = pd.DataFrame(status_data).sort_values(by="6 Aylık RS Skoru", ascending=False)
    st.dataframe(status_df, use_container_width=True)