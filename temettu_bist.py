
# temettu_bist.py
"""
BIST Temettü Odaklı Portföy Yönetim Sistemi
- Backtest (XIRR, Sharpe, Sortino, reel getiri)
- Çok varlıklı mod (hisse + altın + dolar)
- Dönem karşılaştırma, paper trade, canlı portföy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os
import borsapy as bp

# ============================================================
# SAYFA AYARLARI
# ============================================================
st.set_page_config(
    page_title="BIST Temettü Sistemi",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 BIST Temettü Odaklı Portföy Yönetim Paneli")
st.caption("Backtest → Paper Trade → Karşılaştırma → Kırmızı Bayrak → Canlı Portföy")

# ============================================================
# EVDS API ANAHTARI
# ============================================================
EVDS_KEY = os.environ.get("EVDS_API_KEY", "4JcbosMYYp")
if EVDS_KEY:
    try:
        bp.set_evds_key(EVDS_KEY)
    except Exception:
        pass
else:
    st.sidebar.warning(
        "⚠️ EVDS_API_KEY ortam değişkeni ayarlı değil. "
        "Enflasyon verisi çekilemeyecek."
    )

# ============================================================
# OTURUM DURUMU
# ============================================================
defaults = {
    "paper_portfoy": pd.DataFrame(columns=["Tarih", "Hisse", "Lot", "Fiyat", "Tür", "Tutar"]),
    "canli_portfoy": pd.DataFrame(columns=["Tarih", "Hisse", "Lot", "Fiyat", "Tür", "Tutar"]),
    "backtest_sonuc": None,
    "backtest_fiyat_df": None,
    "backtest_hedef": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================
def tz_temizle(obj):
    """DataFrame/Series index'ini tz-naive ve datetime64[ns] yapar."""
    if obj is None:
        return None
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        obj = obj.copy()
        idx = obj.index
        if not isinstance(idx, pd.DatetimeIndex):
            try:
                idx = pd.to_datetime(idx)
            except Exception:
                return obj
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
        idx = idx.astype("datetime64[ns]")
        obj.index = idx
        obj = obj.sort_index()
    return obj


@st.cache_data(ttl=3600)
def veri_cek(hisse_listesi, baslangic, bitis):
    fiyat_dict = {}
    hata_listesi = []
    for h in hisse_listesi:
        try:
            ticker = bp.Ticker(h)
            df = ticker.history(start=baslangic, end=bitis)
            if df is not None and not df.empty:
                fiyat_dict[h] = tz_temizle(df["Close"])
        except Exception as e:
            hata_listesi.append(f"{h}: {str(e)[:60]}")

    if hata_listesi:
        st.warning("Bazı hisseler için veri alınamadı:\n" + "\n".join(hata_listesi))

    if not fiyat_dict:
        return None

    fiyat_df = pd.DataFrame(fiyat_dict)
    fiyat_df = tz_temizle(fiyat_df)
    return fiyat_df.ffill()


@st.cache_data(ttl=3600)
def altin_cek(baslangic, bitis):
    try:
        df = bp.FX("gram-altin").history(start=baslangic, end=bitis)
        if df is not None and not df.empty:
            return tz_temizle(df["Close"])
    except Exception as e:
        st.info(f"Gram altın verisi alınamadı: {str(e)[:60]}")
    return None


@st.cache_data(ttl=3600)
def usd_cek(baslangic, bitis):
    try:
        df = bp.FX("USD").history(start=baslangic, end=bitis)
        if df is not None and not df.empty:
            return tz_temizle(df["Close"])
    except Exception as e:
        st.info(f"USD verisi alınamadı: {str(e)[:60]}")
    return None


@st.cache_data(ttl=3600)
def enflasyon_serisi_cek(baslangic, bitis):
    """TÜFE Genel Endeks (2003=100), EVDS: TP.FG.J0."""
    if not EVDS_KEY:
        return None
    try:
        seri = bp.evds_series(
            "TP.FG.J0", start=baslangic, end=bitis, frequency="monthly"
        )
        if seri is None or seri.empty:
            return None

        if isinstance(seri, pd.DataFrame):
            tarih_col = None
            for col in seri.columns:
                if "tarih" in col.lower() or "date" in col.lower():
                    tarih_col = col
                    break
            if tarih_col:
                seri[tarih_col] = pd.to_datetime(seri[tarih_col])
                seri = seri.set_index(tarih_col)
            sayisal = seri.select_dtypes(include="number")
            if not sayisal.empty:
                seri = sayisal.iloc[:, 0]
            else:
                return None

        if isinstance(seri, pd.Series):
            return tz_temizle(seri)
        return None
    except Exception as e:
        st.warning(f"EVDS enflasyon verisi alınamadı: {str(e)[:100]}")
        return None


def hesapla_sharpe(daily_returns):
    if len(daily_returns) < 2:
        return 0.0
    std = daily_returns.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float((daily_returns.mean() / std) * np.sqrt(252))


def hesapla_sortino(daily_returns):
    if len(daily_returns) < 2:
        return 0.0
    downside = daily_returns[daily_returns < 0]
    if len(downside) < 2:
        return 0.0
    dstd = downside.std()
    if dstd == 0 or pd.isna(dstd):
        return 0.0
    return float((daily_returns.mean() / dstd) * np.sqrt(252))


def hesapla_xirr(nakit_akislari, tarihler):
    """
    XIRR — para ağırlıklı yıllık getiri (%).
    nakit_akislari: negatif = yatırım, pozitif = geri dönüş.
    """
    if len(nakit_akislari) < 2:
        return None
    t0 = tarihler[0]
    gunler = np.array([(t - t0).days for t in tarihler], dtype=float)
    cf = np.array(nakit_akislari, dtype=float)

    if not (np.any(cf > 0) and np.any(cf < 0)):
        return None

    def npv(rate):
        if rate <= -1:
            return float("inf")
        return float(np.sum(cf / (1 + rate) ** (gunler / 365.25)))

    low, high = -0.999, 10.0
    try:
        f_low, f_high = npv(low), npv(high)
        if f_low * f_high > 0:
            return None
        for _ in range(300):
            mid = (low + high) / 2
            f_mid = npv(mid)
            if abs(f_mid) < 1e-6:
                return mid * 100
            if f_low * f_mid < 0:
                high = mid
                f_high = f_mid
            else:
                low = mid
                f_low = f_mid
        return ((low + high) / 2) * 100
    except Exception:
        return None


def backtest_simule(fiyat_df, baslangic_sermaye, aylik_alis, rebalance_esik,
                    hedef_agirliklar=None):
    """
    Düzenli alım + aylık rebalance backtest motoru.

    Düzeltmeler:
    - Nakit, her ay yatırılan tutar kadar ARTAR, sonra harcanır.
    - XIRR için nakit akışları kaydedilir.
    - Rebalance sadece satış yönünde nakit yaratır.
    """
    varliklar = fiyat_df.columns.tolist()
    n = len(varliklar)
    if n == 0:
        return None

    # Hedef ağırlıklar
    if hedef_agirliklar is None:
        hedef = {v: 100.0 / n for v in varliklar}
    else:
        toplam = sum(hedef_agirliklar.values())
        if toplam == 0:
            hedef = {v: 100.0 / n for v in varliklar}
        else:
            hedef = {k: v / toplam * 100 for k, v in hedef_agirliklar.items()}
        for v in varliklar:
            hedef.setdefault(v, 0.0)

    lotlar = {v: 0.0 for v in varliklar}
    nakit = 0.0

    tmp = pd.DataFrame(index=fiyat_df.index)
    tmp["_ym"] = tmp.index.to_period("M")
    aylik_ilk_gunler = set(tmp.groupby("_ym").head(1).index)

    equity_curve = []
    rebalance_kayitlari = []
    nakit_akislari = []      # XIRR için
    tarih_akislari = []
    toplam_yatirim = 0.0

    for i, tarih in enumerate(fiyat_df.index):
        # --- AYLIK YATIRIM ---
        if tarih in aylik_ilk_gunler or i == 0:
            yatirilan = aylik_alis if i > 0 else baslangic_sermaye
            nakit += yatirilan
            toplam_yatirim += yatirilan
            nakit_akislari.append(-yatirilan)
            tarih_akislari.append(tarih)

            fiyatlar = fiyat_df.loc[tarih]
            for v in varliklar:
                fiyat = fiyatlar.get(v, np.nan)
                if pd.notna(fiyat) and fiyat > 0 and hedef.get(v, 0) > 0:
                    hedef_tutar = yatirilan * (hedef[v] / 100)
                    if hedef_tutar > 0 and nakit >= hedef_tutar - 1e-6:
                        lotlar[v] += hedef_tutar / fiyat
                        nakit -= hedef_tutar

        # --- REBALANCE ---
        if i > 0 and tarih in aylik_ilk_gunler:
            fiyatlar = fiyat_df.loc[tarih]
            toplam_deger = nakit + sum(
                lotlar[v] * fiyatlar.get(v, 0) for v in varliklar
            )
            if toplam_deger > 0:
                # 1) Fazla olanları sat
                for v in varliklar:
                    fiyat = fiyatlar.get(v, np.nan)
                    if pd.notna(fiyat) and fiyat > 0:
                        mevcut = (lotlar[v] * fiyat) / toplam_deger * 100
                        sapma = mevcut - hedef[v]
                        if sapma > rebalance_esik:
                            hedef_tutar = toplam_deger * (hedef[v] / 100)
                            hedef_lot = hedef_tutar / fiyat
                            satilan = lotlar[v] - hedef_lot
                            lotlar[v] = hedef_lot
                            nakit += satilan * fiyat
                            rebalance_kayitlari.append({
                                "Tarih": tarih, "Varlık": v, "İşlem": "Satış",
                                "Sapma %": round(sapma, 2)
                            })
                # 2) Eksik olanları nakit varsa al
                for v in varliklar:
                    fiyat = fiyatlar.get(v, np.nan)
                    if pd.notna(fiyat) and fiyat > 0 and nakit > 0:
                        mevcut = (lotlar[v] * fiyat) / toplam_deger * 100
                        sapma = hedef[v] - mevcut
                        if sapma > rebalance_esik:
                            hedef_tutar = toplam_deger * (hedef[v] / 100)
                            alinacak_lot = (hedef_tutar - lotlar[v] * fiyat) / fiyat
                            maliyet = alinacak_lot * fiyat
                            if 0 < maliyet <= nakit + 1e-6:
                                lotlar[v] += alinacak_lot
                                nakit -= maliyet
                                rebalance_kayitlari.append({
                                    "Tarih": tarih, "Varlık": v, "İşlem": "Alış",
                                    "Sapma %": round(sapma, 2)
                                })

        # --- GÜNLÜK DEĞERLEME ---
        fiyatlar = fiyat_df.loc[tarih]
        poz_deger = sum(lotlar[v] * fiyatlar.get(v, 0) for v in varliklar)
        equity_curve.append({"Tarih": tarih, "Portföy Değeri": nakit + poz_deger})

    equity_df = pd.DataFrame(equity_curve).set_index("Tarih")
    equity_df = tz_temizle(equity_df)

    baslangic_deger = equity_df["Portföy Değeri"].iloc[0]
    bitis_deger = equity_df["Portföy Değeri"].iloc[-1]

    # Basit DCA getirisi (para ağırlıklı değil ama dürüst)
    basit_getiri = (bitis_deger / toplam_yatirim - 1) * 100 if toplam_yatirim > 0 else 0.0

    # XIRR (doğru para ağırlıklı yıllık getiri)
    nakit_akislari.append(bitis_deger)
    tarih_akislari.append(equity_df.index[-1])
    xirr = hesapla_xirr(nakit_akislari, tarih_akislari)

    # Sadece grafik referansı
    nominal_buyume = (bitis_deger / baslangic_deger - 1) * 100

    gun_sayisi = (equity_df.index[-1] - equity_df.index[0]).days
    yil = max(gun_sayisi / 365.25, 0.1)

    # Sharpe/Sortino: DCA akışları nedeniyle kirli; sadece referans
    daily_returns = equity_df["Portföy Değeri"].pct_change().dropna()
    daily_returns = daily_returns[daily_returns.abs() < 0.5]
    sharpe = hesapla_sharpe(daily_returns)
    sortino = hesapla_sortino(daily_returns)

    kumulatif_max = equity_df["Portföy Değeri"].cummax()
    drawdown = (equity_df["Portföy Değeri"] - kumulatif_max) / kumulatif_max * 100
    max_drawdown = float(drawdown.min())

    return {
        "equity": equity_df,
        "basit_getiri_pct": basit_getiri,
        "xirr_pct": xirr,
        "nominal_buyume_pct": nominal_buyume,
        "toplam_yatirim": toplam_yatirim,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_drawdown,
        "bitis_deger": bitis_deger,
        "net_kar": bitis_deger - toplam_yatirim,
        "lotlar": lotlar,
        "nakit": nakit,
        "rebalance_kayitlari": rebalance_kayitlari,
        "hedef_agirliklar": hedef,
    }


# ============================================================
# KENAR ÇUBUĞU
# ============================================================
with st.sidebar:
    st.header("⚙️ Sistem Parametreleri")

    varsayilan_hisseler = [
        "TUPRS", "FROTO", "AKBNK", "GARAN", "ISCTR",
        "KCHOL", "SAHOL", "TCELL", "TOASO", "BIMAS"
    ]
    tickers = st.multiselect(
        "Hisse Evreni",
        options=varsayilan_hisseler,
        default=["TUPRS", "FROTO", "AKBNK", "GARAN"]
    )

    st.divider()

    cok_varlikli = st.checkbox(
        "🌍 Çok Varlıklı Mod (Altın + Dolar Ekle)",
        value=False,
        help="Portföye gram altın ve USD/TRY ekler."
    )

    hisse_agirlik, altin_agirlik, dolar_agirlik = 100, 0, 0
    if cok_varlikli:
        hisse_agirlik = st.slider("Hisse Ağırlık %", 0, 100, 60, 5)
        altin_agirlik = st.slider("Altın Ağırlık %", 0, 100, 25, 5)
        dolar_agirlik = st.slider("Dolar Ağırlık %", 0, 100, 15, 5)
        toplam_ag = hisse_agirlik + altin_agirlik + dolar_agirlik
        if toplam_ag != 100:
            st.caption(f"ℹ️ Toplam {toplam_ag}% → 100%'e normalize edilecek.")

    st.divider()

    donem_secim = st.selectbox(
        "📅 Dönem Kısayolu",
        ["Özel", "2010-2014 (Boğa)", "2015-2019 (Normal)",
         "2018-2022 (Kriz)", "2022-2026 (Enflasyon)", "Son 3 Yıl"]
    )

    bugun = datetime.today()
    if donem_secim == "2010-2014 (Boğa)":
        vb, vbit = datetime(2010, 1, 1), datetime(2014, 12, 31)
    elif donem_secim == "2015-2019 (Normal)":
        vb, vbit = datetime(2015, 1, 1), datetime(2019, 12, 31)
    elif donem_secim == "2018-2022 (Kriz)":
        vb, vbit = datetime(2018, 1, 1), datetime(2022, 12, 31)
    elif donem_secim == "2022-2026 (Enflasyon)":
        vb, vbit = datetime(2022, 1, 1), bugun
    elif donem_secim == "Son 3 Yıl":
        vb, vbit = bugun.replace(year=bugun.year - 3), bugun
    else:
        vb, vbit = datetime(2022, 1, 1), bugun

    baslangic = st.date_input("Başlangıç Tarihi", vb)
    bitis = st.date_input("Bitiş Tarihi", vbit)

    baslangic_sermaye = st.number_input(
        "Başlangıç Sermayesi (TL)", value=100_000, step=10_000, min_value=1000
    )
    aylik_alis = st.number_input(
        "Aylık Ek Alım (TL)", value=5_000, step=1_000, min_value=0
    )

    rebalance_esik = st.slider("Rebalance Eşiği (%)", 0, 30, 10)

    st.divider()
    st.caption("⚠️ Bu panel eğitim amaçlıdır. Yatırım tavsiyesi değildir.")

# ============================================================
# ANA SEKMELER
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Backtest", "🧪 Paper Trade", "🔍 Karşılaştırma",
    "🚩 Kırmızı Bayraklar", "💼 Canlı Portföy"
])

# ============================================================
# TAB 1: BACKTEST
# ============================================================
with tab1:
    st.header("Geçmiş Performans Simülasyonu")

    if not tickers:
        st.warning("Lütfen kenar çubuğundan en az bir hisse seçin.")
    else:
        if st.button("🚀 Backtest Çalıştır", use_container_width=True):
            with st.spinner("Veriler çekiliyor ve simülasyon yapılıyor..."):
                fiyat_df = veri_cek(
                    tickers,
                    baslangic.strftime("%Y-%m-%d"),
                    bitis.strftime("%Y-%m-%d")
                )

                if fiyat_df is None or fiyat_df.empty:
                    st.error("Seçilen hisseler için veri alınamadı.")
                else:
                    hedef = None
                    if cok_varlikli and (altin_agirlik + dolar_agirlik) > 0:
                        altin_seri = altin_cek(
                            baslangic.strftime("%Y-%m-%d"),
                            bitis.strftime("%Y-%m-%d")
                        )
                        usd_seri = usd_cek(
                            baslangic.strftime("%Y-%m-%d"),
                            bitis.strftime("%Y-%m-%d")
                        )

                        if altin_seri is not None:
                            fiyat_df["Gram Altın"] = altin_seri.reindex(
                                fiyat_df.index, method="ffill"
                            )
                        if usd_seri is not None:
                            fiyat_df["USD/TRY"] = usd_seri.reindex(
                                fiyat_df.index, method="ffill"
                            )

                        fiyat_df = fiyat_df.ffill().dropna(how="all")

                        hisse_per = hisse_agirlik / len(tickers) if tickers else 0
                        hedef = {h: hisse_per for h in tickers}
                        if altin_seri is not None:
                            hedef["Gram Altın"] = altin_agirlik
                        if usd_seri is not None:
                            hedef["USD/TRY"] = dolar_agirlik

                    sonuc = backtest_simule(
                        fiyat_df,
                        baslangic_sermaye,
                        aylik_alis,
                        rebalance_esik,
                        hedef_agirliklar=hedef
                    )
                    st.session_state.backtest_sonuc = sonuc
                    st.session_state.backtest_fiyat_df = fiyat_df
                    st.session_state.backtest_hedef = hedef

        sonuc = st.session_state.get("backtest_sonuc")
        fiyat_df = st.session_state.get("backtest_fiyat_df")
        hedef = st.session_state.get("backtest_hedef")

        if sonuc is not None and fiyat_df is not None:
            equity = sonuc["equity"]

            st.subheader("📈 Portföy Değer Grafiği")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity.index, y=equity["Portföy Değeri"],
                mode="lines", name="Portföy Değeri",
                line=dict(color="#00CC96", width=2),
                fill="tozeroy", fillcolor="rgba(0,204,150,0.1)"
            ))
            fig.update_layout(
                xaxis_title="Tarih", yaxis_title="Değer (TL)",
                hovermode="x unified", height=450, template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            # ============ METRİKLER ============
            st.subheader("📊 Performans Metrikleri")

            xirr = sonuc.get("xirr_pct")
            xirr_str = f"{xirr:.1f}%" if xirr is not None else "—"

            # 1. Satır — gerçek getiri
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "XIRR (Yıllık, Doğru)",
                xirr_str,
                help="Para ağırlıklı yıllık getiri. DCA için doğru metriktir."
            )
            m2.metric(
                "Basit DCA Getirisi",
                f"{sonuc['basit_getiri_pct']:.1f}%",
                help="(Son Değer / Toplam Yatırım - 1) × 100"
            )
            net_kar = sonuc["net_kar"]
            m3.metric(
                "Net Kâr / Zarar",
                f"{net_kar:,.0f} TL",
                delta=f"{net_kar:,.0f} TL",
                delta_color="normal" if net_kar >= 0 else "inverse"
            )
            m4.metric("Maks. Drawdown", f"{sonuc['max_drawdown_pct']:.1f}%")

            # 2. Satır — risk & portföy
            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Sharpe Oranı", f"{sonuc['sharpe']:.2f}",
                      help=">1 iyi, >2 mükemmel.")
            m6.metric("Sortino Oranı", f"{sonuc['sortino']:.2f}",
                      help="Sadece aşağı yönlü volatiliteyi dikkate alır.")
            m7.metric("Toplam Yatırım", f"{sonuc['toplam_yatirim']:,.0f} TL")
            m8.metric(
                "Son Portföy Değeri",
                f"{sonuc['bitis_deger']:,.0f} TL",
                help=f"Nakit bakiye: {sonuc['nakit']:,.0f} TL"
            )

            # Uyarı: XIRR hesaplanamadıysa
            if xirr is None:
                st.info(
                    "ℹ️ XIRR hesaplanamadı. Bu genelde nakit akışlarının "
                    "tamamının aynı yönde olduğu (hep yatırım, hiç geri dönüş yok) "
                    "durumlarda olur. 'Basit DCA Getirisi' metriğine bakın."
                )

            # Yorum
            if net_kar < 0:
                st.error(
                    f"⚠️ Bu dönemde portföy **{abs(net_kar):,.0f} TL zarar** etti. "
                    f"Toplam yatırım {sonuc['toplam_yatirim']:,.0f} TL, "
                    f"son değer {sonuc['bitis_deger']:,.0f} TL. "
                    "Karşılaştırma sekmesinden enflasyon/altın/dolar ile kıyaslayın."
                )
            else:
                st.success(
                    f"✅ Portföy {net_kar:,.0f} TL kâr etti. "
                    f"Reel getiriyi görmek için Karşılaştırma sekmesine geçin."
                )

            # Hedef ağırlıklar
            if hedef:
                with st.expander("🎯 Hedef Varlık Dağılımı"):
                    hedef_df = pd.DataFrame([
                        {"Varlık": k, "Hedef Ağırlık %": round(v, 1)}
                        for k, v in hedef.items()
                    ])
                    st.dataframe(hedef_df, use_container_width=True, hide_index=True)

            # Pozisyon dağılımı
            st.subheader("💼 Son Pozisyon Dağılımı")
            pozisyon_data = []
            son_fiyatlar = fiyat_df.iloc[-1]
            for v, lot in sonuc["lotlar"].items():
                if v not in son_fiyatlar.index:
                    continue
                fiyat = son_fiyatlar.get(v, 0)
                deger = lot * fiyat
                pozisyon_data.append({
                    "Varlık": v,
                    "Lot": round(lot, 2),
                    "Son Fiyat": round(fiyat, 2),
                    "Değer (TL)": round(deger, 2),
                    "Ağırlık %": 0
                })
            poz_df = pd.DataFrame(pozisyon_data)
            toplam_poz = poz_df["Değer (TL)"].sum()
            if toplam_poz > 0:
                poz_df["Ağırlık %"] = (poz_df["Değer (TL)"] / toplam_poz * 100).round(1)
            st.dataframe(poz_df, use_container_width=True, hide_index=True)

            if sonuc["rebalance_kayitlari"]:
                with st.expander(f"🔄 Rebalance İşlemleri ({len(sonuc['rebalance_kayitlari'])})"):
                    st.dataframe(
                        pd.DataFrame(sonuc["rebalance_kayitlari"]),
                        use_container_width=True, hide_index=True
                    )
        else:
            st.info("Sonuç görmek için 'Backtest Çalıştır' butonuna basın.")

# ============================================================
# TAB 2: PAPER TRADE
# ============================================================
with tab2:
    st.header("🧪 Sanal Portföy Takibi (Paper Trade)")
    paper = st.session_state.paper_portfoy

    with st.expander("➕ Yeni İşlem Ekle", expanded=True):
        with st.form("paper_trade_form", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                p_hisse = st.selectbox("Hisse", tickers if tickers else ["THYAO"])
            with c2:
                p_lot = st.number_input("Lot", min_value=1, value=10, step=1)
            with c3:
                p_fiyat = st.number_input("İşlem Fiyatı (TL)", min_value=0.01, value=100.0, step=0.01)
            with c4:
                p_tarih = st.date_input("Tarih", datetime.today())
            p_tur = st.radio("İşlem Türü", ["Alış", "Satış"], horizontal=True)

            if st.form_submit_button("✅ İşlemi Kaydet", use_container_width=True):
                yeni = pd.DataFrame([{
                    "Tarih": p_tarih, "Hisse": p_hisse, "Lot": p_lot,
                    "Fiyat": p_fiyat, "Tür": p_tur, "Tutar": p_lot * p_fiyat
                }])
                st.session_state.paper_portfoy = pd.concat([paper, yeni], ignore_index=True)
                st.rerun()

    if not paper.empty:
        st.dataframe(paper, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Toplam Alış", f"{paper[paper['Tür']=='Alış']['Tutar'].sum():,.0f} TL")
        c2.metric("Toplam Satış", f"{paper[paper['Tür']=='Satış']['Tutar'].sum():,.0f} TL")
        net = paper[paper['Tür']=='Alış']['Tutar'].sum() - paper[paper['Tür']=='Satış']['Tutar'].sum()
        c3.metric("Net", f"{net:,.0f} TL")
        if st.button("🗑️ Temizle"):
            st.session_state.paper_portfoy = pd.DataFrame(
                columns=["Tarih", "Hisse", "Lot", "Fiyat", "Tür", "Tutar"])
            st.rerun()
    else:
        st.info("Henüz paper trade işlemi yok.")

# ============================================================
# TAB 3: KARŞILAŞTIRMA (REEL GETİRİ)
# ============================================================
with tab3:
    st.header("🔍 Strateji vs Enflasyon / Altın / Döviz")

    sonuc = st.session_state.get("backtest_sonuc")
    if sonuc is None:
        st.info("Önce Backtest sekmesinden bir simülasyon çalıştırın.")
    else:
        equity = sonuc["equity"]

        with st.spinner("Karşılaştırma verileri hazırlanıyor..."):
            karsilastirma = {}

            enf = enflasyon_serisi_cek(
                baslangic.strftime("%Y-%m-%d"), bitis.strftime("%Y-%m-%d"))
            if enf is not None and not enf.empty:
                karsilastirma["Enflasyon (TÜFE)"] = enf

            altin = altin_cek(baslangic.strftime("%Y-%m-%d"), bitis.strftime("%Y-%m-%d"))
            if altin is not None and not altin.empty:
                karsilastirma["Gram Altın"] = altin

            usd = usd_cek(baslangic.strftime("%Y-%m-%d"), bitis.strftime("%Y-%m-%d"))
            if usd is not None and not usd.empty:
                karsilastirma["USD/TRY"] = usd

        if karsilastirma:
            portfoy_norm = equity["Portföy Değeri"] / equity["Portföy Değeri"].iloc[0] * 100
            portfoy_norm = tz_temizle(portfoy_norm)

            st.subheader("📊 Normalize Karşılaştırma (Başlangıç = 100)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfoy_norm.index, y=portfoy_norm,
                mode="lines", name="Strateji Portföyü (nominal)",
                line=dict(color="#00CC96", width=3)
            ))

            renkler = ["#EF553B", "#FFA15A", "#636EFA"]
            getiri_map = {}

            for i, (isim, seri) in enumerate(karsilastirma.items()):
                seri = tz_temizle(seri)
                try:
                    seri_hizali = seri.reindex(portfoy_norm.index, method="ffill").dropna()
                except Exception:
                    ortak = seri.index.intersection(portfoy_norm.index)
                    seri_hizali = seri.loc[ortak] if len(ortak) > 1 else None

                if seri_hizali is not None and len(seri_hizali) > 1:
                    seri_norm = seri_hizali / seri_hizali.iloc[0] * 100
                    fig.add_trace(go.Scatter(
                        x=seri_norm.index, y=seri_norm,
                        mode="lines", name=isim,
                        line=dict(color=renkler[i % len(renkler)], width=2, dash="dash")
                    ))
                    getiri_map[isim] = (seri_hizali.iloc[-1] / seri_hizali.iloc[0] - 1) * 100

            fig.update_layout(
                xaxis_title="Tarih", yaxis_title="Endeks (Başlangıç = 100)",
                hovermode="x unified", height=500, template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Nominal getiri tablosu
            st.subheader("📋 Dönem Sonu Nominal Getiri")
            nominal_rows = [{
                "Varlık": "Strateji Portföyü (nominal büyüme)",
                "Nominal Getiri %": round(sonuc["nominal_buyume_pct"], 1)
            }]
            for k, v in getiri_map.items():
                nominal_rows.append({"Varlık": k, "Nominal Getiri %": round(v, 1)})
            st.dataframe(pd.DataFrame(nominal_rows), use_container_width=True, hide_index=True)

            # Reel getiri tablosu
            st.subheader("💰 Reel Getiri Analizi")
            st.caption(
                "Strateji portföyünün (nominal büyüme bazlı) her bir referans varlığa "
                "göre reel getirisi. Pozitif = referansı geçtiniz, negatif = yenildiniz."
            )

            portfoy_getiri = sonuc["nominal_buyume_pct"]
            reel_rows = []
            for referans in ["Enflasyon (TÜFE)", "Gram Altın", "USD/TRY"]:
                if referans in getiri_map:
                    ref_getiri = getiri_map[referans]
                    reel = ((1 + portfoy_getiri / 100) / (1 + ref_getiri / 100) - 1) * 100
                    durum = "✅ Geçti" if reel > 0 else "❌ Yenildi"
                    reel_rows.append({
                        "Referans": referans,
                        "Referans Getirisi %": round(ref_getiri, 1),
                        "Portföy Reel Getirisi %": round(reel, 1),
                        "Durum": durum
                    })

            if reel_rows:
                reel_df = pd.DataFrame(reel_rows)

                def renklendir_durum(val):
                    if "✅" in str(val):
                        return "background-color: #d4edda"
                    elif "❌" in str(val):
                        return "background-color: #f8d7da"
                    return ""

                st.dataframe(
                    reel_df.style.map(renklendir_durum, subset=["Durum"]),
                    use_container_width=True, hide_index=True
                )

            # XIRR bazlı reel getiri (varsa)
            xirr = sonuc.get("xirr_pct")
            if xirr is not None:
                st.subheader("📐 XIRR Bazlı Reel Getiri (Yıllık)")
                st.caption(
                    "XIRR, para ağırlıklı yıllık getirinizdir. Referans varlıkların "
                    "yıllık bileşik getirileriyle kıyaslanır."
                )
                yil_sayisi = max(
                    (equity.index[-1] - equity.index[0]).days / 365.25, 0.1
                )
                ref_cagr_rows = []
                for referans, getiri in getiri_map.items():
                    cagr = ((1 + getiri / 100) ** (1 / yil_sayisi) - 1) * 100
                    reel = ((1 + xirr / 100) / (1 + cagr / 100) - 1) * 100
                    ref_cagr_rows.append({
                        "Referans": referans,
                        "Yıllık CAGR %": round(cagr, 1),
                        "XIRR %": round(xirr, 1),
                        "Reel Fark %": round(reel, 1)
                    })
                st.dataframe(
                    pd.DataFrame(ref_cagr_rows),
                    use_container_width=True, hide_index=True
                )
        else:
            st.warning("Karşılaştırma verisi alınamadı.")

# ============================================================
# TAB 4: KIRMIZI BAYRAKLAR
# ============================================================
with tab4:
    st.header("🚩 Kırmızı Bayrak Taraması")

    if not tickers:
        st.warning("Lütfen kenar çubuğundan hisse seçin.")
    elif st.button("🔍 Kırmızı Bayrakları Tara", use_container_width=True):
        with st.spinner("Hisse verileri inceleniyor..."):
            sonuclar = []
            for h in tickers:
                try:
                    info = bp.Ticker(h).info
                    bayraklar = []

                    dy = info.get("dividendYield")
                    if isinstance(dy, (int, float)) and dy < 1:
                        bayraklar.append("Düşük temettü verimi (<%1)")

                    pe = info.get("trailingPE")
                    if isinstance(pe, (int, float)):
                        if pe < 0:
                            bayraklar.append("Negatif F/K")
                        elif pe > 30:
                            bayraklar.append(f"Yüksek F/K ({pe:.1f})")

                    pb = info.get("priceToBook")
                    if isinstance(pb, (int, float)) and pb > 5:
                        bayraklar.append(f"Yüksek PD/DD ({pb:.1f})")

                    mc = info.get("marketCap")
                    if isinstance(mc, (int, float)) and mc < 1_000_000_000:
                        bayraklar.append("Küçük piyasa değeri")

                    sonuclar.append({
                        "Hisse": h,
                        "Temettü Verimi %": round(dy, 2) if isinstance(dy, (int, float)) else "—",
                        "F/K": round(pe, 2) if isinstance(pe, (int, float)) else "—",
                        "PD/DD": round(pb, 2) if isinstance(pb, (int, float)) else "—",
                        "Kırmızı Bayraklar": ", ".join(bayraklar) if bayraklar else "✅ Temiz"
                    })
                except Exception as e:
                    sonuclar.append({
                        "Hisse": h, "Temettü Verimi %": "—", "F/K": "—", "PD/DD": "—",
                        "Kırmızı Bayraklar": f"Veri hatası: {str(e)[:40]}"
                    })

            df_sonuc = pd.DataFrame(sonuclar)

            def renklendir(val):
                if "✅" in str(val):
                    return "background-color: #d4edda"
                elif "Veri hatası" in str(val):
                    return "background-color: #fff3cd"
                return "background-color: #f8d7da"

            st.dataframe(
                df_sonuc.style.map(renklendir, subset=["Kırmızı Bayraklar"]),
                use_container_width=True, hide_index=True
            )

            bayrakli = df_sonuc[~df_sonuc["Kırmızı Bayraklar"].str.contains("✅")]
            if not bayrakli.empty:
                st.warning(f"⚠️ {len(bayrakli)} hissede kırmızı bayrak tespit edildi.")
            else:
                st.success("✅ Tüm hisseler temiz.")

# ============================================================
# TAB 5: CANLI PORTFÖY
# ============================================================
with tab5:
    st.header("💼 Gerçek Portföy Takibi")
    st.warning("⚠️ Canlıya geçiş için paper trade'i en az 3-6 ay tamamlayın.")

    canli = st.session_state.canli_portfoy

    with st.expander("➕ Canlı İşlem Ekle", expanded=True):
        with st.form("canli_form", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                c_hisse = st.selectbox("Hisse", tickers if tickers else ["THYAO"], key="canli_hisse")
            with c2:
                c_lot = st.number_input("Lot", min_value=1, value=10, step=1, key="canli_lot")
            with c3:
                c_fiyat = st.number_input("Fiyat (TL)", min_value=0.01, value=100.0, step=0.01, key="canli_fiyat")
            with c4:
                c_tarih = st.date_input("Tarih", datetime.today(), key="canli_tarih")
            c_tur = st.radio("Tür", ["Alış", "Satış"], horizontal=True, key="canli_tur")

            if st.form_submit_button("✅ Kaydet", use_container_width=True):
                yeni = pd.DataFrame([{
                    "Tarih": c_tarih, "Hisse": c_hisse, "Lot": c_lot,
                    "Fiyat": c_fiyat, "Tür": c_tur, "Tutar": c_lot * c_fiyat
                }])
                st.session_state.canli_portfoy = pd.concat([canli, yeni], ignore_index=True)
                st.rerun()

    if not canli.empty:
        st.dataframe(canli, use_container_width=True, hide_index=True)

        st.subheader("🔄 Rebalance Uyarıları")
        alislar = canli[canli["Tür"] == "Alış"].groupby("Hisse").agg(
            Toplam_Lot=("Lot", "sum"), Toplam_Tutar=("Tutar", "sum")
        ).reset_index()

        if not alislar.empty:
            toplam = alislar["Toplam_Tutar"].sum()
            alislar["Ağırlık %"] = (alislar["Toplam_Tutar"] / toplam * 100).round(1)
            hedef_pct = 100 / len(alislar)

            uyarilar = []
            for _, row in alislar.iterrows():
                sapma = abs(row["Ağırlık %"] - hedef_pct)
                if sapma > rebalance_esik:
                    uyarilar.append({
                        "Hisse": row["Hisse"],
                        "Mevcut Ağırlık %": row["Ağırlık %"],
                        "Hedef Ağırlık %": round(hedef_pct, 1),
                        "Sapma %": round(sapma, 1),
                        "Öneri": "Azalt" if row["Ağırlık %"] > hedef_pct else "Artır"
                    })
            if uyarilar:
                st.dataframe(pd.DataFrame(uyarilar), use_container_width=True, hide_index=True)
            else:
                st.success("✅ Portföy ağırlıkları dengeli.")

        if st.button("🗑️ Canlı Portföyü Temizle"):
            st.session_state.canli_portfoy = pd.DataFrame(
                columns=["Tarih", "Hisse", "Lot", "Fiyat", "Tür", "Tutar"])
            st.rerun()
    else:
        st.info("Henüz canlı portföy işlemi yok.")