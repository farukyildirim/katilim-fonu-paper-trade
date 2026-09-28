# temettu_bist.py
"""
BIST Temettü Odaklı Portföy Yönetim Sistemi
- Backtest (Temettü DRIP + DCA modları)
- Otomatik Optimizasyon (tüm dönemler × tüm kombinasyonlar)
- Karşılaştırma, Paper Trade, Kırmızı Bayrak, Canlı Portföy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os
import time
import borsapy as bp

st.set_page_config(
    page_title="BIST Temettü Sistemi",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("📈 BIST Temettü Odaklı Portföy Yönetim Paneli")
st.caption("Backtest → Otomatik Optimizasyon → Paper Trade → Canlı Portföy")

# ============================================================
# EVDS
# ============================================================
EVDS_KEY = os.environ.get("EVDS_API_KEY", "4JcbosMYYp")
if EVDS_KEY:
    try:
        bp.set_evds_key(EVDS_KEY)
    except Exception:
        pass
else:
    st.sidebar.warning("⚠️ EVDS_API_KEY ayarlı değil. Enflasyon verisi çekilemeyecek.")

# ============================================================
# SABITLER
# ============================================================
# Test edilecek dönemler
DONEMLER = {
    "2010-2014 (Boğa, stabil TL)": (datetime(2010, 1, 1), datetime(2014, 12, 31)),
    "2015-2019 (Kur stresi)": (datetime(2015, 1, 1), datetime(2019, 12, 31)),
    "2018-2022 (Enflasyon patlaması)": (datetime(2018, 1, 1), datetime(2022, 12, 31)),
    "2022-2026 (Güncel)": (datetime(2022, 1, 1), datetime.today()),
}

# Test edilecek ağırlık kombinasyonları (hisse/altın/dolar)
KOMBINASYONLAR = [
    ("100/0/0", 100, 0, 0),
    ("80/15/5", 80, 15, 5),
    ("70/20/10", 70, 20, 10),
    ("60/25/15", 60, 25, 15),
    ("55/30/15", 55, 30, 15),
    ("50/35/15", 50, 35, 15),
    ("40/40/20", 40, 40, 20),
]

# ============================================================
# OTURUM DURUMU
# ============================================================
defaults = {
    "paper_portfoy": pd.DataFrame(columns=["Tarih", "Hisse", "Lot", "Fiyat", "Tür", "Tutar"]),
    "canli_portfoy": pd.DataFrame(columns=["Tarih", "Hisse", "Lot", "Fiyat", "Tür", "Tutar"]),
    "backtest_sonuc": None,
    "backtest_fiyat_df": None,
    "backtest_hedef": None,
    "optimizasyon_sonuc": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# YARDIMCILAR
# ============================================================
def tz_temizle(obj):
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
        return obj.sort_index()
    return obj


@st.cache_data(ttl=3600, show_spinner=False)
def veri_cek(hisse_listesi, baslangic, bitis):
    fiyat_dict = {}
    hatalar = []
    for h in hisse_listesi:
        try:
            ticker = bp.Ticker(h)
            df = ticker.history(start=baslangic, end=bitis)
            if df is not None and not df.empty:
                fiyat_dict[h] = tz_temizle(df["Close"])
        except Exception as e:
            hatalar.append(f"{h}: {str(e)[:50]}")
    if hatalar:
        st.warning("Veri alınamadı:\n" + "\n".join(hatalar))
    if not fiyat_dict:
        return None
    return tz_temizle(pd.DataFrame(fiyat_dict)).ffill()


@st.cache_data(ttl=3600, show_spinner=False)
def temettu_verisi_cek(hisse_listesi, baslangic, bitis):
    """
    borsapy.Ticker.dividends DataFrame döndürür.
    Index: Date
    Sütunlar: Amount (hisse başı TL), GrossRate, NetRate, TotalDividend

    Biz 'Amount' sütununu kullanırız (hisse başına ödenen TL).
    """
    temettu_dict = {}
    hata_listesi = []

    for h in hisse_listesi:
        try:
            ticker = bp.Ticker(h)
            df = ticker.dividends

            if df is None or df.empty:
                hata_listesi.append(f"{h}: boş")
                continue

            # Amount sütununu al
            if "Amount" in df.columns:
                seri = df["Amount"].copy()
            else:
                # Bilinmeyen format: ilk sayısal sütunu al
                sayisal = df.select_dtypes(include="number")
                if sayisal.empty:
                    hata_listesi.append(f"{h}: sayısal sütun yok")
                    continue
                seri = sayisal.iloc[:, 0].copy()

            # Index'i datetime yap
            seri.index = pd.to_datetime(seri.index)
            seri = tz_temizle(seri)

            # Tarih aralığına filtrele
            seri = seri[
                (seri.index >= pd.Timestamp(baslangic)) &
                (seri.index <= pd.Timestamp(bitis))
                ]

            # Sıfır/negatif değerleri at
            seri = seri[seri > 0]

            if len(seri) > 0:
                temettu_dict[h] = seri
            else:
                hata_listesi.append(f"{h}: dönemde ödeme yok")

        except Exception as e:
            hata_listesi.append(f"{h}: {str(e)[:60]}")

    if hata_listesi:
        st.info("Temettü verisi durumu:\n" + "\n".join(hata_listesi))

    if not temettu_dict:
        return None

    return tz_temizle(pd.DataFrame(temettu_dict))


@st.cache_data(ttl=3600, show_spinner=False)
def altin_cek(baslangic, bitis):
    try:
        df = bp.FX("gram-altin").history(start=baslangic, end=bitis)
        if df is not None and not df.empty:
            return tz_temizle(df["Close"])
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def usd_cek(baslangic, bitis):
    try:
        df = bp.FX("USD").history(start=baslangic, end=bitis)
        if df is not None and not df.empty:
            return tz_temizle(df["Close"])
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def enflasyon_serisi_cek(baslangic, bitis):
    if not EVDS_KEY:
        return None
    try:
        seri = bp.evds_series("TP.FG.J0", start=baslangic, end=bitis, frequency="monthly")
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
    except Exception:
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
        for _ in range(200):
            mid = (low + high) / 2
            f_mid = npv(mid)
            if abs(f_mid) < 1e-6:
                return mid * 100
            if f_low * f_mid < 0:
                high = mid
            else:
                low = mid
                f_low = f_mid
        return ((low + high) / 2) * 100
    except Exception:
        return None


# ============================================================
# BACKTEST: DCA MODU
# ============================================================
def backtest_dca(fiyat_df, baslangic_sermaye, aylik_alis, rebalance_esik,
                 hedef_agirliklar=None):
    varliklar = fiyat_df.columns.tolist()
    n = len(varliklar)
    if n == 0:
        return None

    if hedef_agirliklar is None:
        hedef = {v: 100.0 / n for v in varliklar}
    else:
        toplam = sum(hedef_agirliklar.values())
        hedef = {k: v / toplam * 100 for k, v in hedef_agirliklar.items()} if toplam > 0 \
                else {v: 100.0 / n for v in varliklar}
        for v in varliklar:
            hedef.setdefault(v, 0.0)

    lotlar = {v: 0.0 for v in varliklar}
    nakit = 0.0

    tmp = pd.DataFrame(index=fiyat_df.index)
    tmp["_ym"] = tmp.index.to_period("M")
    aylik_ilk = set(tmp.groupby("_ym").head(1).index)

    equity_curve = []
    nakit_akislari = []
    tarih_akislari = []
    toplam_yatirim = 0.0

    for i, tarih in enumerate(fiyat_df.index):
        if tarih in aylik_ilk or i == 0:
            yatirilan = aylik_alis if i > 0 else baslangic_sermaye
            nakit += yatirilan
            toplam_yatirim += yatirilan
            nakit_akislari.append(-yatirilan)
            tarih_akislari.append(tarih)

            fiyatlar = fiyat_df.loc[tarih]
            for v in varliklar:
                f = fiyatlar.get(v, np.nan)
                if pd.notna(f) and f > 0 and hedef[v] > 0:
                    tutar = yatirilan * (hedef[v] / 100)
                    if tutar > 0 and nakit >= tutar - 1e-6:
                        lotlar[v] += tutar / f
                        nakit -= tutar

        if i > 0 and tarih in aylik_ilk:
            fiyatlar = fiyat_df.loc[tarih]
            toplam = nakit + sum(lotlar[v] * fiyatlar.get(v, 0) for v in varliklar)
            if toplam > 0:
                for v in varliklar:
                    f = fiyatlar.get(v, np.nan)
                    if pd.notna(f) and f > 0:
                        mevcut = (lotlar[v] * f) / toplam * 100
                        sapma = mevcut - hedef[v]
                        if sapma > rebalance_esik:
                            hedef_t = toplam * (hedef[v] / 100)
                            hedef_l = hedef_t / f
                            satilan = lotlar[v] - hedef_l
                            lotlar[v] = hedef_l
                            nakit += satilan * f

        fiyatlar = fiyat_df.loc[tarih]
        poz = sum(lotlar[v] * fiyatlar.get(v, 0) for v in varliklar)
        equity_curve.append({"Tarih": tarih, "Portföy Değeri": nakit + poz})

    equity_df = tz_temizle(pd.DataFrame(equity_curve).set_index("Tarih"))
    bitis_deger = equity_df["Portföy Değeri"].iloc[-1]

    nakit_akislari.append(bitis_deger)
    tarih_akislari.append(equity_df.index[-1])
    xirr = hesapla_xirr(nakit_akislari, tarih_akislari)

    daily = equity_df["Portföy Değeri"].pct_change().dropna()
    daily = daily[daily.abs() < 0.5]
    sharpe = hesapla_sharpe(daily)
    sortino = hesapla_sortino(daily)

    kum_max = equity_df["Portföy Değeri"].cummax()
    dd = (equity_df["Portföy Değeri"] - kum_max) / kum_max * 100
    max_dd = float(dd.min())

    return {
        "equity": equity_df,
        "xirr_pct": xirr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_dd,
        "bitis_deger": bitis_deger,
        "toplam_yatirim": toplam_yatirim,
        "net_kar": bitis_deger - toplam_yatirim,
        "toplam_getiri_pct": (bitis_deger / baslangic_sermaye - 1) * 100,
        "lotlar": lotlar,
        "nakit": nakit,
        "hedef_agirliklar": hedef,
    }


# ============================================================
# BACKTEST: TEMETTÜ (DRIP) MODU
# ============================================================
def backtest_temettu(fiyat_df, temettu_df, baslangic_sermaye,
                     rebalance_esik=10, hedef_agirliklar=None):
    varliklar = fiyat_df.columns.tolist()
    n = len(varliklar)
    if n == 0:
        return None

    if hedef_agirliklar is None:
        hedef = {v: 100.0 / n for v in varliklar}
    else:
        toplam = sum(hedef_agirliklar.values())
        hedef = {k: v / toplam * 100 for k, v in hedef_agirliklar.items()} if toplam > 0 \
                else {v: 100.0 / n for v in varliklar}
        for v in varliklar:
            hedef.setdefault(v, 0.0)

    lotlar = {v: 0.0 for v in varliklar}
    nakit = 0.0
    toplam_temettu = 0.0

    # Başlangıç alımı
    ilk_tarih = fiyat_df.index[0]
    ilk_fiyatlar = fiyat_df.loc[ilk_tarih]
    for v in varliklar:
        hedef_t = baslangic_sermaye * (hedef[v] / 100)
        f = ilk_fiyatlar.get(v, np.nan)
        if pd.notna(f) and f > 0 and hedef_t > 0:
            lotlar[v] = hedef_t / f

    tmp = pd.DataFrame(index=fiyat_df.index)
    tmp["_ym"] = tmp.index.to_period("M")
    aylik_ilk = set(tmp.groupby("_ym").head(1).index)

    equity_curve = []
    temettu_kayitlari = []
    rebalance_kayitlari = []

    for i, tarih in enumerate(fiyat_df.index):
        fiyatlar = fiyat_df.loc[tarih]

        if i == 0:
            poz = sum(lotlar[v] * fiyatlar.get(v, 0) for v in varliklar)
            equity_curve.append({"Tarih": tarih, "Portföy Değeri": nakit + poz})
            continue

        # Temettü ödemeleri
        if temettu_df is not None and tarih in temettu_df.index:
            gunluk = 0.0
            for v in varliklar:
                if v in temettu_df.columns:
                    hb = temettu_df.loc[tarih, v]
                    if pd.notna(hb) and hb > 0 and lotlar[v] > 0:
                        tutar = lotlar[v] * hb
                        gunluk += tutar
                        toplam_temettu += tutar
                        temettu_kayitlari.append({
                            "Tarih": tarih, "Hisse": v,
                            "Hisse Başı": round(hb, 4),
                            "Temettü (TL)": round(tutar, 2)
                        })
            nakit += gunluk
            # Hemen yeniden yatır
            if gunluk > 0 and nakit > 0:
                for v in varliklar:
                    f = fiyatlar.get(v, np.nan)
                    if pd.notna(f) and f > 0 and hedef[v] > 0:
                        tutar = nakit * (hedef[v] / 100)
                        if tutar > 0:
                            lotlar[v] += tutar / f
                nakit = 0.0

        # Rebalance
        if tarih in aylik_ilk:
            toplam = nakit + sum(lotlar[v] * fiyatlar.get(v, 0) for v in varliklar)
            if toplam > 0:
                for v in varliklar:
                    f = fiyatlar.get(v, np.nan)
                    if pd.notna(f) and f > 0:
                        mevcut = (lotlar[v] * f) / toplam * 100
                        sapma = mevcut - hedef[v]
                        if sapma > rebalance_esik:
                            hedef_t = toplam * (hedef[v] / 100)
                            hedef_l = hedef_t / f
                            satilan = lotlar[v] - hedef_l
                            lotlar[v] = hedef_l
                            nakit += satilan * f
                            rebalance_kayitlari.append({
                                "Tarih": tarih, "Varlık": v, "İşlem": "Satış",
                                "Sapma %": round(sapma, 2)
                            })
                for v in varliklar:
                    f = fiyatlar.get(v, np.nan)
                    if pd.notna(f) and f > 0 and nakit > 0:
                        mevcut = (lotlar[v] * f) / toplam * 100
                        sapma = hedef[v] - mevcut
                        if sapma > rebalance_esik:
                            hedef_t = toplam * (hedef[v] / 100)
                            alinacak = (hedef_t - lotlar[v] * f) / f
                            maliyet = alinacak * f
                            if 0 < maliyet <= nakit + 1e-6:
                                lotlar[v] += alinacak
                                nakit -= maliyet

        poz = sum(lotlar[v] * fiyatlar.get(v, 0) for v in varliklar)
        equity_curve.append({"Tarih": tarih, "Portföy Değeri": nakit + poz})

    equity_df = tz_temizle(pd.DataFrame(equity_curve).set_index("Tarih"))
    bitis_deger = equity_df["Portföy Değeri"].iloc[-1]

    gun_sayisi = (equity_df.index[-1] - equity_df.index[0]).days
    yil = max(gun_sayisi / 365.25, 0.1)
    cagr = ((bitis_deger / baslangic_sermaye) ** (1 / yil) - 1) * 100
    toplam_getiri = (bitis_deger / baslangic_sermaye - 1) * 100

    daily = equity_df["Portföy Değeri"].pct_change().dropna()
    daily = daily[daily.abs() < 0.5]
    sharpe = hesapla_sharpe(daily)
    sortino = hesapla_sortino(daily)

    kum_max = equity_df["Portföy Değeri"].cummax()
    dd = (equity_df["Portföy Değeri"] - kum_max) / kum_max * 100
    max_dd = float(dd.min())

    return {
        "equity": equity_df,
        "cagr_pct": cagr,
        "xirr_pct": cagr,  # DRIP'te tek nakit girişi, CAGR = XIRR
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_dd,
        "baslangic_deger": baslangic_sermaye,
        "bitis_deger": bitis_deger,
        "net_kar": bitis_deger - baslangic_sermaye,
        "toplam_getiri_pct": toplam_getiri,
        "toplam_temettu": toplam_temettu,
        "lotlar": lotlar,
        "nakit": nakit,
        "temettu_kayitlari": temettu_kayitlari,
        "rebalance_kayitlari": rebalance_kayitlari,
        "hedef_agirliklar": hedef,
    }


# ============================================================
# OTOMATİK OPTİMİZASYON MOTORU
# ============================================================
def otomatik_optimizasyon(donemler, kombinasyonlar, hisseler,
                          mod="temettu", baslangic_sermaye=100_000,
                          aylik_alis=5_000, rebalance_esik=10,
                          progress_cb=None):
    """
    Tüm dönemleri × tüm kombinasyonları test eder.
    """
    # Tarih normalizasyonu
    def _dt(x):
        if isinstance(x, str):
            return pd.to_datetime(x).to_pydatetime()
        if isinstance(x, pd.Timestamp):
            return x.to_pydatetime()
        if isinstance(x, datetime):
            return x
        return pd.to_datetime(x).to_pydatetime()

    donemler = {k: (_dt(v[0]), _dt(v[1])) for k, v in donemler.items()}

    # ✅ En geniş tarih aralığını bul — .values() ile
    min_bas = min(v[0] for v in donemler.values())
    max_bit = max(v[1] for v in donemler.values())

    # ... geri kalanı aynı ...

    # Tüm verileri bir kere çek
    fiyat_full = veri_cek(
        hisseler,
        min_bas.strftime("%Y-%m-%d"),
        max_bit.strftime("%Y-%m-%d")
    )
    if fiyat_full is None or fiyat_full.empty:
        return None, "Hisse verisi alınamadı."

    temettu_full = None
    if mod == "temettu":
        temettu_full = temettu_verisi_cek(
            hisseler,
            min_bas.strftime("%Y-%m-%d"),
            max_bit.strftime("%Y-%m-%d")
        )
        if temettu_full is None:
            return None, "Temettü verisi alınamadı. 'DCA' moduna geçin veya API'yi kontrol edin."

    altin_full = altin_cek(min_bas.strftime("%Y-%m-%d"), max_bit.strftime("%Y-%m-%d"))
    usd_full = usd_cek(min_bas.strftime("%Y-%m-%d"), max_bit.strftime("%Y-%m-%d"))

    # Sonuçları topla
    sonuclar = []
    toplam_is = len(donemler) * len(kombinasyonlar)
    is_sayaci = 0

    for d_adi, (d_bas, d_bit) in donemler.items():
        # Bu döneme ait verileri kes
        try:
            fiyat_kes = fiyat_full.loc[d_bas:d_bit].copy()
        except Exception:
            continue
        if fiyat_kes.empty or len(fiyat_kes) < 30:
            continue

        temettu_kes = None
        if temettu_full is not None:
            try:
                temettu_kes = temettu_full.loc[d_bas:d_bit]
            except Exception:
                temettu_kes = None

        altin_kes = None
        if altin_full is not None:
            try:
                altin_kes = altin_full.loc[d_bas:d_bit]
            except Exception:
                altin_kes = None

        usd_kes = None
        if usd_full is not None:
            try:
                usd_kes = usd_full.loc[d_bas:d_bit]
            except Exception:
                usd_kes = None

        for k_adi, h_ag, a_ag, d_ag in kombinasyonlar:
            is_sayaci += 1
            if progress_cb:
                progress_cb(is_sayaci, toplam_is, f"{d_adi} × {k_adi}")

            # Fiyat verisini oluştur
            if a_ag == 0 and d_ag == 0:
                # Saf hisse
                kullan_fiyat = fiyat_kes.copy()
                hedef = {h: 100.0 / len(hisseler) for h in hisseler}
            else:
                kullan_fiyat = fiyat_kes.copy()
                if altin_kes is not None:
                    kullan_fiyat["Gram Altın"] = altin_kes.reindex(
                        kullan_fiyat.index, method="ffill")
                if usd_kes is not None:
                    kullan_fiyat["USD/TRY"] = usd_kes.reindex(
                        kullan_fiyat.index, method="ffill")
                kullan_fiyat = kullan_fiyat.ffill().dropna(how="all")

                hisse_per = h_ag / len(hisseler)
                hedef = {h: hisse_per for h in hisseler}
                if altin_kes is not None:
                    hedef["Gram Altın"] = a_ag
                if usd_kes is not None:
                    hedef["USD/TRY"] = d_ag

            # Backtest
            try:
                if mod == "temettu" and temettu_kes is not None:
                    sonuc = backtest_temettu(
                        kullan_fiyat, temettu_kes, baslangic_sermaye,
                        rebalance_esik, hedef_agirliklar=hedef
                    )
                else:
                    sonuc = backtest_dca(
                        kullan_fiyat, baslangic_sermaye, aylik_alis,
                        rebalance_esik, hedef_agirliklar=hedef
                    )
            except Exception as e:
                sonuc = None

            if sonuc is None:
                continue

            # Reel getiri hesapla (enflasyona karşı)
            enf_seri = enflasyon_serisi_cek(
                d_bas.strftime("%Y-%m-%d"), d_bit.strftime("%Y-%m-%d")
            )
            enf_cagr = None
            reel_fark = None
            if enf_seri is not None and len(enf_seri) > 1:
                try:
                    yil_sayisi = (d_bit - d_bas).days / 365.25
                    enf_getiri = (enf_seri.iloc[-1] / enf_seri.iloc[0] - 1) * 100
                    enf_cagr = ((1 + enf_getiri / 100) ** (1 / yil_sayisi) - 1) * 100
                    port_cagr = sonuc.get("xirr_pct") or sonuc.get("cagr_pct")
                    if port_cagr is not None:
                        reel_fark = ((1 + port_cagr / 100) / (1 + enf_cagr / 100) - 1) * 100
                except Exception:
                    pass

            sonuclar.append({
                "Dönem": d_adi,
                "Kombinasyon": k_adi,
                "Hisse %": h_ag,
                "Altın %": a_ag,
                "Dolar %": d_ag,
                "XIRR/CAGR %": round(sonuc.get("xirr_pct") or sonuc.get("cagr_pct") or 0, 1),
                "Sharpe": round(sonuc.get("sharpe", 0), 2),
                "Sortino": round(sonuc.get("sortino", 0), 2),
                "Maks DD %": round(sonuc.get("max_drawdown_pct", 0), 1),
                "Net Kâr": round(sonuc.get("net_kar", 0), 0),
                "Enf CAGR %": round(enf_cagr, 1) if enf_cagr is not None else None,
                "Reel Fark": round(reel_fark, 1) if reel_fark is not None else None,
            })

    return pd.DataFrame(sonuclar), None


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
    donem_secim = st.selectbox(
        "📅 Backtest Dönemi",
        ["Özel", "2010-2014", "2015-2019", "2018-2022", "2022-2026", "Son 3 Yıl"]
    )
    bugun = datetime.today()
    if donem_secim == "2010-2014":
        vb, vbit = datetime(2010, 1, 1), datetime(2014, 12, 31)
    elif donem_secim == "2015-2019":
        vb, vbit = datetime(2015, 1, 1), datetime(2019, 12, 31)
    elif donem_secim == "2018-2022":
        vb, vbit = datetime(2018, 1, 1), datetime(2022, 12, 31)
    elif donem_secim == "2022-2026":
        vb, vbit = datetime(2022, 1, 1), bugun
    elif donem_secim == "Son 3 Yıl":
        vb, vbit = bugun.replace(year=bugun.year - 3), bugun
    else:
        vb, vbit = datetime(2018, 1, 1), bugun

    baslangic = st.date_input("Başlangıç", vb)
    bitis = st.date_input("Bitiş", vbit)

    baslangic_sermaye = st.number_input("Başlangıç Sermayesi (TL)",
                                         value=100_000, step=10_000, min_value=1000)
    aylik_alis = st.number_input("Aylık Ek Alım (TL) — sadece DCA",
                                  value=5_000, step=1_000, min_value=0)
    rebalance_esik = st.slider("Rebalance Eşiği (%)", 0, 30, 10)
    st.caption("⚠️ Eğitim amaçlıdır. Yatırım tavsiyesi değildir.")

# ============================================================
# ANA SEKMELER
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Backtest",
    "🎯 Otomatik Optimizasyon",
    "🧪 Paper Trade",
    "🔍 Karşılaştırma",
    "🚩 Kırmızı Bayraklar",
    "💼 Canlı Portföy",
])

# ============================================================
# TAB 1: BACKTEST (tek konfigürasyon)
# ============================================================
with tab1:
    st.header("Tek Konfigürasyon Backtest'i")

    mod_sec = st.radio(
        "Model",
        ["Temettü Yeniden Yatırım (DRIP)", "Düzenli Ek Alım (DCA)"],
        horizontal=True,
        help="DRIP: Sadece başlangıç sermayesi + temettüler. DCA: Her ay ek alım."
    )

    # Ağırlık seçimi
    c1, c2, c3 = st.columns(3)
    with c1:
        h_ag = st.slider("Hisse Ağırlık %", 0, 100, 60, 5)
    with c2:
        a_ag = st.slider("Altın Ağırlık %", 0, 100, 25, 5)
    with c3:
        d_ag = st.slider("Dolar Ağırlık %", 0, 100, 15, 5)

    cok_varlikli = (a_ag + d_ag) > 0

    if st.button("🚀 Backtest Çalıştır", use_container_width=True):
        with st.spinner("Çalışıyor..."):
            fiyat_df = veri_cek(tickers,
                                baslangic.strftime("%Y-%m-%d"),
                                bitis.strftime("%Y-%m-%d"))
            if fiyat_df is None or fiyat_df.empty:
                st.error("Hisse verisi alınamadı.")
            else:
                temettu_df = None
                if "DRIP" in mod_sec:
                    temettu_df = temettu_verisi_cek(
                        tickers,
                        baslangic.strftime("%Y-%m-%d"),
                        bitis.strftime("%Y-%m-%d")
                    )
                    if temettu_df is None:
                        st.warning("⚠️ Temettü verisi alınamadı. DCA modu kullanılıyor.")
                        mod_sec = "Düzenli Ek Alım (DCA)"

                hedef = None
                if cok_varlikli:
                    altin = altin_cek(baslangic.strftime("%Y-%m-%d"),
                                      bitis.strftime("%Y-%m-%d"))
                    usd = usd_cek(baslangic.strftime("%Y-%m-%d"),
                                  bitis.strftime("%Y-%m-%d"))
                    if altin is not None:
                        fiyat_df["Gram Altın"] = altin.reindex(fiyat_df.index, method="ffill")
                    if usd is not None:
                        fiyat_df["USD/TRY"] = usd.reindex(fiyat_df.index, method="ffill")
                    fiyat_df = fiyat_df.ffill().dropna(how="all")
                    hisse_per = h_ag / len(tickers) if tickers else 0
                    hedef = {h: hisse_per for h in tickers}
                    if altin is not None:
                        hedef["Gram Altın"] = a_ag
                    if usd is not None:
                        hedef["USD/TRY"] = d_ag

                if "DRIP" in mod_sec and temettu_df is not None:
                    sonuc = backtest_temettu(
                        fiyat_df, temettu_df, baslangic_sermaye,
                        rebalance_esik, hedef_agirliklar=hedef
                    )
                else:
                    sonuc = backtest_dca(
                        fiyat_df, baslangic_sermaye, aylik_alis,
                        rebalance_esik, hedef_agirliklar=hedef
                    )
                st.session_state.backtest_sonuc = sonuc
                st.session_state.backtest_fiyat_df = fiyat_df
                st.session_state.backtest_hedef = hedef

    sonuc = st.session_state.get("backtest_sonuc")
    fiyat_df = st.session_state.get("backtest_fiyat_df")

    if sonuc is not None and fiyat_df is not None:
        equity = sonuc["equity"]

        st.subheader("📈 Portföy Değer Grafiği")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity["Portföy Değeri"],
            mode="lines", name="Portföy",
            line=dict(color="#00CC96", width=2),
            fill="tozeroy", fillcolor="rgba(0,204,150,0.1)"
        ))
        fig.update_layout(height=400, template="plotly_white",
                          xaxis_title="Tarih", yaxis_title="Değer (TL)")
        st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("XIRR / CAGR", f"{(sonuc.get('xirr_pct') or sonuc.get('cagr_pct') or 0):.1f}%")
        m2.metric("Sharpe", f"{sonuc['sharpe']:.2f}")
        m3.metric("Maks DD", f"{sonuc['max_drawdown_pct']:.1f}%")
        m4.metric("Net Kâr", f"{sonuc['net_kar']:,.0f} TL")

        if "toplam_temettu" in sonuc and sonuc["toplam_temettu"] > 0:
            st.info(f"💰 Toplam temettü geliri: **{sonuc['toplam_temettu']:,.0f} TL**")
    else:
        st.info("Backtest çalıştırın.")

# ============================================================
# TAB 2: OTOMATİK OPTİMİZASYON
# ============================================================
with tab2:
    st.header("🎯 Otomatik Dönem × Kombinasyon Optimizasyonu")
    st.caption(
        "Tüm seçili dönemleri × tüm kombinasyonları otomatik test eder ve "
        "hangi dönemde hangi ağırlığın en iyi olduğunu özetler."
    )

    col1, col2 = st.columns(2)
    with col1:
        sec_donemler = st.multiselect(
            "Test Edilecek Dönemler",
            options=list(DONEMLER.keys()),
            default=list(DONEMLER.keys())[:3]
        )
    with col2:
        sec_kombinasyonlar = st.multiselect(
            "Test Edilecek Kombinasyonlar",
            options=[k[0] for k in KOMBINASYONLAR],
            default=["100/0/0", "70/20/10", "60/25/15", "50/35/15"]
        )

    mod_opt = st.radio(
        "Model",
        ["Temettü Yeniden Yatırım (DRIP)", "Düzenli Ek Alım (DCA)"],
        horizontal=True,
        key="opt_mod"
    )

    if st.button("🧪 Tüm Testleri Çalıştır", use_container_width=True, type="primary"):
        if not sec_donemler or not sec_kombinasyonlar:
            st.warning("En az bir dönem ve bir kombinasyon seçin.")
        elif not tickers:
            st.warning("Kenar çubuğundan en az bir hisse seçin.")
        else:
            donemler_sec = {d: DONEMLER[d] for d in sec_donemler}
            kombs_sec = [k for k in KOMBINASYONLAR if k[0] in sec_kombinasyonlar]

            prog = st.progress(0)
            durum = st.empty()

            def cb(i, toplam, mesaj):
                prog.progress(min(i / toplam, 1.0))
                durum.text(f"⏳ {i}/{toplam} — {mesaj}")

            with st.spinner("Optimizasyon çalışıyor... (Bu işlem 1-2 dakika sürebilir)"):
                df_sonuc, hata = otomatik_optimizasyon(
                    donemler=donemler_sec,
                    kombinasyonlar=kombs_sec,
                    hisseler=tickers,
                    mod="temettu" if "DRIP" in mod_opt else "dca",
                    baslangic_sermaye=baslangic_sermaye,
                    aylik_alis=aylik_alis,
                    rebalance_esik=rebalance_esik,
                    progress_cb=cb
                )

            prog.empty()
            durum.empty()

            if hata:
                st.error(hata)
            elif df_sonuc is None or df_sonuc.empty:
                st.error("Hiçbir test sonuç üretemedi.")
            else:
                st.session_state.optimizasyon_sonuc = df_sonuc
                st.success(f"✅ {len(df_sonuc)} test tamamlandı.")

    # Sonuçları göster
    df_sonuc = st.session_state.get("optimizasyon_sonuc")

    if df_sonuc is not None and not df_sonuc.empty:
        st.subheader("📋 Tüm Sonuçlar")
        st.dataframe(df_sonuc, use_container_width=True, hide_index=True)

        # --- DÖNEM BAZLI EN İYİ KOMBİNASYON ---
        st.subheader("🏆 Her Dönem İçin En İyi Kombinasyon")
        st.caption("Kriter: En yüksek Sharpe. Beraberlikte en düşük drawdown.")

        en_iyi_rows = []
        for d in df_sonuc["Dönem"].unique():
            alt = df_sonuc[df_sonuc["Dönem"] == d].copy()
            alt = alt.sort_values(
                by=["Sharpe", "Maks DD %"],
                ascending=[False, False]
            )
            en_iyi = alt.iloc[0]
            en_iyi_rows.append({
                "Dönem": d,
                "En İyi Kombinasyon": en_iyi["Kombinasyon"],
                "XIRR/CAGR %": en_iyi["XIRR/CAGR %"],
                "Sharpe": en_iyi["Sharpe"],
                "Maks DD %": en_iyi["Maks DD %"],
                "Reel Fark": en_iyi.get("Reel Fark", None),
            })
        df_en_iyi = pd.DataFrame(en_iyi_rows)
        st.dataframe(df_en_iyi, use_container_width=True, hide_index=True)

        # --- GENEL ORTALAMA ---
        st.subheader("📊 Kombinasyonların Ortalama Performansı")
        ozet = df_sonuc.groupby("Kombinasyon").agg(
            **{
                "Ort. Sharpe": ("Sharpe", "mean"),
                "Ort. XIRR/CAGR %": ("XIRR/CAGR %", "mean"),
                "Ort. Maks DD %": ("Maks DD %", "mean"),
                "Ort. Reel Fark": ("Reel Fark", "mean"),
                "Test Sayısı": ("Sharpe", "count"),
            }
        ).reset_index().sort_values("Ort. Sharpe", ascending=False)

        def renkli_satir(row):
            if row.name == 0:
                return ["background-color: #d4edda"] * len(row)
            return [""] * len(row)

        st.dataframe(
            ozet.style.apply(renkli_satir, axis=1),
            use_container_width=True, hide_index=True
        )

        # --- KARAR ÖNERİSİ ---
        st.subheader("🎯 Sistemin Karar Önerisi")

        en_iyi_genel = ozet.iloc[0]
        kazanan = en_iyi_genel["Kombinasyon"]
        ort_sharpe = en_iyi_genel["Ort. Sharpe"]
        ort_dd = en_iyi_genel["Ort. Maks DD %"]

        st.success(
            f"**Önerilen Kombinasyon: {kazanan}**\n\n"
            f"- Ortalama Sharpe: **{ort_sharpe:.2f}**\n"
            f"- Ortalama Maks Drawdown: **{ort_dd:.1f}%**\n"
            f"- Tüm dönemlerde en yüksek ortalama Sharpe'ı verdi."
        )

        # Dönem bazlı notlar
        st.markdown("### 📝 Dönem Bazlı Notlar")
        for _, row in df_en_iyi.iterrows():
            reelfark = row.get("Reel Fark")
            reelfark_str = f", reel fark: {reelfark:+.1f} puan" if pd.notna(reelfark) else ""
            st.markdown(
                f"- **{row['Dönem']}** → `{row['En İyi Kombinasyon']}` "
                f"(Sharpe {row['Sharpe']:.2f}, DD {row['Maks DD %']:.1f}%{reelfark_str})"
            )

        # Uyarı
        st.warning(
            "⚠️ **Uyarı:** Bu sonuçlar geçmiş performansa dayanır. "
            "En iyi kombinasyon gelecekte de en iyi olmayabilir. "
            "Farklı rejimlerde **tutarlı Sharpe > 1.5** veren kombinasyon tercih edilmelidir."
        )

# ============================================================
# TAB 3: PAPER TRADE
# ============================================================
with tab3:
    st.header("🧪 Paper Trade")
    paper = st.session_state.paper_portfoy

    with st.expander("➕ İşlem Ekle", expanded=True):
        with st.form("paper_form", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                p_hisse = st.selectbox("Hisse", tickers if tickers else ["THYAO"])
            with c2:
                p_lot = st.number_input("Lot", min_value=1, value=10)
            with c3:
                p_fiyat = st.number_input("Fiyat", min_value=0.01, value=100.0)
            with c4:
                p_tarih = st.date_input("Tarih", datetime.today())
            p_tur = st.radio("Tür", ["Alış", "Satış", "Temettü"], horizontal=True)
            if st.form_submit_button("✅ Kaydet"):
                yeni = pd.DataFrame([{
                    "Tarih": p_tarih, "Hisse": p_hisse, "Lot": p_lot,
                    "Fiyat": p_fiyat, "Tür": p_tur, "Tutar": p_lot * p_fiyat
                }])
                st.session_state.paper_portfoy = pd.concat([paper, yeni], ignore_index=True)
                st.rerun()

    if not paper.empty:
        st.dataframe(paper, use_container_width=True, hide_index=True)
    else:
        st.info("İşlem yok.")

# ============================================================
# TAB 4: KARŞILAŞTIRMA
# ============================================================
with tab4:
    st.header("🔍 Karşılaştırma")
    sonuc = st.session_state.get("backtest_sonuc")
    if sonuc is None:
        st.info("Önce Backtest çalıştırın.")
    else:
        equity = sonuc["equity"]
        karsilastirma = {}
        enf = enflasyon_serisi_cek(baslangic.strftime("%Y-%m-%d"), bitis.strftime("%Y-%m-%d"))
        if enf is not None and not enf.empty:
            karsilastirma["Enflasyon"] = enf
        altin = altin_cek(baslangic.strftime("%Y-%m-%d"), bitis.strftime("%Y-%m-%d"))
        if altin is not None and not altin.empty:
            karsilastirma["Gram Altın"] = altin
        usd = usd_cek(baslangic.strftime("%Y-%m-%d"), bitis.strftime("%Y-%m-%d"))
        if usd is not None and not usd.empty:
            karsilastirma["USD/TRY"] = usd

        if karsilastirma:
            portfoy_norm = equity["Portföy Değeri"] / equity["Portföy Değeri"].iloc[0] * 100
            portfoy_norm = tz_temizle(portfoy_norm)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfoy_norm.index, y=portfoy_norm,
                mode="lines", name="Portföy",
                line=dict(color="#00CC96", width=3)
            ))
            renkler = ["#EF553B", "#FFA15A", "#636EFA"]
            getiri_map = {}
            for i, (isim, seri) in enumerate(karsilastirma.items()):
                seri = tz_temizle(seri)
                try:
                    s = seri.reindex(portfoy_norm.index, method="ffill").dropna()
                except Exception:
                    continue
                if len(s) > 1:
                    sn = s / s.iloc[0] * 100
                    fig.add_trace(go.Scatter(
                        x=sn.index, y=sn, mode="lines", name=isim,
                        line=dict(color=renkler[i % 3], width=2, dash="dash")
                    ))
                    getiri_map[isim] = (s.iloc[-1] / s.iloc[0] - 1) * 100

            fig.update_layout(height=450, template="plotly_white",
                              xaxis_title="Tarih", yaxis_title="Endeks")
            st.plotly_chart(fig, use_container_width=True)

            rows = [{"Varlık": "Portföy",
                     "Getiri %": round(sonuc.get("toplam_getiri_pct", 0), 1)}]
            for k, v in getiri_map.items():
                rows.append({"Varlık": k, "Getiri %": round(v, 1)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ============================================================
# TAB 5: KIRMIZI BAYRAKLAR
# ============================================================
with tab5:
    st.header("🚩 Kırmızı Bayraklar")
    if tickers and st.button("🔍 Tara"):
        sonuclar = []
        for h in tickers:
            try:
                info = bp.Ticker(h).info
                bayraklar = []
                dy = info.get("dividendYield")
                if isinstance(dy, (int, float)) and dy < 1:
                    bayraklar.append("Düşük temettü verimi")
                pe = info.get("trailingPE")
                if isinstance(pe, (int, float)):
                    if pe < 0:
                        bayraklar.append("Negatif F/K")
                    elif pe > 30:
                        bayraklar.append(f"Yüksek F/K ({pe:.1f})")
                sonuclar.append({
                    "Hisse": h,
                    "Temettü Verimi %": round(dy, 2) if isinstance(dy, (int, float)) else "—",
                    "F/K": round(pe, 2) if isinstance(pe, (int, float)) else "—",
                    "Bayraklar": ", ".join(bayraklar) if bayraklar else "✅ Temiz"
                })
            except Exception as e:
                sonuclar.append({"Hisse": h, "Temettü Verimi %": "—", "F/K": "—",
                                 "Bayraklar": f"Hata: {str(e)[:40]}"})
        st.dataframe(pd.DataFrame(sonuclar), use_container_width=True, hide_index=True)

# ============================================================
# TAB 6: CANLI PORTFÖY
# ============================================================
with tab6:
    st.header("💼 Canlı Portföy")
    st.warning("Paper trade'i 3-6 ay tamamladıktan sonra canlıya geçin.")
    canli = st.session_state.canli_portfoy

    with st.expander("➕ İşlem Ekle", expanded=True):
        with st.form("canli_form", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                c_hisse = st.selectbox("Hisse", tickers if tickers else ["THYAO"], key="ch")
            with c2:
                c_lot = st.number_input("Lot", min_value=1, value=10, key="cl")
            with c3:
                c_fiyat = st.number_input("Fiyat", min_value=0.01, value=100.0, key="cf")
            with c4:
                c_tarih = st.date_input("Tarih", datetime.today(), key="ct")
            c_tur = st.radio("Tür", ["Alış", "Satış", "Temettü"], horizontal=True, key="ctur")
            if st.form_submit_button("✅ Kaydet"):
                yeni = pd.DataFrame([{
                    "Tarih": c_tarih, "Hisse": c_hisse, "Lot": c_lot,
                    "Fiyat": c_fiyat, "Tür": c_tur, "Tutar": c_lot * c_fiyat
                }])
                st.session_state.canli_portfoy = pd.concat([canli, yeni], ignore_index=True)
                st.rerun()

    if not canli.empty:
        st.dataframe(canli, use_container_width=True, hide_index=True)
        if st.button("🗑️ Temizle"):
            st.session_state.canli_portfoy = pd.DataFrame(
                columns=["Tarih", "Hisse", "Lot", "Fiyat", "Tür", "Tutar"])
            st.rerun()
    else:
        st.info("Canlı işlem yok.")