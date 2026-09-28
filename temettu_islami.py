# temettu_islami.py
"""
BIST Temettü Odaklı, Faizsiz (İslami) Portföy Yönetim Sistemi
- Backtest (Temettü DRIP + DCA modları)
- Otomatik Optimizasyon (tüm dönemler × tüm kombinasyonlar)
- CSV yükleme: Sukuk fonu ve Katılma Hesabı için gerçek veri
- Sukuk fonu seçimi (ZPG, GLS, IAT vb.)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import borsapy as bp

st.set_page_config(
    page_title="Faizsiz Portföy Sistemi",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🕌 BIST Temettü Odaklı Faizsiz Portföy Yönetim Paneli")
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
    st.sidebar.warning("⚠️ EVDS_API_KEY ortam değişkeni ayarlı değil. Enflasyon verisi çekilemeyecek.")

# ============================================================
# SABITLER
# ============================================================
VARSAYILAN_HISSELER = [
    "TUPRS", "FROTO", "TOASO", "BIMAS", "ASELS",
    "EREGL", "SISE", "TCELL", "ENJSA", "ISDMR",
    "Z30KP", "Z30KE", "OPK30"
]

KATILMA_HESABI_YILLIK_ORAN_VARSAYILAN = 0.30

DONEMLER = {
    "2010-2014 (Boğa, stabil TL)": (datetime(2010, 1, 1), datetime(2014, 12, 31)),
    "2015-2019 (Kur stresi)": (datetime(2015, 1, 1), datetime(2019, 12, 31)),
    "2018-2022 (Enflasyon patlaması)": (datetime(2018, 1, 1), datetime(2022, 12, 31)),
    "2022-2026 (Güncel)": (datetime(2022, 1, 1), datetime.today()),
}

KOMBINASYONLAR = [
    ("100/0/0/0", 100, 0, 0, 0),
    ("70/20/10/0", 70, 20, 10, 0),
    ("60/25/15/0", 60, 25, 15, 0),
    ("50/35/15/0", 50, 35, 15, 0),
    ("55/25/10/10", 55, 25, 10, 10),
    ("50/30/10/10", 50, 30, 10, 10),
    ("40/40/10/10", 40, 40, 10, 10),
]

# ============================================================
# OTURUM DURUMU
# ============================================================
defaults = {
    "paper_portfoy": pd.DataFrame(columns=["Tarih", "Varlık", "Lot", "Fiyat", "Tür", "Tutar"]),
    "canli_portfoy": pd.DataFrame(columns=["Tarih", "Varlık", "Lot", "Fiyat", "Tür", "Tutar"]),
    "backtest_sonuc": None,
    "backtest_fiyat_df": None,
    "backtest_hedef": None,
    "optimizasyon_sonuc": None,
    "sukuk_df": None,
    "katilma_df": None,
    "sukuk_kaynak": None,
    "katilma_kaynak": None,
    "secili_sukuk_fon": None,
    "secili_katilma_kaynak": None,
    "katilim_listesi": [
        # BIST Katılım Endeksi üyeliği (2024-2025 güncel liste)
        # Bu listeyi BIST'in resmi sitesinden güncelleyebilirsiniz
        "TUPRS", "FROTO", "TOASO", "BIMAS", "SISE", "TCELL",
        "ENJSA", "ISDMR", "ASELS", "EREGL", "PETKM", "AYGAZ",
        "TTKOM", "VESTL", "ARCLK", "DOAS", "OTKAR", "CIMSA",
        "KORDS", "GUBRF", "SOKM", "MAVI", "LOGO", "NETAS",
    ],
    "manuel_faizli_borc": {},   # {hisse: oran} manuel giriş
    "manuel_faiz_geliri": {},   # {hisse: oran} manuel giriş
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


def _dt(x):
    if isinstance(x, str):
        return pd.to_datetime(x).to_pydatetime()
    if isinstance(x, pd.Timestamp):
        return x.to_pydatetime()
    if isinstance(x, datetime):
        return x
    return pd.to_datetime(x).to_pydatetime()


# ============================================================
# CSV PARSING
# ============================================================
def parse_sukuk_csv(df):
    """Beklenen format: tarih, fon1, fon2, ... (her fon fiyatı)"""
    if df is None or df.empty:
        return None
    df = df.copy()

    tarih_col = None
    for c in df.columns:
        cl = str(c).lower()
        if "tarih" in cl or "date" in cl:
            tarih_col = c
            break
    if tarih_col is None:
        tarih_col = df.columns[0]

    try:
        df[tarih_col] = pd.to_datetime(df[tarih_col], errors="coerce")
    except Exception:
        return None
    df = df.dropna(subset=[tarih_col])
    df = df.set_index(tarih_col)

    # Sayısal sütunları al
    sayisal = df.select_dtypes(include="number")
    if sayisal.empty:
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        sayisal = df.select_dtypes(include="number")
        if sayisal.empty:
            return None

    sayisal = sayisal[sayisal > 0]
    sayisal = sayisal.dropna(how="all").ffill()
    return tz_temizle(sayisal)


def parse_katilma_csv(df):
    """Beklenen format: tarih, gunluk_oran VEYA birikimli_deger"""
    if df is None or df.empty:
        return None
    df = df.copy()

    tarih_col = None
    for c in df.columns:
        cl = str(c).lower()
        if "tarih" in cl or "date" in cl:
            tarih_col = c
            break
    if tarih_col is None:
        tarih_col = df.columns[0]

    try:
        df[tarih_col] = pd.to_datetime(df[tarih_col], errors="coerce")
    except Exception:
        return None
    df = df.dropna(subset=[tarih_col])
    df = df.set_index(tarih_col)

    sayisal = df.select_dtypes(include="number")
    if sayisal.empty:
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        sayisal = df.select_dtypes(include="number")
        if sayisal.empty:
            return None

    seri = sayisal.iloc[:, 0].dropna()
    seri = seri[seri > 0]
    if len(seri) < 2:
        return None

    # Değerler < 1.5 ise günlük oran kabul et ve birikimli değere çevir
    if seri.median() < 1.5:
        seri = (1 + seri).cumprod()

    return tz_temizle(seri)


# ============================================================
# VERİ ÇEKME
# ============================================================
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
    temettu_dict = {}
    hata_listesi = []
    for h in hisse_listesi:
        try:
            ticker = bp.Ticker(h)
            df = ticker.dividends
            if df is None or df.empty:
                continue
            if "Amount" in df.columns:
                seri = df["Amount"].copy()
            else:
                sayisal = df.select_dtypes(include="number")
                if sayisal.empty:
                    continue
                seri = sayisal.iloc[:, 0].copy()
            seri.index = pd.to_datetime(seri.index)
            seri = tz_temizle(seri)
            seri = seri[(seri.index >= pd.Timestamp(baslangic)) & (seri.index <= pd.Timestamp(bitis))]
            seri = seri[seri > 0]
            if len(seri) > 0:
                temettu_dict[h] = seri
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


# ============================================================
# SUKUK VE KATILMA HESABI — Series döndüren fonksiyonlar
# ============================================================
def sukuk_serisi_uret(baslangic, bitis, kaynak="otomatik"):
    """
    Sukuk fonu fiyat serisi (Series) döndürür.
    - CSV varsa: kullanıcının seçtiği fonun serisini döndürür
    - Yoksa: simülasyon serisi döndürür
    """
    # CSV'den yükleme
    if kaynak in ("csv", "otomatik") and st.session_state.get("sukuk_df") is not None:
        df = st.session_state["sukuk_df"]
        secili = st.session_state.get("secili_sukuk_fon")

        if secili and secili in df.columns:
            seri = df[secili].copy()
        else:
            seri = df.iloc[:, 0].copy()  # ilk fon

        try:
            bas_ts = pd.Timestamp(baslangic)
            bit_ts = pd.Timestamp(bitis)
            kes = seri.loc[(seri.index >= bas_ts) & (seri.index <= bit_ts)]
            if not kes.empty and len(kes) > 5:
                return tz_temizle(kes)
        except Exception:
            pass

    if kaynak == "csv":
        return None

    # Simülasyon fallback
    tarihler = pd.date_range(start=baslangic, end=bitis, freq="D")
    gunluk_oran = (1 + 0.35) ** (1/365) - 1
    fiyatlar = 1.0 * (1 + gunluk_oran) ** np.arange(len(tarihler))
    return pd.Series(fiyatlar, index=tarihler, name="SUKUK_simule")


def katilma_serisi_uret(baslangic, bitis, kaynak="otomatik",
                        yillik_oran=KATILMA_HESABI_YILLIK_ORAN_VARSAYILAN):
    """Katılma hesabı getiri serisi (Series) döndürür."""
    if kaynak in ("csv", "otomatik") and st.session_state.get("katilma_df") is not None:
        seri = st.session_state["katilma_df"]
        try:
            bas_ts = pd.Timestamp(baslangic)
            bit_ts = pd.Timestamp(bitis)
            kes = seri.loc[(seri.index >= bas_ts) & (seri.index <= bit_ts)]
            if not kes.empty and len(kes) > 5:
                return tz_temizle(kes)
        except Exception:
            pass

    if kaynak == "csv":
        return None

    tarihler = pd.date_range(start=baslangic, end=bitis, freq="D")
    gunluk_oran = (1 + yillik_oran) ** (1/365) - 1
    fiyatlar = 1.0 * (1 + gunluk_oran) ** np.arange(len(tarihler))
    return pd.Series(fiyatlar, index=tarihler, name="KATILMA_simule")


# ============================================================
# METRİKLER
# ============================================================
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
# BACKTEST: DCA
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
# BACKTEST: TEMETTÜ (DRIP)
# ============================================================
def backtest_temettu(fiyat_df, temettu_df, baslangic_sermaye,
                     rebalance_esik=10, hedef_agirliklar=None,
                     arindirma_orani=0.05):
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
    toplam_arindirma = 0.0

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

        if temettu_df is not None and tarih in temettu_df.index:
            gunluk_brut = 0.0
            gunluk_arindirma = 0.0
            for v in varliklar:
                if v in temettu_df.columns:
                    hb = temettu_df.loc[tarih, v]
                    if pd.notna(hb) and hb > 0 and lotlar[v] > 0:
                        brut = lotlar[v] * hb
                        gunluk_brut += brut
                        arindirma = brut * arindirma_orani
                        gunluk_arindirma += arindirma
                        toplam_arindirma += arindirma
                        net = brut - arindirma
                        toplam_temettu += net
                        temettu_kayitlari.append({
                            "Tarih": tarih, "Hisse": v,
                            "Hisse Başı": round(hb, 4),
                            "Brüt Temettü": round(brut, 2),
                            "Arındırma": round(arindirma, 2),
                            "Net Temettü": round(net, 2)
                        })
            nakit += (gunluk_brut - gunluk_arindirma)
            if nakit > 0:
                for v in varliklar:
                    f = fiyatlar.get(v, np.nan)
                    if pd.notna(f) and f > 0 and hedef[v] > 0:
                        tutar = nakit * (hedef[v] / 100)
                        if tutar > 0:
                            lotlar[v] += tutar / f
                nakit = 0.0

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
        "xirr_pct": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_dd,
        "baslangic_deger": baslangic_sermaye,
        "bitis_deger": bitis_deger,
        "net_kar": bitis_deger - baslangic_sermaye,
        "toplam_getiri_pct": toplam_getiri,
        "toplam_temettu": toplam_temettu,
        "toplam_arindirma": toplam_arindirma,
        "lotlar": lotlar,
        "nakit": nakit,
        "temettu_kayitlari": temettu_kayitlari,
        "rebalance_kayitlari": rebalance_kayitlari,
        "hedef_agirliklar": hedef,
    }


# ============================================================
# OTOMATİK OPTİMİZASYON
# ============================================================
def otomatik_optimizasyon(donemler, kombinasyonlar, hisseler,
                          mod="temettu", baslangic_sermaye=100_000,
                          aylik_alis=5_000, rebalance_esik=10,
                          progress_cb=None, sukuk_kaynak="otomatik",
                          katilma_kaynak="otomatik"):
    donemler = {k: (_dt(v[0]), _dt(v[1])) for k, v in donemler.items()}
    min_bas = min(v[0] for v in donemler.values())
    max_bit = max(v[1] for v in donemler.values())

    fiyat_full = veri_cek(hisseler, min_bas.strftime("%Y-%m-%d"), max_bit.strftime("%Y-%m-%d"))
    if fiyat_full is None or fiyat_full.empty:
        return None, "Hisse verisi alınamadı."

    temettu_full = None
    if mod == "temettu":
        temettu_full = temettu_verisi_cek(hisseler,
                                          min_bas.strftime("%Y-%m-%d"),
                                          max_bit.strftime("%Y-%m-%d"))
        if temettu_full is None:
            return None, "Temettü verisi alınamadı. 'DCA' moduna geçin."

    altin_full = altin_cek(min_bas.strftime("%Y-%m-%d"), max_bit.strftime("%Y-%m-%d"))
    sukuk_full = sukuk_serisi_uret(min_bas, max_bit, kaynak=sukuk_kaynak)
    katilma_full = katilma_serisi_uret(min_bas, max_bit, kaynak=katilma_kaynak)

    sonuclar = []
    toplam_is = len(donemler) * len(kombinasyonlar)
    is_sayaci = 0

    for d_adi, (d_bas, d_bit) in donemler.items():
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

        sukuk_kes = None
        if sukuk_full is not None:
            try:
                if isinstance(sukuk_full, pd.Series):
                    sukuk_kes = sukuk_full.loc[d_bas:d_bit]
                elif isinstance(sukuk_full, pd.DataFrame):
                    # Fallback: ilk sütunu al
                    sukuk_kes = sukuk_full.loc[d_bas:d_bit].iloc[:, 0]
            except Exception:
                sukuk_kes = None

        katilma_kes = None
        if katilma_full is not None:
            try:
                if isinstance(katilma_full, pd.Series):
                    katilma_kes = katilma_full.loc[d_bas:d_bit]
                elif isinstance(katilma_full, pd.DataFrame):
                    katilma_kes = katilma_full.loc[d_bas:d_bit].iloc[:, 0]
            except Exception:
                katilma_kes = None

        for k_adi, h_ag, s_ag, a_ag, n_ag in kombinasyonlar:
            is_sayaci += 1
            if progress_cb:
                progress_cb(is_sayaci, toplam_is, f"{d_adi} × {k_adi}")

            kullan_fiyat = fiyat_kes.copy()
            hedef = {h: h_ag / len(hisseler) for h in hisseler}

            # Sukuk ekle (Series olarak)
            if s_ag > 0 and sukuk_kes is not None and len(sukuk_kes) > 0:
                sukuk_hizali = sukuk_kes.reindex(kullan_fiyat.index, method="ffill")
                if sukuk_hizali.notna().sum() > 5:
                    kullan_fiyat["SUKUK"] = sukuk_hizali
                    hedef["SUKUK"] = s_ag

            # Altın ekle
            if a_ag > 0 and altin_kes is not None and len(altin_kes) > 0:
                altin_hizali = altin_kes.reindex(kullan_fiyat.index, method="ffill")
                if altin_hizali.notna().sum() > 5:
                    kullan_fiyat["Gram Altın"] = altin_hizali
                    hedef["Gram Altın"] = a_ag

            # Katılma hesabı ekle
            if n_ag > 0 and katilma_kes is not None and len(katilma_kes) > 0:
                katilma_hizali = katilma_kes.reindex(kullan_fiyat.index, method="ffill")
                if katilma_hizali.notna().sum() > 5:
                    kullan_fiyat["Katılma Hesabı"] = katilma_hizali
                    hedef["Katılma Hesabı"] = n_ag

            kullan_fiyat = kullan_fiyat.ffill().dropna(how="all")

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
            except Exception:
                sonuc = None

            if sonuc is None:
                continue

            enf_seri = enflasyon_serisi_cek(d_bas.strftime("%Y-%m-%d"),
                                            d_bit.strftime("%Y-%m-%d"))
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
                "Sukuk %": s_ag,
                "Altın %": a_ag,
                "Nakit %": n_ag,
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

    tickers = st.multiselect(
        "Hisse Evreni (Faizsiz)",
        options=VARSAYILAN_HISSELER,
        default=["TUPRS", "FROTO", "TOASO", "BIMAS", "ASELS"]
    )

    st.divider()

    # ============================================================
    # CSV YÜKLEME
    # ============================================================
    st.subheader("📁 Veri Kaynakları")

    with st.expander("📊 Sukuk Fonu CSV Yükle", expanded=(st.session_state.sukuk_df is None)):
        st.caption(
            "**Format:** İlk sütun tarih (YYYY-MM-DD), diğer sütunlar fon fiyatları. "
            "Örnek başlık: `tarih, ZPG, GLS, IAT`"
        )
        sukuk_file = st.file_uploader("Sukuk CSV", type=["csv"], key="sukuk_uploader")
        if sukuk_file is not None:
            try:
                raw = pd.read_csv(sukuk_file)
                parsed = parse_sukuk_csv(raw)
                if parsed is not None and not parsed.empty:
                    st.session_state.sukuk_df = parsed
                    st.session_state.sukuk_kaynak = "csv"
                    st.success(f"✅ {len(parsed)} satır yüklendi.")
                    st.dataframe(parsed.head(3), use_container_width=True)
                else:
                    st.error("❌ CSV parse edilemedi.")
            except Exception as e:
                st.error(f"❌ Hata: {str(e)[:100]}")

        # Sukuk fonu seçimi
        if st.session_state.sukuk_df is not None:
            kolonlar = st.session_state.sukuk_df.columns.tolist()
            st.info(f"📌 Fonlar: {', '.join(kolonlar)}")
            secili = st.selectbox(
                "Backtest'te kullanılacak fon",
                options=kolonlar,
                index=0,
                key="sukuk_fon_secim"
            )
            st.session_state.secili_sukuk_fon = secili

            if st.button("🗑️ Sukuk CSV'yi Sil", key="sukuk_clear"):
                st.session_state.sukuk_df = None
                st.session_state.sukuk_kaynak = None
                st.session_state.secili_sukuk_fon = None
                st.rerun()

    with st.expander("💰 Katılma Hesabı CSV Yükle", expanded=(st.session_state.katilma_df is None)):
        st.caption(
            "**Format:** İlk sütun tarih, ikinci sütun ya **günlük kâr payı oranı** "
            "(örn: 0.0008) ya da **birikimli değer** olmalıdır."
        )
        katilma_file = st.file_uploader("Katılma CSV", type=["csv"], key="katilma_uploader")
        if katilma_file is not None:
            try:
                raw = pd.read_csv(katilma_file)
                parsed = parse_katilma_csv(raw)
                if parsed is not None and not parsed.empty:
                    st.session_state.katilma_df = parsed
                    st.session_state.katilma_kaynak = "csv"
                    st.success(f"✅ {len(parsed)} satır yüklendi.")
                    st.dataframe(parsed.head(3).to_frame("Değer"), use_container_width=True)
                else:
                    st.error("❌ CSV parse edilemedi.")
            except Exception as e:
                st.error(f"❌ Hata: {str(e)[:100]}")

        if st.session_state.katilma_df is not None:
            if st.button("🗑️ Katılma CSV'yi Sil", key="katilma_clear"):
                st.session_state.katilma_df = None
                st.session_state.katilma_kaynak = None
                st.rerun()

    # Veri kaynak durumu
    sukuk_ok = st.session_state.sukuk_df is not None
    katilma_ok = st.session_state.katilma_df is not None
    if not sukuk_ok or not katilma_ok:
        st.warning(
            f"⚠️ **Simülasyon modu:** "
            f"{'Sukuk ' if not sukuk_ok else ''}"
            f"{'Katılma ' if not katilma_ok else ''}"
            "verisi simüle ediliyor."
        )
    else:
        st.success("✅ Tüm veriler gerçek (CSV'den)")

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
        vb, vbit = datetime(2022, 1, 1), bugun

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
# TAB 1: BACKTEST
# ============================================================
with tab1:
    st.header("Tek Konfigürasyon Backtest'i (Faizsiz)")

    mod_sec = st.radio(
        "Model",
        ["Temettü Yeniden Yatırım (DRIP)", "Düzenli Ek Alım (DCA)"],
        horizontal=True,
        help="DRIP: Sadece başlangıç sermayesi + temettüler. DCA: Her ay ek alım."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        h_ag = st.slider("Hisse Ağırlık %", 0, 100, 55, 5)
    with c2:
        s_ag = st.slider("Sukuk Ağırlık %", 0, 100, 25, 5)
    with c3:
        a_ag = st.slider("Altın Ağırlık %", 0, 100, 10, 5)
    with c4:
        n_ag = st.slider("Katılma Hesabı %", 0, 100, 10, 5)

    if st.button("🚀 Backtest Çalıştır", use_container_width=True):
        with st.spinner("Çalışıyor..."):
            fiyat_df = veri_cek(tickers, baslangic.strftime("%Y-%m-%d"), bitis.strftime("%Y-%m-%d"))
            if fiyat_df is None or fiyat_df.empty:
                st.error("Hisse verisi alınamadı.")
            else:
                temettu_df = None
                if "DRIP" in mod_sec:
                    temettu_df = temettu_verisi_cek(tickers,
                                                    baslangic.strftime("%Y-%m-%d"),
                                                    bitis.strftime("%Y-%m-%d"))
                    if temettu_df is None:
                        st.warning("⚠️ Temettü verisi alınamadı. DCA modu kullanılıyor.")
                        mod_sec = "Düzenli Ek Alım (DCA)"

                hedef = {h: h_ag / len(tickers) for h in tickers}

                if s_ag > 0:
                    sukuk = sukuk_serisi_uret(
                        baslangic, bitis,
                        kaynak=st.session_state.sukuk_kaynak or "otomatik"
                    )
                    if sukuk is not None and len(sukuk) > 0:
                        if isinstance(sukuk, pd.DataFrame):
                            sukuk = sukuk.iloc[:, 0]
                        hizali = sukuk.reindex(fiyat_df.index, method="ffill")
                        if hizali.notna().sum() > 5:
                            fiyat_df["SUKUK"] = hizali
                            hedef["SUKUK"] = s_ag
                if a_ag > 0:
                    altin = altin_cek(baslangic.strftime("%Y-%m-%d"), bitis.strftime("%Y-%m-%d"))
                    if altin is not None:
                        fiyat_df["Gram Altın"] = altin.reindex(fiyat_df.index, method="ffill")
                        hedef["Gram Altın"] = a_ag
                if n_ag > 0:
                    katilma = katilma_serisi_uret(
                        baslangic, bitis,
                        kaynak=st.session_state.katilma_kaynak or "otomatik"
                    )
                    if katilma is not None and len(katilma) > 0:
                        if isinstance(katilma, pd.DataFrame):
                            katilma = katilma.iloc[:, 0]
                        hizali = katilma.reindex(fiyat_df.index, method="ffill")
                        if hizali.notna().sum() > 5:
                            fiyat_df["Katılma Hesabı"] = hizali
                            hedef["Katılma Hesabı"] = n_ag

                fiyat_df = fiyat_df.ffill().dropna(how="all")

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
            st.info(f"💰 Toplam net temettü geliri: **{sonuc['toplam_temettu']:,.0f} TL**")
            if "toplam_arindirma" in sonuc and sonuc["toplam_arindirma"] > 0:
                st.warning(
                    f"🧼 Arındırma için ayrılan tutar: **{sonuc['toplam_arindirma']:,.0f} TL** "
                    f"(hayır kurumlarına bağışlanmalıdır)"
                )
    else:
        st.info("Backtest çalıştırın.")


# ============================================================
# TAB 2: OTOMATİK OPTİMİZASYON
# ============================================================
with tab2:
    st.header("🎯 Otomatik Dönem × Kombinasyon Optimizasyonu (Faizsiz)")

    # Uyarıyı sadece gerçekten simülasyon kullanılıyorsa göster
    sukuk_gercek = st.session_state.sukuk_df is not None
    katilma_gercek = st.session_state.katilma_df is not None
    katilma_kullaniliyor = any(k[4] > 0 for k in kombs_sec) if 'kombs_sec' in dir() else False

    if sukuk_gercek and (katilma_gercek or not katilma_kullaniliyor):
        st.success(f"✅ Tüm kullanılan veriler gerçek (Sukuk: {st.session_state.secili_sukuk_fon})")
    elif not sukuk_gercek:
        st.warning("⚠️ Sukuk verisi simüle ediliyor.")
    elif katilma_kullaniliyor and not katilma_gercek:
        st.warning("⚠️ Katılma hesabı verisi simüle ediliyor.")

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
            default=["100/0/0/0", "70/20/10/0", "60/25/15/0", "50/35/15/0"]
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
            st.warning("Kenar çubuktan en az bir hisse seçin.")
        else:
            donemler_sec = {d: DONEMLER[d] for d in sec_donemler}
            kombs_sec = [k for k in KOMBINASYONLAR if k[0] in sec_kombinasyonlar]

            prog = st.progress(0)
            durum = st.empty()

            def cb(i, toplam, mesaj):
                prog.progress(min(i / toplam, 1.0))
                durum.text(f"⏳ {i}/{toplam} — {mesaj}")

            with st.spinner("Optimizasyon çalışıyor... (1-2 dakika sürebilir)"):
                df_sonuc, hata = otomatik_optimizasyon(
                    donemler=donemler_sec,
                    kombinasyonlar=kombs_sec,
                    hisseler=tickers,
                    mod="temettu" if "DRIP" in mod_opt else "dca",
                    baslangic_sermaye=baslangic_sermaye,
                    aylik_alis=aylik_alis,
                    rebalance_esik=rebalance_esik,
                    progress_cb=cb,
                    sukuk_kaynak=st.session_state.sukuk_kaynak or "otomatik",
                    katilma_kaynak=st.session_state.katilma_kaynak or "otomatik",
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

    df_sonuc = st.session_state.get("optimizasyon_sonuc")

    if df_sonuc is not None and not df_sonuc.empty:
        st.subheader("📋 Tüm Sonuçlar")
        st.dataframe(df_sonuc, use_container_width=True, hide_index=True)

        st.subheader("🏆 Her Dönem İçin En İyi Kombinasyon")
        en_iyi_rows = []
        for d in df_sonuc["Dönem"].unique():
            alt = df_sonuc[df_sonuc["Dönem"] == d].copy()
            alt = alt.sort_values(by=["Sharpe", "Maks DD %"], ascending=[False, False])
            en_iyi = alt.iloc[0]
            en_iyi_rows.append({
                "Dönem": d,
                "En İyi Kombinasyon": en_iyi["Kombinasyon"],
                "XIRR/CAGR %": en_iyi["XIRR/CAGR %"],
                "Sharpe": en_iyi["Sharpe"],
                "Maks DD %": en_iyi["Maks DD %"],
                "Reel Fark": en_iyi.get("Reel Fark", None),
            })
        st.dataframe(pd.DataFrame(en_iyi_rows), use_container_width=True, hide_index=True)

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

        st.subheader("🎯 Sistemin Karar Önerisi")
        en_iyi_genel = ozet.iloc[0]
        st.success(
            f"**Önerilen Kombinasyon: {en_iyi_genel['Kombinasyon']}**\n\n"
            f"- Ortalama Sharpe: **{en_iyi_genel['Ort. Sharpe']:.2f}**\n"
            f"- Ortalama Maks Drawdown: **{en_iyi_genel['Ort. Maks DD %']:.1f}%**"
        )

        if st.session_state.sukuk_df is None or st.session_state.katilma_df is None:
            st.error(
                "🚨 **ÖNEMLİ:** Bu sonuçlar simüle edilmiş Sukuk/Katılma verisi içeriyor. "
                "Sharpe oranları olduğundan **yüksek** çıkmıştır."
            )


# ============================================================
# TAB 3: PAPER TRADE
# ============================================================
with tab3:
    st.header("🧪 Paper Trade (Faizsiz)")
    paper = st.session_state.paper_portfoy

    with st.expander("➕ İşlem Ekle", expanded=True):
        with st.form("paper_form", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                varlik_listesi = list(tickers) + ["SUKUK", "Gram Altın", "Katılma Hesabı"]
                p_varlik = st.selectbox("Varlık", varlik_listesi)
            with c2:
                p_lot = st.number_input("Lot", min_value=1, value=10)
            with c3:
                p_fiyat = st.number_input("Fiyat", min_value=0.01, value=100.0)
            with c4:
                p_tarih = st.date_input("Tarih", datetime.today())
            p_tur = st.radio("Tür", ["Alış", "Satış", "Temettü"], horizontal=True)
            if st.form_submit_button("✅ Kaydet"):
                yeni = pd.DataFrame([{
                    "Tarih": p_tarih, "Varlık": p_varlik, "Lot": p_lot,
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
    st.header("🔍 Karşılaştırma (Faizsiz Ölçütler)")
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
        sukuk = sukuk_serisi_uret(
            baslangic, bitis,
            kaynak=st.session_state.sukuk_kaynak or "otomatik"
        )
        if sukuk is not None and not sukuk.empty:
            if isinstance(sukuk, pd.DataFrame):
                sukuk = sukuk.iloc[:, 0]
            karsilastirma["Sukuk"] = sukuk

        if karsilastirma:
            portfoy_norm = equity["Portföy Değeri"] / equity["Portföy Değeri"].iloc[0] * 100
            portfoy_norm = tz_temizle(portfoy_norm)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfoy_norm.index, y=portfoy_norm,
                mode="lines", name="Portföy",
                line=dict(color="#00CC96", width=3)
            ))
            renkler = ["#EF553B", "#FFA15A", "#636EFA", "#AB63FA"]
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
                        line=dict(color=renkler[i % 4], width=2, dash="dash")
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
    st.header("🚩 Kırmızı Bayraklar (İslami Uyum + Finansal Sağlık)")

    # --- Manuel Veri Girişi Expander'ı ---
    with st.expander("📝 Manuel Veri Girişi (Faizli Borç / Faiz Geliri)", expanded=False):
        st.caption(
            "Borsapy finansal tablo verilerinden **faizli borç oranı** ve "
            "**faiz geliri oranı** otomatik çekilemiyorsa, buradan manuel girin. "
            "KAP'tan (kamuyu aydınlatma platformu) finansal tabloları inceleyip "
            "değerleri girin."
        )
        if tickers:
            secili_manuel = st.selectbox(
                "Hisse Seç",
                options=tickers,
                key="manuel_hisse_sec"
            )
            col1, col2 = st.columns(2)
            with col1:
                fb_orani = st.number_input(
                    "Faizli Borç Oranı (%)",
                    min_value=0.0, max_value=200.0, value=0.0, step=1.0,
                    help="Faizli borçlar / Toplam varlıklar × 100. İslami eşik: <%33",
                    key="manuel_fb_input"
                )
            with col2:
                fg_orani = st.number_input(
                    "Faiz Geliri Oranı (%)",
                    min_value=0.0, max_value=100.0, value=0.0, step=1.0,
                    help="Faiz geliri / Toplam gelir × 100. İslami eşik: <%5",
                    key="manuel_fg_input"
                )
            if st.button("💾 Kaydet", key="manuel_kaydet"):
                st.session_state.manuel_faizli_borc[secili_manuel] = fb_orani
                st.session_state.manuel_faiz_geliri[secili_manuel] = fg_orani
                st.success(f"✅ {secili_manuel}: Faizli borç %{fb_orani}, Faiz geliri %{fg_orani} kaydedildi.")
                st.rerun()

        # Kaydedilmiş manuel verileri göster
        if st.session_state.manuel_faizli_borc or st.session_state.manuel_faiz_geliri:
            manuel_data = []
            for h in tickers:
                fb = st.session_state.manuel_faizli_borc.get(h)
                fg = st.session_state.manuel_faiz_geliri.get(h)
                if fb is not None or fg is not None:
                    manuel_data.append({
                        "Hisse": h,
                        "Faizli Borç %": fb if fb is not None else "—",
                        "Faiz Geliri %": fg if fg is not None else "—",
                    })
            if manuel_data:
                st.dataframe(pd.DataFrame(manuel_data), use_container_width=True, hide_index=True)
                if st.button("🗑️ Tüm Manuel Verileri Sil", key="manuel_temizle"):
                    st.session_state.manuel_faizli_borc = {}
                    st.session_state.manuel_faiz_geliri = {}
                    st.rerun()

    # --- Katılım Listesi Yönetimi ---
    with st.expander("📋 Katılım Endeksi Listesi", expanded=False):
        st.caption(
            "Bu liste, BIST Katılım Endeksi'ne uygun hisseleri içerir. "
            "Güncel listeyi BIST'in resmi sitesinden kontrol edebilirsiniz."
        )
        katilim_str = st.text_area(
            "Virgülle ayrılmış hisse kodları",
            value=", ".join(st.session_state.katilim_listesi),
            height=80,
            key="katilim_text"
        )
        if st.button("💾 Listeyi Güncelle", key="katilim_guncelle"):
            yeni_liste = [h.strip().upper() for h in katilim_str.split(",") if h.strip()]
            st.session_state.katilim_listesi = yeni_liste
            st.success(f"✅ {len(yeni_liste)} hisse kaydedildi.")
            st.rerun()

    st.divider()

    # --- Tarama ---
    if not tickers:
        st.warning("Kenar çubuktan en az bir hisse seçin.")
    elif st.button("🔍 Kırmızı Bayrakları Tara", use_container_width=True, type="primary"):
        with st.spinner("İnceleniyor... (10 hisse için ~30 saniye)"):
            sonuclar = []

            for h in tickers:
                satir = {
                    "Hisse": h,
                    "Katılım Uyumlu": "—",
                    "Faizli Borç %": "—",
                    "Faiz Geliri %": "—",
                    "Temettü Verimi %": "—",
                    "F/K": "—",
                    "Payout %": "—",
                    "Bayrak Sayısı": 0,
                    "Seviye": "🟢 Temiz",
                    "Aksiyon": "Tut",
                    "Detay": "",
                }
                bayraklar = []
                ciddi_bayraklar = []

                try:
                    # --- İslami Uyum: Katılım Listesi ---
                    if h in st.session_state.katilim_listesi:
                        satir["Katılım Uyumlu"] = "✅ Evet"
                    else:
                        satir["Katılım Uyumlu"] = "❌ Hayır"
                        ciddi_bayraklar.append("Katılım endeksinde değil")

                    # --- İslami Uyum: Faizli Borç ---
                    fb = st.session_state.manuel_faizli_borc.get(h)
                    if fb is not None:
                        satir["Faizli Borç %"] = round(fb, 1)
                        if fb > 33:
                            ciddi_bayraklar.append(f"Faizli borç > %33 ({fb:.1f}%)")
                        elif fb > 25:
                            bayraklar.append(f"Faizli borç %25-33 ({fb:.1f}%)")
                    else:
                        # Otomatik çekmeyi dene (borsapy'den)
                        try:
                            ticker_obj = bp.Ticker(h)
                            # Deneme: bilanço veya info içinden
                            info = ticker_obj.info
                            # Çoğu hissede bu veri yok; manuel giriş gerekli
                            satir["Faizli Borç %"] = "❓ Gir"
                        except Exception:
                            satir["Faizli Borç %"] = "❓ Gir"

                    # --- İslami Uyum: Faiz Geliri ---
                    fg = st.session_state.manuel_faiz_geliri.get(h)
                    if fg is not None:
                        satir["Faiz Geliri %"] = round(fg, 1)
                        if fg > 5:
                            ciddi_bayraklar.append(f"Faiz geliri > %5 ({fg:.1f}%)")
                        elif fg > 3:
                            bayraklar.append(f"Faiz geliri %3-5 ({fg:.1f}%)")
                    else:
                        satir["Faiz Geliri %"] = "❓ Gir"

                    # --- Finansal Sağlık: Temettü ve Değerleme ---
                    ticker_obj = bp.Ticker(h)
                    info = ticker_obj.info

                    dy = info.get("dividendYield")
                    if isinstance(dy, (int, float)):
                        satir["Temettü Verimi %"] = round(dy, 2)
                        if dy < 1:
                            bayraklar.append(f"Düşük temettü (%{dy:.2f})")
                    else:
                        bayraklar.append("Temettü verimi yok")

                    pe = info.get("trailingPE")
                    if isinstance(pe, (int, float)):
                        satir["F/K"] = round(pe, 2)
                        if pe < 0:
                            ciddi_bayraklar.append("Negatif F/K (zarar)")
                        elif pe > 30:
                            bayraklar.append(f"Yüksek F/K ({pe:.1f})")

                    # Payout oranı (ödenen temettü / net kâr)
                    payout = info.get("payoutRatio")
                    if isinstance(payout, (int, float)):
                        satir["Payout %"] = round(payout * 100, 1)
                        if payout > 1.0:
                            ciddi_bayraklar.append(f"Payout >%100 ({payout*100:.0f}%)")
                        elif payout > 0.85:
                            bayraklar.append(f"Yüksek payout (%{payout*100:.0f})")

                except Exception as e:
                    satir["Detay"] = f"Hata: {str(e)[:50]}"

                # --- Seviye ve Aksiyon Belirleme ---
                toplam_bayrak = len(bayraklar) + len(ciddi_bayraklar)
                satir["Bayrak Sayısı"] = toplam_bayrak

                if len(ciddi_bayraklar) >= 2:
                    satir["Seviye"] = "🔴 Kritik"
                    satir["Aksiyon"] = "Derhal Çıkar"
                elif len(ciddi_bayraklar) == 1:
                    satir["Seviye"] = "🟠 Orta"
                    satir["Aksiyon"] = "Portföyden Çıkar"
                elif len(bayraklar) >= 2:
                    satir["Seviye"] = "🟠 Orta"
                    satir["Aksiyon"] = "Ağırlığı Azalt"
                elif len(bayraklar) == 1:
                    satir["Seviye"] = "🟡 Hafif"
                    satir["Aksiyon"] = "İzle, Yeni Alım Yapma"
                else:
                    satir["Seviye"] = "🟢 Temiz"
                    satir["Aksiyon"] = "Tut"

                # --- Detay Metni ---
                detaylar = []
                if ciddi_bayraklar:
                    detaylar.append("🔴 " + " | ".join(ciddi_bayraklar))
                if bayraklar:
                    detaylar.append("🟡 " + " | ".join(bayraklar))
                satir["Detay"] = " • ".join(detaylar) if detaylar else "✅ Sorun yok"

                sonuclar.append(satir)

            df_sonuc = pd.DataFrame(sonuclar)

            # --- Renklendirme ---
            def renkli_seviye(val):
                v = str(val)
                if "🔴" in v:
                    return "background-color: #f8d7da; color: #721c24; font-weight: bold"
                if "🟠" in v:
                    return "background-color: #ffe5b4; color: #8a4b00; font-weight: bold"
                if "🟡" in v:
                    return "background-color: #fff3cd; color: #856404"
                if "🟢" in v:
                    return "background-color: #d4edda; color: #155724"
                return ""

            st.dataframe(
                df_sonuc.style.map(renkli_seviye, subset=["Seviye"]),
                use_container_width=True,
                hide_index=True,
                height=min(600, 60 + len(df_sonuc) * 38)
            )

            # --- Özet İstatistikler ---
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            temiz = len(df_sonuc[df_sonuc["Seviye"].str.contains("🟢")])
            hafif = len(df_sonuc[df_sonuc["Seviye"].str.contains("🟡")])
            orta = len(df_sonuc[df_sonuc["Seviye"].str.contains("🟠")])
            kritik = len(df_sonuc[df_sonuc["Seviye"].str.contains("🔴")])

            c1.metric("🟢 Temiz", temiz)
            c2.metric("🟡 Hafif", hafif)
            c3.metric("🟠 Orta", orta)
            c4.metric("🔴 Kritik", kritik)

            # --- Aksiyon Önerileri ---
            st.divider()
            st.subheader("📋 Aksiyon Planı")

            if kritik > 0:
                st.error(
                    f"🔴 **{kritik} hissedе kritik bayrak var.** "
                    "Bu hisseleri **derhal portföyden çıkarın**. "
                    "Aşağıdaki tabloda hangi hisseler olduğunu görebilirsiniz."
                )
                kritik_df = df_sonuc[df_sonuc["Seviye"].str.contains("🔴")]
                st.dataframe(
                    kritik_df[["Hisse", "Detay"]],
                    use_container_width=True, hide_index=True
                )

            if orta > 0:
                st.warning(
                    f"🟠 **{orta} hissede orta seviye bayrak var.** "
                    "Ağırlıklarını azaltın veya bir sonraki rebalance'da çıkarın."
                )
                orta_df = df_sonuc[df_sonuc["Seviye"].str.contains("🟠")]
                st.dataframe(
                    orta_df[["Hisse", "Aksiyon", "Detay"]],
                    use_container_width=True, hide_index=True
                )

            if hafif > 0:
                st.info(
                    f"🟡 **{hafif} hissede hafif bayrak var.** "
                    "Yeni alım yapmayın, mevcut pozisyonu izleyin."
                )

            if temiz > 0:
                st.success(
                    f"🟢 **{temiz} hisse temiz.** "
                    "Hedef ağırlıklarını koruyun."
                )

            # --- Uyarı ---
            st.caption(
                "⚠️ **Not:** Otomatik finansal sağlık taraması yapılır, ancak "
                "**faizli borç oranı** ve **faiz geliri oranı** için manuel veri "
                "girişi gerekir. Bu verileri KAP'tan temin edebilirsiniz."
            )

            # --- Bilgi Kutusu ---
            with st.expander("ℹ️ Kriterler ve Eşikler"):
                st.markdown("""
                **İslami Uyum Kriterleri:**
                - **Katılım Endeksi Üyeliği:** Hisse, BIST Katılım Endeksi'nde olmalı
                - **Faizli Borç Oranı:** < %33 (faizli borç / toplam varlıklar)
                - **Faiz Geliri Oranı:** < %5 (faiz geliri / toplam gelir)

                **Finansal Sağlık Kriterleri:**
                - **Temettü Verimi:** > %1 (temettü odaklı portföy için)
                - **F/K:** 0-30 arası (negatif = zarar, >30 = pahalı)
                - **Payout Oranı:** < %85 (sürdürülebilir temettü)

                **Seviye ve Aksiyon:**
                | Seviye | Kriter | Aksiyon |
                |---|---|---|
                | 🔴 Kritik | 2+ ciddi ihlal | Derhal çıkar |
                | 🟠 Orta | 1 ciddi VEYA 2+ hafif | Çıkar / azalt |
                | 🟡 Hafif | 1 hafif ihlal | İzle, alım yapma |
                | 🟢 Temiz | İhlal yok | Tut |
                """)
    else:
        st.info("Tarama başlatmak için yukarıdaki butona basın.")

# ============================================================
# TAB 6: CANLI PORTFÖY
# ============================================================
with tab6:
    st.header("💼 Canlı Portföy (Faizsiz)")
    st.warning("Paper trade'i 3-6 ay tamamladıktan sonra canlıya geçin.")
    canli = st.session_state.canli_portfoy

    with st.expander("➕ İşlem Ekle", expanded=True):
        with st.form("canli_form", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                varlik_listesi = list(tickers) + ["SUKUK", "Gram Altın", "Katılma Hesabı"]
                c_varlik = st.selectbox("Varlık", varlik_listesi, key="cv")
            with c2:
                c_lot = st.number_input("Lot", min_value=1, value=10, key="cl")
            with c3:
                c_fiyat = st.number_input("Fiyat", min_value=0.01, value=100.0, key="cf")
            with c4:
                c_tarih = st.date_input("Tarih", datetime.today(), key="ct")
            c_tur = st.radio("Tür", ["Alış", "Satış", "Temettü"], horizontal=True, key="ctur")
            if st.form_submit_button("✅ Kaydet"):
                yeni = pd.DataFrame([{
                    "Tarih": c_tarih, "Varlık": c_varlik, "Lot": c_lot,
                    "Fiyat": c_fiyat, "Tür": c_tur, "Tutar": c_lot * c_fiyat
                }])
                st.session_state.canli_portfoy = pd.concat([canli, yeni], ignore_index=True)
                st.rerun()

    if not canli.empty:
        st.dataframe(canli, use_container_width=True, hide_index=True)
        if st.button("🗑️ Temizle"):
            st.session_state.canli_portfoy = pd.DataFrame(
                columns=["Tarih", "Varlık", "Lot", "Fiyat", "Tür", "Tutar"])
            st.rerun()
    else:
        st.info("Canlı işlem yok.")