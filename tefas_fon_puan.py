"""
TEFAS Akıllı Para Sinyal Sistemi — SQLite Kalıcı Depolama Sürümü
=================================================================
Her çalıştırmada sadece YENİ tarihli veriyi TEFAS'tan çeker,
eski veriyi SQLite'ta saklar ve birleştirerek analiz eder.

Gereksinimler:
    pip install pytefas pandas numpy streamlit plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import plotly.graph_objects as go
import plotly.express as px
import warnings
import os
import json
import sqlite3
import time

warnings.filterwarnings('ignore')

# =============================================
# 0. KONFİGÜRASYON
# =============================================
DB_PATH = "tefas_data.db"
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

try:
    from pytefas import Crawler as TefasCrawler
    PYTEFAS_AVAILABLE = True
except ImportError:
    PYTEFAS_AVAILABLE = False
    TefasCrawler = None


# =============================================
# 1. SQLITE KATMANI
# =============================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS fund_info (
        date TEXT NOT NULL,
        kind TEXT NOT NULL,
        fund_code TEXT NOT NULL,
        fund_name TEXT,
        price REAL,
        shares_outstanding REAL,
        investor_count REAL,
        portfolio_size REAL,
        PRIMARY KEY (date, kind, fund_code)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS fund_breakdown (
        date TEXT NOT NULL,
        kind TEXT NOT NULL,
        fund_code TEXT NOT NULL,
        breakdown_json TEXT,
        PRIMARY KEY (date, kind, fund_code)
    )''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_info_kind_date ON fund_info(kind, date)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_info_code ON fund_info(fund_code)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_breakdown_kind_date ON fund_breakdown(kind, date)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_breakdown_code ON fund_breakdown(fund_code)')
    conn.commit()
    conn.close()


init_db()


def get_last_date(kind, table='fund_info'):
    """DB'deki en son veri tarihini döndürür. Yoksa None."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            f"SELECT MAX(date) AS son FROM {table} WHERE kind=?",
            conn, params=(kind,)
        )
        son = df['son'].iloc[0]
        if son is None or pd.isna(son):
            return None
        return pd.to_datetime(son).date()
    finally:
        conn.close()


def get_total_rows(kind, table='fund_info'):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            f"SELECT COUNT(*) AS n FROM {table} WHERE kind=?",
            conn, params=(kind,)
        )
        return int(df['n'].iloc[0])
    finally:
        conn.close()


def save_fund_info(df):
    """Fon bilgilerini DB'ye yazar (INSERT OR REPLACE)."""
    if df.empty:
        return 0

    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    # Beklenen kolonları garanti et
    for col in ['fund_name', 'price', 'shares_outstanding',
                'investor_count', 'portfolio_size']:
        if col not in df.columns:
            df[col] = None

    if 'kind' not in df.columns:
        df['kind'] = 'YAT'

    rows = list(df[['date', 'kind', 'fund_code', 'fund_name', 'price',
                    'shares_outstanding', 'investor_count', 'portfolio_size']]
                .itertuples(index=False, name=None))

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executemany(
            """INSERT OR REPLACE INTO fund_info
               (date, kind, fund_code, fund_name, price,
                shares_outstanding, investor_count, portfolio_size)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows
        )
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def save_breakdown(df):
    """Varlık dağılımlarını JSON olarak DB'ye yazar."""
    if df.empty:
        return 0

    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    if 'kind' not in df.columns:
        df['kind'] = 'YAT'

    meta_cols = {'date', 'kind', 'fund_code', 'fund_name'}
    numeric_cols = [c for c in df.columns if c not in meta_cols]

    rows = []
    for _, row in df.iterrows():
        breakdown = {}
        for col in numeric_cols:
            v = row[col]
            if pd.notna(v) and isinstance(v, (int, float, np.number)):
                breakdown[col] = float(v)
        rows.append((
            row['date'], row['kind'], row['fund_code'],
            json.dumps(breakdown, ensure_ascii=False)
        ))

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executemany(
            """INSERT OR REPLACE INTO fund_breakdown
               (date, kind, fund_code, breakdown_json)
               VALUES (?, ?, ?, ?)""",
            rows
        )
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def load_fund_info(kind, start_date=None, end_date=None):
    """DB'den fon bilgilerini çeker, date sütununu datetime yapar."""
    conn = sqlite3.connect(DB_PATH)
    try:
        query = "SELECT * FROM fund_info WHERE kind=?"
        params = [kind]
        if start_date is not None:
            query += " AND date >= ?"
            params.append(pd.to_datetime(start_date).strftime('%Y-%m-%d'))
        if end_date is not None:
            query += " AND date <= ?"
            params.append(pd.to_datetime(end_date).strftime('%Y-%m-%d'))
        query += " ORDER BY date, fund_code"
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    # KRİTİK: date sütununu datetime'a çevir (hata düzeltmesi)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    return df


def load_breakdown(kind, start_date=None, end_date=None, fund_code=None):
    """DB'den varlık dağılımını çeker ve JSON'u açarak geniş DataFrame yapar."""
    conn = sqlite3.connect(DB_PATH)
    try:
        query = "SELECT * FROM fund_breakdown WHERE kind=?"
        params = [kind]
        if start_date is not None:
            query += " AND date >= ?"
            params.append(pd.to_datetime(start_date).strftime('%Y-%m-%d'))
        if end_date is not None:
            query += " AND date <= ?"
            params.append(pd.to_datetime(end_date).strftime('%Y-%m-%d'))
        if fund_code is not None:
            query += " AND fund_code = ?"
            params.append(fund_code)
        query += " ORDER BY date, fund_code"
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    # JSON'u aç
    breakdowns = df['breakdown_json'].apply(
        lambda x: json.loads(x) if x else {}
    )
    df_break = pd.json_normalize(breakdowns)
    df = pd.concat([df.drop(columns=['breakdown_json']), df_break], axis=1)

    # KRİTİK: date sütununu datetime'a çevir
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    return df


# =============================================
# 2. TEFAS VERİ SAĞLAYICI (SQLite ile senkronize)
# =============================================
class TefasDataProvider:
    """
    pytefas üzerinden TEFAS verilerini çeker.
    Her çağrıda SQLite'taki son tarihi kontrol eder,
    sadece YENİ tarihleri çeker ve DB'ye yazar.
    """

    def __init__(self):
        if not PYTEFAS_AVAILABLE:
            raise ImportError(
                "pytefas kütüphanesi bulunamadı. "
                "Lütfen 'pip install pytefas' komutunu çalıştırın."
            )
        self.crawler = TefasCrawler()

    # -----------------------------------------
    # 2.1 İnkremental senkronizasyon: fund_info
    # -----------------------------------------
    def sync_fund_info(self, kind, start_date, end_date, force_refresh=False,
                       progress_cb=None):
        """
        DB'deki son tarihten bugüne kadar olan YENİ günleri çeker.
        force_refresh=True ise tüm aralığı baştan çeker.
        """
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()

        # DB'deki son tarihi bul
        son_db = get_last_date(kind, 'fund_info')

        if force_refresh or son_db is None:
            fetch_start = start_date
        else:
            # Son tarihten bir gün sonrasından başla (üzerine yazmasın)
            fetch_start = son_db + timedelta(days=1)

        if fetch_start > end_date:
            if progress_cb:
                progress_cb(f"Bilgi: yeni veri yok (son tarih: {son_db})")
            return load_fund_info(kind, start_date, end_date)

        if progress_cb:
            progress_cb(f"Çekiliyor: {fetch_start} → {end_date} (info)")

        try:
            df_new = self.crawler.fetch(
                start=fetch_start.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                kind=kind,
                columns="info",
            )
        except Exception as e:
            if progress_cb:
                progress_cb(f"Hata: {e}")
            df_new = pd.DataFrame()

        if not df_new.empty:
            if 'kind' not in df_new.columns:
                df_new['kind'] = kind
            n = save_fund_info(df_new)
            if progress_cb:
                progress_cb(f"✔ {n} yeni kayıt kaydedildi (info)")

        return load_fund_info(kind, start_date, end_date)

    # -----------------------------------------
    # 2.2 İnkremental senkronizasyon: breakdown
    # -----------------------------------------
    def sync_breakdown(self, kind, start_date, end_date, force_refresh=False,
                       progress_cb=None):
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()

        son_db = get_last_date(kind, 'fund_breakdown')

        if force_refresh or son_db is None:
            fetch_start = start_date
        else:
            fetch_start = son_db + timedelta(days=1)

        if fetch_start > end_date:
            if progress_cb:
                progress_cb(f"Bilgi: yeni breakdown verisi yok (son: {son_db})")
            return load_breakdown(kind, start_date, end_date)

        if progress_cb:
            progress_cb(f"Çekiliyor: {fetch_start} → {end_date} (breakdown)")

        try:
            df_new = self.crawler.fetch(
                start=fetch_start.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                kind=kind,
                columns="breakdown",
            )
        except Exception as e:
            if progress_cb:
                progress_cb(f"Hata (breakdown): {e}")
            df_new = pd.DataFrame()

        if not df_new.empty:
            if 'kind' not in df_new.columns:
                df_new['kind'] = kind
            n = save_breakdown(df_new)
            if progress_cb:
                progress_cb(f"✔ {n} yeni kayıt kaydedildi (breakdown)")

        return load_breakdown(kind, start_date, end_date)

    # -----------------------------------------
    # 2.3 Tek fon için geçmiş
    # -----------------------------------------
    def get_fund_history(self, fund_code, kind, start_date, end_date):
        df = load_fund_info(kind, start_date, end_date)
        if df.empty:
            return df
        return df[df['fund_code'] == fund_code].copy()

    def get_fund_breakdown(self, fund_code, kind, start_date, end_date):
        return load_breakdown(kind, start_date, end_date, fund_code=fund_code)


# =============================================
# 3. NAKİT AKIŞI HESAPLAMA (DÜZELTİLMİŞ)
# =============================================
class CashFlowAnalyzer:

    @staticmethod
    def _ensure_datetime(df):
        """date sütununu datetime'a zorlar — hata önleyici."""
        if df.empty:
            return df
        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
        return df

    @staticmethod
    def calculate_net_flow(df_info):
        if df_info.empty:
            return df_info

        df = CashFlowAnalyzer._ensure_datetime(df_info)
        df = df.sort_values(["fund_code", "date"]).copy()

        # Sayısal dönüşüm garantisi
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['portfolio_size'] = pd.to_numeric(df['portfolio_size'], errors='coerce')

        df["daily_return"] = df.groupby("fund_code")["price"].pct_change()
        df["aum_change"] = df.groupby("fund_code")["portfolio_size"].diff()
        df["prev_aum"] = df.groupby("fund_code")["portfolio_size"].shift(1)
        df["market_effect"] = df["daily_return"] * df["prev_aum"]
        df["net_flow"] = df["aum_change"] - df["market_effect"]
        df["cumulative_flow"] = df.groupby("fund_code")["net_flow"].cumsum()

        return df

    @staticmethod
    def aggregate_flow(df_flow, window_days=30):
        if df_flow.empty or "net_flow" not in df_flow.columns:
            return pd.DataFrame()

        df = CashFlowAnalyzer._ensure_datetime(df_flow)
        df = df.dropna(subset=["net_flow"])
        if df.empty:
            return pd.DataFrame()

        # HATA DÜZELTMESİ: max() artık datetime, Timedelta çıkarma geçerli
        max_date = df["date"].max()
        if not isinstance(max_date, (pd.Timestamp, datetime.datetime, datetime.date)):
            return pd.DataFrame()

        cutoff = pd.to_datetime(max_date) - pd.Timedelta(days=int(window_days))
        df_recent = df[df["date"] >= cutoff]

        if df_recent.empty:
            return pd.DataFrame()

        agg = df_recent.groupby("fund_code").agg(
            toplam_net_akis=("net_flow", "sum"),
            gunluk_ortalama_akis=("net_flow", "mean"),
            akis_gun_sayisi=("net_flow", "count"),
            son_aum=("portfolio_size", "last"),
            ilk_aum=("portfolio_size", "first"),
        ).reset_index()

        agg["aum_degisim_pct"] = (
            (agg["son_aum"] / agg["ilk_aum"] - 1) * 100
        )

        return agg


# =============================================
# 4. AKILLI PARA SKORU
# =============================================
class SmartMoneyScorer:

    @staticmethod
    def _ensure_datetime(df):
        if df.empty:
            return df
        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
        return df

    @staticmethod
    def calculate_investor_score(df_info, fund_code, window_days=30):
        if df_info.empty:
            return 0, 0

        df = SmartMoneyScorer._ensure_datetime(df_info)
        df = df[df["fund_code"] == fund_code].sort_values("date")
        if len(df) < 2:
            return 0, 0

        max_date = df["date"].max()
        cutoff = max_date - pd.Timedelta(days=int(window_days))
        df_recent = df[df["date"] >= cutoff]
        if len(df_recent) < 2:
            return 0, 0

        ilk = df_recent["investor_count"].iloc[0]
        son = df_recent["investor_count"].iloc[-1]
        if pd.isna(ilk) or pd.isna(son) or ilk == 0:
            return 0, 0

        degisim_pct = (son / ilk - 1) * 100

        if degisim_pct > 10:   return 30, degisim_pct
        elif degisim_pct > 5:  return 20, degisim_pct
        elif degisim_pct > 0:  return 10, degisim_pct
        elif degisim_pct > -5: return -10, degisim_pct
        else:                  return -30, degisim_pct

    @staticmethod
    def calculate_flow_score(df_flow_agg, fund_code):
        if df_flow_agg.empty:
            return 0, 0
        row = df_flow_agg[df_flow_agg["fund_code"] == fund_code]
        if row.empty:
            return 0, 0
        net_akis = row["toplam_net_akis"].iloc[0]
        if pd.isna(net_akis):
            return 0, 0
        aum = row["son_aum"].iloc[0] if "son_aum" in row.columns else 0
        akis_pct = (net_akis / aum * 100) if aum and aum > 0 else 0

        if akis_pct > 5:   return 30, net_akis
        elif akis_pct > 2: return 20, net_akis
        elif akis_pct > 0: return 10, net_akis
        elif akis_pct > -2: return -10, net_akis
        else:              return -30, net_akis

    @staticmethod
    def calculate_allocation_score(df_breakdown, fund_code, window_days=30):
        if df_breakdown.empty:
            return 0, {}

        df = SmartMoneyScorer._ensure_datetime(df_breakdown)
        df = df[df["fund_code"] == fund_code].sort_values("date")
        if len(df) < 2:
            return 0, {}

        max_date = df["date"].max()
        cutoff = max_date - pd.Timedelta(days=int(window_days))
        df_recent = df[df["date"] >= cutoff]
        if len(df_recent) < 2:
            return 0, {}

        score = 0
        detay = {}

        hisse_kolonlari = [c for c in df.columns
                           if "stock" in c.lower() and c not in ['fund_code', 'date', 'kind']]
        for col in hisse_kolonlari:
            ilk = df_recent[col].iloc[0]
            son = df_recent[col].iloc[-1]
            if pd.notna(ilk) and pd.notna(son):
                degisim = son - ilk
                detay[col] = round(degisim, 2)
                if degisim > 2:   score += 10
                elif degisim > 0: score += 5
                elif degisim < -2: score -= 10

        risk_off_kolonlari = [c for c in df.columns
                              if any(k in c.lower() for k in
                                     ["gold", "fx", "foreign_currency"])
                              and c not in ['fund_code', 'date', 'kind']]
        for col in risk_off_kolonlari:
            ilk = df_recent[col].iloc[0]
            son = df_recent[col].iloc[-1]
            if pd.notna(ilk) and pd.notna(son):
                degisim = son - ilk
                detay[col] = round(degisim, 2)
                if degisim > 2:   score -= 10
                elif degisim < -2: score += 5

        return int(np.clip(score, -20, 20)), detay

    @staticmethod
    def calculate_size_score(df_info, fund_code, min_aum=50_000_000):
        if df_info.empty:
            return 0, 0
        df = df_info[df_info["fund_code"] == fund_code]
        if df.empty:
            return 0, 0
        son_aum = df["portfolio_size"].iloc[-1]
        if pd.isna(son_aum):
            return 0, 0

        if son_aum > 1_000_000_000:    return 10, son_aum
        elif son_aum > 500_000_000:    return 8, son_aum
        elif son_aum > 100_000_000:    return 5, son_aum
        elif son_aum > min_aum:        return 2, son_aum
        else:                          return -10, son_aum

    @staticmethod
    def calculate_total_score(df_info, df_flow_agg, df_breakdown, fund_code):
        inv_score, inv_degisim = SmartMoneyScorer.calculate_investor_score(df_info, fund_code)
        flow_score, net_akis = SmartMoneyScorer.calculate_flow_score(df_flow_agg, fund_code)
        alloc_score, alloc_detay = SmartMoneyScorer.calculate_allocation_score(df_breakdown, fund_code)
        size_score, aum = SmartMoneyScorer.calculate_size_score(df_info, fund_code)

        if not df_info.empty:
            nokta_sayisi = len(df_info[df_info["fund_code"] == fund_code])
            if nokta_sayisi > 200:   yas_score = 10
            elif nokta_sayisi > 100: yas_score = 7
            elif nokta_sayisi > 50:  yas_score = 4
            else:                    yas_score = 0
        else:
            yas_score = 0

        toplam = inv_score + flow_score + alloc_score + size_score + yas_score
        toplam = float(np.clip(toplam, 0, 100))

        detay = {
            "Yatırımcı Skoru": inv_score,
            "Yatırımcı Değişim (%)": round(inv_degisim, 2),
            "Akış Skoru": flow_score,
            "Net Akış (TL)": round(net_akis, 2) if pd.notna(net_akis) else 0,
            "Dağılım Skoru": alloc_score,
            "Büyüklük Skoru": size_score,
            "Fon Büyüklüğü (TL)": round(aum, 2) if pd.notna(aum) else 0,
            "Yaş/İstikrar Skoru": yas_score,
            "Toplam Skor": toplam,
        }
        return toplam, detay


# =============================================
# 5. SİNYAL ÜRETİCİ
# =============================================
class SignalGenerator:

    @staticmethod
    def generate_signals(df_info, df_flow_agg, df_breakdown):
        if df_info.empty:
            return pd.DataFrame()

        fonlar = df_info["fund_code"].dropna().unique()
        sonuclar = []

        for fon in fonlar:
            try:
                skor, detay = SmartMoneyScorer.calculate_total_score(
                    df_info, df_flow_agg, df_breakdown, fon
                )

                if skor >= 70:   sinyal = "GÜÇLÜ AL"
                elif skor >= 50: sinyal = "AL"
                elif skor >= 40: sinyal = "NÖTR"
                elif skor >= 25: sinyal = "ZAYIF"
                else:            sinyal = "UZAK DUR"

                fon_df = df_info[df_info["fund_code"] == fon]
                fon_adi = ""
                if not fon_df.empty and "fund_name" in fon_df.columns:
                    fon_adi = fon_df["fund_name"].iloc[-1] or ""

                sonuclar.append({
                    "Fon Kodu": fon,
                    "Fon Adı": fon_adi,
                    "Toplam Skor": round(skor, 1),
                    "Sinyal": sinyal,
                    **{k: v for k, v in detay.items() if k != "Toplam Skor"},
                })
            except Exception:
                continue

        df_sonuc = pd.DataFrame(sonuclar)
        if not df_sonuc.empty:
            df_sonuc = df_sonuc.sort_values("Toplam Skor", ascending=False)

        return df_sonuc


def signal_to_portfolio(df_signals, top_n=5, min_aum=50_000_000):
    if df_signals.empty:
        return {}
    df = df_signals[df_signals["Sinyal"].isin(["GÜÇLÜ AL", "AL"])].copy()

    if "Fon Büyüklüğü (TL)" in df.columns:
        df = df[df["Fon Büyüklüğü (TL)"] >= min_aum]

    if df.empty:
        return {}

    df = df.head(top_n)
    toplam_skor = df["Toplam Skor"].sum()
    if toplam_skor == 0:
        return {}

    return {row["Fon Kodu"]: row["Toplam Skor"] / toplam_skor
            for _, row in df.iterrows()}


# =============================================
# 6. STREAMLIT ARAYÜZ
# =============================================
st.set_page_config(page_title="TEFAS Akıllı Para Sinyal Sistemi", layout="wide")
st.title("🏦 TEFAS Akıllı Para Sinyal Sistemi")
st.caption("Varlık dağılımı + Yatırımcı sayısı + Fon büyüklüğü + Nakit akışı analizi • SQLite kalıcı depolama")

with st.sidebar:
    st.header("⚙️ Ayarlar")

    fon_tipi = st.selectbox(
        "Fon Tipi",
        ["YAT (Yatırım Fonları)", "EMK (Emeklilik Fonları)", "BYF (Borsa Yatırım Fonları)"],
        index=0
    )
    kind_map = {"YAT (Yatırım Fonları)": "YAT",
                "EMK (Emeklilik Fonları)": "EMK",
                "BYF (Borsa Yatırım Fonları)": "BYF"}
    kind = kind_map[fon_tipi]

    analiz_gun = st.slider("Analiz Periyodu (Gün)", 30, 365, 90, 30)

    st.divider()
    st.subheader("🔄 Veri Senkronizasyonu")
    force_refresh = st.checkbox(
        "Tüm veriyi baştan çek (force refresh)",
        value=False,
        help="İşaretlenirse DB'deki tüm veri yeniden çekilir. "
             "Normalde sadece yeni tarihler çekilir."
    )

    # DB özeti
    try:
        n_info = get_total_rows(kind, 'fund_info')
        n_brk = get_total_rows(kind, 'fund_breakdown')
        son_info = get_last_date(kind, 'fund_info')
        son_brk = get_last_date(kind, 'fund_breakdown')
        st.info(
            f"📦 **DB'deki Kayıtlar ({kind})**\n\n"
            f"• Info: {n_info:,} satır\n"
            f"• Breakdown: {n_brk:,} satır\n"
            f"• Son info: `{son_info or '—'}`\n"
            f"• Son breakdown: `{son_brk or '—'}`"
        )
    except Exception as e:
        st.warning(f"DB okunamadı: {e}")

    st.divider()
    st.subheader("🎯 Sinyal Filtreleri")
    min_skor = st.slider("Minimum Skor", 0, 100, 40, 5)
    max_fon = st.slider("Maksimum Fon Sayısı", 3, 20, 10, 1)
    min_aum_milyon = st.slider("Minimum Fon Büyüklüğü (Milyon TL)", 10, 1000, 50, 10)

    st.divider()
    st.caption("⚠️ Bu araç yatırım tavsiyesi değildir.")


tab_sinyal, tab_fon_detay, tab_akis, tab_db = st.tabs(
    ["📡 Sinyaller", "📋 Fon Detayı", "💰 Nakit Akışı", "🗄️ Veritabanı"]
)


# =============================================
# SEKME 1: SİNYALLER
# =============================================
with tab_sinyal:
    st.subheader(f"Akıllı Para Skorları — {fon_tipi}")

    if not PYTEFAS_AVAILABLE:
        st.error("❌ `pytefas` yüklü değil. Terminalde `pip install pytefas` çalıştırın.")
    else:
        if st.button("🔍 Analizi Çalıştır", type="primary"):
            end = datetime.date.today()
            start = end - timedelta(days=analiz_gun + 30)

            progress_placeholder = st.empty()
            log_messages = []

            def log_cb(msg):
                log_messages.append(msg)
                progress_placeholder.info("📋 " + " | ".join(log_messages[-3:]))

            with st.spinner("Veri senkronize ediliyor..."):
                try:
                    provider = TefasDataProvider()
                except Exception as e:
                    st.error(f"Provider oluşturulamadı: {e}")
                    st.stop()

                # 1. Info senkronizasyonu
                df_info = provider.sync_fund_info(
                    kind, start, end,
                    force_refresh=force_refresh,
                    progress_cb=log_cb,
                )

                if df_info.empty:
                    st.error("Hiç fon bilgisi yok. force_refresh'i deneyin.")
                    st.stop()

                # 2. Breakdown senkronizasyonu
                df_breakdown = provider.sync_breakdown(
                    kind, start, end,
                    force_refresh=force_refresh,
                    progress_cb=log_cb,
                )

                # 3. Akış analizi
                with st.spinner("Nakit akışları hesaplanıyor..."):
                    df_flow = CashFlowAnalyzer.calculate_net_flow(df_info)
                    df_flow_agg = CashFlowAnalyzer.aggregate_flow(
                        df_flow, window_days=analiz_gun
                    )

                # 4. Sinyaller
                with st.spinner("Sinyaller üretiliyor..."):
                    df_signals = SignalGenerator.generate_signals(
                        df_info, df_flow_agg, df_breakdown
                    )

            progress_placeholder.success(
                f"✅ Tamamlandı. {len(df_info):,} info kaydı, "
                f"{len(df_breakdown):,} breakdown kaydı analiz edildi."
            )

            if df_signals.empty:
                st.warning("Sinyal üretilemedi. Yeterli veri yok.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Toplam Fon", len(df_signals))
                c2.metric("Güçlü Al", len(df_signals[df_signals["Sinyal"] == "GÜÇLÜ AL"]))
                c3.metric("Al", len(df_signals[df_signals["Sinyal"] == "AL"]))
                c4.metric("Uzak Dur", len(df_signals[df_signals["Sinyal"] == "UZAK DUR"]))

                st.markdown("### 📡 Fon Sinyalleri")
                goster_kolonlar = ["Fon Kodu", "Fon Adı", "Toplam Skor", "Sinyal",
                                   "Yatırımcı Skoru", "Akış Skoru", "Dağılım Skoru",
                                   "Büyüklük Skoru", "Fon Büyüklüğü (TL)"]
                goster_kolonlar = [c for c in goster_kolonlar if c in df_signals.columns]

                def renk_sinyal(val):
                    renkler = {
                        "GÜÇLÜ AL": "background-color: #d4edda",
                        "AL": "background-color: #e8f5e9",
                        "NÖTR": "background-color: #fff3cd",
                        "ZAYIF": "background-color: #ffe0b2",
                        "UZAK DUR": "background-color: #f8d7da",
                    }
                    return renkler.get(val, "")

                df_filtreli = df_signals[df_signals["Toplam Skor"] >= min_skor]

                st.dataframe(
                    df_filtreli[goster_kolonlar].style.map(
                        renk_sinyal, subset=["Sinyal"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("### 💼 Önerilen Portföy")
                agirliklar = signal_to_portfolio(
                    df_signals, top_n=max_fon,
                    min_aum=min_aum_milyon * 1_000_000
                )

                if agirliklar:
                    portfoy_df = pd.DataFrame({
                        "Fon": list(agirliklar.keys()),
                        "Ağırlık (%)": [f"%{v*100:.1f}" for v in agirliklar.values()],
                        "100.000 TL Karşılığı": [f"{v*100000:,.0f} ₺"
                                                 for v in agirliklar.values()],
                    })
                    st.dataframe(portfoy_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("Portföy oluşturulamadı. Filtreleri gevşetin.")

                csv = df_signals.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Sinyalleri CSV İndir", csv,
                                   "tefas_sinyaller.csv", "text/csv")


# =============================================
# SEKME 2: FON DETAYI
# =============================================
with tab_fon_detay:
    st.subheader("📋 Fon Bazlı Detaylı Analiz")

    if not PYTEFAS_AVAILABLE:
        st.error("pytefas yüklü değil.")
    else:
        fon_kodu = st.text_input("Fon Kodu", value="AAK")

        if st.button("🔎 Fonu Analiz Et"):
            end = datetime.date.today()
            start = end - timedelta(days=analiz_gun + 30)

            with st.spinner(f"{fon_kodu} analiz ediliyor..."):
                provider = TefasDataProvider()
                provider.sync_fund_info(kind, start, end, force_refresh=force_refresh)
                provider.sync_breakdown(kind, start, end, force_refresh=force_refresh)

                df_info = provider.get_fund_history(fon_kodu, kind, start, end)
                df_breakdown = provider.get_fund_breakdown(fon_kodu, kind, start, end)

            if df_info.empty:
                st.warning(f"{fon_kodu} için veri yok. force_refresh'i deneyin.")
            else:
                son = df_info.iloc[-1]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Son Fiyat", f"{son.get('price', 0) or 0:.4f} ₺")
                c2.metric("Fon Büyüklüğü",
                          f"{son.get('portfolio_size', 0) or 0:,.0f} ₺")
                c3.metric("Yatırımcı Sayısı",
                          f"{son.get('investor_count', 0) or 0:,.0f}")
                c4.metric("Pay Sayısı",
                          f"{son.get('shares_outstanding', 0) or 0:,.0f}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_info["date"], y=df_info["price"],
                                         name="Fiyat", line=dict(color="green", width=2)))
                fig.update_layout(title=f"{fon_kodu} Fiyat Geçmişi",
                                  xaxis_title="Tarih", yaxis_title="Fiyat")
                st.plotly_chart(fig, use_container_width=True)

                if "investor_count" in df_info.columns:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=df_info["date"],
                                              y=df_info["investor_count"],
                                              name="Yatırımcı",
                                              line=dict(color="blue", width=2)))
                    fig2.update_layout(title="Yatırımcı Sayısı Trendi",
                                       xaxis_title="Tarih", yaxis_title="Kişi")
                    st.plotly_chart(fig2, use_container_width=True)

                if "portfolio_size" in df_info.columns:
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(x=df_info["date"],
                                              y=df_info["portfolio_size"],
                                              name="AUM", fill="tozeroy",
                                              line=dict(color="purple", width=2)))
                    fig3.update_layout(title="Fon Büyüklüğü Trendi",
                                       xaxis_title="Tarih", yaxis_title="TL")
                    st.plotly_chart(fig3, use_container_width=True)

                if not df_breakdown.empty:
                    st.markdown("### 🥧 Varlık Dağılımı (Son)")
                    son_dagilim = df_breakdown.iloc[-1]
                    pct_kolonlar = [c for c in df_breakdown.columns
                                    if c.endswith("_pct")
                                    and c not in ['date', 'fund_code', 'kind']]
                    if pct_kolonlar:
                        dagilim_data = []
                        for col in pct_kolonlar:
                            val = son_dagilim.get(col, 0)
                            if pd.notna(val) and val > 0:
                                dagilim_data.append({
                                    "Varlık Sınıfı": col.replace("_pct", "")
                                                       .replace("_", " ").title(),
                                    "Oran (%)": val
                                })
                        if dagilim_data:
                            df_dagilim = pd.DataFrame(dagilim_data)
                            fig_pie = px.pie(df_dagilim, values="Oran (%)",
                                             names="Varlık Sınıfı",
                                             title=f"{fon_kodu} Portföy Dağılımı")
                            st.plotly_chart(fig_pie, use_container_width=True)


# =============================================
# SEKME 3: NAKİT AKIŞI
# =============================================
with tab_akis:
    st.subheader("💰 Fon Nakit Akışları (Net Giriş/Çıkış)")

    if not PYTEFAS_AVAILABLE:
        st.error("pytefas yüklü değil.")
    else:
        if st.button("💸 Akış Analizini Çalıştır"):
            end = datetime.date.today()
            start = end - timedelta(days=analiz_gun + 30)

            with st.spinner("Nakit akışları hesaplanıyor..."):
                provider = TefasDataProvider()
                df_info = provider.sync_fund_info(kind, start, end,
                                                  force_refresh=force_refresh)

            if df_info.empty:
                st.warning("Veri yok.")
            else:
                df_flow = CashFlowAnalyzer.calculate_net_flow(df_info)
                df_flow_agg = CashFlowAnalyzer.aggregate_flow(
                    df_flow, window_days=analiz_gun
                )

                if df_flow_agg.empty:
                    st.warning("Akış hesaplanamadı. Daha uzun periyot seçin.")
                else:
                    df_flow_agg = df_flow_agg.sort_values(
                        "toplam_net_akis", ascending=False
                    )

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("#### 🟢 En Çok Para Giren")
                        top_in = df_flow_agg.head(10)[
                            ["fund_code", "toplam_net_akis", "aum_degisim_pct"]
                        ].copy()
                        top_in["toplam_net_akis"] = top_in["toplam_net_akis"].apply(
                            lambda x: f"{x:,.0f} ₺" if pd.notna(x) else "-"
                        )
                        st.dataframe(top_in, use_container_width=True, hide_index=True)

                    with c2:
                        st.markdown("#### 🔴 En Çok Para Çıkan")
                        top_out = df_flow_agg.tail(10)[
                            ["fund_code", "toplam_net_akis", "aum_degisim_pct"]
                        ].copy()
                        top_out["toplam_net_akis"] = top_out["toplam_net_akis"].apply(
                            lambda x: f"{x:,.0f} ₺" if pd.notna(x) else "-"
                        )
                        st.dataframe(top_out, use_container_width=True, hide_index=True)

                    head20 = df_flow_agg.head(20)
                    fig_flow = go.Figure()
                    fig_flow.add_trace(go.Bar(
                        x=head20["fund_code"],
                        y=head20["toplam_net_akis"],
                        name="Net Akış",
                        marker_color=["green" if v > 0 else "red"
                                      for v in head20["toplam_net_akis"]]
                    ))
                    fig_flow.update_layout(
                        title=f"Fon Bazında Net Nakit Akışı (Son {analiz_gun} Gün)",
                        xaxis_title="Fon", yaxis_title="Net Akış (TL)"
                    )
                    st.plotly_chart(fig_flow, use_container_width=True)


# =============================================
# SEKME 4: VERİTABANI YÖNETİMİ
# =============================================
with tab_db:
    st.subheader("🗄️ Veritabanı Durumu")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📊 Genel İstatistikler")
        try:
            conn = sqlite3.connect(DB_PATH)
            stats_info = pd.read_sql_query(
                """SELECT kind, COUNT(DISTINCT date) AS gun_sayisi,
                          COUNT(*) AS satir_sayisi,
                          MIN(date) AS ilk_tarih, MAX(date) AS son_tarih
                   FROM fund_info GROUP BY kind""", conn)
            stats_brk = pd.read_sql_query(
                """SELECT kind, COUNT(DISTINCT date) AS gun_sayisi,
                          COUNT(*) AS satir_sayisi,
                          MIN(date) AS ilk_tarih, MAX(date) AS son_tarih
                   FROM fund_breakdown GROUP BY kind""", conn)
            conn.close()

            st.markdown("**fund_info:**")
            st.dataframe(stats_info, use_container_width=True, hide_index=True)
            st.markdown("**fund_breakdown:**")
            st.dataframe(stats_brk, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"İstatistik okunamadı: {e}")

    with c2:
        st.markdown("#### 🧹 Bakım İşlemleri")
        st.caption(f"DB dosyası: `{os.path.abspath(DB_PATH)}`")
        if os.path.exists(DB_PATH):
            boyut_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
            st.metric("DB Boyutu", f"{boyut_mb:.2f} MB")

        sil_kind = st.selectbox("Silinecek fon tipi", ["YAT", "EMK", "BYF"], key="sil_kind")
        sil_tablo = st.selectbox("Silinecek tablo",
                                 ["fund_info", "fund_breakdown", "her ikisi"],
                                 key="sil_tablo")

        if st.button("🗑️ Seçili Veriyi Sil", type="secondary"):
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            if sil_tablo == "her ikisi":
                c.execute("DELETE FROM fund_info WHERE kind=?", (sil_kind,))
                c.execute("DELETE FROM fund_breakdown WHERE kind=?", (sil_kind,))
            else:
                c.execute(f"DELETE FROM {sil_tablo} WHERE kind=?", (sil_kind,))
            conn.commit()
            conn.close()
            st.success(f"{sil_kind} için {sil_tablo} silindi. Sayfayı yenileyin.")

        if st.button("⚠️ TÜM VERİYİ SİL", type="secondary"):
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("DELETE FROM fund_info")
            c.execute("DELETE FROM fund_breakdown")
            conn.commit()
            conn.close()
            st.warning("Tüm veri silindi. Sayfayı yenileyin.")