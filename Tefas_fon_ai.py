"""
TEFAS Kalıcı Öğrenmeli AI Fon Yönetim Sistemi
==============================================
Katmanlar:
  1. SQLite kalıcı depolama (fund_info, fund_breakdown, ai_signals,
     ai_learner_state, ai_regime_state)
  2. TEFAS veri sağlayıcı (pytefas) — incremental sync
  3. Akıllı Para Skoru (yatırımcı + akış + dağılım + büyüklük + yaş)
  4. Profesyonel metrikler (Sharpe, Sortino, Calmar, VaR, CVaR, Treynor,
     Info Ratio, Up/Down Capture)
  5. Adaptif pozisyon boyutlandırma (Kelly + Risk Parity)
  6. Online öğrenme (EMA + decay) — KALICI (SQLite)
  7. Rejim tespiti (Gaussian HMM) — KALICI (pickle BLOB)
  8. Sinyal füzyonu motoru
  9. Streamlit UI

Kurulum:
    pip install pytefas pandas numpy streamlit plotly yfinance hmmlearn scikit-learn

Çalıştırma:
    streamlit run tefas_ai_fund.py
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
import pickle
import sqlite3
import time

warnings.filterwarnings('ignore')

# =============================================
# 0. KONFİGÜRASYON VE OPSİYONEL KÜTÜPHANELER
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

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


# =============================================
# 1. SQLITE KATMANI
# =============================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Fon bilgileri
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
    # Varlık dağılımı (JSON)
    c.execute('''CREATE TABLE IF NOT EXISTS fund_breakdown (
        date TEXT NOT NULL,
        kind TEXT NOT NULL,
        fund_code TEXT NOT NULL,
        breakdown_json TEXT,
        PRIMARY KEY (date, kind, fund_code)
    )''')
    # AI sinyal geçmişi
    c.execute('''CREATE TABLE IF NOT EXISTS ai_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tarih TEXT,
        kind TEXT,
        regime INTEGER,
        regime_name TEXT,
        weights_json TEXT,
        metrics_json TEXT,
        learning_rate REAL,
        confidence REAL,
        created_at TEXT
    )''')
    # >>> KALICI ÖĞRENME: Online learner state
    c.execute('''CREATE TABLE IF NOT EXISTS ai_learner_state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        weights_json TEXT,
        learning_rate REAL,
        initial_lr REAL,
        decay REAL,
        history_json TEXT,
        updated_at TEXT
    )''')
    # >>> KALICI ÖĞRENME: HMM rejim model state
    c.execute('''CREATE TABLE IF NOT EXISTS ai_regime_state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        model_blob BLOB,
        updated_at TEXT
    )''')
    # İndeksler
    c.execute('CREATE INDEX IF NOT EXISTS idx_info_kind_date ON fund_info(kind, date)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_info_code ON fund_info(fund_code)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_breakdown_kind_date ON fund_breakdown(kind, date)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_breakdown_code ON fund_breakdown(fund_code)')
    conn.commit()
    conn.close()


init_db()


# ---- FUND_INFO / BREAKDOWN YARDIMCILARI ----
def get_last_date(kind, table='fund_info'):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            f"SELECT MAX(date) AS son FROM {table} WHERE kind=?",
            conn, params=(kind,))
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
            conn, params=(kind,))
        return int(df['n'].iloc[0])
    finally:
        conn.close()


def save_fund_info(df):
    if df.empty:
        return 0
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
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
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", rows)
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def save_breakdown(df):
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
        rows.append((row['date'], row['kind'], row['fund_code'],
                     json.dumps(breakdown, ensure_ascii=False)))
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executemany(
            """INSERT OR REPLACE INTO fund_breakdown
               (date, kind, fund_code, breakdown_json)
               VALUES (?, ?, ?, ?)""", rows)
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def load_fund_info(kind, start_date=None, end_date=None):
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
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    for c in ['price', 'shares_outstanding', 'investor_count', 'portfolio_size']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def load_breakdown(kind, start_date=None, end_date=None, fund_code=None):
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
    breakdowns = df['breakdown_json'].apply(lambda x: json.loads(x) if x else {})
    df_break = pd.json_normalize(breakdowns)
    df = pd.concat([df.drop(columns=['breakdown_json']), df_break], axis=1)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    return df


# ---- AI SİNYAL KAYDI ----
def save_ai_signal(signal, kind):
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("""INSERT INTO ai_signals
                     (tarih, kind, regime, regime_name, weights_json,
                      metrics_json, learning_rate, confidence, created_at)
                     VALUES (?,?,?,?,?,?,?,?,?)""",
                  (str(signal['date'])[:10], kind, int(signal['regime']),
                   signal['regime_name'],
                   json.dumps(signal['weights'], ensure_ascii=False),
                   json.dumps(signal['metrics'], ensure_ascii=False, default=str),
                   float(signal['learning_rate']),
                   float(signal['confidence']),
                   datetime.datetime.now().isoformat(timespec='seconds')))
        conn.commit()
    finally:
        conn.close()


def load_ai_signals(kind=None, limit=50):
    conn = sqlite3.connect(DB_PATH)
    try:
        if kind:
            df = pd.read_sql_query(
                "SELECT * FROM ai_signals WHERE kind=? ORDER BY id DESC LIMIT ?",
                conn, params=(kind, limit))
        else:
            df = pd.read_sql_query(
                "SELECT * FROM ai_signals ORDER BY id DESC LIMIT ?",
                conn, params=(limit,))
    finally:
        conn.close()
    return df


# =============================================
# 2. KALICI ÖĞRENME — ONLINE LEARNER STATE
# =============================================
def save_learner_state(learner):
    """Online learner state'ini DB'ye yazar."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""INSERT OR REPLACE INTO ai_learner_state
                     (id, weights_json, learning_rate, initial_lr,
                      decay, history_json, updated_at)
                     VALUES (1, ?, ?, ?, ?, ?, ?)""",
                  (json.dumps(learner.weights, ensure_ascii=False),
                   float(learner.learning_rate),
                   float(learner.initial_lr),
                   float(learner.decay),
                   json.dumps(learner.history[-500:], default=str),
                   datetime.datetime.now().isoformat(timespec='seconds')))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Learner state kaydedilemedi: {e}")
        return False


def load_learner_state(learner):
    """DB'den online learner state'ini yükler (varsa)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT weights_json, learning_rate, initial_lr, decay, "
                  "history_json FROM ai_learner_state WHERE id=1")
        row = c.fetchone()
        conn.close()
        if row is None:
            return False
        weights_json, lr, init_lr, decay, hist_json = row
        if weights_json:
            learner.weights = json.loads(weights_json)
        if lr is not None:
            learner.learning_rate = float(lr)
        if init_lr is not None:
            learner.initial_lr = float(init_lr)
        if decay is not None:
            learner.decay = float(decay)
        if hist_json:
            try:
                learner.history = json.loads(hist_json)
            except Exception:
                learner.history = []
        return True
    except Exception as e:
        print(f"Learner state yüklenemedi: {e}")
        return False


def reset_learner_state():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM ai_learner_state WHERE id=1")
    conn.commit()
    conn.close()


# =============================================
# 3. KALICI ÖĞRENME — HMM REJİM STATE
# =============================================
def save_regime_model(detector):
    if detector.model is None:
        return False
    try:
        blob = pickle.dumps(detector.model)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""INSERT OR REPLACE INTO ai_regime_state
                     (id, model_blob, updated_at)
                     VALUES (1, ?, ?)""",
                  (blob, datetime.datetime.now().isoformat(timespec='seconds')))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"HMM kaydedilemedi: {e}")
        return False


def load_regime_model(detector):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT model_blob FROM ai_regime_state WHERE id=1")
        row = c.fetchone()
        conn.close()
        if row and row[0]:
            detector.model = pickle.loads(row[0])
            return True
        return False
    except Exception as e:
        print(f"HMM yüklenemedi: {e}")
        return False


def reset_regime_model():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM ai_regime_state WHERE id=1")
    conn.commit()
    conn.close()


# =============================================
# 4. TEFAS VERİ SAĞLAYICI
# =============================================
class TefasDataProvider:
    def __init__(self):
        if not PYTEFAS_AVAILABLE:
            raise ImportError("pytefas bulunamadı. Kur: pip install pytefas")
        self.crawler = TefasCrawler()

    def sync_fund_info(self, kind, start_date, end_date,
                       force_refresh=False, progress_cb=None):
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()
        son_db = get_last_date(kind, 'fund_info')

        if force_refresh or son_db is None:
            fetch_start = start_date
        else:
            fetch_start = son_db + timedelta(days=1)

        if fetch_start > end_date:
            if progress_cb:
                progress_cb(f"Info: yeni veri yok (son: {son_db})")
            return load_fund_info(kind, start_date, end_date)

        if progress_cb:
            progress_cb(f"Çekiliyor (info): {fetch_start} → {end_date}")

        try:
            df_new = self.crawler.fetch(
                start=fetch_start.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                kind=kind, columns="info")
        except Exception as e:
            if progress_cb:
                progress_cb(f"Hata (info): {e}")
            df_new = pd.DataFrame()

        if not df_new.empty:
            if 'kind' not in df_new.columns:
                df_new['kind'] = kind
            n = save_fund_info(df_new)
            if progress_cb:
                progress_cb(f"✔ {n} yeni info kaydı")

        return load_fund_info(kind, start_date, end_date)

    def sync_breakdown(self, kind, start_date, end_date,
                       force_refresh=False, progress_cb=None):
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()
        son_db = get_last_date(kind, 'fund_breakdown')

        if force_refresh or son_db is None:
            fetch_start = start_date
        else:
            fetch_start = son_db + timedelta(days=1)

        if fetch_start > end_date:
            if progress_cb:
                progress_cb(f"Breakdown: yeni veri yok (son: {son_db})")
            return load_breakdown(kind, start_date, end_date)

        if progress_cb:
            progress_cb(f"Çekiliyor (breakdown): {fetch_start} → {end_date}")

        try:
            df_new = self.crawler.fetch(
                start=fetch_start.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                kind=kind, columns="breakdown")
        except Exception as e:
            if progress_cb:
                progress_cb(f"Hata (breakdown): {e}")
            df_new = pd.DataFrame()

        if not df_new.empty:
            if 'kind' not in df_new.columns:
                df_new['kind'] = kind
            n = save_breakdown(df_new)
            if progress_cb:
                progress_cb(f"✔ {n} yeni breakdown kaydı")

        return load_breakdown(kind, start_date, end_date)

    def get_fund_history(self, fund_code, kind, start_date, end_date):
        df = load_fund_info(kind, start_date, end_date)
        if df.empty:
            return df
        return df[df['fund_code'] == fund_code].copy()

    def get_fund_breakdown(self, fund_code, kind, start_date, end_date):
        return load_breakdown(kind, start_date, end_date, fund_code=fund_code)


# =============================================
# 5. NAKİT AKIŞI ANALİZCİSİ
# =============================================
class CashFlowAnalyzer:

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
    def calculate_net_flow(df_info):
        if df_info.empty:
            return df_info
        df = CashFlowAnalyzer._ensure_datetime(df_info)
        df = df.sort_values(["fund_code", "date"]).copy()
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
        agg["aum_degisim_pct"] = (agg["son_aum"] / agg["ilk_aum"] - 1) * 100
        return agg


# =============================================
# 6. AKILLI PARA SKORU
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
        if akis_pct > 5:    return 30, net_akis
        elif akis_pct > 2:  return 20, net_akis
        elif akis_pct > 0:  return 10, net_akis
        elif akis_pct > -2: return -10, net_akis
        else:               return -30, net_akis

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
        for col in [c for c in df.columns if "stock" in c.lower()
                    and c not in ['fund_code', 'date', 'kind']]:
            ilk = df_recent[col].iloc[0]
            son = df_recent[col].iloc[-1]
            if pd.notna(ilk) and pd.notna(son):
                degisim = son - ilk
                detay[col] = round(degisim, 2)
                if degisim > 2:    score += 10
                elif degisim > 0:  score += 5
                elif degisim < -2: score -= 10
        for col in [c for c in df.columns
                    if any(k in c.lower() for k in ["gold", "fx", "foreign_currency"])
                    and c not in ['fund_code', 'date', 'kind']]:
            ilk = df_recent[col].iloc[0]
            son = df_recent[col].iloc[-1]
            if pd.notna(ilk) and pd.notna(son):
                degisim = son - ilk
                detay[col] = round(degisim, 2)
                if degisim > 2:    score -= 10
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
        inv_s, inv_d = SmartMoneyScorer.calculate_investor_score(df_info, fund_code)
        flow_s, na = SmartMoneyScorer.calculate_flow_score(df_flow_agg, fund_code)
        alloc_s, ad = SmartMoneyScorer.calculate_allocation_score(df_breakdown, fund_code)
        size_s, aum = SmartMoneyScorer.calculate_size_score(df_info, fund_code)

        if not df_info.empty:
            n = len(df_info[df_info["fund_code"] == fund_code])
            yas = 10 if n > 200 else 7 if n > 100 else 4 if n > 50 else 0
        else:
            yas = 0

        toplam = float(np.clip(inv_s + flow_s + alloc_s + size_s + yas, 0, 100))
        return toplam, {
            "Yatırımcı Skoru": inv_s,
            "Yatırımcı Değişim (%)": round(inv_d, 2),
            "Akış Skoru": flow_s,
            "Net Akış (TL)": round(na, 2) if pd.notna(na) else 0,
            "Dağılım Skoru": alloc_s,
            "Büyüklük Skoru": size_s,
            "Fon Büyüklüğü (TL)": round(aum, 2) if pd.notna(aum) else 0,
            "Yaş/İstikrar Skoru": yas,
            "Toplam Skor": toplam,
        }


# =============================================
# 7. PROFESYONEL FON YÖNETİMİ METRİKLERİ
# =============================================
class ProfessionalMetrics:

    @staticmethod
    def calculate_all(df_prices, df_benchmark=None, risk_free=0.40,
                      window=90, confidence=0.95):
        if df_prices is None or df_prices.empty or len(df_prices) < window:
            return {}
        recent = df_prices.tail(window)
        returns = recent.pct_change().dropna()
        if returns.empty:
            return {}

        bench_returns = None
        if df_benchmark is not None and not df_benchmark.empty:
            b = df_benchmark.iloc[:, 0] if isinstance(df_benchmark, pd.DataFrame) else df_benchmark
            b = pd.to_numeric(b, errors='coerce').dropna()
            bench_returns = b.pct_change().dropna()

        metrics = {}
        for fund in returns.columns:
            r = returns[fund].dropna()
            if len(r) < 20:
                continue

            ann_return = (1 + r.mean()) ** 252 - 1
            ann_vol = r.std() * np.sqrt(252)
            excess = ann_return - risk_free
            sharpe = excess / (ann_vol + 1e-9)

            daily_rf = (1 + risk_free) ** (1 / 252) - 1
            neg = r[r < daily_rf] - daily_rf
            downside = np.sqrt((neg ** 2).mean()) * np.sqrt(252) if len(neg) > 0 else 1e-9
            sortino = excess / (downside + 1e-9)

            cum = (1 + r).cumprod()
            rolling_max = cum.cummax()
            dd = (cum - rolling_max) / rolling_max
            max_dd = dd.min()
            calmar = ann_return / (abs(max_dd) + 1e-9)

            var_95 = np.percentile(r, (1 - confidence) * 100)
            tail = r[r <= var_95]
            cvar_95 = tail.mean() if len(tail) > 0 else var_95

            treynor = info_ratio = up_capture = down_capture = beta = None

            if bench_returns is not None:
                aligned = pd.concat([r, bench_returns], axis=1).dropna()
                aligned.columns = ['fund', 'bench']
                if len(aligned) > 20:
                    cov = aligned['fund'].cov(aligned['bench'])
                    bench_var = aligned['bench'].var()
                    beta = cov / (bench_var + 1e-9)
                    treynor = excess / (beta + 1e-9)
                    active = aligned['fund'] - aligned['bench']
                    te = active.std() * np.sqrt(252)
                    info_ratio = (active.mean() * 252) / (te + 1e-9)
                    up = aligned[aligned['bench'] > 0]
                    down = aligned[aligned['bench'] < 0]
                    if len(up) > 0 and up['bench'].mean() != 0:
                        up_capture = up['fund'].mean() / up['bench'].mean()
                    if len(down) > 0 and down['bench'].mean() != 0:
                        down_capture = down['fund'].mean() / down['bench'].mean()

            metrics[fund] = {
                'Sharpe': round(sharpe, 3),
                'Sortino': round(sortino, 3),
                'Calmar': round(calmar, 3),
                'Max_DD': round(max_dd * 100, 2),
                'Ann_Return': round(ann_return * 100, 2),
                'Ann_Vol': round(ann_vol * 100, 2),
                'VaR_95': round(var_95 * 100, 3),
                'CVaR_95': round(cvar_95 * 100, 3),
                'Treynor': round(treynor, 3) if treynor is not None else None,
                'Beta': round(beta, 3) if beta is not None else None,
                'Info_Ratio': round(info_ratio, 3) if info_ratio is not None else None,
                'Up_Capture': round(up_capture, 3) if up_capture is not None else None,
                'Down_Capture': round(down_capture, 3) if down_capture is not None else None,
            }
        return metrics


# =============================================
# 8. ADAPTİF POZİSYON BOYUTLANDIRMA
# =============================================
class AdaptiveSizing:

    @staticmethod
    def risk_parity_weights(df_prices, window=60):
        if df_prices is None or df_prices.empty or len(df_prices) < window:
            return {}
        returns = df_prices.tail(window).pct_change().dropna()
        if returns.empty:
            return {}
        vols = returns.std() * np.sqrt(252)
        vols = vols.replace(0, np.nan).dropna()
        if vols.empty:
            return {}
        inv = 1.0 / (vols + 1e-6)
        w = inv / inv.sum()
        corr = returns[vols.index].corr()
        avg_corr = corr.mean()
        penalty = (1.0 / (1.0 + avg_corr - 0.5)).clip(0.5, 1.5)
        w = (w * penalty).dropna()
        if w.sum() == 0:
            return {}
        return (w / w.sum()).to_dict()

    @staticmethod
    def kelly_optimal_weights(metrics, max_weight=0.40, min_weight=0.02):
        if not metrics:
            return {}
        scores = {}
        for fund, m in metrics.items():
            if m.get('Sharpe') is None:
                continue
            s = np.clip(m['Sharpe'] / 3.0, 0, 1)
            so = np.clip(m['Sortino'] / 4.0, 0, 1)
            dd = 1.0 - np.clip(abs(m['Max_DD']) / 50.0, 0, 1)
            comp = 0.4 * s + 0.4 * so + 0.2 * dd
            scores[fund] = max(0, comp)
        if not scores or sum(scores.values()) == 0:
            return {}
        total = sum(scores.values())
        w = {f: s / total for f, s in scores.items()}
        w = {f: float(np.clip(x, min_weight, max_weight)) for f, x in w.items()}
        t = sum(w.values())
        return {f: x / t for f, x in w.items()}


# =============================================
# 9. ONLINE ÖĞRENME (KALICI)
# =============================================
class OnlineLearner:
    """EMA tabanlı online öğrenme, decay ile yakınsama.
    save()/load() metotları SQLite'a bağlanır."""

    def __init__(self, learning_rate=0.10, decay=0.995):
        self.learning_rate = float(learning_rate)
        self.initial_lr = float(learning_rate)
        self.decay = float(decay)
        self.weights = {}
        self.history = []

    def update(self, target_weights):
        if not target_weights:
            return self.weights
        if not self.weights:
            self.weights = dict(target_weights)
        else:
            all_f = set(self.weights) | set(target_weights)
            new_w = {}
            for f in all_f:
                old = self.weights.get(f, 0.0)
                tgt = target_weights.get(f, 0.0)
                new_w[f] = (1 - self.learning_rate) * old + self.learning_rate * tgt
            total = sum(new_w.values())
            if total > 0:
                new_w = {f: x / total for f, x in new_w.items()}
            self.weights = new_w
        self.learning_rate *= self.decay
        self.history.append({
            'n_funds': len(self.weights),
            'learning_rate': self.learning_rate,
            'ts': datetime.datetime.now().isoformat(timespec='seconds'),
        })
        return self.weights

    def get_confidence(self):
        n = len(self.history)
        if n < 10:
            return n / 10.0
        return min(1.0, 0.5 + n / 200.0)

    # --- Kalıcılık ---
    def save(self):
        return save_learner_state(self)

    def load(self):
        return load_learner_state(self)

    def reset(self):
        self.weights = {}
        self.learning_rate = self.initial_lr
        self.history = []
        reset_learner_state()


# =============================================
# 10. REJİM TESPİTİ (KALICI)
# =============================================
class RegimeDetector:
    def __init__(self, n_regimes=3, lookback=252):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.model = None

    def fit(self, df_benchmark):
        if not HMM_AVAILABLE or df_benchmark is None or df_benchmark.empty:
            return self
        b = df_benchmark.iloc[:, 0] if isinstance(df_benchmark, pd.DataFrame) else df_benchmark
        b = pd.to_numeric(b, errors='coerce').dropna()
        returns = b.pct_change().dropna()
        if len(returns) < self.lookback // 2:
            return self
        feats = pd.DataFrame({
            'return': returns,
            'vol': returns.rolling(20).std(),
        }).dropna()
        if len(feats) < 50:
            return self
        try:
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=42)
            self.model.fit(feats.values)
            save_regime_model(self)  # Otomatik kaydet
        except Exception:
            self.model = None
        return self

    def predict_current_regime(self, df_benchmark):
        if self.model is None or df_benchmark is None or df_benchmark.empty:
            return 1
        b = df_benchmark.iloc[:, 0] if isinstance(df_benchmark, pd.DataFrame) else df_benchmark
        b = pd.to_numeric(b, errors='coerce').dropna()
        returns = b.pct_change().dropna()
        if len(returns) < 20:
            return 1
        feats = pd.DataFrame({
            'return': returns,
            'vol': returns.rolling(20).std(),
        }).dropna()
        if feats.empty:
            return 1
        try:
            raw = self.model.predict(feats.values)[-1]
            means = self.model.means_[:, 1]
            order = np.argsort(means)
            mapping = {order[0]: 0, order[1]: 1, order[2]: 2}
            return int(mapping.get(raw, 1))
        except Exception:
            return 1

    @staticmethod
    def get_regime_name(regime):
        return {0: "🐂 BOĞA", 1: "😐 NÖTR", 2: "🐻 AYI"}.get(regime, "BİLİNMİYOR")

    # --- Kalıcılık ---
    def save(self):
        return save_regime_model(self)

    def load(self):
        return load_regime_model(self)

    def reset(self):
        self.model = None
        reset_regime_model()


# =============================================
# 11. SİNYAL ÜRETİCİ (klasik skor tablosu)
# =============================================
class SignalGenerator:
    @staticmethod
    def generate_signals(df_info, df_flow_agg, df_breakdown):
        if df_info.empty:
            return pd.DataFrame()
        fonlar = df_info["fund_code"].dropna().unique()
        rows = []
        for fon in fonlar:
            try:
                skor, detay = SmartMoneyScorer.calculate_total_score(
                    df_info, df_flow_agg, df_breakdown, fon)
                if skor >= 70:   sinyal = "GÜÇLÜ AL"
                elif skor >= 50: sinyal = "AL"
                elif skor >= 40: sinyal = "NÖTR"
                elif skor >= 25: sinyal = "ZAYIF"
                else:            sinyal = "UZAK DUR"
                f_df = df_info[df_info["fund_code"] == fon]
                ad = f_df["fund_name"].iloc[-1] if not f_df.empty and "fund_name" in f_df.columns else ""
                rows.append({
                    "Fon Kodu": fon, "Fon Adı": ad or "",
                    "Toplam Skor": round(skor, 1),
                    "Sinyal": sinyal,
                    **{k: v for k, v in detay.items() if k != "Toplam Skor"},
                })
            except Exception:
                continue
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("Toplam Skor", ascending=False)
        return df


# =============================================
# 12. KENDİNİ GELİŞTİREN SİNYAL MOTORU (FÜZYON)
# =============================================
class SelfLearningSignalEngine:
    def __init__(self, learning_rate=0.10, decay=0.995, auto_load=True):
        self.online = OnlineLearner(learning_rate=learning_rate, decay=decay)
        self.regime_detector = RegimeDetector()
        self.last_signal = None

        # Başlangıçta DB'den state yükle
        if auto_load:
            loaded_learner = self.online.load()
            loaded_regime = self.regime_detector.load()
            self._loaded_from_db = bool(loaded_learner or loaded_regime)
        else:
            self._loaded_from_db = False

    def generate(self, df_info, df_breakdown, df_benchmark=None,
                 window=90, top_n=5, regime_weights=None):
        if regime_weights is None:
            regime_weights = {0: 1.0, 1: 0.8, 2: 0.5}

        df_prices = df_info.pivot_table(
            index='date', columns='fund_code', values='price').ffill()
        if df_prices.empty:
            return None
        df_prices = df_prices.dropna(axis=1, how='all')
        df_prices = df_prices.loc[:, df_prices.notna().sum() >= 20]
        if df_prices.empty:
            return None

        # Katman 1: Profesyonel metrikler
        metrics = ProfessionalMetrics.calculate_all(
            df_prices, df_benchmark, risk_free=0.40, window=window)

        # Katman 2: Akıllı para skoru
        df_flow = CashFlowAnalyzer.calculate_net_flow(df_info)
        df_flow_agg = CashFlowAnalyzer.aggregate_flow(df_flow, window_days=window)
        smart_scores = {}
        for fund in df_prices.columns:
            try:
                s, _ = SmartMoneyScorer.calculate_total_score(
                    df_info, df_flow_agg, df_breakdown, fund)
                smart_scores[fund] = s
            except Exception:
                smart_scores[fund] = 0.0

        # Katman 3: Adaptif boyutlandırma
        kelly_w = AdaptiveSizing.kelly_optimal_weights(metrics)
        rp_w = AdaptiveSizing.risk_parity_weights(df_prices, window=min(60, window))

        # Katman 4: Online öğrenme
        target = {}
        all_f = set(kelly_w) | set(smart_scores)
        for f in all_f:
            k = kelly_w.get(f, 0.0)
            s = smart_scores.get(f, 0.0) / 100.0
            target[f] = 0.5 * k + 0.5 * s
        if sum(target.values()) > 0:
            tot = sum(target.values())
            target = {f: v / tot for f, v in target.items()}
        online_w = self.online.update(target)
        self.online.save()  # >>> Her güncellemede DB'ye kaydet

        # Katman 5: Rejim tespiti
        if self.regime_detector.model is None:
            self.regime_detector.fit(df_benchmark)
        regime = self.regime_detector.predict_current_regime(df_benchmark)
        regime_name = self.regime_detector.get_regime_name(regime)
        regime_mult = regime_weights.get(regime, 1.0)

        # Sinyal füzyonu
        all_funds = set(kelly_w) | set(online_w) | set(rp_w) | set(smart_scores)
        if not all_funds:
            return None

        fused = {}
        for f in all_funds:
            k = kelly_w.get(f, 0.0)
            o = online_w.get(f, 0.0)
            r = rp_w.get(f, 0.0)
            s = smart_scores.get(f, 0.0) / 100.0
            comp = 0.35 * k + 0.25 * o + 0.20 * r + 0.20 * s
            fused[f] = max(0.0, comp * regime_mult)

        total = sum(fused.values())
        if total <= 0:
            return None
        fused = {f: v / total for f, v in fused.items()}
        top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_n]
        final = dict(top)
        tot = sum(final.values())
        final = {f: v / tot for f, v in final.items()}

        signal = {
            'date': df_prices.index[-1],
            'regime': regime,
            'regime_name': regime_name,
            'weights': final,
            'metrics': {f: metrics.get(f, {}) for f in final},
            'smart_scores': {f: smart_scores.get(f, 0) for f in final},
            'learning_rate': self.online.learning_rate,
            'confidence': self.online.get_confidence(),
            'n_funds_analyzed': len(all_funds),
        }
        self.last_signal = signal
        return signal

    @staticmethod
    def explain(signal):
        if not signal:
            return "Sinyal üretilemedi."
        lines = [
            f"📅 **Tarih:** {signal['date']}",
            f"🚦 **Rejim:** {signal['regime_name']}",
            f"🎯 **Güven:** {signal['confidence']:.0%}",
            f"📚 **Öğrenme Oranı:** {signal['learning_rate']:.4f}",
            f"🔍 **Analiz Edilen Fon:** {signal['n_funds_analyzed']}",
            "",
            "**Önerilen Portföy:**",
        ]
        for f, w in signal['weights'].items():
            m = signal['metrics'].get(f, {})
            s = signal['smart_scores'].get(f, 0)
            lines.append(
                f"• **{f}** → %{w*100:.1f} "
                f"(Sharpe: {m.get('Sharpe', 'N/A')}, "
                f"Sortino: {m.get('Sortino', 'N/A')}, "
                f"Max DD: {m.get('Max_DD', 'N/A')}%, "
                f"Akıllı Para: {s:.0f})")
        return "\n".join(lines)

    def reset_all(self):
        self.online.reset()
        self.regime_detector.reset()
        self._loaded_from_db = False

    def get_state_summary(self):
        return {
            'weights_count': len(self.online.weights),
            'learning_rate': self.online.learning_rate,
            'initial_lr': self.online.initial_lr,
            'decay': self.online.decay,
            'history_length': len(self.online.history),
            'has_regime_model': self.regime_detector.model is not None,
            'loaded_from_db': self._loaded_from_db,
        }


# =============================================
# 13. STREAMLIT ARAYÜZ
# =============================================
st.set_page_config(page_title="TEFAS AI Fon Yönetimi", layout="wide")
st.title("🧠 TEFAS Kalıcı Öğrenmeli AI Fon Yönetim Sistemi")
st.caption("Akıllı Para + Profesyonel Metrikler + Kelly/Risk Parity + Online Öğrenme + HMM Rejim")

# Engine'i session state'te tut (öğrenme KALICI olarak DB'de saklı)
if 'engine' not in st.session_state:
    st.session_state.engine = SelfLearningSignalEngine(
        learning_rate=0.10,
        decay=0.995,
        auto_load=True,
    )

with st.sidebar:
    st.header("⚙️ Ayarlar")

    fon_tipi = st.selectbox(
        "Fon Tipi",
        ["YAT (Yatırım Fonları)", "EMK (Emeklilik Fonları)", "BYF (Borsa Yatırım Fonları)"],
        index=0)
    kind_map = {"YAT (Yatırım Fonları)": "YAT",
                "EMK (Emeklilik Fonları)": "EMK",
                "BYF (Borsa Yatırım Fonları)": "BYF"}
    kind = kind_map[fon_tipi]

    analiz_gun = st.slider("Analiz Periyodu (Gün)", 30, 365, 90, 30)
    window_metrics = st.slider("Metrik Penceresi (Gün)", 60, 252, 90, 15)

    st.divider()
    st.subheader("🔄 Veri Senkronizasyonu")
    force_refresh = st.checkbox(
        "Tüm veriyi baştan çek", value=False,
        help="İşaretlenirse DB'deki tüm veri yeniden çekilir. "
             "Normalde sadece yeni tarihler çekilir.")

    try:
        n_info = get_total_rows(kind, 'fund_info')
        n_brk = get_total_rows(kind, 'fund_breakdown')
        son_info = get_last_date(kind, 'fund_info')
        son_brk = get_last_date(kind, 'fund_breakdown')
        st.info(
            f"📦 **DB ({kind})**\n\n"
            f"• Info: {n_info:,} satır\n"
            f"• Breakdown: {n_brk:,} satır\n"
            f"• Son info: `{son_info or '—'}`\n"
            f"• Son breakdown: `{son_brk or '—'}`")
    except Exception as e:
        st.warning(f"DB: {e}")

    st.divider()
    st.subheader("🧠 Motor State")

    try:
        state = st.session_state.engine.get_state_summary()
        ca, cb = st.columns(2)
        ca.metric("Fon Sayısı", state['weights_count'])
        cb.metric("Geçmiş Adım", state['history_length'])
        st.caption(f"Öğrenme Oranı: `{state['learning_rate']:.4f}` "
                   f"(başlangıç: `{state['initial_lr']:.4f}`)")
        st.caption(f"Decay: `{state['decay']}`")
        if state['has_regime_model']:
            st.success("✅ HMM modeli yüklü")
        else:
            st.warning("⚠️ HMM henüz eğitilmedi")
        if state['loaded_from_db']:
            st.info("💾 Önceki oturumdan yüklendi")
        else:
            st.info("🆕 İlk çalıştırma")
    except Exception as e:
        st.warning(f"State: {e}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("♻️ Sıfırla", help="Öğrenme geçmişini sıfırlar"):
            st.session_state.engine.reset_all()
            st.success("Sıfırlandı.")
            st.rerun()
    with c2:
        if st.button("💾 Kaydet", help="State'i DB'ye yazar"):
            st.session_state.engine.online.save()
            st.session_state.engine.regime_detector.save()
            st.success("Kaydedildi.")

    st.divider()
    st.subheader("🎛️ Motor Parametreleri")
    learning_rate = st.slider("Online Öğrenme Oranı", 0.01, 0.30, 0.10, 0.01)
    decay = st.slider("Öğrenme Decay", 0.990, 1.000, 0.995, 0.001)
    top_n = st.slider("Maksimum Fon", 3, 15, 5, 1)
    min_skor = st.slider("Minimum Akıllı Para Skoru", 0, 100, 40, 5)
    min_aum_milyon = st.slider("Minimum Fon Büyüklüğü (Milyon TL)", 10, 1000, 50, 10)

    if st.button("⚙️ Parametreleri Uygula"):
        engine = st.session_state.engine
        engine.online.learning_rate = learning_rate
        engine.online.decay = decay
        engine.online.save()
        st.success(f"Yeni LR: {learning_rate}, Decay: {decay}")

    st.divider()
    st.caption("⚠️ Bu araç yatırım tavsiyesi değildir.")


tab_ai, tab_sinyal, tab_fon_detay, tab_akis, tab_db = st.tabs(
    ["🧠 AI Motoru", "📡 Sinyaller", "📋 Fon Detayı", "💰 Nakit Akışı", "🗄️ Veritabanı"])


# =============================================
# SEKME: AI MOTORU
# =============================================
with tab_ai:
    st.subheader("🧠 Kendini Geliştiren Sinyal Motoru")
    st.caption("Profesyonel metrikler + Online öğrenme + HMM rejim + Kelly/Risk Parity füzyonu")

    if not PYTEFAS_AVAILABLE:
        st.error("❌ `pytefas` yüklü değil. Terminalde `pip install pytefas` çalıştırın.")
    else:
        if st.button("🧠 AI Motorunu Çalıştır", type="primary"):
            end = datetime.date.today()
            start = end - timedelta(days=analiz_gun + 90)
            log_box = st.empty()
            logs = []

            def log_cb(msg):
                logs.append(msg)
                log_box.info("📋 " + " | ".join(logs[-3:]))

            with st.spinner("Veri senkronize ediliyor..."):
                try:
                    provider = TefasDataProvider()
                except Exception as e:
                    st.error(f"Provider: {e}")
                    st.stop()

                df_info = provider.sync_fund_info(kind, start, end, force_refresh, log_cb)
                if df_info.empty:
                    st.error("Info verisi yok.")
                    st.stop()
                df_breakdown = provider.sync_breakdown(kind, start, end, force_refresh, log_cb)

            df_bench = pd.DataFrame()
            if YF_AVAILABLE:
                try:
                    b = yf.download("XU100.IS", start=start, end=end,
                                    progress=False, auto_adjust=True)['Close']
                    df_bench = b.to_frame("XU100") if isinstance(b, pd.Series) else b
                except Exception:
                    df_bench = pd.DataFrame()

            with st.spinner("AI motoru hesaplıyor..."):
                signal = st.session_state.engine.generate(
                    df_info, df_breakdown, df_bench,
                    window=window_metrics, top_n=top_n)

            log_box.success("✅ AI motoru tamamlandı. Öğrenme state'i DB'ye kaydedildi.")

            if signal:
                regime_icons = {0: "🟢", 1: "🟡", 2: "🔴"}
                st.markdown(f"## {regime_icons.get(signal['regime'], '⚪')} "
                            f"{signal['regime_name']}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Güven", f"{signal['confidence']:.0%}")
                c2.metric("Öğrenme Oranı", f"{signal['learning_rate']:.4f}")
                c3.metric("Analiz Edilen", signal['n_funds_analyzed'])
                c4.metric("Seçilen Fon", len(signal['weights']))

                sinyal_rows = []
                for f, w in signal['weights'].items():
                    m = signal['metrics'].get(f, {})
                    sinyal_rows.append({
                        'Fon': f,
                        'Ağırlık (%)': f"%{w*100:.1f}",
                        'Sharpe': m.get('Sharpe', '-'),
                        'Sortino': m.get('Sortino', '-'),
                        'Calmar': m.get('Calmar', '-'),
                        'Max DD (%)': m.get('Max_DD', '-'),
                        'VaR 95%': m.get('VaR_95', '-'),
                        'CVaR 95%': m.get('CVaR_95', '-'),
                        'Info Ratio': m.get('Info_Ratio', '-'),
                        'Akıllı Para': f"{signal['smart_scores'].get(f, 0):.0f}",
                    })
                st.dataframe(pd.DataFrame(sinyal_rows),
                             use_container_width=True, hide_index=True)

                st.markdown("### 💰 100.000 TL İçin Örnek Dağıtım")
                df_prices_check = df_info.pivot_table(
                    index='date', columns='fund_code', values='price').ffill()
                son_fiyat = df_prices_check.iloc[-1].to_dict() if not df_prices_check.empty else {}
                dagitim = []
                for f, w in signal['weights'].items():
                    tutar = w * 100000
                    fp = son_fiyat.get(f, np.nan)
                    adet = tutar / fp if fp and not pd.isna(fp) and fp > 0 else 0
                    dagitim.append({
                        'Fon': f,
                        'TL Tutar': f"{tutar:,.2f} ₺",
                        'Son Fiyat': f"{fp:,.4f}" if fp and not pd.isna(fp) else "-",
                        'Adet (yaklaşık)': f"{adet:,.4f}" if adet else "-",
                    })
                st.dataframe(pd.DataFrame(dagitim),
                             use_container_width=True, hide_index=True)

                st.markdown("### 📝 AI Açıklaması")
                st.markdown(SelfLearningSignalEngine.explain(signal))

                if st.button("💾 Sinyali Veritabanına Kaydet"):
                    save_ai_signal(signal, kind)
                    st.success("Sinyal kaydedildi.")
            else:
                st.warning("Sinyal üretilemedi. Filtreleri gevşetin.")


# =============================================
# SEKME: SİNYALLER
# =============================================
with tab_sinyal:
    st.subheader(f"Akıllı Para Skorları — {fon_tipi}")

    if not PYTEFAS_AVAILABLE:
        st.error("pytefas yüklü değil.")
    else:
        if st.button("🔍 Analizi Çalıştır", type="primary"):
            end = datetime.date.today()
            start = end - timedelta(days=analiz_gun + 30)
            log_box = st.empty()
            logs = []

            def log_cb(msg):
                logs.append(msg)
                log_box.info("📋 " + " | ".join(logs[-3:]))

            with st.spinner("Veri senkronize ediliyor..."):
                provider = TefasDataProvider()
                df_info = provider.sync_fund_info(kind, start, end, force_refresh, log_cb)
                df_breakdown = provider.sync_breakdown(kind, start, end, force_refresh, log_cb)

            if df_info.empty:
                st.error("Veri yok.")
                st.stop()

            with st.spinner("Hesaplanıyor..."):
                df_flow = CashFlowAnalyzer.calculate_net_flow(df_info)
                df_flow_agg = CashFlowAnalyzer.aggregate_flow(df_flow, analiz_gun)
                df_signals = SignalGenerator.generate_signals(
                    df_info, df_flow_agg, df_breakdown)

            log_box.success("✅ Tamamlandı.")

            if df_signals.empty:
                st.warning("Sinyal yok.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Toplam Fon", len(df_signals))
                c2.metric("Güçlü Al", len(df_signals[df_signals["Sinyal"] == "GÜÇLÜ AL"]))
                c3.metric("Al", len(df_signals[df_signals["Sinyal"] == "AL"]))
                c4.metric("Uzak Dur", len(df_signals[df_signals["Sinyal"] == "UZAK DUR"]))

                goster = [c for c in ["Fon Kodu", "Fon Adı", "Toplam Skor", "Sinyal",
                                      "Yatırımcı Skoru", "Akış Skoru", "Dağılım Skoru",
                                      "Büyüklük Skoru", "Fon Büyüklüğü (TL)"]
                          if c in df_signals.columns]

                def renk(val):
                    return {
                        "GÜÇLÜ AL": "background-color: #d4edda",
                        "AL": "background-color: #e8f5e9",
                        "NÖTR": "background-color: #fff3cd",
                        "ZAYIF": "background-color: #ffe0b2",
                        "UZAK DUR": "background-color: #f8d7da",
                    }.get(val, "")

                df_f = df_signals[df_signals["Toplam Skor"] >= min_skor]
                st.dataframe(df_f[goster].style.map(renk, subset=["Sinyal"]),
                             use_container_width=True, hide_index=True)

                csv = df_signals.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ CSV İndir", csv,
                                   "tefas_sinyaller.csv", "text/csv")


# =============================================
# SEKME: FON DETAYI
# =============================================
with tab_fon_detay:
    st.subheader("📋 Fon Bazlı Detaylı Analiz")

    if not PYTEFAS_AVAILABLE:
        st.error("pytefas yüklü değil.")
    else:
        fon_kodu = st.text_input("Fon Kodu", value="AAK")

        if st.button("🔎 Fonu Analiz Et"):
            end = datetime.date.today()
            start = end - timedelta(days=analiz_gun + 90)
            with st.spinner(f"{fon_kodu} analiz ediliyor..."):
                provider = TefasDataProvider()
                provider.sync_fund_info(kind, start, end, force_refresh)
                provider.sync_breakdown(kind, start, end, force_refresh)
                df_info = provider.get_fund_history(fon_kodu, kind, start, end)
                df_bd = provider.get_fund_breakdown(fon_kodu, kind, start, end)

            if df_info.empty:
                st.warning(f"{fon_kodu} verisi yok.")
            else:
                son = df_info.iloc[-1]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Son Fiyat", f"{son.get('price', 0) or 0:.4f} ₺")
                c2.metric("AUM", f"{son.get('portfolio_size', 0) or 0:,.0f} ₺")
                c3.metric("Yatırımcı", f"{son.get('investor_count', 0) or 0:,.0f}")
                c4.metric("Pay", f"{son.get('shares_outstanding', 0) or 0:,.0f}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_info["date"], y=df_info["price"],
                                         name="Fiyat", line=dict(color="green", width=2)))
                fig.update_layout(title=f"{fon_kodu} Fiyat",
                                  xaxis_title="Tarih", yaxis_title="Fiyat")
                st.plotly_chart(fig, use_container_width=True)

                if "investor_count" in df_info.columns:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=df_info["date"],
                                              y=df_info["investor_count"],
                                              name="Yatırımcı",
                                              line=dict(color="blue", width=2)))
                    fig2.update_layout(title="Yatırımcı Sayısı",
                                       xaxis_title="Tarih", yaxis_title="Kişi")
                    st.plotly_chart(fig2, use_container_width=True)

                if "portfolio_size" in df_info.columns:
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(x=df_info["date"],
                                              y=df_info["portfolio_size"],
                                              name="AUM", fill="tozeroy",
                                              line=dict(color="purple", width=2)))
                    fig3.update_layout(title="Fon Büyüklüğü",
                                       xaxis_title="Tarih", yaxis_title="TL")
                    st.plotly_chart(fig3, use_container_width=True)

                if len(df_info) > 20:
                    st.markdown("### 📊 Profesyonel Metrikler")
                    pivot = df_info.pivot_table(
                        index='date', columns='fund_code', values='price').ffill()
                    df_bench_local = pd.DataFrame()
                    if YF_AVAILABLE:
                        try:
                            b = yf.download("XU100.IS", start=start, end=end,
                                            progress=False, auto_adjust=True)['Close']
                            if isinstance(b, pd.Series):
                                df_bench_local = b.to_frame("XU100")
                        except Exception:
                            pass
                    m = ProfessionalMetrics.calculate_all(
                        pivot, df_bench_local, risk_free=0.40, window=window_metrics)
                    if fon_kodu in m:
                        mm = m[fon_kodu]
                        cols = st.columns(4)
                        cols[0].metric("Sharpe", mm.get('Sharpe', '-'))
                        cols[1].metric("Sortino", mm.get('Sortino', '-'))
                        cols[2].metric("Calmar", mm.get('Calmar', '-'))
                        cols[3].metric("Max DD", f"%{mm.get('Max_DD', '-')}")
                        cols2 = st.columns(4)
                        cols2[0].metric("VaR 95%", f"%{mm.get('VaR_95', '-')}")
                        cols2[1].metric("CVaR 95%", f"%{mm.get('CVaR_95', '-')}")
                        cols2[2].metric("Info Ratio", mm.get('Info_Ratio', '-'))
                        cols2[3].metric("Beta", mm.get('Beta', '-'))

                if not df_bd.empty:
                    st.markdown("### 🥧 Varlık Dağılımı (Son)")
                    son_d = df_bd.iloc[-1]
                    pct_kolonlar = [c for c in df_bd.columns
                                    if c.endswith("_pct")
                                    and c not in ['date', 'fund_code', 'kind']]
                    if pct_kolonlar:
                        dd = []
                        for col in pct_kolonlar:
                            v = son_d.get(col, 0)
                            if pd.notna(v) and v > 0:
                                dd.append({
                                    "Varlık": col.replace("_pct", "").replace("_", " ").title(),
                                    "Oran (%)": v})
                        if dd:
                            fig_pie = px.pie(pd.DataFrame(dd),
                                             values="Oran (%)", names="Varlık",
                                             title=f"{fon_kodu} Dağılım")
                            st.plotly_chart(fig_pie, use_container_width=True)


# =============================================
# SEKME: NAKİT AKIŞI
# =============================================
with tab_akis:
    st.subheader("💰 Fon Nakit Akışları")

    if not PYTEFAS_AVAILABLE:
        st.error("pytefas yüklü değil.")
    else:
        if st.button("💸 Akış Analizini Çalıştır"):
            end = datetime.date.today()
            start = end - timedelta(days=analiz_gun + 30)
            with st.spinner("Hesaplanıyor..."):
                provider = TefasDataProvider()
                df_info = provider.sync_fund_info(kind, start, end, force_refresh)
            if df_info.empty:
                st.warning("Veri yok.")
            else:
                df_flow = CashFlowAnalyzer.calculate_net_flow(df_info)
                df_agg = CashFlowAnalyzer.aggregate_flow(df_flow, analiz_gun)
                if df_agg.empty:
                    st.warning("Akış hesaplanamadı.")
                else:
                    df_agg = df_agg.sort_values("toplam_net_akis", ascending=False)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("#### 🟢 En Çok Giren")
                        ti = df_agg.head(10)[
                            ["fund_code", "toplam_net_akis", "aum_degisim_pct"]].copy()
                        ti["toplam_net_akis"] = ti["toplam_net_akis"].apply(
                            lambda x: f"{x:,.0f} ₺" if pd.notna(x) else "-")
                        st.dataframe(ti, use_container_width=True, hide_index=True)
                    with c2:
                        st.markdown("#### 🔴 En Çok Çıkan")
                        to = df_agg.tail(10)[
                            ["fund_code", "toplam_net_akis", "aum_degisim_pct"]].copy()
                        to["toplam_net_akis"] = to["toplam_net_akis"].apply(
                            lambda x: f"{x:,.0f} ₺" if pd.notna(x) else "-")
                        st.dataframe(to, use_container_width=True, hide_index=True)

                    h20 = df_agg.head(20)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=h20["fund_code"], y=h20["toplam_net_akis"],
                        marker_color=["green" if v > 0 else "red"
                                      for v in h20["toplam_net_akis"]]))
                    fig.update_layout(title=f"Net Akış (Son {analiz_gun} Gün)",
                                      xaxis_title="Fon", yaxis_title="TL")
                    st.plotly_chart(fig, use_container_width=True)


# =============================================
# SEKME: VERİTABANI
# =============================================
with tab_db:
    st.subheader("🗄️ Veritabanı Yönetimi")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📊 İstatistikler")
        try:
            conn = sqlite3.connect(DB_PATH)
            si = pd.read_sql_query(
                """SELECT kind, COUNT(DISTINCT date) AS gun, COUNT(*) AS satir,
                          MIN(date) AS ilk, MAX(date) AS son
                   FROM fund_info GROUP BY kind""", conn)
            sb = pd.read_sql_query(
                """SELECT kind, COUNT(DISTINCT date) AS gun, COUNT(*) AS satir,
                          MIN(date) AS ilk, MAX(date) AS son
                   FROM fund_breakdown GROUP BY kind""", conn)
            ss = pd.read_sql_query("SELECT COUNT(*) AS n FROM ai_signals", conn)
            conn.close()
            st.markdown("**fund_info:**")
            st.dataframe(si, use_container_width=True, hide_index=True)
            st.markdown("**fund_breakdown:**")
            st.dataframe(sb, use_container_width=True, hide_index=True)
            st.metric("AI Sinyalleri", int(ss['n'].iloc[0]))
        except Exception as e:
            st.error(f"Okunamadı: {e}")

    with c2:
        st.markdown("#### 🧹 Bakım")
        st.caption(f"DB: `{os.path.abspath(DB_PATH)}`")
        if os.path.exists(DB_PATH):
            st.metric("Boyut", f"{os.path.getsize(DB_PATH) / 1024 / 1024:.2f} MB")

        sil_kind = st.selectbox("Fon tipi", ["YAT", "EMK", "BYF"], key="sk")
        sil_tablo = st.selectbox(
            "Tablo", ["fund_info", "fund_breakdown", "ai_signals", "her ikisi"],
            key="st")
        if st.button("🗑️ Seçili Veriyi Sil", type="secondary"):
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            if sil_tablo == "her ikisi":
                c.execute("DELETE FROM fund_info WHERE kind=?", (sil_kind,))
                c.execute("DELETE FROM fund_breakdown WHERE kind=?", (sil_kind,))
            elif sil_tablo == "ai_signals":
                c.execute("DELETE FROM ai_signals WHERE kind=?", (sil_kind,))
            else:
                c.execute(f"DELETE FROM {sil_tablo} WHERE kind=?", (sil_kind,))
            conn.commit()
            conn.close()
            st.success(f"{sil_kind} / {sil_tablo} silindi.")

        if st.button("⚠️ TÜMÜNÜ SİL", type="secondary"):
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("DELETE FROM fund_info")
            c.execute("DELETE FROM fund_breakdown")
            c.execute("DELETE FROM ai_signals")
            conn.commit()
            conn.close()
            st.warning("Her şey silindi.")

    # Kalıcı öğrenme state
    st.divider()
    st.markdown("### 🧠 Kalıcı Öğrenme State")
    try:
        conn = sqlite3.connect(DB_PATH)
        df_state = pd.read_sql_query(
            "SELECT weights_json, learning_rate, initial_lr, decay, "
            "history_json, updated_at FROM ai_learner_state WHERE id=1", conn)
        df_regime = pd.read_sql_query(
            "SELECT updated_at FROM ai_regime_state WHERE id=1", conn)
        conn.close()

        if df_state.empty:
            st.info("Henüz kayıtlı öğrenme state'i yok.")
        else:
            row = df_state.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Öğrenme Oranı", f"{row['learning_rate']:.4f}")
            c2.metric("İlk LR", f"{row['initial_lr']:.4f}")
            c3.metric("Decay", f"{row['decay']}")

            weights = json.loads(row['weights_json']) if row['weights_json'] else {}
            st.caption(f"**Öğrenilen fon sayısı:** {len(weights)}")
            st.caption(f"**Son güncelleme:** {row['updated_at']}")

            if weights:
                df_w = pd.DataFrame([
                    {'Fon': k, 'Ağırlık': f"%{v*100:.2f}"}
                    for k, v in sorted(weights.items(),
                                       key=lambda x: x[1], reverse=True)[:20]])
                st.dataframe(df_w, use_container_width=True, hide_index=True)

            if not df_regime.empty:
                st.success(f"✅ HMM modeli kayıtlı: {df_regime['updated_at'].iloc[0]}")
            else:
                st.warning("⚠️ HMM modeli henüz kaydedilmedi")
    except Exception as e:
        st.error(f"State okunamadı: {e}")

    # AI sinyal geçmişi
    st.divider()
    st.markdown("#### 🧠 AI Sinyal Geçmişi")
    df_sig = load_ai_signals(kind=kind, limit=100)
    if df_sig.empty:
        st.info("Kayıtlı sinyal yok.")
    else:
        df_sig = df_sig.rename(columns={
            'tarih': 'Tarih', 'regime_name': 'Rejim',
            'learning_rate': 'Öğrenme', 'confidence': 'Güven',
            'created_at': 'Kayıt'})
        st.dataframe(
            df_sig[['Tarih', 'Rejim', 'Öğrenme', 'Güven', 'Kayıt']],
            use_container_width=True, hide_index=True)