"""
TEFAS AI Fon Yönetim Sistemi v2.2
==================================
Düzeltmeler (v2.1 → v2.2):
  • Overlapping window fix: kümülatif getiri artık non-overlap subset'ten
  • HMM cache: backtest'te 30 günde bir retrain (19 → ~3 eğitim)
  • n_non_overlap metriği eklendi (gerçek gözlem sayısı)
  • avg_return metriği eklendi (ortalama getiri)

Modlar:
  UI:   streamlit run tefas_ai_fund.py
  Cron: python tefas_ai_fund.py --cron --kind YAT
  Stat: python tefas_ai_fund.py --status
  Test: python tefas_ai_fund.py --test
  BT:   python tefas_ai_fund.py --backtest --kind YAT --freq 7
"""

import sys
import argparse
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
import traceback
import logging

warnings.filterwarnings('ignore')

# =============================================
# 0. KONFİGÜRASYON
# =============================================
OPERATION_PROFILE = {
    'name': 'Haftalık Dengeli (Cuma)',
    'signal_weekday': 4,
    'signal_hour': 19,
    'signal_minute': 35,
    'rebalance_week_of_month': 1,
    'decay': 0.995,
    'learning_rate': 0.10,
    'hmm_retrain_days': 30,
    'rf_update_days': 30,
    'default_rf': 0.40,
    'corr_threshold': 0.85,
    'defensive_fund': 'PPF',
    'defensive_ratio': 0.60,
    'backtest_freq_days': 7,
    'backtest_hmm_interval_days': 30,  # >>> YENİ: backtest'te HMM retrain aralığı
}

DATA_LOOKBACK_DAYS = 400
METRICS_MIN_DAYS = 30
HMM_MIN_DAYS = 60

DB_PATH = "tefas_data.db"
LOG_PATH = "tefas_ai.log"
CACHE_DIR = "cache"

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ])
logger = logging.getLogger("tefas_ai")


# =============================================
# 1. ZAMANLAMA
# =============================================
def get_week_of_month(d):
    return (d.day - 1) // 7 + 1


def is_signal_day(d=None):
    if d is None:
        d = datetime.date.today()
    return d.weekday() == OPERATION_PROFILE['signal_weekday']


def is_rebalance_week(d=None):
    if d is None:
        d = datetime.date.today()
    return (is_signal_day(d) and
            get_week_of_month(d) == OPERATION_PROFILE['rebalance_week_of_month'])


def get_next_signal_date(from_date=None):
    if from_date is None:
        from_date = datetime.date.today()
    days_ahead = (OPERATION_PROFILE['signal_weekday'] - from_date.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return from_date + timedelta(days=days_ahead)


def get_next_rebalance_date(from_date=None):
    if from_date is None:
        from_date = datetime.date.today()
    for i in range(1, 60):
        d = from_date + timedelta(days=i)
        if is_rebalance_week(d):
            return d
    return None


# =============================================
# 2. OPSİYONEL KÜTÜPHANELER
# =============================================
TG_TOKEN = os.environ.get("TG_TOKEN", "")
TG_CHAT = os.environ.get("TG_CHAT", "")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

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
# 3. SQLITE
# =============================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS fund_info (
        date TEXT NOT NULL, kind TEXT NOT NULL, fund_code TEXT NOT NULL,
        fund_name TEXT, price REAL, shares_outstanding REAL,
        investor_count REAL, portfolio_size REAL,
        PRIMARY KEY (date, kind, fund_code))''')
    c.execute('''CREATE TABLE IF NOT EXISTS fund_breakdown (
        date TEXT NOT NULL, kind TEXT NOT NULL, fund_code TEXT NOT NULL,
        breakdown_json TEXT,
        PRIMARY KEY (date, kind, fund_code))''')
    c.execute('''CREATE TABLE IF NOT EXISTS ai_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT, tarih TEXT, kind TEXT,
        regime INTEGER, regime_name TEXT, risk_level INTEGER,
        weights_json TEXT, metrics_json TEXT, learning_rate REAL,
        confidence REAL, is_rebalance INTEGER DEFAULT 0,
        is_preview INTEGER DEFAULT 0, created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS ai_learner_state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        weights_json TEXT, learning_rate REAL, initial_lr REAL,
        decay REAL, history_json TEXT, updated_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS ai_regime_state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        model_blob BLOB, feats_mean_json TEXT, feats_std_json TEXT,
        label_mapping_json TEXT, updated_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS auto_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_at TEXT, mode TEXT, status TEXT, message TEXT,
        n_funds INTEGER, is_rebalance INTEGER, duration_sec REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS rf_rates (
        date TEXT PRIMARY KEY, rate REAL, source TEXT,
        updated_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_portfolio (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT, fund_code TEXT, units REAL, avg_cost REAL,
        updated_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS trade_orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tarih TEXT, kind TEXT, fund_code TEXT,
        action TEXT, units REAL, price REAL, cost REAL,
        signal_id INTEGER, status TEXT, created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS backtest_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_at TEXT, kind TEXT, start_date TEXT, end_date TEXT,
        strategy TEXT, lookback INTEGER, top_n INTEGER,
        freq_days INTEGER,
        hit_rate REAL, cumulative_return REAL, annualized_return REAL,
        max_drawdown REAL, sharpe REAL, n_signals INTEGER,
        details_json TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS backtest_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        backtest_id INTEGER, signal_date TEXT,
        selected_funds TEXT, weights_json TEXT,
        forward_return_20d REAL, forward_return_60d REAL,
        benchmark_return_20d REAL, benchmark_return_60d REAL,
        is_hit_20d INTEGER, is_hit_60d INTEGER,
        max_dd_realized REAL,
        regime INTEGER, regime_name TEXT, risk_level INTEGER,
        created_at TEXT)''')

    for idx in [
        'CREATE INDEX IF NOT EXISTS idx_info_kind_date ON fund_info(kind, date)',
        'CREATE INDEX IF NOT EXISTS idx_info_code ON fund_info(fund_code)',
        'CREATE INDEX IF NOT EXISTS idx_breakdown_kind_date ON fund_breakdown(kind, date)',
        'CREATE INDEX IF NOT EXISTS idx_breakdown_code ON fund_breakdown(fund_code)',
        'CREATE INDEX IF NOT EXISTS idx_bt_signals ON backtest_signals(backtest_id)',
    ]:
        c.execute(idx)

    conn.commit()

    migrations = [
        ('ai_signals', 'is_rebalance', 'INTEGER DEFAULT 0'),
        ('ai_signals', 'is_preview', 'INTEGER DEFAULT 0'),
        ('ai_signals', 'risk_level', 'INTEGER DEFAULT 1'),
        ('ai_regime_state', 'feats_mean_json', 'TEXT'),
        ('ai_regime_state', 'feats_std_json', 'TEXT'),
        ('ai_regime_state', 'label_mapping_json', 'TEXT'),
        ('backtest_runs', 'freq_days', 'INTEGER DEFAULT 7'),
        ('backtest_signals', 'max_dd_realized', 'REAL'),
    ]
    for table, col, typ in migrations:
        try:
            c.execute(f"PRAGMA table_info({table})")
            cols = [r[1] for r in c.fetchall()]
            if col not in cols:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
                conn.commit()
                logger.info(f"[DB] Migration: {table}.{col} eklendi")
        except Exception as e:
            logger.warning(f"[DB] Migration {table}.{col}: {e}")

    conn.close()


init_db()


# =============================================
# 4. CRUD YARDIMCILARI
# =============================================
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
    except Exception as e:
        logger.warning(f"get_last_date: {e}")
        return None
    finally:
        conn.close()


def get_total_rows(kind, table='fund_info'):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            f"SELECT COUNT(*) AS n FROM {table} WHERE kind=?",
            conn, params=(kind,))
        return int(df['n'].iloc[0])
    except Exception:
        return 0
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
    except Exception as e:
        logger.error(f"save_fund_info: {e}")
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
    except Exception as e:
        logger.error(f"save_breakdown: {e}")
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
    except Exception as e:
        logger.error(f"load_fund_info: {e}")
        return pd.DataFrame()
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
    except Exception as e:
        logger.error(f"load_breakdown: {e}")
        return pd.DataFrame()
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


def save_rf_rate(rate, source='manual', date=None):
    if date is None:
        date = datetime.date.today().isoformat()
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("""INSERT OR REPLACE INTO rf_rates
                     (date, rate, source, updated_at)
                     VALUES (?, ?, ?, ?)""",
                  (date, float(rate), source,
                   datetime.datetime.now().isoformat(timespec='seconds')))
        conn.commit()
    except Exception as e:
        logger.error(f"save_rf_rate: {e}")
    finally:
        conn.close()


def get_current_rf_rate():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT rate, source, date FROM rf_rates ORDER BY date DESC LIMIT 1",
            conn)
    except Exception:
        return OPERATION_PROFILE['default_rf'], 'default'
    finally:
        conn.close()
    if df.empty:
        return OPERATION_PROFILE['default_rf'], 'default'
    return float(df['rate'].iloc[0]), df['source'].iloc[0]


def save_ai_signal(signal, kind, is_rebalance=False, is_preview=False):
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("""INSERT INTO ai_signals
                     (tarih, kind, regime, regime_name, risk_level,
                      weights_json, metrics_json, learning_rate, confidence,
                      is_rebalance, is_preview, created_at)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (str(signal['date'])[:10], kind, int(signal['regime']),
                   signal['regime_name'], int(signal.get('risk_level', 1)),
                   json.dumps(signal['weights'], ensure_ascii=False),
                   json.dumps(signal['metrics'], ensure_ascii=False, default=str),
                   float(signal['learning_rate']),
                   float(signal['confidence']),
                   int(is_rebalance), int(is_preview),
                   datetime.datetime.now().isoformat(timespec='seconds')))
        signal_id = c.lastrowid
        conn.commit()
        return signal_id
    except Exception as e:
        logger.error(f"save_ai_signal: {e}")
        return None
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
    except Exception as e:
        logger.warning(f"load_ai_signals: {e}")
        return pd.DataFrame()
    finally:
        conn.close()
    return df


def log_auto_run(mode, status, message, n_funds=0, is_rebalance=False, duration=0):
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("""INSERT INTO auto_runs
                     (run_at, mode, status, message, n_funds,
                      is_rebalance, duration_sec)
                     VALUES (?,?,?,?,?,?,?)""",
                  (datetime.datetime.now().isoformat(timespec='seconds'),
                   mode, status, message, int(n_funds),
                   int(is_rebalance), float(duration)))
        conn.commit()
    except Exception as e:
        logger.error(f"log_auto_run: {e}")
    finally:
        conn.close()


def load_auto_runs(limit=50):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM auto_runs ORDER BY id DESC LIMIT ?",
            conn, params=(limit,))
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()
    return df


def save_learner_state(learner):
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
        logger.error(f"save_learner_state: {e}")
        return False


def load_learner_state(learner):
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
        logger.warning(f"load_learner_state: {e}")
        return False


def reset_learner_state():
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("DELETE FROM ai_learner_state WHERE id=1")
        conn.commit()
    except Exception as e:
        logger.error(f"reset_learner_state: {e}")
    finally:
        conn.close()


def save_regime_model(detector):
    if detector.model is None:
        return False
    try:
        blob = pickle.dumps(detector.model)
        fm = json.dumps(detector.feats_mean.to_dict()) if detector.feats_mean is not None else None
        fs = json.dumps(detector.feats_std.to_dict()) if detector.feats_std is not None else None
        lm = json.dumps(detector.label_mapping) if detector.label_mapping else None
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""INSERT OR REPLACE INTO ai_regime_state
                     (id, model_blob, feats_mean_json, feats_std_json,
                      label_mapping_json, updated_at)
                     VALUES (1, ?, ?, ?, ?, ?)""",
                  (blob, fm, fs, lm,
                   datetime.datetime.now().isoformat(timespec='seconds')))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"save_regime_model: {e}")
        return False


def load_regime_model(detector):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT model_blob, feats_mean_json, feats_std_json, "
                  "label_mapping_json, updated_at FROM ai_regime_state WHERE id=1")
        row = c.fetchone()
        conn.close()
        if row and row[0]:
            detector.model = pickle.loads(row[0])
            if row[1]:
                detector.feats_mean = pd.Series(json.loads(row[1]))
            if row[2]:
                detector.feats_std = pd.Series(json.loads(row[2]))
            if row[3]:
                detector.label_mapping = json.loads(row[3])
                detector.label_mapping = {int(k): v for k, v in detector.label_mapping.items()}
            if row[4]:
                detector.last_train_date = pd.to_datetime(row[4]).date()
            return True
        return False
    except Exception as e:
        logger.warning(f"load_regime_model: {e}")
        return False


def reset_regime_model():
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("DELETE FROM ai_regime_state WHERE id=1")
        conn.commit()
    except Exception as e:
        logger.error(f"reset_regime_model: {e}")
    finally:
        conn.close()


def get_regime_model_age_days():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT updated_at FROM ai_regime_state WHERE id=1", conn)
    except Exception:
        return None
    finally:
        conn.close()
    if df.empty:
        return None
    try:
        d = pd.to_datetime(df['updated_at'].iloc[0]).date()
        return (datetime.date.today() - d).days
    except Exception:
        return None


# =============================================
# 5. RF RATE
# =============================================
class RiskFreeRateProvider:
    @staticmethod
    def update_from_tcmb():
        return None

    @staticmethod
    def get_rate():
        return get_current_rf_rate()

    @staticmethod
    def update_manual(rate):
        save_rf_rate(rate, source='manual')
        logger.info(f"RF güncellendi: {rate}")


# =============================================
# 6. TEFAS PROVIDER
# =============================================
class TefasDataProvider:
    def __init__(self):
        if not PYTEFAS_AVAILABLE:
            raise ImportError("pytefas yok. pip install pytefas")
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
                progress_cb(f"Info: yeni yok (son: {son_db})")
            return load_fund_info(kind, start_date, end_date)
        if progress_cb:
            progress_cb(f"Info: {fetch_start} → {end_date}")
        try:
            df_new = self.crawler.fetch(
                start=fetch_start.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                kind=kind, columns="info")
        except Exception as e:
            logger.error(f"sync_fund_info: {e}")
            df_new = pd.DataFrame()
        if not df_new.empty:
            if 'kind' not in df_new.columns:
                df_new['kind'] = kind
            n = save_fund_info(df_new)
            if progress_cb:
                progress_cb(f"✔ {n} info kaydı")
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
                progress_cb(f"Breakdown: yeni yok (son: {son_db})")
            return load_breakdown(kind, start_date, end_date)
        if progress_cb:
            progress_cb(f"Breakdown: {fetch_start} → {end_date}")
        try:
            df_new = self.crawler.fetch(
                start=fetch_start.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                kind=kind, columns="breakdown")
        except Exception as e:
            logger.error(f"sync_breakdown: {e}")
            df_new = pd.DataFrame()
        if not df_new.empty:
            if 'kind' not in df_new.columns:
                df_new['kind'] = kind
            n = save_breakdown(df_new)
            if progress_cb:
                progress_cb(f"✔ {n} breakdown kaydı")
        return load_breakdown(kind, start_date, end_date)

    def get_fund_history(self, fund_code, kind, start_date, end_date):
        df = load_fund_info(kind, start_date, end_date)
        if df.empty:
            return df
        return df[df['fund_code'] == fund_code].copy()

    def get_fund_breakdown(self, fund_code, kind, start_date, end_date):
        return load_breakdown(kind, start_date, end_date, fund_code=fund_code)


# =============================================
# 7. NAKİT AKIŞI
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
        cutoff = pd.to_datetime(max_date) - pd.Timedelta(days=int(window_days))
        df_recent = df[df["date"] >= cutoff]
        if df_recent.empty:
            return pd.DataFrame()
        agg = df_recent.groupby("fund_code").agg(
            toplam_net_akis=("net_flow", "sum"),
            son_aum=("portfolio_size", "last"),
            ilk_aum=("portfolio_size", "first"),
        ).reset_index()
        agg["aum_degisim_pct"] = (agg["son_aum"] / agg["ilk_aum"] - 1) * 100
        return agg


# =============================================
# 8. AKILLI PARA SKORU
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
        cutoff = df["date"].max() - pd.Timedelta(days=int(window_days))
        df_r = df[df["date"] >= cutoff]
        if len(df_r) < 2:
            return 0, 0
        ilk = df_r["investor_count"].iloc[0]
        son = df_r["investor_count"].iloc[-1]
        if pd.isna(ilk) or pd.isna(son) or ilk == 0:
            return 0, 0
        d = (son / ilk - 1) * 100
        if d > 10:   return 30, d
        elif d > 5:  return 20, d
        elif d > 0:  return 10, d
        elif d > -5: return -10, d
        else:        return -30, d

    @staticmethod
    def calculate_flow_score(df_flow_agg, fund_code):
        if df_flow_agg.empty:
            return 0, 0
        row = df_flow_agg[df_flow_agg["fund_code"] == fund_code]
        if row.empty:
            return 0, 0
        na = row["toplam_net_akis"].iloc[0]
        if pd.isna(na):
            return 0, 0
        aum = row["son_aum"].iloc[0] if "son_aum" in row.columns else 0
        pct = (na / aum * 100) if aum and aum > 0 else 0
        if pct > 5:    return 30, na
        elif pct > 2:  return 20, na
        elif pct > 0:  return 10, na
        elif pct > -2: return -10, na
        else:          return -30, na

    @staticmethod
    def calculate_allocation_score(df_breakdown, fund_code, window_days=30):
        if df_breakdown.empty:
            return 0, {}
        df = SmartMoneyScorer._ensure_datetime(df_breakdown)
        df = df[df["fund_code"] == fund_code].sort_values("date")
        if len(df) < 2:
            return 0, {}
        cutoff = df["date"].max() - pd.Timedelta(days=int(window_days))
        df_r = df[df["date"] >= cutoff]
        if len(df_r) < 2:
            return 0, {}
        score = 0
        detay = {}
        for col in [c for c in df.columns if "stock" in c.lower()
                    and c not in ['fund_code', 'date', 'kind']]:
            i, s = df_r[col].iloc[0], df_r[col].iloc[-1]
            if pd.notna(i) and pd.notna(s):
                dc = s - i
                detay[col] = round(dc, 2)
                if dc > 2:    score += 10
                elif dc > 0:  score += 5
                elif dc < -2: score -= 10
        for col in [c for c in df.columns
                    if any(k in c.lower() for k in ["gold", "fx", "foreign_currency"])
                    and c not in ['fund_code', 'date', 'kind']]:
            i, s = df_r[col].iloc[0], df_r[col].iloc[-1]
            if pd.notna(i) and pd.notna(s):
                dc = s - i
                detay[col] = round(dc, 2)
                if dc > 2:    score -= 10
                elif dc < -2: score += 5
        return int(np.clip(score, -20, 20)), detay

    @staticmethod
    def calculate_size_score(df_info, fund_code, min_aum=50_000_000):
        if df_info.empty:
            return 0, 0
        df = df_info[df_info["fund_code"] == fund_code]
        if df.empty:
            return 0, 0
        aum = df["portfolio_size"].iloc[-1]
        if pd.isna(aum):
            return 0, 0
        if aum > 1_000_000_000:    return 10, aum
        elif aum > 500_000_000:    return 8, aum
        elif aum > 100_000_000:    return 5, aum
        elif aum > min_aum:        return 2, aum
        else:                      return -10, aum

    @staticmethod
    def calculate_total_score(df_info, df_flow_agg, df_breakdown, fund_code):
        inv_s, inv_d = SmartMoneyScorer.calculate_investor_score(df_info, fund_code)
        flow_s, na = SmartMoneyScorer.calculate_flow_score(df_flow_agg, fund_code)
        alloc_s, _ = SmartMoneyScorer.calculate_allocation_score(df_breakdown, fund_code)
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
# 9. PROFESYONEL METRİKLER
# =============================================
class ProfessionalMetrics:
    @staticmethod
    def calculate_all(df_prices, df_benchmark=None, risk_free=None,
                      window=90, confidence=0.95):
        if risk_free is None:
            risk_free, _ = get_current_rf_rate()

        if df_prices is None or df_prices.empty:
            return {}
        if len(df_prices) < METRICS_MIN_DAYS:
            return {}

        effective_window = min(window, len(df_prices) - 2)
        if effective_window < METRICS_MIN_DAYS:
            return {}

        recent = df_prices.tail(effective_window)
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
            if len(r) < 15:
                continue
            ann_return = (1 + r.mean()) ** 252 - 1
            ann_vol = r.std() * np.sqrt(252)
            excess = ann_return - risk_free
            sharpe = excess / (ann_vol + 1e-9)
            daily_rf = (1 + risk_free) ** (1 / 252) - 1
            neg = r[r < daily_rf] - daily_rf
            down = np.sqrt((neg ** 2).mean()) * np.sqrt(252) if len(neg) > 0 else 1e-9
            sortino = excess / (down + 1e-9)
            cum = (1 + r).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax()
            max_dd = dd.min()
            calmar = ann_return / (abs(max_dd) + 1e-9)
            var_95 = np.percentile(r, (1 - confidence) * 100)
            tail = r[r <= var_95]
            cvar_95 = tail.mean() if len(tail) > 0 else var_95
            treynor = info_ratio = up_capture = down_capture = beta = None
            if bench_returns is not None:
                aligned = pd.concat([r, bench_returns], axis=1).dropna()
                aligned.columns = ['fund', 'bench']
                if len(aligned) > 15:
                    beta = aligned['fund'].cov(aligned['bench']) / (aligned['bench'].var() + 1e-9)
                    treynor = excess / (beta + 1e-9)
                    active = aligned['fund'] - aligned['bench']
                    te = active.std() * np.sqrt(252)
                    info_ratio = (active.mean() * 252) / (te + 1e-9)
                    up = aligned[aligned['bench'] > 0]
                    down_b = aligned[aligned['bench'] < 0]
                    if len(up) > 0 and up['bench'].mean() != 0:
                        up_capture = up['fund'].mean() / up['bench'].mean()
                    if len(down_b) > 0 and down_b['bench'].mean() != 0:
                        down_capture = down_b['fund'].mean() / down_b['bench'].mean()
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
                'risk_free_used': round(risk_free, 4),
            }
        return metrics


# =============================================
# 10. KORELASYON FİLTRESİ
# =============================================
class CorrelationFilter:
    @staticmethod
    def filter_top_n(df_prices, candidates, top_n=5,
                     threshold=0.85, window=60):
        if not candidates:
            return []
        if len(candidates) <= top_n:
            return candidates
        recent = df_prices.tail(window).pct_change().dropna()
        if recent.empty:
            return candidates[:top_n]
        available = [c for c in candidates if c in recent.columns]
        if len(available) <= 1:
            return candidates[:top_n]
        corr_matrix = recent[available].corr().abs()
        selected = []
        for cand in candidates:
            if cand not in corr_matrix.columns:
                continue
            if len(selected) == 0:
                selected.append(cand)
                continue
            corrs = [corr_matrix.loc[cand, s] for s in selected
                     if s in corr_matrix.columns]
            max_corr = max(corrs) if corrs else 0
            if max_corr < threshold:
                selected.append(cand)
            if len(selected) >= top_n:
                break
        return selected if selected else candidates[:top_n]


# =============================================
# 11. ADAPTİF BOYUTLANDIRMA
# =============================================
class AdaptiveSizing:
    @staticmethod
    def risk_parity_weights(df_prices, window=60):
        if df_prices is None or df_prices.empty or len(df_prices) < 20:
            return {}
        effective = min(window, len(df_prices) - 2)
        if effective < 20:
            return {}
        returns = df_prices.tail(effective).pct_change().dropna()
        if returns.empty:
            return {}
        vols = returns.std() * np.sqrt(252)
        vols = vols.replace(0, np.nan).dropna()
        if vols.empty:
            return {}
        w = (1.0 / (vols + 1e-6))
        w = w / w.sum()
        corr = returns[vols.index].corr()
        avg = corr.mean()
        penalty = (1.0 / (1.0 + avg - 0.5)).clip(0.5, 1.5)
        w = (w * penalty).dropna()
        if w.sum() == 0:
            return {}
        return (w / w.sum()).to_dict()

    @staticmethod
    def kelly_optimal_weights(metrics, max_weight=0.40, min_weight=0.02):
        if not metrics:
            return {}
        scores = {}
        for f, m in metrics.items():
            if m.get('Sharpe') is None:
                continue
            s = np.clip(m['Sharpe'] / 3.0, 0, 1)
            so = np.clip(m['Sortino'] / 4.0, 0, 1)
            dd = 1.0 - np.clip(abs(m['Max_DD']) / 50.0, 0, 1)
            scores[f] = max(0, 0.4 * s + 0.4 * so + 0.2 * dd)
        if not scores or sum(scores.values()) == 0:
            return {}
        t = sum(scores.values())
        w = {f: s / t for f, s in scores.items()}
        w = {f: float(np.clip(x, min_weight, max_weight)) for f, x in w.items()}
        t2 = sum(w.values())
        return {f: x / t2 for f, x in w.items()}


# =============================================
# 12. ONLINE LEARNER
# =============================================
class OnlineLearner:
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
            t = sum(new_w.values())
            if t > 0:
                new_w = {f: x / t for f, x in new_w.items()}
            self.weights = new_w
        self.learning_rate *= self.decay
        self.history.append({
            'n_funds': len(self.weights),
            'learning_rate': self.learning_rate,
            'ts': datetime.datetime.now().isoformat(timespec='seconds'),
        })
        return self.weights

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
# 13. HMM REJİM
# =============================================
class RegimeDetector:
    def __init__(self, n_regimes=3, lookback=252):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.model = None
        self.feats_mean = None
        self.feats_std = None
        self.label_mapping = {}
        self.last_train_date = None

    def fit(self, df_benchmark):
        if not HMM_AVAILABLE:
            logger.warning("[HMM] hmmlearn yok")
            return self
        if df_benchmark is None or df_benchmark.empty:
            logger.warning("[HMM] Benchmark boş")
            return self

        b = df_benchmark.iloc[:, 0] if isinstance(df_benchmark, pd.DataFrame) else df_benchmark
        b = pd.to_numeric(b, errors='coerce').dropna()
        returns = b.pct_change().dropna()

        if len(returns) < HMM_MIN_DAYS:
            logger.warning(f"[HMM] Yetersiz: {len(returns)} iş günü")
            return self

        feats = pd.DataFrame({
            'return': returns,
            'vol': returns.rolling(20).std()
        }).dropna()
        if len(feats) < 40:
            return self

        feats_mean = feats.mean()
        feats_std = feats.std().replace(0, 1)
        feats_scaled = (feats - feats_mean) / feats_std
        np.random.seed(42)
        jitter = np.random.normal(0, 1e-6, feats_scaled.shape)
        X = feats_scaled.values + jitter

        best_model = None
        best_score = -np.inf
        best_seed = None
        for seed in [42, 7, 123, 2024, 999]:
            try:
                m = hmm.GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type="diag",
                    n_iter=200, tol=1e-4, min_covar=1e-3,
                    random_state=seed,
                    init_params="stmc", params="stmc")
                m.fit(X)
                score = m.score(X)
                if score > best_score:
                    best_score = score
                    best_model = m
                    best_seed = seed
            except Exception as e:
                logger.debug(f"[HMM] seed {seed}: {e}")
                continue

        if best_model is None:
            logger.warning("[HMM] Hiçbir seed OK olmadı")
            self.model = None
            return self

        self.model = best_model
        self.feats_mean = feats_mean
        self.feats_std = feats_std
        self.last_train_date = datetime.date.today()

        return_means = best_model.means_[:, 0]
        order = np.argsort(return_means)
        self.label_mapping = {
            int(order[0]): 2,  # AYI
            int(order[1]): 1,  # NÖTR
            int(order[2]): 0,  # BOĞA
        }

        logger.info(f"[HMM] Eğitildi: {len(X)} örnek, seed={best_seed}, "
                    f"skor={best_score:.2f}, map={self.label_mapping}")
        return self

    def should_retrain(self):
        if self.model is None:
            return True
        if self.last_train_date is None:
            return True
        age = (datetime.date.today() - self.last_train_date).days
        return age >= OPERATION_PROFILE['hmm_retrain_days']

    def predict_current_regime(self, df_benchmark):
        if self.model is None or df_benchmark is None or df_benchmark.empty:
            return 1, self.get_regime_name(1), 1
        b = df_benchmark.iloc[:, 0] if isinstance(df_benchmark, pd.DataFrame) else df_benchmark
        b = pd.to_numeric(b, errors='coerce').dropna()
        returns = b.pct_change().dropna()
        if len(returns) < 20:
            return 1, self.get_regime_name(1), 1
        feats = pd.DataFrame({'return': returns,
                              'vol': returns.rolling(20).std()}).dropna()
        if feats.empty:
            return 1, self.get_regime_name(1), 1
        try:
            if self.feats_mean is not None and self.feats_std is not None:
                feats_scaled = (feats - self.feats_mean) / self.feats_std
                X = feats_scaled.values
            else:
                X = feats.values
            raw = int(self.model.predict(X)[-1])
            regime = self.label_mapping.get(raw, 1)
            vol_mean = self.model.means_[raw, 1]
            all_vol_means = np.sort(self.model.means_[:, 1])
            if vol_mean <= all_vol_means[0]:
                risk_level = 0
            elif vol_mean >= all_vol_means[-1]:
                risk_level = 2
            else:
                risk_level = 1
            return regime, self.get_regime_name(regime), int(risk_level)
        except Exception as e:
            logger.warning(f"[HMM] Predict: {e}")
            return 1, self.get_regime_name(1), 1

    @staticmethod
    def get_regime_name(r):
        return {0: "🐂 BOĞA", 1: "😐 NÖTR", 2: "🐻 AYI"}.get(r, "BİLİNMİYOR")

    @staticmethod
    def get_risk_level_name(rl):
        return {0: "🟢 Düşük Vol", 1: "🟡 Orta Vol",
                2: "🔴 Yüksek Vol"}.get(rl, "—")

    def save(self):
        return save_regime_model(self)

    def load(self):
        return load_regime_model(self)

    def reset(self):
        self.model = None
        self.feats_mean = None
        self.feats_std = None
        self.label_mapping = {}
        self.last_train_date = None
        reset_regime_model()


# =============================================
# 14. DEFANSİF ALLOCATION
# =============================================
class DefensiveAllocator:
    @staticmethod
    def apply(weights, regime, risk_level, defensive_fund=None,
              defensive_ratio=None):
        if defensive_fund is None:
            defensive_fund = OPERATION_PROFILE['defensive_fund']
        if defensive_ratio is None:
            defensive_ratio = OPERATION_PROFILE['defensive_ratio']

        if regime == 2:
            shift = defensive_ratio
        elif regime == 1 and risk_level == 2:
            shift = defensive_ratio * 0.5
        elif regime == 1:
            shift = defensive_ratio * 0.25
        else:
            return weights, 0.0

        new_w = {f: w * (1 - shift) for f, w in weights.items()}
        new_w[defensive_fund] = new_w.get(defensive_fund, 0.0) + shift
        total = sum(new_w.values())
        if total > 0:
            new_w = {f: w / total for f, w in new_w.items()}
        return new_w, shift


# =============================================
# 15. PORTFÖY TAKİBİ
# =============================================
class PortfolioTracker:
    @staticmethod
    def load_user_portfolio(kind):
        conn = sqlite3.connect(DB_PATH)
        try:
            df = pd.read_sql_query(
                "SELECT fund_code, units, avg_cost FROM user_portfolio WHERE kind=?",
                conn, params=(kind,))
        except Exception as e:
            logger.warning(f"load_user_portfolio: {e}")
            return {}
        finally:
            conn.close()
        if df.empty:
            return {}
        return {row['fund_code']: {
            'units': float(row['units'] or 0),
            'avg_cost': float(row['avg_cost'] or 0)
        } for _, row in df.iterrows()}

    @staticmethod
    def save_user_position(kind, fund_code, units, avg_cost):
        conn = sqlite3.connect(DB_PATH)
        try:
            c = conn.cursor()
            c.execute("DELETE FROM user_portfolio WHERE kind=? AND fund_code=?",
                      (kind, fund_code))
            c.execute("""INSERT INTO user_portfolio
                         (kind, fund_code, units, avg_cost, updated_at)
                         VALUES (?, ?, ?, ?, ?)""",
                      (kind, fund_code, float(units), float(avg_cost),
                       datetime.datetime.now().isoformat(timespec='seconds')))
            conn.commit()
        except Exception as e:
            logger.error(f"save_user_position: {e}")
        finally:
            conn.close()

    @staticmethod
    def clear_user_portfolio(kind):
        conn = sqlite3.connect(DB_PATH)
        try:
            c = conn.cursor()
            c.execute("DELETE FROM user_portfolio WHERE kind=?", (kind,))
            conn.commit()
        except Exception as e:
            logger.error(f"clear_user_portfolio: {e}")
        finally:
            conn.close()

    @staticmethod
    def compute_delta(current_positions, target_weights, latest_prices,
                      total_value=None, commission=0.001):
        if total_value is None:
            total_value = sum(
                pos['units'] * latest_prices.get(f, 0)
                for f, pos in current_positions.items())

        target_value = {f: total_value * w for f, w in target_weights.items()}
        orders = []
        all_funds = set(current_positions) | set(target_weights)
        for fund in all_funds:
            price = latest_prices.get(fund, 0)
            if price <= 0:
                continue
            current_units = current_positions.get(fund, {}).get('units', 0)
            current_val = current_units * price
            target_val = target_value.get(fund, 0)
            delta_val = target_val - current_val
            if abs(delta_val) < total_value * 0.01:
                continue
            delta_units = delta_val / price
            cost = abs(delta_val) * commission
            orders.append({
                'fund': fund,
                'action': 'AL' if delta_units > 0 else 'SAT',
                'units': round(abs(delta_units), 4),
                'price': round(price, 4),
                'value': round(abs(delta_val), 2),
                'cost': round(cost, 2),
                'target_weight': round(target_weights.get(fund, 0) * 100, 2),
                'current_units': round(current_units, 4),
            })
        return sorted(orders, key=lambda x: -x['value'])

    @staticmethod
    def save_trade_orders(orders, kind, tarih, signal_id=None):
        conn = sqlite3.connect(DB_PATH)
        try:
            c = conn.cursor()
            for o in orders:
                c.execute("""INSERT INTO trade_orders
                             (tarih, kind, fund_code, action, units,
                              price, cost, signal_id, status, created_at)
                             VALUES (?,?,?,?,?,?,?,?,?,?)""",
                          (tarih, kind, o['fund'], o['action'],
                           o['units'], o['price'], o['cost'],
                           signal_id, 'PENDING',
                           datetime.datetime.now().isoformat(timespec='seconds')))
            conn.commit()
        except Exception as e:
            logger.error(f"save_trade_orders: {e}")
        finally:
            conn.close()


# =============================================
# 16. WALK-FORWARD BACKTESTER (v2.2 DÜZELTİLDİ)
# =============================================
class WalkForwardBacktester:
    """
    Walk-forward backtest.

    >>> v2.2 DÜZELTMELERİ:
      • Overlapping window fix: kümülatif getiri non-overlap subset'ten
      • HMM cache: 30 günde bir retrain (19 eğitim → ~3 eğitim)
      • n_non_overlap metriği eklendi
      • avg_return metriği eklendi
    """

    def __init__(self):
        # HMM RAM cache
        self._hmm_cache_model = None
        self._hmm_cache_feats_mean = None
        self._hmm_cache_feats_std = None
        self._hmm_cache_label_map = None
        self._hmm_cache_date = None
        self._hmm_retrain_count = 0
        # >>> YENİ: HMM disk cache istatistikleri
        self._hmm_disk_cache_hits = 0
        # >>> YENİ: Disk cache dizini
        self._disk_cache_dir = "hmm_cache"
        os.makedirs(self._disk_cache_dir, exist_ok=True)

    def run(self, df_info, df_breakdown, df_benchmark,
            kind='YAT',
            signal_freq_days=None,
            forward_windows=(20, 60),
            window=90, top_n=5,
            lookback_days=DATA_LOOKBACK_DAYS,
            use_correlation_filter=True,
            corr_threshold=0.85,
            regime_weights=None):
        if signal_freq_days is None:
            signal_freq_days = OPERATION_PROFILE['backtest_freq_days']

        if df_info.empty:
            return {}, []

        if regime_weights is None:
            regime_weights = {0: 1.0, 1: 0.8, 2: 0.5}

        # Reset HMM RAM cache at start (disk cache korunur)
        self._hmm_cache_model = None
        self._hmm_cache_feats_mean = None
        self._hmm_cache_feats_std = None
        self._hmm_cache_label_map = None
        self._hmm_cache_date = None
        self._hmm_retrain_count = 0
        self._hmm_disk_cache_hits = 0  # >>> YENİ

        df_info = df_info.sort_values('date')
        dates = sorted(df_info['date'].unique())
        if len(dates) < window + max(forward_windows) + 30:
            logger.warning(f"[BT] Yetersiz veri: {len(dates)} gün")
            return {}, []

        df_prices_full = df_info.pivot_table(
            index='date', columns='fund_code', values='price').ffill()

        first_idx = window + 20
        last_idx = len(dates) - max(forward_windows) - 5
        if last_idx <= first_idx:
            logger.warning("[BT] Backtest penceresi boş")
            return {}, []

        signal_indices = list(range(first_idx, last_idx, signal_freq_days))
        logger.info(f"[BT] {len(signal_indices)} sinyal tarihi işlenecek "
                    f"(her {signal_freq_days} iş günü)")

        signal_details = []
        for i, idx in enumerate(signal_indices):
            signal_date = dates[idx]
            try:
                detail = self._run_single_backtest(
                    df_info, df_breakdown, df_benchmark,
                    df_prices_full, signal_date, dates, idx,
                    forward_windows, window, top_n,
                    use_correlation_filter, corr_threshold,
                    regime_weights)
                if detail:
                    signal_details.append(detail)
            except Exception as e:
                logger.warning(f"[BT] {signal_date}: {e}")
                logger.debug(traceback.format_exc())
                continue

            if (i + 1) % 5 == 0:
                logger.info(f"[BT] {i+1}/{len(signal_indices)} tamam")

        if not signal_details:
            return {}, []

        logger.info(f"[BT] HMM toplam {self._hmm_retrain_count} kez eğitildi")
        return self._summarize(signal_details, forward_windows,
                                freq_days=signal_freq_days), signal_details
    def _save_hmm_to_disk(self, engine, cache_path, signal_date):
        """Eğitilen HMM modelini diske kaydeder ve RAM cache'i güncelle."""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'model': engine.regime_detector.model,
                    'feats_mean': engine.regime_detector.feats_mean,
                    'feats_std': engine.regime_detector.feats_std,
                    'label_map': engine.regime_detector.label_mapping,
                }, f)

            # RAM cache güncelle
            self._hmm_cache_model = engine.regime_detector.model
            self._hmm_cache_feats_mean = engine.regime_detector.feats_mean
            self._hmm_cache_feats_std = engine.regime_detector.feats_std
            self._hmm_cache_label_map = engine.regime_detector.label_mapping
            self._hmm_cache_date = signal_date
            self._hmm_retrain_count += 1
            logger.info(f"[BT] HMM retrain #{self._hmm_retrain_count} "
                        f"@ {str(signal_date)[:10]} (diske kaydedildi)")
        except Exception as e:
            logger.warning(f"HMM cache yazılamadı: {e}")

    def _run_single_backtest(self, df_info, df_breakdown, df_benchmark,
                             df_prices_full, signal_date, all_dates,
                             signal_idx, forward_windows,
                             window, top_n, use_corr, corr_thresh,
                             regime_weights):
        df_hist = df_info[df_info['date'] <= signal_date].copy()
        df_breakdown_hist = df_breakdown[df_breakdown['date'] <= signal_date].copy() \
            if not df_breakdown.empty else pd.DataFrame()

        if df_hist.empty:
            return None

        df_bench_hist = pd.DataFrame()
        if df_benchmark is not None and not df_benchmark.empty:
            try:
                df_bench_hist = df_benchmark[df_benchmark.index <= signal_date].copy()
            except Exception:
                df_bench_hist = pd.DataFrame()

        # >>> DÜZELTİLDİ: Fresh engine ama HMM cache'li
        engine = SelfLearningSignalEngine(auto_load=False)

        # HMM retrain kontrolü (30 günde bir)
        should_fit = False
        if self._hmm_cache_model is None or self._hmm_cache_date is None:
            should_fit = True
        else:
            days_since = (signal_date - self._hmm_cache_date).days
            if days_since >= OPERATION_PROFILE['backtest_hmm_interval_days']:
                should_fit = True

        # >>> YENİ: Disk cache kontrolü
        if should_fit and not df_bench_hist.empty:
            cache_key = f"hmm_{str(signal_date)[:10]}.pkl"
            cache_path = os.path.join(self._disk_cache_dir, cache_key)

            if os.path.exists(cache_path):
                # >>> Disk cache'ten yükle
                try:
                    with open(cache_path, 'rb') as f:
                        cached = pickle.load(f)
                    engine.regime_detector.model = cached['model']
                    engine.regime_detector.feats_mean = cached['feats_mean']
                    engine.regime_detector.feats_std = cached['feats_std']
                    engine.regime_detector.label_mapping = cached['label_map']

                    # RAM cache'i de güncelle
                    self._hmm_cache_model = cached['model']
                    self._hmm_cache_feats_mean = cached['feats_mean']
                    self._hmm_cache_feats_std = cached['feats_std']
                    self._hmm_cache_label_map = cached['label_map']
                    self._hmm_cache_date = signal_date
                    self._hmm_disk_cache_hits += 1
                    logger.info(f"[BT] HMM disk cache hit @ "
                                f"{str(signal_date)[:10]}")
                except Exception as e:
                    logger.warning(f"HMM cache okunamadı: {e}, yeniden eğitiliyor")
                    engine.regime_detector.fit(df_bench_hist)
                    self._save_hmm_to_disk(engine, cache_path, signal_date)
            else:
                # >>> Yeni eğit ve diske kaydet
                engine.regime_detector.fit(df_bench_hist)
                if engine.regime_detector.model is not None:
                    self._save_hmm_to_disk(engine, cache_path, signal_date)

        elif self._hmm_cache_model is not None:
            # RAM cache'ten kullan
            engine.regime_detector.model = self._hmm_cache_model
            engine.regime_detector.feats_mean = self._hmm_cache_feats_mean
            engine.regime_detector.feats_std = self._hmm_cache_feats_std
            engine.regime_detector.label_mapping = self._hmm_cache_label_map or {}

        # Online learner warmup
        try:
            df_flow_hist = CashFlowAnalyzer.calculate_net_flow(df_hist)
            df_flow_agg_hist = CashFlowAnalyzer.aggregate_flow(
                df_flow_hist, window_days=window)
            target = {}
            for fund in df_hist['fund_code'].unique():
                try:
                    s, _ = SmartMoneyScorer.calculate_total_score(
                        df_hist, df_flow_agg_hist, df_breakdown_hist, fund)
                    if s > 0:
                        target[fund] = s / 100.0
                except Exception:
                    continue
            if target:
                t = sum(target.values())
                target = {f: v / t for f, v in target.items()}
                engine.online.update(target)
        except Exception as e:
            logger.debug(f"[BT] online warmup: {e}")

        signal = engine.generate_preview(
            df_hist, df_breakdown_hist, df_bench_hist,
            window=window, top_n=top_n,
            use_correlation_filter=use_corr,
            corr_threshold=corr_thresh,
            regime_weights=regime_weights)

        if signal is None:
            return None
        weights = signal['weights']
        if not weights:
            return None

        # Günlük equity curve
        max_forward = max(forward_windows)
        daily_values = []
        for j in range(signal_idx, min(signal_idx + max_forward + 1, len(all_dates))):
            day_date = all_dates[j]
            day_val = 1.0
            try:
                for fund, w in weights.items():
                    try:
                        p0 = df_prices_full.loc[signal_date, fund]
                        pj = df_prices_full.loc[day_date, fund]
                        if pd.notna(p0) and pd.notna(pj) and p0 > 0:
                            day_val += w * ((pj / p0) - 1)
                    except Exception:
                        continue
            except Exception:
                pass
            daily_values.append((str(day_date)[:10], round(day_val, 6)))

        result = {
            'signal_date': signal_date,
            'selected_funds': list(weights.keys()),
            'weights': weights,
            'regime': signal['regime'],
            'regime_name': signal['regime_name'],
            'risk_level': signal.get('risk_level', 1),
            'forward_returns': {},
            'benchmark_returns': {},
            'is_hit': {},
            'daily_values': daily_values,
        }

        for fwd in forward_windows:
            future_date = self._get_future_date(all_dates, signal_date, fwd)
            if future_date is None:
                result['forward_returns'][fwd] = None
                result['is_hit'][fwd] = None
                continue

            portfolio_return = 0.0
            for fund, w in weights.items():
                try:
                    p0 = df_prices_full.loc[signal_date, fund]
                    p1 = df_prices_full.loc[future_date, fund]
                    if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                        portfolio_return += w * ((p1 / p0) - 1)
                except Exception:
                    continue
            result['forward_returns'][fwd] = round(portfolio_return * 100, 3)

            bench_r = None
            if df_benchmark is not None and not df_benchmark.empty:
                try:
                    b0 = df_benchmark.loc[:signal_date].iloc[-1, 0]
                    b1 = df_benchmark.loc[:future_date].iloc[-1, 0]
                    if pd.notna(b0) and pd.notna(b1) and b0 > 0:
                        bench_r = (b1 / b0) - 1
                except Exception:
                    pass
            result['benchmark_returns'][fwd] = round(bench_r * 100, 3) \
                if bench_r is not None else None
            result['is_hit'][fwd] = (
                portfolio_return > (bench_r if bench_r is not None else 0))

        # Gerçek max drawdown
        if len(daily_values) > 2:
            vals = np.array([v for _, v in daily_values])
            rolling_max = np.maximum.accumulate(vals)
            dd = (vals - rolling_max) / rolling_max
            result['max_dd_realized'] = round(float(dd.min()) * 100, 3)
        else:
            result['max_dd_realized'] = None

        return result

    @staticmethod
    def _get_future_date(all_dates, signal_date, forward_days):
        try:
            idx = list(all_dates).index(signal_date)
        except ValueError:
            return None
        future_idx = idx + forward_days
        if future_idx >= len(all_dates):
            return None
        return all_dates[future_idx]

    def _summarize(self, details, forward_windows, freq_days=7):
        """
        >>> DÜZELTİLDİ: Overlapping windows sorunu çözüldü.

        Sinyaller arası boşluk (freq_days) forward pencereden (fwd) küçükse
        pencereler çakışır. Bu durumda:
          - Kümülatif getiri için SADECE çakışmayan alt-kümeyi kullan
          - Hit rate için tüm sinyalleri kullan
          - Sharpe için tüm sinyalleri kullan (daha stabil)
        """
        summary = {'n_signals': len(details), 'freq_days': freq_days}
        details_sorted = sorted(details, key=lambda d: d['signal_date'])

        for fwd in forward_windows:
            returns = [d['forward_returns'].get(fwd) for d in details_sorted
                       if d['forward_returns'].get(fwd) is not None]
            bench_returns = [d['benchmark_returns'].get(fwd) for d in details_sorted
                             if d['benchmark_returns'].get(fwd) is not None]
            hits = [d['is_hit'].get(fwd) for d in details_sorted
                    if d['is_hit'].get(fwd) is not None]

            if not returns:
                continue

            returns_arr = np.array(returns) / 100
            bench_arr = np.array(bench_returns) / 100 if bench_returns else None

            # >>> DÜZELTME: Non-overlapping subset seç
            step = max(1, int(np.ceil(fwd / freq_days)))
            non_overlap = returns_arr[::step]
            non_overlap_bench = bench_arr[::step] if bench_arr is not None else None

            cum = float(np.prod(1 + non_overlap) - 1)
            n_eff = len(non_overlap)

            periods_per_year = 252 / fwd
            if n_eff > 0:
                try:
                    ann_return = (1 + cum) ** (periods_per_year / n_eff) - 1
                except Exception:
                    ann_return = 0
            else:
                ann_return = 0

            avg_r = returns_arr.mean()
            std = returns_arr.std()
            sharpe = (avg_r / (std + 1e-9)) * np.sqrt(periods_per_year) \
                if std > 0 else 0

            hit_rate = float(np.mean(hits)) if hits else None

            summary[f'hit_rate_{fwd}d'] = round(hit_rate * 100, 2) \
                if hit_rate is not None else None
            summary[f'avg_return_{fwd}d'] = round(avg_r * 100, 3)
            summary[f'cumulative_return_{fwd}d'] = round(cum * 100, 2)
            summary[f'annualized_return_{fwd}d'] = round(ann_return * 100, 2)
            summary[f'sharpe_{fwd}d'] = round(sharpe, 3)
            summary[f'n_observations_{fwd}d'] = len(returns_arr)
            summary[f'n_non_overlap_{fwd}d'] = n_eff
            summary[f'overlap_step_{fwd}d'] = step

            if non_overlap_bench is not None and len(non_overlap_bench) > 0:
                bench_cum = float(np.prod(1 + non_overlap_bench) - 1)
                summary[f'benchmark_cumulative_{fwd}d'] = round(bench_cum * 100, 2)
                summary[f'excess_return_{fwd}d'] = round((cum - bench_cum) * 100, 2)

        # Gerçek drawdown
        dds = [d.get('max_dd_realized') for d in details
               if d.get('max_dd_realized') is not None]
        if dds:
            summary['max_drawdown_realized_avg'] = round(float(np.mean(dds)), 2)
            summary['max_drawdown_realized_worst'] = round(float(np.min(dds)), 2)

        summary['hmm_retrain_count'] = self._hmm_retrain_count
        summary['hmm_disk_cache_hits'] = self._hmm_disk_cache_hits  # >>> YENİ
        return summary


# =============================================
# 17. SİNYAL ÜRETİCİ
# =============================================
class SignalGenerator:
    @staticmethod
    def generate_signals(df_info, df_flow_agg, df_breakdown):
        if df_info.empty:
            return pd.DataFrame()
        rows = []
        for fon in df_info["fund_code"].dropna().unique():
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
                    "Toplam Skor": round(skor, 1), "Sinyal": sinyal,
                    **{k: v for k, v in detay.items() if k != "Toplam Skor"},
                })
            except Exception:
                continue
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("Toplam Skor", ascending=False)
        return df


# =============================================
# 18. SELF-LEARNING ENGINE
# =============================================
class SelfLearningSignalEngine:
    def __init__(self, learning_rate=0.10, decay=0.995, auto_load=True):
        self.online = OnlineLearner(learning_rate=learning_rate, decay=decay)
        self.regime_detector = RegimeDetector()
        self.last_signal = None
        self.last_hit_rate = None
        if auto_load:
            ll = self.online.load()
            lr = self.regime_detector.load()
            self._loaded_from_db = bool(ll or lr)
        else:
            self._loaded_from_db = False
        self._load_last_hit_rate()

    def _load_last_hit_rate(self, kind='YAT'):
        conn = sqlite3.connect(DB_PATH)
        try:
            df = pd.read_sql_query(
                "SELECT hit_rate, n_signals FROM backtest_runs "
                "WHERE kind=? ORDER BY id DESC LIMIT 1", conn, params=(kind,))
            if not df.empty and df['hit_rate'].iloc[0] is not None:
                self.last_hit_rate = float(df['hit_rate'].iloc[0])
            else:
                self.last_hit_rate = None
        except Exception:
            pass
        finally:
            conn.close()

    def generate(self, df_info, df_breakdown, df_benchmark=None,
                 window=90, top_n=5, regime_weights=None,
                 use_correlation_filter=True, corr_threshold=0.85,
                 commit_state=True):
        return self._generate(
            df_info, df_breakdown, df_benchmark,
            window, top_n, regime_weights,
            use_correlation_filter, corr_threshold,
            commit_state=commit_state)

    def generate_preview(self, df_info, df_breakdown, df_benchmark=None,
                         window=90, top_n=5, regime_weights=None,
                         use_correlation_filter=True, corr_threshold=0.85):
        return self._generate(
            df_info, df_breakdown, df_benchmark,
            window, top_n, regime_weights,
            use_correlation_filter, corr_threshold,
            commit_state=False)

    def _generate(self, df_info, df_breakdown, df_benchmark,
                  window, top_n, regime_weights,
                  use_correlation_filter, corr_threshold,
                  commit_state=True):
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

        metrics = ProfessionalMetrics.calculate_all(
            df_prices, df_benchmark, window=window)

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

        kelly_w = AdaptiveSizing.kelly_optimal_weights(metrics)
        rp_w = AdaptiveSizing.risk_parity_weights(df_prices,
                                                   window=min(60, window))

        target = {}
        for f in set(kelly_w) | set(smart_scores):
            target[f] = 0.5 * kelly_w.get(f, 0.0) + 0.5 * smart_scores.get(f, 0.0) / 100.0
        if sum(target.values()) > 0:
            t = sum(target.values())
            target = {f: v / t for f, v in target.items()}

        if commit_state:
            online_w = self.online.update(target)
            self.online.save()
        else:
            online_w = dict(self.online.weights)

        if commit_state and self.regime_detector.should_retrain():
            logger.info("[HMM] Otomatik retrain (commit)")
            self.regime_detector.fit(df_benchmark)
            self.regime_detector.save()
        elif self.regime_detector.model is None:
            logger.info("[HMM] İlk eğitim")
            self.regime_detector.fit(df_benchmark)
            if commit_state:
                self.regime_detector.save()

        regime, regime_name, risk_level = self.regime_detector.predict_current_regime(df_benchmark)
        regime_mult = regime_weights.get(regime, 1.0)

        all_f = set(kelly_w) | set(online_w) | set(rp_w) | set(smart_scores)
        if not all_f:
            return None
        fused = {}
        for f in all_f:
            k = kelly_w.get(f, 0.0)
            o = online_w.get(f, 0.0)
            r = rp_w.get(f, 0.0)
            s = smart_scores.get(f, 0.0) / 100.0
            fused[f] = max(0.0, (0.35 * k + 0.25 * o + 0.20 * r + 0.20 * s) * regime_mult)

        top_candidates = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        if use_correlation_filter:
            candidate_funds = [f for f, _ in top_candidates[:top_n * 3]]
            selected = CorrelationFilter.filter_top_n(
                df_prices, candidate_funds, top_n=top_n, threshold=corr_threshold)
            final = {f: fused[f] for f in selected}
        else:
            final = dict(top_candidates[:top_n])

        if not final:
            return None
        t = sum(final.values())
        if t > 0:
            final = {f: v / t for f, v in final.items()}

        final, def_shift = DefensiveAllocator.apply(final, regime, risk_level)

        confidence = self.last_hit_rate if self.last_hit_rate is not None else 0.0

        signal = {
            'date': df_prices.index[-1],
            'regime': regime,
            'regime_name': regime_name,
            'risk_level': risk_level,
            'weights': final,
            'metrics': {f: metrics.get(f, {}) for f in final},
            'smart_scores': {f: smart_scores.get(f, 0) for f in final},
            'learning_rate': self.online.learning_rate,
            'confidence': confidence,
            'n_funds_analyzed': len(all_f),
            'defensive_shift': def_shift,
            'preview_mode': not commit_state,
        }
        if commit_state:
            self.last_signal = signal
        return signal

    @staticmethod
    def explain(signal):
        if not signal:
            return "Sinyal üretilemedi."
        mode = "🔍 **ÖNİZLEME**" if signal.get('preview_mode') else "✅ **KOMİT**"
        lines = [
            f"{mode}",
            f"📅 **Tarih:** {signal['date']}",
            f"🚦 **Rejim:** {signal['regime_name']}  |  "
            f"**Risk:** {RegimeDetector.get_risk_level_name(signal.get('risk_level', 1))}",
        ]
        if signal['confidence'] > 0:
            lines.append(f"🎯 **Güven (Hit Rate):** {signal['confidence']:.0%}")
        else:
            lines.append("🎯 **Güven:** Backtest çalıştırılmadı")
        lines.append(f"📚 **Öğrenme Oranı:** {signal['learning_rate']:.4f}")
        lines.append(f"🔍 **Analiz Edilen Fon:** {signal['n_funds_analyzed']}")
        if signal.get('defensive_shift', 0) > 0:
            lines.append(f"🛡️ **Defansif Kayma:** %{signal['defensive_shift']*100:.1f}")
        lines.append("")
        lines.append("**Önerilen Portföy:**")
        for f, w in signal['weights'].items():
            m = signal['metrics'].get(f, {})
            s = signal['smart_scores'].get(f, 0)
            lines.append(
                f"• **{f}** → %{w*100:.1f} "
                f"(Sharpe: {m.get('Sharpe', 'N/A')}, "
                f"Sortino: {m.get('Sortino', 'N/A')}, "
                f"Max DD: {m.get('Max_DD', 'N/A')}%)")
        return "\n".join(lines)

    def reset_all(self):
        self.online.reset()
        self.regime_detector.reset()
        self._loaded_from_db = False
        self.last_hit_rate = None

    def get_state_summary(self):
        age = get_regime_model_age_days()
        return {
            'weights_count': len(self.online.weights),
            'learning_rate': self.online.learning_rate,
            'initial_lr': self.online.initial_lr,
            'decay': self.online.decay,
            'history_length': len(self.online.history),
            'has_regime_model': self.regime_detector.model is not None,
            'loaded_from_db': self._loaded_from_db,
            'regime_age_days': age,
            'last_hit_rate': self.last_hit_rate,
        }


# =============================================
# 19. BİRİM TESTLERİ
# =============================================
def run_unit_tests():
    results = []
    def check(name, fn):
        try:
            fn()
            results.append((name, 'PASS', ''))
            logger.info(f"[TEST] PASS: {name}")
        except AssertionError as e:
            results.append((name, 'FAIL', str(e)))
            logger.error(f"[TEST] FAIL: {name} - {e}")
        except Exception as e:
            results.append((name, 'ERROR', f"{type(e).__name__}: {e}"))
            logger.error(f"[TEST] ERROR: {name} - {e}")

    def t_corr_filter():
        dates = pd.date_range('2024-01-01', periods=100)
        df = pd.DataFrame({
            'A': np.random.randn(100).cumsum() + 100,
            'C': np.random.randn(100).cumsum() + 100,
        }, index=dates)
        df['B'] = df['A'] * 1.0
        result = CorrelationFilter.filter_top_n(
            df, ['A', 'B', 'C'], top_n=2, threshold=0.9, window=60)
        assert 'A' in result
        if 'A' in result:
            assert 'B' not in result
    check("CorrelationFilter", t_corr_filter)

    def t_regime():
        d = RegimeDetector()
        assert d.get_regime_name(0) == "🐂 BOĞA"
        assert d.get_regime_name(2) == "🐻 AYI"
    check("RegimeDetector labels", t_regime)

    def t_defensive():
        w = {'A': 0.5, 'B': 0.5}
        new_w, shift = DefensiveAllocator.apply(w, regime=2, risk_level=2,
                                                defensive_fund='PPF',
                                                defensive_ratio=0.6)
        assert abs(sum(new_w.values()) - 1.0) < 0.01
        assert 'PPF' in new_w
        assert new_w['PPF'] > 0.5
    check("DefensiveAllocator", t_defensive)

    def t_delta():
        current = {'A': {'units': 100, 'avg_cost': 1.0}}
        target_w = {'A': 0.5, 'B': 0.5}
        prices = {'A': 1.0, 'B': 2.0}
        orders = PortfolioTracker.compute_delta(
            current, target_w, prices, total_value=1000, commission=0.001)
        assert len(orders) >= 1
    check("PortfolioTracker.compute_delta", t_delta)

    def t_metrics_empty():
        m = ProfessionalMetrics.calculate_all(pd.DataFrame())
        assert m == {}
    check("ProfessionalMetrics empty", t_metrics_empty)

    def t_cashflow_empty():
        r = CashFlowAnalyzer.aggregate_flow(pd.DataFrame())
        assert r.empty
    check("CashFlowAnalyzer empty", t_cashflow_empty)

    # >>> YENİ: Overlap fix testi
    def t_overlap_fix():
        """_summarize'ın non-overlap subset kullandığını doğrula."""
        bt = WalkForwardBacktester()
        # 10 sinyal, her biri %2 getiri, freq=7, fwd=20
        details = []
        for i in range(10):
            details.append({
                'signal_date': datetime.date(2025, 1, 1) + timedelta(days=i * 7),
                'forward_returns': {20: 2.0},
                'benchmark_returns': {20: 1.0},
                'is_hit': {20: True},
                'regime': 1,
                'regime_name': 'NÖTR',
                'risk_level': 1,
                'max_dd_realized': -1.0,
            })
        summary = bt._summarize(details, (20,), freq_days=7)
        # step = ceil(20/7) = 3
        # non_overlap = [0, 3, 6, 9] = 4 gözlem
        assert summary['n_non_overlap_20d'] == 4, \
            f"Beklenen 4, gelen {summary['n_non_overlap_20d']}"
        # 4 × %2 = ~%8.24 (compound)
        assert 8.0 < summary['cumulative_return_20d'] < 8.5, \
            f"Compound getiri yanlış: {summary['cumulative_return_20d']}"
    check("Overlap fix", t_overlap_fix)

    return results


# =============================================
# 20. CRON
# =============================================
def run_cron_job(kind='YAT', window=90, top_n=5,
                 analiz_gun=DATA_LOOKBACK_DAYS,
                 force_refresh=False, send_notification=True,
                 run_backtest_first=False):
    start_time = time.time()
    today = datetime.date.today()
    is_reb = is_rebalance_week(today)

    logger.info(f"Cron başladı — kind={kind}, rebalance={is_reb}")

    try:
        if not PYTEFAS_AVAILABLE:
            msg = "pytefas yok"
            log_auto_run('cron', 'ERROR', msg)
            return False

        end = today
        start = end - timedelta(days=max(analiz_gun, DATA_LOOKBACK_DAYS))

        provider = TefasDataProvider()
        df_info = provider.sync_fund_info(kind, start, end, force_refresh)
        if df_info.empty:
            msg = "Info boş"
            log_auto_run('cron', 'ERROR', msg, duration=time.time() - start_time)
            return False

        df_breakdown = provider.sync_breakdown(kind, start, end, force_refresh)

        df_bench = pd.DataFrame()
        if YF_AVAILABLE:
            try:
                b = yf.download("XU100.IS", start=start, end=end,
                                progress=False, auto_adjust=True)
                if not b.empty:
                    if 'Close' in b.columns:
                        b = b['Close']
                    df_bench = b.to_frame("XU100") if isinstance(b, pd.Series) else b
                    logger.info(f"[BENCH] {len(df_bench)} satır")
            except Exception as e:
                logger.warning(f"[BENCH] {e}")

        if run_backtest_first or is_reb:
            logger.info("[CRON] Backtest çalıştırılıyor...")
            try:
                bt = WalkForwardBacktester()
                summary, details = bt.run(
                    df_info, df_breakdown, df_bench, kind=kind,
                    window=window, top_n=top_n)
                if summary:
                    _save_backtest_run(summary, kind, start, end,
                                       window, top_n,
                                       OPERATION_PROFILE['backtest_freq_days'],
                                       details)
                    logger.info(f"[CRON] BT hit_rate={summary.get('hit_rate_20d')}%")
            except Exception as e:
                logger.warning(f"[CRON] BT: {e}")

        engine = SelfLearningSignalEngine(
            learning_rate=OPERATION_PROFILE['learning_rate'],
            decay=OPERATION_PROFILE['decay'],
            auto_load=True)
        signal = engine.generate(
            df_info, df_breakdown, df_bench,
            window=window, top_n=top_n, commit_state=True)

        if signal is None:
            msg = "Sinyal yok"
            log_auto_run('cron', 'ERROR', msg, duration=time.time() - start_time)
            return False

        signal_id = save_ai_signal(signal, kind, is_rebalance=is_reb,
                                    is_preview=False)

        duration = time.time() - start_time
        log_auto_run('cron', 'SUCCESS',
                     f"{len(signal['weights'])} fon, {signal['regime_name']}, "
                     f"risk={signal.get('risk_level', 1)}",
                     n_funds=len(signal['weights']),
                     is_rebalance=is_reb, duration=duration)

        if send_notification and TG_TOKEN and TG_CHAT:
            _send_signal_telegram(signal, is_reb)

        logger.info(f"Cron OK ({duration:.1f}s)")
        return True

    except Exception as e:
        msg = f"Hata: {e}"
        logger.error(msg)
        logger.debug(traceback.format_exc())
        log_auto_run('cron', 'ERROR', msg, duration=time.time() - start_time)
        return False


def _send_signal_telegram(signal, is_reb):
    lines = [
        f"📡 <b>Haftalık TEFAS Sinyali</b>",
        f"📅 {signal['date']}",
        f"🚦 Rejim: {signal['regime_name']}",
        f"⚠️ Risk: {RegimeDetector.get_risk_level_name(signal.get('risk_level', 1))}",
    ]
    if signal['confidence'] > 0:
        lines.append(f"🎯 Güven: {signal['confidence']:.0%}")
    if is_reb:
        lines.insert(0, "⚠️ <b>REBALANCE HAFTASI</b>\n")
    if signal.get('defensive_shift', 0) > 0:
        lines.append(f"🛡️ Defansif: %{signal['defensive_shift']*100:.1f}")
    lines.append("")
    lines.append("<b>Portföy:</b>")
    for f, w in signal['weights'].items():
        lines.append(f"• {f}: %{w*100:.1f}")
    send_telegram("\n".join(lines))


def _save_backtest_run(summary, kind, start_date, end_date,
                       lookback, top_n, freq_days, details):
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("""INSERT INTO backtest_runs
                     (run_at, kind, start_date, end_date, strategy,
                      lookback, top_n, freq_days, hit_rate,
                      cumulative_return, annualized_return, max_drawdown,
                      sharpe, n_signals, details_json)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (datetime.datetime.now().isoformat(timespec='seconds'),
                   kind, str(start_date)[:10], str(end_date)[:10],
                   'fused', int(lookback), int(top_n), int(freq_days),
                   summary.get('hit_rate_20d'),
                   summary.get('cumulative_return_20d'),
                   summary.get('annualized_return_20d'),
                   summary.get('max_drawdown_realized_avg'),
                   summary.get('sharpe_20d'),
                   summary.get('n_signals', 0),
                   json.dumps(summary, default=str)))
        bt_id = c.lastrowid

        for d in details:
            c.execute("""INSERT INTO backtest_signals
                         (backtest_id, signal_date, selected_funds,
                          weights_json, forward_return_20d, forward_return_60d,
                          benchmark_return_20d, benchmark_return_60d,
                          is_hit_20d, is_hit_60d, max_dd_realized,
                          regime, regime_name, risk_level, created_at)
                         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                      (bt_id, str(d['signal_date'])[:10],
                       ', '.join(d['selected_funds']),
                       json.dumps(d['weights'], default=str),
                       d['forward_returns'].get(20),
                       d['forward_returns'].get(60),
                       d['benchmark_returns'].get(20),
                       d['benchmark_returns'].get(60),
                       int(bool(d['is_hit'].get(20))) if d['is_hit'].get(20) is not None else None,
                       int(bool(d['is_hit'].get(60))) if d['is_hit'].get(60) is not None else None,
                       d.get('max_dd_realized'),
                       int(d['regime']), d['regime_name'],
                       int(d.get('risk_level', 1)),
                       datetime.datetime.now().isoformat(timespec='seconds')))
        conn.commit()
        return bt_id
    except Exception as e:
        logger.error(f"_save_backtest_run: {e}")
        return None
    finally:
        conn.close()


def load_backtest_runs(kind=None, limit=20):
    conn = sqlite3.connect(DB_PATH)
    try:
        if kind:
            df = pd.read_sql_query(
                "SELECT * FROM backtest_runs WHERE kind=? ORDER BY id DESC LIMIT ?",
                conn, params=(kind, limit))
        else:
            df = pd.read_sql_query(
                "SELECT * FROM backtest_runs ORDER BY id DESC LIMIT ?",
                conn, params=(limit,))
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()
    return df


# =============================================
# 21. TELEGRAM
# =============================================
def send_telegram(message, token=None, chat_id=None):
    token = token or TG_TOKEN
    chat_id = chat_id or TG_CHAT
    if not REQUESTS_AVAILABLE or not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, data={
            'chat_id': chat_id, 'text': message, 'parse_mode': 'HTML'
        }, timeout=10)
        return r.status_code == 200
    except Exception as e:
        logger.warning(f"send_telegram: {e}")
        return False


# =============================================
# 22. CLI
# =============================================
def main_cli():
    parser = argparse.ArgumentParser(description="TEFAS AI Fon v2.2")
    parser.add_argument("--cron", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--kind", default="YAT", choices=["YAT", "EMK", "BYF"])
    parser.add_argument("--window", type=int, default=90)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--days", type=int, default=DATA_LOOKBACK_DAYS)
    parser.add_argument("--freq", type=int,
                        default=OPERATION_PROFILE['backtest_freq_days'])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--update-rf", type=float, default=None)
    args = parser.parse_args()

    if args.update_rf is not None:
        RiskFreeRateProvider.update_manual(args.update_rf)
        print(f"RF güncellendi: {args.update_rf}")
        return

    if args.test:
        print("=== BİRİM TESTLERİ ===")
        results = run_unit_tests()
        for name, status, msg in results:
            icon = "✅" if status == 'PASS' else "❌"
            print(f"{icon} {name}: {status} {msg}")
        failed = [r for r in results if r[1] != 'PASS']
        sys.exit(0 if not failed else 1)

    if args.status:
        rf, src = get_current_rf_rate()
        print(f"Profil: {OPERATION_PROFILE['name']}")
        print(f"Bugün: {datetime.date.today()}")
        print(f"Sinyal günü: {is_signal_day()}")
        print(f"Rebalance haftası: {is_rebalance_week()}")
        print(f"Sonraki sinyal: {get_next_signal_date()}")
        print(f"Sonraki rebalance: {get_next_rebalance_date()}")
        print(f"RF Rate: {rf} (kaynak: {src})")
        print(f"HMM yaşı: {get_regime_model_age_days()} gün")
        print(f"DB: {os.path.abspath(DB_PATH)}")
        return

    if args.backtest:
        print(f"=== BACKTEST ({args.kind}) ===")
        end = datetime.date.today()
        start = end - timedelta(days=max(args.days, DATA_LOOKBACK_DAYS))
        provider = TefasDataProvider()
        df_info = provider.sync_fund_info(args.kind, start, end, args.force)
        df_breakdown = provider.sync_breakdown(args.kind, start, end, args.force)

        df_bench = pd.DataFrame()
        if YF_AVAILABLE:
            try:
                b = yf.download("XU100.IS", start=start, end=end,
                                progress=False, auto_adjust=True)
                if not b.empty:
                    if 'Close' in b.columns:
                        b = b['Close']
                    df_bench = b.to_frame("XU100") if isinstance(b, pd.Series) else b
            except Exception:
                pass

        bt = WalkForwardBacktester()
        summary, details = bt.run(
            df_info, df_breakdown, df_bench, kind=args.kind,
            signal_freq_days=args.freq,
            window=args.window, top_n=args.top)
        print(json.dumps(summary, indent=2, default=str))
        if summary:
            _save_backtest_run(summary, args.kind, start, end,
                               args.window, args.top, args.freq, details)
            print(f"Kaydedildi. {len(details)} sinyal detayı.")
        return

    if args.cron:
        success = run_cron_job(
            kind=args.kind, window=args.window, top_n=args.top,
            analiz_gun=args.days, force_refresh=args.force,
            send_notification=not args.no_notify)
        sys.exit(0 if success else 1)


# =============================================
# 23. STREAMLIT UI
# =============================================
def main_streamlit():
    st.set_page_config(page_title="TEFAS AI v2.2", layout="wide")
    st.title("🧠 TEFAS AI Fon Yönetim Sistemi v2.2")
    st.caption("Walk-Forward Backtest (Overlap Fix) + HMM Cache + Portföy Takibi")

    if 'engine' not in st.session_state:
        st.session_state.engine = SelfLearningSignalEngine(
            learning_rate=OPERATION_PROFILE['learning_rate'],
            decay=OPERATION_PROFILE['decay'],
            auto_load=True)

    with st.sidebar:
        st.header("⚙️ Ayarlar")
        fon_tipi = st.selectbox(
            "Fon Tipi",
            ["YAT (Yatırım Fonları)", "EMK (Emeklilik Fonları)",
             "BYF (Borsa Yatırım Fonları)"], index=0)
        kind_map = {"YAT (Yatırım Fonları)": "YAT",
                    "EMK (Emeklilik Fonları)": "EMK",
                    "BYF (Borsa Yatırım Fonları)": "BYF"}
        kind = kind_map[fon_tipi]

        analiz_gun = st.slider("Analiz Periyodu (Gün)", 30, 730, 365, 30)
        window_metrics = st.slider("Metrik Penceresi (Gün)", 60, 252, 90, 15)

        st.divider()
        st.subheader("💰 Risksiz Faiz")
        rf_now, rf_src = get_current_rf_rate()
        st.metric("Mevcut RF", f"%{rf_now*100:.2f}", help=f"Kaynak: {rf_src}")
        new_rf = st.number_input("Manuel Güncelle",
                                 value=float(rf_now), step=0.01, format="%.4f")
        if st.button("💾 RF Güncelle"):
            RiskFreeRateProvider.update_manual(new_rf)
            st.success(f"RF = {new_rf}")
            st.rerun()

        st.divider()
        st.subheader("🧠 Motor State")
        try:
            state = st.session_state.engine.get_state_summary()
            ca, cb = st.columns(2)
            ca.metric("Fon", state['weights_count'])
            cb.metric("Geçmiş", state['history_length'])
            st.caption(f"LR: `{state['learning_rate']:.4f}` | "
                       f"Decay: `{state['decay']}`")
            if state['has_regime_model']:
                age = state.get('regime_age_days')
                age_txt = f" ({age}g)" if age is not None else ""
                st.success(f"✅ HMM yüklü{age_txt}")
            else:
                st.warning("⚠️ HMM eğitilmedi")
            if state.get('last_hit_rate') is not None:
                st.info(f"🎯 Hit Rate: %{state['last_hit_rate']:.1f}")
            else:
                st.info("🎯 Hit Rate: Backtest gerekli")
        except Exception as e:
            st.warning(f"State: {e}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("♻️ Sıfırla"):
                st.session_state.engine.reset_all()
                st.success("Sıfırlandı.")
                st.rerun()
        with c2:
            if st.button("💾 Kaydet"):
                st.session_state.engine.online.save()
                st.session_state.engine.regime_detector.save()
                st.success("Kaydedildi.")

        st.divider()
        st.subheader("🎛️ Parametreler")
        top_n = st.slider("Maksimum Fon", 3, 15, 5, 1)
        min_skor = st.slider("Min Akıllı Para", 0, 100, 40, 5)
        use_corr = st.checkbox("Korelasyon filtresi", value=True)
        corr_th = st.slider("Korelasyon eşiği", 0.5, 0.99, 0.85, 0.01)

        st.divider()
        st.caption("⚠️ Yatırım tavsiyesi değildir.")

    tabs = st.tabs([
        "🧠 AI Motoru", "📊 Backtest", "📡 Sinyaller",
        "💼 Portföyüm", "📋 Fon Detayı", "💰 Nakit Akışı",
        "📅 Takvim", "🗄️ Veritabanı", "🧪 Testler"])

    # === AI MOTORU ===
    with tabs[0]:
        st.subheader("🧠 Sinyal Motoru (Önizleme Modu)")
        st.caption("Bu modülde state DEĞİŞMEZ. Commit için cron veya "
                   "'✅ Commit' butonu.")

        if not PYTEFAS_AVAILABLE:
            st.error("❌ pytefas yok.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Önizleme Üret", type="primary"):
                    end = datetime.date.today()
                    start = end - timedelta(days=max(analiz_gun, DATA_LOOKBACK_DAYS))
                    with st.spinner("Veri + önizleme..."):
                        provider = TefasDataProvider()
                        df_info = provider.sync_fund_info(kind, start, end, False)
                        df_breakdown = provider.sync_breakdown(kind, start, end, False)
                    df_bench = _fetch_benchmark(start, end)
                    with st.spinner("Motor (preview)..."):
                        signal = st.session_state.engine.generate_preview(
                            df_info, df_breakdown, df_bench,
                            window=window_metrics, top_n=top_n,
                            use_correlation_filter=use_corr,
                            corr_threshold=corr_th)
                    if signal:
                        st.session_state['preview_signal'] = signal
                        st.success("✅ Önizleme hazır (state değişmedi)")
                    else:
                        st.error("Sinyal üretilemedi.")

            with col2:
                if st.button("✅ Commit (State Kaydet)"):
                    end = datetime.date.today()
                    start = end - timedelta(days=max(analiz_gun, DATA_LOOKBACK_DAYS))
                    with st.spinner("Commit..."):
                        provider = TefasDataProvider()
                        df_info = provider.sync_fund_info(kind, start, end, False)
                        df_breakdown = provider.sync_breakdown(kind, start, end, False)
                    df_bench = _fetch_benchmark(start, end)
                    signal = st.session_state.engine.generate(
                        df_info, df_breakdown, df_bench,
                        window=window_metrics, top_n=top_n,
                        use_correlation_filter=use_corr,
                        corr_threshold=corr_th,
                        commit_state=True)
                    if signal:
                        save_ai_signal(signal, kind,
                                      is_rebalance=is_rebalance_week(),
                                      is_preview=False)
                        st.success("✅ Commit + DB kaydedildi.")
                        st.session_state['committed_signal'] = signal

            signal = st.session_state.get('preview_signal') or \
                     st.session_state.get('committed_signal')
            if signal:
                _render_signal(signal)

    # === BACKTEST ===
    with tabs[1]:
        st.subheader("📊 Walk-Forward Backtest")
        st.caption("Overlap fix + HMM cache. Non-overlap subset kümülatif "
                   "getiri için kullanılır.")

        col1, col2, col3 = st.columns(3)
        with col1:
            bt_freq = st.slider("Sinyal aralığı (iş günü)", 5, 30,
                                 OPERATION_PROFILE['backtest_freq_days'], 1)
        with col2:
            bt_forward = st.multiselect("Forward pencereleri",
                                         [10, 20, 30, 60, 90],
                                         default=[20, 60])
        with col3:
            bt_top = st.slider("Backtest Top-N", 3, 10, 5, 1)

        if st.button("🚀 Backtest Çalıştır", type="primary"):
            end = datetime.date.today()
            start = end - timedelta(days=max(analiz_gun, DATA_LOOKBACK_DAYS))
            with st.spinner("Veri..."):
                provider = TefasDataProvider()
                df_info = provider.sync_fund_info(kind, start, end, False)
                df_breakdown = provider.sync_breakdown(kind, start, end, False)
            df_bench = _fetch_benchmark(start, end)

            with st.spinner("Backtest (HMM cache sayesinde hızlı)..."):
                bt = WalkForwardBacktester()
                summary, details = bt.run(
                    df_info, df_breakdown, df_bench,
                    kind=kind, signal_freq_days=bt_freq,
                    forward_windows=tuple(bt_forward) if bt_forward else (20,),
                    window=window_metrics, top_n=bt_top,
                    use_correlation_filter=use_corr,
                    corr_threshold=corr_th)

            if summary:
                st.session_state['bt_summary'] = summary
                st.session_state['bt_details'] = details
                _save_backtest_run(summary, kind, start, end,
                                   window_metrics, bt_top, bt_freq, details)
                st.success(f"✅ {len(details)} sinyal işlendi. "
                           f"HMM {summary.get('hmm_retrain_count', 0)} kez eğitildi.")
            else:
                st.warning("Backtest sonuç yok.")

        if 'bt_summary' in st.session_state:
            _render_backtest(st.session_state['bt_summary'],
                             st.session_state.get('bt_details', []))

        st.divider()
        st.markdown("### 📜 Geçmiş Backtest'ler")
        df_bt = load_backtest_runs(kind=kind, limit=10)
        if not df_bt.empty:
            st.dataframe(df_bt[['run_at', 'start_date', 'end_date', 'freq_days',
                                 'hit_rate', 'cumulative_return',
                                 'annualized_return', 'max_drawdown',
                                 'sharpe', 'n_signals']],
                         use_container_width=True, hide_index=True)

    # === SİNYALLER ===
    with tabs[2]:
        st.subheader(f"Akıllı Para Skorları — {fon_tipi}")
        if PYTEFAS_AVAILABLE and st.button("🔍 Hesapla", type="primary"):
            end = datetime.date.today()
            start = end - timedelta(days=max(analiz_gun, DATA_LOOKBACK_DAYS))
            with st.spinner("..."):
                provider = TefasDataProvider()
                df_info = provider.sync_fund_info(kind, start, end, False)
                df_breakdown = provider.sync_breakdown(kind, start, end, False)
                df_flow = CashFlowAnalyzer.calculate_net_flow(df_info)
                df_flow_agg = CashFlowAnalyzer.aggregate_flow(df_flow, analiz_gun)
                df_signals = SignalGenerator.generate_signals(
                    df_info, df_flow_agg, df_breakdown)
            if df_signals.empty:
                st.warning("Sinyal yok.")
            else:
                df_f = df_signals[df_signals["Toplam Skor"] >= min_skor]
                st.dataframe(df_f, use_container_width=True, hide_index=True)

    # === PORTFÖYÜM ===
    with tabs[3]:
        st.subheader("💼 Gerçek Portföy Takibi")
        with st.expander("📝 Pozisyonları Düzenle", expanded=False):
            user_port = PortfolioTracker.load_user_portfolio(kind)
            if user_port:
                df_up = pd.DataFrame([
                    {'Fon': k, 'Adet': v['units'], 'Ort. Maliyet': v['avg_cost']}
                    for k, v in user_port.items()])
                st.dataframe(df_up, use_container_width=True, hide_index=True)

            st.markdown("**Yeni/Güncelle**")
            c1, c2, c3 = st.columns(3)
            with c1:
                new_fund = st.text_input("Fon kodu", value="").strip().upper()
            with c2:
                new_units = st.number_input("Adet", value=0.0, step=1.0)
            with c3:
                new_cost = st.number_input("Ort. maliyet", value=0.0, step=0.01)
            if st.button("➕ Ekle"):
                if new_fund:
                    PortfolioTracker.save_user_position(
                        kind, new_fund, new_units, new_cost)
                    st.success(f"{new_fund} kaydedildi.")
                    st.rerun()
            if st.button("🗑️ Tümünü Sil"):
                PortfolioTracker.clear_user_portfolio(kind)
                st.warning("Silindi.")
                st.rerun()

        if st.button("🔄 Al/Sat Delta Hesapla", type="primary"):
            user_port = PortfolioTracker.load_user_portfolio(kind)
            signal = st.session_state.get('preview_signal') or \
                     st.session_state.get('committed_signal')
            if not user_port:
                st.warning("Pozisyon girin.")
            elif not signal:
                st.warning("Sinyal üretin.")
            else:
                end = datetime.date.today()
                start = end - timedelta(days=30)
                provider = TefasDataProvider()
                df_info = provider.sync_fund_info(kind, start, end, False)
                df_prices = df_info.pivot_table(
                    index='date', columns='fund_code', values='price').ffill()
                latest = df_prices.iloc[-1].to_dict() if not df_prices.empty else {}
                orders = PortfolioTracker.compute_delta(
                    user_port, signal['weights'], latest,
                    total_value=None, commission=0.001)
                if orders:
                    df_orders = pd.DataFrame(orders)
                    st.dataframe(df_orders, use_container_width=True,
                                 hide_index=True)
                    total_cost = sum(o['cost'] for o in orders)
                    st.metric("Toplam Maliyet", f"{total_cost:,.2f} ₺")
                    if st.button("💾 Emirleri Kaydet"):
                        PortfolioTracker.save_trade_orders(
                            orders, kind, str(signal['date'])[:10])
                        st.success("Kaydedildi.")
                else:
                    st.info("Delta yok.")

    # === FON DETAYI ===
    with tabs[4]:
        st.subheader("📋 Fon Detayı")
        if PYTEFAS_AVAILABLE:
            fon_kodu = st.text_input("Fon Kodu", value="AAK")
            if st.button("🔎 Analiz Et"):
                end = datetime.date.today()
                start = end - timedelta(days=max(analiz_gun, DATA_LOOKBACK_DAYS))
                with st.spinner(f"{fon_kodu}..."):
                    provider = TefasDataProvider()
                    provider.sync_fund_info(kind, start, end, False)
                    provider.sync_breakdown(kind, start, end, False)
                    df_info = provider.get_fund_history(fon_kodu, kind, start, end)
                if df_info.empty:
                    st.warning("Veri yok.")
                else:
                    son = df_info.iloc[-1]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Fiyat", f"{son.get('price', 0) or 0:.4f} ₺")
                    c2.metric("AUM", f"{son.get('portfolio_size', 0) or 0:,.0f}")
                    c3.metric("Yatırımcı", f"{son.get('investor_count', 0) or 0:,.0f}")
                    c4.metric("Pay", f"{son.get('shares_outstanding', 0) or 0:,.0f}")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_info["date"], y=df_info["price"],
                                             name="Fiyat"))
                    st.plotly_chart(fig, use_container_width=True)

    # === NAKİT AKIŞI ===
    with tabs[5]:
        st.subheader("💰 Nakit Akışları")
        if PYTEFAS_AVAILABLE and st.button("💸 Analiz"):
            end = datetime.date.today()
            start = end - timedelta(days=max(analiz_gun, DATA_LOOKBACK_DAYS))
            provider = TefasDataProvider()
            df_info = provider.sync_fund_info(kind, start, end, False)
            df_flow = CashFlowAnalyzer.calculate_net_flow(df_info)
            df_agg = CashFlowAnalyzer.aggregate_flow(df_flow, analiz_gun)
            if not df_agg.empty:
                df_agg = df_agg.sort_values("toplam_net_akis", ascending=False)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 🟢 Giren")
                    st.dataframe(df_agg.head(10), use_container_width=True,
                                 hide_index=True)
                with c2:
                    st.markdown("#### 🔴 Çıkan")
                    st.dataframe(df_agg.tail(10), use_container_width=True,
                                 hide_index=True)

    # === TAKVİM ===
    with tabs[6]:
        st.subheader("📅 Takvim")
        today = datetime.date.today()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bugün", today.strftime("%d %b (%a)"))
        c2.metric("Sinyal", "✅" if is_signal_day(today) else "❌")
        c3.metric("Rebalance", "✅" if is_rebalance_week(today) else "❌")
        c4.metric("Sonraki", get_next_signal_date(today).strftime("%d %b"))

        st.markdown(f"""
| Parametre | Değer |
|---|---|
| Sinyal | Her Cuma 19:35 |
| Rebalance | Ayın ilk Cuma'sı |
| LR / Decay | {OPERATION_PROFILE['learning_rate']} / {OPERATION_PROFILE['decay']} |
| HMM retrain | Her {OPERATION_PROFILE['hmm_retrain_days']} gün |
| Backtest freq | Her {OPERATION_PROFILE['backtest_freq_days']} iş günü |
| Backtest HMM | Her {OPERATION_PROFILE['backtest_hmm_interval_days']} gün retrain |
| Defansif | {OPERATION_PROFILE['defensive_fund']} |
        """)

        st.divider()
        st.markdown("### 📊 Son Cron Çalışmaları")
        df_runs = load_auto_runs(limit=20)
        if not df_runs.empty:
            st.dataframe(df_runs, use_container_width=True, hide_index=True)

    # === VERİTABANI ===
    with tabs[7]:
        st.subheader("🗄️ Veritabanı")
        try:
            conn = sqlite3.connect(DB_PATH)
            si = pd.read_sql_query(
                """SELECT kind, COUNT(DISTINCT date) AS gun, COUNT(*) AS satir,
                          MIN(date) AS ilk, MAX(date) AS son
                   FROM fund_info GROUP BY kind""", conn)
            conn.close()
            st.dataframe(si, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"DB: {e}")

        if os.path.exists(DB_PATH):
            st.metric("DB Boyutu",
                      f"{os.path.getsize(DB_PATH) / 1024 / 1024:.2f} MB")

        st.divider()
        st.markdown("### 📜 AI Sinyal Geçmişi")
        df_sig = load_ai_signals(kind=kind, limit=50)
        if not df_sig.empty:
            st.dataframe(df_sig, use_container_width=True, hide_index=True)

    # === TESTLER ===
    with tabs[8]:
        st.subheader("🧪 Birim Testleri")
        if st.button("▶️ Testleri Çalıştır"):
            with st.spinner("..."):
                results = run_unit_tests()
            for name, status, msg in results:
                if status == 'PASS':
                    st.success(f"✅ {name}: PASS")
                else:
                    st.error(f"❌ {name}: {status} — {msg}")

        st.divider()
        st.markdown("### 📋 Loglar")
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, 'r', encoding='utf-8') as f:
                logs = f.readlines()
            st.code(''.join(logs[-100:]), language='text')


# ---------- UI YARDIMCILARI ----------
def _fetch_benchmark(start, end):
    if not YF_AVAILABLE:
        return pd.DataFrame()
    try:
        b = yf.download("XU100.IS", start=start, end=end,
                        progress=False, auto_adjust=True)
        if b.empty:
            return pd.DataFrame()
        if 'Close' in b.columns:
            b = b['Close']
        return b.to_frame("XU100") if isinstance(b, pd.Series) else b
    except Exception as e:
        logger.warning(f"_fetch_benchmark: {e}")
        return pd.DataFrame()


def _render_signal(signal):
    icons = {0: "🟢", 1: "🟡", 2: "🔴"}
    st.markdown(f"## {icons.get(signal['regime'], '⚪')} {signal['regime_name']} "
                f"| {RegimeDetector.get_risk_level_name(signal.get('risk_level', 1))}")
    c1, c2, c3, c4 = st.columns(4)
    conf_txt = f"{signal['confidence']:.0%}" if signal['confidence'] > 0 else "—"
    c1.metric("Güven (Hit Rate)", conf_txt)
    c2.metric("Öğrenme Oranı", f"{signal['learning_rate']:.4f}")
    c3.metric("Analiz Edilen", signal['n_funds_analyzed'])
    c4.metric("Seçilen", len(signal['weights']))

    if signal.get('defensive_shift', 0) > 0:
        st.warning(f"🛡️ Defansif: %{signal['defensive_shift']*100:.1f} "
                   f"→ {OPERATION_PROFILE['defensive_fund']}")

    rows = []
    for f, w in signal['weights'].items():
        m = signal['metrics'].get(f, {})
        rows.append({
            'Fon': f, 'Ağırlık (%)': f"%{w*100:.1f}",
            'Sharpe': m.get('Sharpe', '-'),
            'Sortino': m.get('Sortino', '-'),
            'Max DD (%)': m.get('Max_DD', '-'),
            'Akıllı Para': f"{signal['smart_scores'].get(f, 0):.0f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("### 📝 Açıklama")
    st.markdown(SelfLearningSignalEngine.explain(signal))


def _render_backtest(summary, details):
    st.markdown("### 📊 Backtest Özeti")
    n_sig = summary.get('n_signals', 0)
    n_retrain = summary.get('hmm_retrain_count', 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sinyal Sayısı", n_sig)
    c2.metric("HMM Retrain", n_retrain)
    c3.metric("Frekans (gün)", summary.get('freq_days', '—'))
    c4.metric("Overlap Step", summary.get('overlap_step_20d', '—'))

    if n_sig < 15:
        st.warning(f"⚠️ Sadece {n_sig} sinyal — istatistik zayıf.")

    st.divider()
    st.markdown("#### 20 Günlük Forward")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hit Rate",
              f"%{summary.get('hit_rate_20d', 0):.1f}"
              if summary.get('hit_rate_20d') is not None else "—")
    c2.metric("Kümülatif (non-overlap)",
              f"%{summary.get('cumulative_return_20d', 0):.2f}")
    c3.metric("Yıllık",
              f"%{summary.get('annualized_return_20d', 0):.2f}")
    c4.metric("Sharpe", f"{summary.get('sharpe_20d', 0):.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Ort. Getiri (avg)",
              f"%{summary.get('avg_return_20d', 0):.3f}")
    c6.metric("Bench Cum",
              f"%{summary.get('benchmark_cumulative_20d', 0):.2f}"
              if summary.get('benchmark_cumulative_20d') is not None else "—")
    c7.metric("Fazla Getiri",
              f"%{summary.get('excess_return_20d', 0):.2f}"
              if summary.get('excess_return_20d') is not None else "—")
    c8.metric("Non-overlap N",
              summary.get('n_non_overlap_20d', '—'))

    st.divider()
    st.markdown("#### 60 Günlük Forward")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hit Rate",
              f"%{summary.get('hit_rate_60d', 0):.1f}"
              if summary.get('hit_rate_60d') is not None else "—")
    c2.metric("Kümülatif (non-overlap)",
              f"%{summary.get('cumulative_return_60d', 0):.2f}")
    c3.metric("Yıllık",
              f"%{summary.get('annualized_return_60d', 0):.2f}")
    c4.metric("Sharpe", f"{summary.get('sharpe_60d', 0):.2f}")

    st.divider()
    st.markdown("#### Risk Metrikleri")
    c1, c2 = st.columns(2)
    c1.metric("Ort. Drawdown (gerçek)",
              f"%{summary.get('max_drawdown_realized_avg', 0):.2f}"
              if summary.get('max_drawdown_realized_avg') is not None else "—")
    c2.metric("En Kötü Drawdown",
              f"%{summary.get('max_drawdown_realized_worst', 0):.2f}"
              if summary.get('max_drawdown_realized_worst') is not None else "—")

    if details:
        st.markdown("### 📈 Sinyal Bazında")
        rows = []
        for d in details:
            rows.append({
                'Tarih': str(d['signal_date'])[:10],
                'Rejim': d['regime_name'],
                'Fonlar': ', '.join(d['selected_funds'][:3]) +
                          ('...' if len(d['selected_funds']) > 3 else ''),
                'Get 20g (%)': d['forward_returns'].get(20, '-'),
                'Bench 20g (%)': d['benchmark_returns'].get(20, '-'),
                'Hit 20g': '✅' if d['is_hit'].get(20) else '❌'
                    if d['is_hit'].get(20) is not None else '—',
                'Max DD (%)': d.get('max_dd_realized', '-'),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

        try:
            returns_20d = [d['forward_returns'].get(20, 0) or 0 for d in details]
            dates = [d['signal_date'] for d in details]
            equity = (1 + np.array(returns_20d) / 100).cumprod()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=equity, name='Portföy',
                                     line=dict(color='green')))
            fig.update_layout(title="Kümülatif (20g forward, overlap)",
                              xaxis_title="Tarih", yaxis_title="Equity")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Grafik: {e}")


# =============================================
# 24. ENTRY POINT
# =============================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and any(a in sys.argv for a in
                                  ['--cron', '--status', '--test',
                                   '--backtest', '--help', '--update-rf']):
        main_cli()
    else:
        main_streamlit()