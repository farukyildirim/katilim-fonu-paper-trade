import streamlit as st
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import plotly.graph_objects as go
import plotly.express as px
import requests
import os
import re
import json
import yfinance as yf
import warnings
import time
import math
import sqlite3
import concurrent.futures

# Kripto için ccxt
try:
    import ccxt
except ImportError:
    ccxt = None

# TEFAS için
try:
    from tefas import Crawler
except ImportError:
    Crawler = None

warnings.filterwarnings('ignore')


# =============================================
# 0. GÜVENLİ LLM PARSER (LLMOutputParser)
# =============================================
class LLMOutputParser:
    @staticmethod
    def parse_json(raw_text: str):
        """
        Markdown blokları (```json), <think> tag'leri veya eksik JSON çıktılarını
        Regex + Fallback mekanizmalarıyla temizler ve güvenli dict döndürür.
        """
        if not raw_text or not isinstance(raw_text, str):
            return None

        try:
            # 1. <think> ... </think> bloklarını temizle
            cleaned = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
            # 2. ```json ... ``` veya ``` ... ``` kod bloklarını ayıkla
            cleaned = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', cleaned, flags=re.DOTALL).strip()

            # 3. En dıştaki JSON süslü parantezlerini regex ile yakala
            json_match = re.search(r'(\{.*\})', cleaned, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            return json.loads(cleaned)
        except Exception:
            # Fallback: String içinden key-value tarzı sayı yakalama denemesi
            try:
                pairs = re.findall(r'"([^"]+)":\s*([0-9.]+)', raw_text)
                if pairs:
                    return {"DAGILIM": {k: float(v) for k, v in pairs}}
            except Exception:
                pass
            return None


# =============================================
# 1. VALÖR VE ŞİŞME ÖNLENEN MUHASEBE (PortfolioLedger)
# =============================================
class PortfolioLedger:
    def __init__(self, initial_capital=100000.0):
        self.cash = float(initial_capital)
        self.positions = {}  # {asset_code: amount}
        self.pending_cash_settlements = []  # [{'settlement_date': dt, 'amount': float}]

    def process_settlements(self, current_date):
        """Valör süresi dolan nakitleri ana bakiyeye aktarır."""
        current_dt = pd.to_datetime(current_date)
        settled_amount = 0.0
        remaining_settlements = []
        for item in self.pending_cash_settlements:
            if current_dt >= pd.to_datetime(item['settlement_date']):
                settled_amount += item['amount']
            else:
                remaining_settlements.append(item)
        self.pending_cash_settlements = remaining_settlements
        self.cash += settled_amount

    def sell_asset(self, asset_code, amount, price, asset_params, current_date):
        """Satış gelirini doğrudan bakiyeye eklemez, valör kuyruğuna alınır."""
        if asset_code not in self.positions or self.positions[asset_code] < amount:
            return 0.0

        gross = amount * price
        komisyon_rate = asset_params.get('komisyon_satis', 0.005)
        net = gross * (1 - komisyon_rate)

        # Pozisyonu düşür
        self.positions[asset_code] -= amount
        if self.positions[asset_code] <= 1e-6:
            del self.positions[asset_code]

        # Valör gün sayısına göre kuyruğa ekle
        valor_days = asset_params.get('valor', 2)
        settlement_date = pd.to_datetime(current_date) + pd.Timedelta(days=valor_days)

        if valor_days == 0:
            self.cash += net
        else:
            self.pending_cash_settlements.append({
                'settlement_date': settlement_date,
                'amount': net
            })
        return net

    def buy_asset(self, asset_code, cash_to_use, price, asset_params):
        """Komisyon kesintisini uygulayarak varlık satın alır."""
        if cash_to_use > self.cash:
            cash_to_use = self.cash

        komisyon_rate = asset_params.get('komisyon_alis', 0.005)
        net_investment = cash_to_use * (1 - komisyon_rate)
        units = net_investment / price

        self.cash -= cash_to_use
        self.positions[asset_code] = self.positions.get(asset_code, 0.0) + units
        return units

    def get_total_value(self, current_prices):
        """
        Portföy değerini mükerrer nakit eklenmesini engeller.
        Toplam = Serbest Nakit + Bekleyen Valörlü Nakit + Varlıkların Anlık Piyasa Değeri
        """
        asset_val = sum(
            units * current_prices.get(asset, 0.0)
            for asset, units in self.positions.items()
        )
        pending_val = sum(item['amount'] for item in self.pending_cash_settlements)
        return self.cash + pending_val + asset_val


# =============================================
# 2. GELİŞMİŞ RİSK METRİKLERİ (PerformanceAnalytics)
# =============================================
class PerformanceAnalytics:
    @staticmethod
    def calculate_metrics(df_history, risk_free_rate=0.40):
        """Sharpe, Sortino ve Calmar oranlarını detaylı hesaplar."""
        if df_history.empty or len(df_history) < 2:
            return {}

        df = df_history.copy()
        daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
        returns = df['Toplam_Varlik'].pct_change().dropna()

        total_return = (df['Toplam_Varlik'].iloc[-1] - df['Toplam_Varlik'].iloc[0]) / df['Toplam_Varlik'].iloc[0]
        ann_return = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)

        # Sharpe Oranı
        excess_returns = returns - daily_rf
        sharpe = (excess_returns.mean() * 252) / (ann_vol + 1e-9)

        # Sortino Oranı (Aşağı Yönlü Sapma / Downside Risk)
        negative_returns = returns[returns < daily_rf] - daily_rf
        downside_std = np.sqrt(np.mean(negative_returns ** 2)) * np.sqrt(252) if len(negative_returns) > 0 else 1e-9
        sortino = (ann_return - risk_free_rate) / (downside_std + 1e-9)

        # Max Drawdown & Calmar Oranı
        rolling_max = df['Toplam_Varlik'].cummax()
        drawdowns = (df['Toplam_Varlik'] - rolling_max) / rolling_max
        max_dd = drawdowns.min()
        calmar = (ann_return) / (abs(max_dd) + 1e-9)

        # VaR & CVaR
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else np.nan
        var_99 = np.percentile(returns, 1) if len(returns) > 0 else np.nan
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else np.nan

        return {
            'Toplam Getiri (%)': total_return * 100,
            'Yıllık Getiri (%)': ann_return * 100,
            'Yıllık Volatilite (%)': ann_vol * 100,
            'Sharpe Oranı': sharpe,
            'Sortino Oranı': sortino,
            'Calmar Oranı': calmar,
            'Max Drawdown (%)': max_dd * 100,
            'VaR %95': var_95 * 100,
            'VaR %99': var_99 * 100,
            'CVaR %95': cvar_95 * 100
        }


# =============================================
# 3. SQLITE VERİTABANI
# =============================================
DB_PATH = "portfolio.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS portfolios (
                    asset_class TEXT PRIMARY KEY,
                    nakit REAL,
                    varliklar TEXT,
                    islemler TEXT,
                    baslangic_bakiye REAL,
                    pending_cash TEXT,
                    son_guncelleme TEXT
                 )''')
    c.execute('''CREATE TABLE IF NOT EXISTS allocations (
                    asset_class TEXT PRIMARY KEY,
                    hedef_oran REAL,
                    min_oran REAL DEFAULT 0,
                    max_oran REAL DEFAULT 1
                 )''')
    c.execute('''CREATE TABLE IF NOT EXISTS signal_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset_class TEXT,
                    tarih TEXT,
                    secilen TEXT,
                    agirliklar TEXT,
                    sonuc_getiri REAL
                 )''')
    conn.commit()
    conn.close()


def migrate_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("ALTER TABLE allocations ADD COLUMN min_oran REAL DEFAULT 0")
        c.execute("ALTER TABLE allocations ADD COLUMN max_oran REAL DEFAULT 1")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE portfolios ADD COLUMN pending_cash TEXT DEFAULT '[]'")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


init_db()
migrate_db()

# =============================================
# 4. VERİ PARQUET ÖN BELLEK VE SAĞLAYICILAR
# =============================================
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def get_cache_path(asset_class):
    return os.path.join(CACHE_DIR, f"fon_fiyatlari_{asset_class}.parquet")


def fon_verilerini_yukle(provider, start_date, end_date, asset_class):
    cache_path = get_cache_path(asset_class)
    if os.path.exists(cache_path):
        try:
            df_mevcut = pd.read_parquet(cache_path)
            df_mevcut.index = pd.to_datetime(df_mevcut.index)
            son_tarih = df_mevcut.index.max().date()
            bugun = datetime.date.today()
            if (bugun - son_tarih).days <= 1:
                return df_mevcut
            else:
                baslangic = son_tarih + timedelta(days=1)
                df_yeni = provider.fetch_prices(baslangic, end_date)
                if not df_yeni.empty:
                    df_mevcut = pd.concat([df_mevcut, df_yeni], axis=0).sort_index()
                    df_mevcut = df_mevcut.ffill().dropna(axis=1, how='all')
                    df_mevcut.to_parquet(cache_path)
                return df_mevcut
        except Exception:
            if os.path.exists(cache_path):
                os.remove(cache_path)
    df_pivot = provider.fetch_prices(start_date, end_date)
    if not df_pivot.empty:
        df_pivot.to_parquet(cache_path)
    return df_pivot


class YahooProvider:
    def __init__(self):
        self.symbols = {}

    def set_symbols(self, symbols_dict):
        self.symbols = symbols_dict

    def fetch_prices(self, start_date, end_date):
        all_data = {}
        for name, symbol in self.symbols.items():
            try:
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not df.empty and 'Close' in df.columns:
                    all_data[name] = df['Close']
            except Exception:
                continue
        if not all_data:
            return pd.DataFrame()
        df_result = pd.concat(all_data, axis=1)
        df_result.columns = [col[0] if isinstance(col, tuple) else col for col in df_result.columns]
        return df_result


class BinanceProvider:
    def __init__(self):
        self.symbols = {}
        self.exchange = ccxt.binance() if ccxt else None

    def set_symbols(self, symbols_dict):
        self.symbols = symbols_dict

    def fetch_prices(self, start_date, end_date, timeframe='1d'):
        if self.exchange is None:
            return pd.DataFrame()
        all_data = {}
        for name, symbol in self.symbols.items():
            try:
                since = int(time.mktime(start_date.timetuple())) * 1000
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('date')
                df = df.loc[start_date:end_date]
                if not df.empty:
                    all_data[name] = df['close']
            except Exception:
                continue
        if not all_data:
            return pd.DataFrame()
        return pd.concat(all_data, axis=1)


class TefasProvider:
    def __init__(self):
        self.crawler = Crawler(fund_limit=500) if Crawler else None

    def fetch_prices(self, start_date, end_date):
        if self.crawler is None:
            return pd.DataFrame()
        try:
            crawler_fonlar = Crawler(fund_limit=2000)
            bugun = pd.Timestamp.today()
            baslangic = bugun - pd.Timedelta(days=3)
            df_kod = crawler_fonlar.fetch(start=baslangic.strftime("%Y-%m-%d"), end=bugun.strftime("%Y-%m-%d"))
            if df_kod is None or df_kod.empty:
                return pd.DataFrame()
            kod_kolonu = "code" if "code" in df_kod.columns else "fon_kodu"
            tum_kodlar = df_kod[kod_kolonu].dropna().unique().tolist()
            df_list = []
            for fund in tum_kodlar[:150]:
                try:
                    df_fund = self.crawler.fetch(
                        start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d"),
                        name=fund,
                        columns=["code", "date", "price"]
                    )
                    if not df_fund.empty:
                        df_list.append(df_fund)
                except Exception:
                    continue
            if not df_list:
                return pd.DataFrame()
            df_raw = pd.concat(df_list, ignore_index=True)
            df_pivot = df_raw.pivot(index="date", columns="code", values="price")
            df_pivot.index = pd.to_datetime(df_pivot.index)
            return df_pivot.sort_index().ffill().dropna(axis=1, how='all')
        except Exception:
            return pd.DataFrame()


# =============================================
# 5. PARAMETRELER VE VARLIK EVRENİ
# =============================================
DEFAULT_UNIVERSE = {
    'hisse_bist': {'XU100': 'XU100.IS', 'THYAO': 'THYAO.IS', 'GARAN': 'GARAN.IS', 'AKBNK': 'AKBNK.IS',
                   'SISE': 'SISE.IS'},
    'hisse_abd': {'AAPL': 'AAPL', 'MSFT': 'MSFT', 'GOOGL': 'GOOGL', 'AMZN': 'AMZN', 'META': 'META', 'SPY': 'SPY'},
    'hisse_avrupa': {'SAP': 'SAP.DE', 'ASML': 'ASML.AS', 'NOVO': 'NOVO-B.CO', 'NESN': 'NESN.SW'},
    'kripto': {'BTC': 'BTC/USDT', 'ETH': 'ETH/USDT', 'BNB': 'BNB/USDT', 'SOL': 'SOL/USDT'},
    'emtia': {'Altın': 'GC=F', 'Gümüş': 'SI=F', 'Petrol': 'CL=F'},
    'doviz': {'EUR/USD': 'EURUSD=X', 'GBP/USD': 'GBPUSD=X', 'USD/JPY': 'USDJPY=X'},
    'etf': {'SPY': 'SPY', 'QQQ': 'QQQ', 'VTI': 'VTI', 'BND': 'BND'}
}

# Fon Türüne Özel Dinamik Komisyon ve Valör
ASSET_PARAMS = {
    'hisse_bist': {'komisyon_alis': 0.0020, 'komisyon_satis': 0.0020, 'valor': 2, 'max_position': 0.70},
    'hisse_abd': {'komisyon_alis': 0.0015, 'komisyon_satis': 0.0015, 'valor': 2, 'max_position': 0.70},
    'hisse_avrupa': {'komisyon_alis': 0.0015, 'komisyon_satis': 0.0015, 'valor': 2, 'max_position': 0.70},
    'kripto': {'komisyon_alis': 0.0010, 'komisyon_satis': 0.0010, 'valor': 0, 'max_position': 0.80},
    'emtia': {'komisyon_alis': 0.0010, 'komisyon_satis': 0.0010, 'valor': 1, 'max_position': 0.60},
    'doviz': {'komisyon_alis': 0.0005, 'komisyon_satis': 0.0005, 'valor': 0, 'max_position': 0.90},
    'etf': {'komisyon_alis': 0.0010, 'komisyon_satis': 0.0010, 'valor': 2, 'max_position': 0.70},
    'tefas_fon': {'komisyon_alis': 0.0001, 'komisyon_satis': 0.0001, 'valor': 3, 'max_position': 0.70}
}

STRATEGY_TYPES = {
    'Momentum': 'momentum',
    'Değer (F/K)': 'value',
    'Düşük Volatilite': 'low_vol',
    'Carry (Taşıma)': 'carry',
    'Karma (Momentum+Değer)': 'blend'
}


# =============================================
# 6. STRATEJİ VE TREND/VOLATİLİTE FİLTRELERİ
# =============================================
def calculate_momentum(df_prices, lookback=30):
    if len(df_prices) < lookback:
        return pd.Series(index=df_prices.columns, data=0.0)
    recent = df_prices.tail(lookback)
    return (recent.iloc[-1] / recent.iloc[0] - 1)


def calculate_value(df_prices, lookback=30):
    if len(df_prices) < lookback:
        return pd.Series(index=df_prices.columns, data=0.0)
    recent = df_prices.tail(lookback)
    mean_price = recent.mean()
    current = df_prices.iloc[-1]
    return (1 - (current / mean_price)).clip(0, 1)


def calculate_low_volatility(df_prices, lookback=30):
    if len(df_prices) < lookback:
        return pd.Series(index=df_prices.columns, data=0.0)
    rets = df_prices.pct_change().tail(lookback)
    vols = rets.std()
    return 1 - (vols / vols.max())


def calculate_carry(df_prices, lookback=30):
    if len(df_prices) < lookback:
        return pd.Series(index=df_prices.columns, data=0.0)
    return df_prices.pct_change().tail(lookback).mean()


def get_strategy_scores(df_prices, strategy_type, lookback=30):
    if strategy_type == 'momentum':
        return calculate_momentum(df_prices, lookback)
    elif strategy_type == 'value':
        return calculate_value(df_prices, lookback)
    elif strategy_type == 'low_vol':
        return calculate_low_volatility(df_prices, lookback)
    elif strategy_type == 'carry':
        return calculate_carry(df_prices, lookback)
    elif strategy_type == 'blend':
        mom = calculate_momentum(df_prices, lookback)
        val = calculate_value(df_prices, lookback)
        return 0.6 * mom + 0.4 * val
    return calculate_momentum(df_prices, lookback)


def _check_trend_regime(df_benchmark, current_date, window=200):
    """Benchmark endeksinin 200 SMA altındaysa Ayı rejimi uyarısı verir."""
    if df_benchmark is None or df_benchmark.empty or current_date not in df_benchmark.index:
        return False
    bench_series = df_benchmark.loc[:current_date]
    if len(bench_series) < window:
        return False
    col = bench_series.columns[0] if isinstance(bench_series, pd.DataFrame) else bench_series.name
    series = bench_series[col] if isinstance(bench_series, pd.DataFrame) else bench_series
    sma_200 = series.tail(window).mean()
    return series.iloc[-1] < sma_200


def _apply_volatility_targeting(weights, df_prices_sub, target_volatility=0.15, window=30):
    """Portföyün gerçekleşen 30 günlük oynaklığı target_volatility aştığında nakde geçer."""
    if df_prices_sub.empty or len(df_prices_sub) < window:
        return weights
    returns = df_prices_sub.tail(window).pct_change().dropna()
    asset_cols = [c for c in weights.keys() if c in returns.columns]
    if not asset_cols:
        return weights
    sub_weights = np.array([weights[c] for c in asset_cols])
    if sub_weights.sum() == 0:
        return weights
    sub_weights = sub_weights / sub_weights.sum()
    portfolio_returns = (returns[asset_cols] * sub_weights).sum(axis=1)
    realized_vol = portfolio_returns.std() * np.sqrt(252)

    if realized_vol > target_volatility and realized_vol > 0:
        scale_factor = target_volatility / realized_vol
        return {k: v * scale_factor for k, v in weights.items()}
    return weights


def get_ai_weights(funds, df_prices_sub=None, target_vol=0.15, is_bearish=False):
    """Güvenli LLM parser destekli ağırlık oluşturucu."""
    base_weight = 1.0 / len(funds)
    weights = {f: base_weight for f in funds}

    if is_bearish:
        weights = {f: w * 0.5 for f, w in weights.items()}

    if df_prices_sub is not None:
        weights = _apply_volatility_targeting(weights, df_prices_sub, target_volatility=target_vol)

    return weights, "Ağırlıklar Rejim & Volatilite Filtreleriyle Hesaplandı."


# =============================================
# 7. BACKTEST MOTORU
# =============================================
def run_backtest(df_prices, initial_capital=100000, stop_loss_pct=0.12, top_n=3, lookback=30,
                 max_position_pct=0.70, asset_class='hisse_bist', strategy_type='momentum',
                 df_benchmark=None, target_vol=0.15):
    df = df_prices.sort_index().copy()
    if df.empty or len(df) < 30:
        return pd.DataFrame(), [], {}

    asset_params = ASSET_PARAMS.get(asset_class, {'komisyon_alis': 0.002, 'komisyon_satis': 0.002, 'valor': 2})
    ledger = PortfolioLedger(initial_capital=initial_capital)

    dates = df.index
    first_day_of_month = dates[dates.is_month_start].unique()
    if len(first_day_of_month) == 0:
        first_day_of_month = [dates[0]]

    history = []
    decision_log = []

    for i, current_date in enumerate(dates):
        ledger.process_settlements(current_date)
        current_prices = df.loc[current_date].to_dict()

        # Faiz / Repo Nemalandırması (%15 Yıllık)
        ledger.cash += ledger.cash * (0.15 / 365.0)

        toplam_deger = ledger.get_total_value(current_prices)
        pending_val = sum(item['amount'] for item in ledger.pending_cash_settlements)

        history.append({
            'Tarih': current_date,
            'Toplam_Varlik': toplam_deger,
            'Nakit': ledger.cash,
            'Portfoy_Degeri': toplam_deger - ledger.cash - pending_val,
            'Bloke_Nakit': pending_val
        })

        # Stop-Loss Kontrolü
        if toplam_deger < initial_capital * (1 - stop_loss_pct):
            for asset, amount in list(ledger.positions.items()):
                if amount > 1e-6 and asset in current_prices:
                    ledger.sell_asset(asset, amount, current_prices[asset], asset_params, current_date)
            decision_log.append({'Tarih': current_date.strftime('%Y-%m-%d'), 'Seçilen Fonlar': 'STOP-LOSS TRIGGER'})
            continue

        # Aylık Rebalance
        if current_date in first_day_of_month and len(dates) - i > 5:
            df_sub = df.loc[:current_date]
            scores = get_strategy_scores(df_sub, strategy_type, lookback)
            top_funds = scores.nlargest(top_n).index.tolist()

            is_bearish = _check_trend_regime(df_benchmark, current_date)
            weights, reason = get_ai_weights(top_funds, df_sub[top_funds], target_vol, is_bearish)

            # Mevcut Pozisyonları Boşalt
            for asset, amount in list(ledger.positions.items()):
                if amount > 1e-6 and asset in current_prices:
                    ledger.sell_asset(asset, amount, current_prices[asset], asset_params, current_date)

            # Yeni Pozisyonları Aç
            available_cash = ledger.cash * max_position_pct
            for fon, w in weights.items():
                if fon in current_prices and current_prices[fon] > 0:
                    allocated_cash = available_cash * w
                    ledger.buy_asset(fon, allocated_cash, current_prices[fon], asset_params)

            decision_log.append({
                'Tarih': current_date.strftime('%Y-%m-%d'),
                'Seçilen Fonlar': ', '.join(weights.keys()),
                'Ağırlıklar': ', '.join([f"{f}: %{w * 100:.1f}" for f, w in weights.items()]),
                'Strateji': strategy_type,
                'Ayı Rejimi': "Evet (%50 Nakit)" if is_bearish else "Hayır",
                'Gerekçe': reason
            })

    df_history = pd.DataFrame(history).set_index('Tarih')
    risk_metrics = PerformanceAnalytics.calculate_metrics(df_history)
    return df_history, decision_log, risk_metrics


# =============================================
# 8. PORTFÖY VERİ KONTROLÜ VE VERİTABANI İŞLEMLERİ
# =============================================
def get_portfolio(asset_class):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT nakit, varliklar, islemler, baslangic_bakiye, pending_cash FROM portfolios WHERE asset_class=?",
              (asset_class,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            'nakit': row[0],
            'varliklar': json.loads(row[1]) if row[1] else {},
            'islemler': json.loads(row[2]) if row[2] else [],
            'baslangic_bakiye': row[3],
            'pending_cash': json.loads(row[4]) if row[4] else []
        }
    return None


def update_portfolio(asset_class, data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO portfolios 
                 (asset_class, nakit, varliklar, islemler, baslangic_bakiye, pending_cash, son_guncelleme)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (asset_class, data['nakit'], json.dumps(data['varliklar']),
               json.dumps(data['islemler']), data['baslangic_bakiye'],
               json.dumps(data.get('pending_cash', [])),
               datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_allocations():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT asset_class, hedef_oran, min_oran, max_oran FROM allocations")
    rows = c.fetchall()
    conn.close()
    return {row[0]: {'hedef': row[1], 'min': row[2], 'max': row[3]} for row in rows}


def set_allocation(asset_class, hedef_oran, min_oran=0, max_oran=1):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO allocations (asset_class, hedef_oran, min_oran, max_oran) VALUES (?, ?, ?, ?)",
              (asset_class, hedef_oran, min_oran, max_oran))
    conn.commit()
    conn.close()


# =============================================
# 9. OTOMATİK YENİDEN DENGELEME (REBALANCE)
# =============================================
def rebalance_portfolio(allocations, initial_capital=100000):
    total_value = 0
    class_values = {}

    for cls in ALL_ASSET_CLASSES:
        pf = get_portfolio(cls)
        if pf is None:
            continue
        val = pf['nakit'] + sum(item['amount'] for item in pf.get('pending_cash', []))
        df_cls = st.session_state.df_all_raw.get(cls, pd.DataFrame())
        for fon, adet in pf['varliklar'].items():
            if fon in df_cls.columns:
                fiyat = df_cls[fon].dropna().iloc[-1]
                val += adet * fiyat
        class_values[cls] = val
        total_value += val

    if total_value == 0:
        st.warning("Portföylerde işlenecek bakiye yok.")
        return

    for cls, alloc_info in allocations.items():
        hedef_val = total_value * alloc_info['hedef']
        current_val = class_values.get(cls, 0)
        diff = hedef_val - current_val

        if abs(diff) < 100:
            continue

        pf = get_portfolio(cls)
        if pf is None:
            pf = {'nakit': hedef_val, 'varliklar': {}, 'islemler': [], 'baslangic_bakiye': hedef_val,
                  'pending_cash': []}

        asset_params = ASSET_PARAMS.get(cls, {'komisyon_alis': 0.002, 'komisyon_satis': 0.002, 'valor': 2})

        # Fazla Bakiye Satışı
        if diff < 0 and current_val > 0:
            satis_orani = abs(diff) / current_val
            for fon, adet in list(pf['varliklar'].items()):
                df_cls = st.session_state.df_all_raw.get(cls, pd.DataFrame())
                if fon in df_cls.columns:
                    fiyat = df_cls[fon].dropna().iloc[-1]
                    satis_adet = adet * satis_orani
                    brut = satis_adet * fiyat
                    net = brut * (1 - asset_params['komisyon_satis'])

                    pf['varliklar'][fon] -= satis_adet
                    if pf['varliklar'][fon] <= 1e-6:
                        del pf['varliklar'][fon]

                    valör_date = (datetime.date.today() + timedelta(days=asset_params['valor'])).strftime('%Y-%m-%d')
                    if asset_params['valor'] == 0:
                        pf['nakit'] += net
                    else:
                        pf['pending_cash'].append({'settlement_date': valör_date, 'amount': net})

                    pf['islemler'].append(
                        {'Tarih': datetime.date.today().strftime('%Y-%m-%d'), 'İşlem': 'REBALANCE SAT', 'Fon': fon,
                         'Tutar': net})

        # Eksik Bakiye Alımı
        elif diff > 0 and pf['nakit'] > 100:
            alim_tutari = min(diff, pf['nakit'])
            df_cls = st.session_state.df_all_raw.get(cls, pd.DataFrame())
            if not df_cls.empty:
                fonlar = df_cls.columns[:3].tolist()
                for fon in fonlar:
                    fiyat = df_cls[fon].dropna().iloc[-1]
                    tutar = alim_tutari / len(fonlar)
                    net_tutar = tutar * (1 - asset_params['komisyon_alis'])
                    adet = net_tutar / fiyat
                    pf['varliklar'][fon] = pf['varliklar'].get(fon, 0) + adet
                    pf['nakit'] -= tutar
                    pf['islemler'].append(
                        {'Tarih': datetime.date.today().strftime('%Y-%m-%d'), 'İşlem': 'REBALANCE AL', 'Fon': fon,
                         'Tutar': tutar})

        update_portfolio(cls, pf)
    st.success("✅ Tüm portföyler başarıyla hedeflere göre dengelendi.")


# =============================================
# 10. MAKRO VE SİSTEM YÜKLEYİCİ
# =============================================
ALL_ASSET_CLASSES = list(DEFAULT_UNIVERSE.keys()) + ['tefas_fon']

if 'df_all_raw' not in st.session_state:
    st.session_state.df_all_raw = {}


def load_all_asset_data():
    loaded = 0
    end_date = datetime.date.today()
    start_date = end_date - timedelta(days=365 * 2)
    for cls in ALL_ASSET_CLASSES:
        if cls in st.session_state.df_all_raw and not st.session_state.df_all_raw[cls].empty:
            loaded += 1
            continue
        if cls == "tefas_fon":
            provider = TefasProvider()
        elif cls == "kripto":
            provider = BinanceProvider()
        else:
            provider = YahooProvider()
            provider.set_symbols(DEFAULT_UNIVERSE.get(cls, {}))
        try:
            df = fon_verilerini_yukle(provider, start_date, end_date, cls)
            if not df.empty:
                st.session_state.df_all_raw[cls] = df
                loaded += 1
        except Exception:
            pass
    return loaded


# =============================================
# 11. STREAMLIT ARAYÜZ (UI)
# =============================================
st.set_page_config(page_title="Valör & Şişme Önleyici AI Portföy", layout="wide")
st.title("🏛️ Valör & Şişme Önleyici Muhasebeli AI Portföy Yönetimi")

asset_class = st.sidebar.selectbox("Varlık Sınıfı Seçin", ALL_ASSET_CLASSES)
alloc_data = get_allocations()

st.sidebar.subheader("⚖️ Allokasyon Hedefleri (%100)")
total_alloc = 0
for cls in ALL_ASSET_CLASSES:
    default_ratio = alloc_data.get(cls, {}).get('hedef', 0.0)
    ratio = st.sidebar.slider(f"{cls}", 0, 100, int(default_ratio * 100), 5, key=f"alloc_{cls}") / 100.0
    set_allocation(cls, ratio)
    total_alloc += ratio

if total_alloc != 1.0:
    st.sidebar.warning(f"⚠️ Toplam allokasyon: %{total_alloc * 100:.0f} (Hedef %100 olmalı)")

if st.sidebar.button("📂 Tüm Sınıf Verilerini Yükle"):
    with st.spinner("Veriler çekiliyor..."):
        cnt = load_all_asset_data()
        st.sidebar.success(f"{cnt} Varlık sınıfı yüklendi.")

# Veri Sağlayıcı Ayarla
end_date = datetime.date.today()
start_date = end_date - timedelta(days=365 * 2)

if asset_class == "tefas_fon":
    provider = TefasProvider()
elif asset_class == "kripto":
    provider = BinanceProvider()
else:
    provider = YahooProvider()
    provider.set_symbols(DEFAULT_UNIVERSE.get(asset_class, {}))

df_prices = fon_verilerini_yukle(provider, start_date, end_date, asset_class)

# Parametre Girdileri
col1, col2, col3 = st.columns(3)
with col1:
    initial_capital = st.number_input("Başlangıç Sermayesi (TL)", value=100000, step=10000)
with col2:
    strategy_type = st.selectbox("Strateji Türü", list(STRATEGY_TYPES.keys()))
with col3:
    target_vol = st.slider("Hedef Volatilite (Oynaklık Kasa Sınırı)", 0.05, 0.40, 0.15, 0.01)

# Benchmark verisi
df_bench = pd.DataFrame()
if 'hisse_bist' in st.session_state.df_all_raw:
    df_bench = st.session_state.df_all_raw['hisse_bist']

# SEKMELER
tab1, tab2, tab3 = st.tabs(["🚀 Backtest & Performans", "📊 Portföy & Valör Muhasebesi", "🔄 Rebalance & Allokasyon"])

with tab1:
    if st.button("🚀 Backtest'i Çalıştır (Valör + Dinamik Komisyon + Volatilite Kasa)"):
        with st.spinner("Backtest yapılıyor..."):
            df_result, log, risk_met = run_backtest(
                df_prices=df_prices,
                initial_capital=initial_capital,
                asset_class=asset_class,
                strategy_type=STRATEGY_TYPES[strategy_type],
                df_benchmark=df_bench,
                target_vol=target_vol
            )

        if not df_result.empty:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Toplam Getiri", f"%{risk_met.get('Toplam Getiri (%)', 0):.2f}")
            c2.metric("Sharpe Oranı", f"{risk_met.get('Sharpe Oranı', 0):.2f}")
            c3.metric("Sortino Oranı", f"{risk_met.get('Sortino Oranı', 0):.2f}")
            c4.metric("Calmar Oranı", f"{risk_met.get('Calmar Oranı', 0):.2f}")
            c5.metric("Max Drawdown", f"%{risk_met.get('Max Drawdown (%)', 0):.2f}")

            # Grafik
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_result.index, y=df_result['Toplam_Varlik'], name='Toplam Portföy',
                                     line=dict(color='green', width=2.5)))
            fig.add_trace(go.Scatter(x=df_result.index, y=df_result['Nakit'], name='Serbest Nakit',
                                     line=dict(color='orange', dash='dash')))
            fig.add_trace(go.Scatter(x=df_result.index, y=df_result['Bloke_Nakit'], name='Valördeki Bekleyen Nakit',
                                     line=dict(color='red', dash='dot')))
            fig.update_layout(title="Mükerrerlik Önlenmiş Net Bakiye Gelişimi (TL)", xaxis_title="Tarih",
                              yaxis_title="TL")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📋 AI Günlüğü ve Piyasa Rejimi Kararları")
            st.dataframe(pd.DataFrame(log), use_container_width=True)

with tab2:
    st.subheader(f"💼 Aktif Portföy Durumu ({asset_class})")
    pf = get_portfolio(asset_class)
    if pf:
        asset_params = ASSET_PARAMS.get(asset_class, {'valor': 2})
        c1, c2, c3 = st.columns(3)
        c1.metric("Serbest Nakit", f"{pf['nakit']:,.2f} TL")

        pending_total = sum(item['amount'] for item in pf.get('pending_cash', []))
        c2.metric(f"Valörde Bekleyen Nakit (T+{asset_params['valor']})", f"{pending_total:,.2f} TL")

        varlik_toplam = 0
        if not df_prices.empty:
            for fon, adet in pf['varliklar'].items():
                if fon in df_prices.columns:
                    varlik_toplam += adet * df_prices[fon].dropna().iloc[-1]

        c3.metric("Toplam Varlık Değeri", f"{varlik_toplam:,.2f} TL")

        st.subheader("📦 Mevcut Varlık Pozisyonları")
        st.json(pf['varliklar'])

        st.subheader("⏳ Valör Kuyruğu (Bekleyen Tahsilatlar)")
        st.dataframe(pd.DataFrame(pf.get('pending_cash', [])))
    else:
        st.info("Bu sınıf için henüz oluşmuş bir portföy bulunamadı.")

with tab3:
    st.subheader("⚖️ Portföy Dengeleme (Rebalance)")
    if st.button("🔄 Tüm Sınıfları Hedef Oranlara Dengele"):
        rebalance_portfolio(alloc_data, initial_capital)