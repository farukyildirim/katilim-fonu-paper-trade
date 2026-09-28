import streamlit as st
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import plotly.graph_objects as go
import plotly.express as px
import os
import json
import yfinance as yf
import warnings
import time
import sqlite3
import requests  # >>> YENİ: Telegram bildirimi için

try:
    import ccxt
except ImportError:
    ccxt = None

try:
    from tefas import Crawler
except ImportError:
    Crawler = None

warnings.filterwarnings('ignore')

# =============================================
# 0. KONFİGÜRASYON
# =============================================
DB_PATH = "portfoliod.db"
CACHE_DIR = "cache"
CACHE_TTL_HOURS = 12  # >>> YENİ: Cache geçerlilik süresi (saat)

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# =============================================
# 1. VALÖR VE ŞİŞME ÖNLEYİCİ MUHASEBE
# =============================================
class PortfolioLedger:
    def __init__(self, initial_capital=100000.0):
        self.cash = float(initial_capital)
        self.positions = {}
        self.entry_prices = {}  # >>> YENİ: Stop-loss için ortalama giriş fiyatı
        self.pending_cash_settlements = []

    def process_settlements(self, current_date):
        current_dt = pd.to_datetime(current_date)
        settled_amount = 0.0
        remaining = []
        for item in self.pending_cash_settlements:
            if current_dt >= pd.to_datetime(item['settlement_date']):
                settled_amount += item['amount']
            else:
                remaining.append(item)
        self.pending_cash_settlements = remaining
        self.cash += settled_amount

    def sell_asset(self, asset_code, amount, price, asset_params, current_date):
        if asset_code not in self.positions or self.positions[asset_code] < amount:
            return 0.0

        if isinstance(price, dict):
            price = next((v for v in price.values() if isinstance(v, (int, float)) and not np.isnan(v)), 0.0)
        else:
            price = float(price) if price is not None and not np.isnan(price) else 0.0

        if price <= 0:
            return 0.0

        gross = amount * price
        komisyon_rate = asset_params.get('komisyon_satis', 0.0010)
        net = gross * (1 - komisyon_rate)

        self.positions[asset_code] -= amount
        if self.positions[asset_code] <= 1e-6:
            del self.positions[asset_code]
            self.entry_prices.pop(asset_code, None)  # >>> YENİ

        valor_days = asset_params.get('valor', 1)
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
        if cash_to_use > self.cash:
            cash_to_use = self.cash

        if isinstance(price, dict):
            price = next((v for v in price.values() if isinstance(v, (int, float)) and not np.isnan(v)), 0.0)
        else:
            price = float(price) if price is not None and not np.isnan(price) else 0.0

        if price <= 0:
            return 0.0

        komisyon_rate = asset_params.get('komisyon_alis', 0.0010)
        net_investment = cash_to_use * (1 - komisyon_rate)
        units = net_investment / price

        self.cash -= cash_to_use

        # >>> YENİ: Ortalama giriş fiyatı güncelle (stop-loss için)
        old_units = self.positions.get(asset_code, 0.0)
        old_cost = old_units * self.entry_prices.get(asset_code, price)
        new_cost = units * price
        total_units = old_units + units
        self.entry_prices[asset_code] = (old_cost + new_cost) / total_units if total_units > 0 else price
        self.positions[asset_code] = total_units
        return units

    def get_total_value(self, current_prices):
        asset_val = 0.0
        for asset, units in self.positions.items():
            price = current_prices.get(asset, 0.0)
            if isinstance(price, dict):
                price = next((v for v in price.values() if isinstance(v, (int, float)) and not np.isnan(v)), 0.0)
            elif isinstance(price, (pd.Series, np.ndarray)):
                price = float(price.iloc[0]) if len(price) > 0 and not np.isnan(price.iloc[0]) else 0.0
            else:
                try:
                    price = float(price) if price is not None and not np.isnan(price) else 0.0
                except (ValueError, TypeError):
                    price = 0.0
            asset_val += units * price

        pending_val = sum(item['amount'] for item in self.pending_cash_settlements)
        return self.cash + pending_val + asset_val

# =============================================
# 2. RİSK METRİKLERİ
# =============================================
class PerformanceAnalytics:
    @staticmethod
    def calculate_metrics(df_history, risk_free_rate=0.035):
        if df_history.empty or len(df_history) < 2:
            return {}

        df = df_history.dropna(subset=['Toplam_Varlik']).copy()
        if len(df) < 2:
            return {}

        returns = df['Toplam_Varlik'].pct_change().dropna()
        if returns.empty or returns.std() == 0:
            return {}

        total_return = (df['Toplam_Varlik'].iloc[-1] - df['Toplam_Varlik'].iloc[0]) / df['Toplam_Varlik'].iloc[0]
        n_days = len(df)

        ann_return = ((1 + max(total_return, -0.99)) ** (252 / max(n_days, 1))) - 1
        ann_vol = returns.std() * np.sqrt(252)

        excess_return = ann_return - risk_free_rate
        sharpe = excess_return / (ann_vol + 1e-9)

        daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
        negative_returns = returns[returns < daily_rf] - daily_rf
        downside_std = np.sqrt(np.mean(negative_returns ** 2)) * np.sqrt(252) if len(negative_returns) > 0 else 1e-9
        sortino = excess_return / (downside_std + 1e-9)

        rolling_max = df['Toplam_Varlik'].cummax()
        drawdowns = (df['Toplam_Varlik'] - rolling_max) / rolling_max
        max_dd = drawdowns.min()
        calmar = ann_return / (abs(max_dd) + 1e-9)

        return {
            'Toplam Getiri (%)': total_return * 100,
            'Yıllık Getiri (%)': ann_return * 100,
            'Yıllık Volatilite (%)': ann_vol * 100,
            'Sharpe Oranı': sharpe,
            'Sortino Oranı': sortino,
            'Calmar Oranı': calmar,
            'Max Drawdown (%)': max_dd * 100
        }

# =============================================
# 3. SQLITE — PORTFÖY + SİNYAL TABLOLARI
# =============================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS portfolios (
                    asset_class TEXT PRIMARY KEY,
                    nakit REAL, varliklar TEXT, islemler TEXT,
                    baslangic_bakiye REAL, pending_cash TEXT,
                    son_guncelleme TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS allocations (
                    asset_class TEXT PRIMARY KEY,
                    hedef_oran REAL, min_oran REAL DEFAULT 0,
                    max_oran REAL DEFAULT 1)''')
    # >>> YENİ: Sinyal kayıt tablosu
    c.execute('''CREATE TABLE IF NOT EXISTS sinyaller (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tarih TEXT, asset_class TEXT,
                    varliklar TEXT, agirliklar TEXT,
                    risk_off INTEGER, gerekce TEXT,
                    created_at TEXT)''')
    conn.commit()
    conn.close()

init_db()

# >>> YENİ: Sinyal kaydetme / okuma
def save_signal(asset_class, sinyal):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO sinyaller
                 (tarih, asset_class, varliklar, agirliklar, risk_off, gerekce, created_at)
                 VALUES (?,?,?,?,?,?,?)""",
              (str(sinyal['tarih'])[:10], asset_class,
               ', '.join(sinyal['varliklar']),
               ', '.join([f"{k}: %{v*100:.1f}" for k, v in sinyal['agirliklar'].items()]),
               int(sinyal['risk_off']),
               sinyal['gerekce'],
               datetime.datetime.now().isoformat(timespec='seconds')))
    conn.commit()
    conn.close()

def load_signals(asset_class=None, limit=50):
    conn = sqlite3.connect(DB_PATH)
    if asset_class:
        df = pd.read_sql_query(
            "SELECT * FROM sinyaller WHERE asset_class=? ORDER BY id DESC LIMIT ?",
            conn, params=(asset_class, limit))
    else:
        df = pd.read_sql_query(
            "SELECT * FROM sinyaller ORDER BY id DESC LIMIT ?", conn, params=(limit,))
    conn.close()
    return df

# >>> YENİ: Opsiyonel Telegram bildirim
def send_telegram(message, token, chat_id):
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, data={'chat_id': chat_id, 'text': message, 'parse_mode': 'HTML'}, timeout=10)
        return r.status_code == 200
    except Exception:
        return False

# =============================================
# 4. VERİ SAĞLAYICILAR VE CACHE (TTL EKLENDİ)
# =============================================
def get_cache_path(asset_class):
    return os.path.join(CACHE_DIR, f"fon_fiyatlari_{asset_class}.parquet")

# >>> DÜZELTİLDİ: TTL kontrolü eklendi
def cache_is_fresh(path, ttl_hours=CACHE_TTL_HOURS):
    if not os.path.exists(path):
        return False
    age_seconds = time.time() - os.path.getmtime(path)
    return age_seconds < ttl_hours * 3600

def fon_verilerini_yukle(provider, start_date, end_date, asset_class, force_refresh=False):
    cache_path = get_cache_path(asset_class)

    # >>> DÜZELTİLDİ: TTL dolmadıysa cache kullan
    if not force_refresh and cache_is_fresh(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            df.index = pd.to_datetime(df.index)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception:
            if os.path.exists(cache_path):
                os.remove(cache_path)

    df_pivot = provider.fetch_prices(start_date, end_date)
    if not df_pivot.empty:
        if isinstance(df_pivot.columns, pd.MultiIndex):
            df_pivot.columns = df_pivot.columns.get_level_values(0)
        df_pivot = df_pivot.loc[:, ~df_pivot.columns.duplicated()]
        try:
            df_pivot.to_parquet(cache_path)
        except Exception:
            pass
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
                df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if not df.empty and 'Close' in df.columns:
                    close_val = df['Close']
                    if isinstance(close_val, pd.DataFrame):
                        close_val = close_val.iloc[:, 0]
                    all_data[name] = close_val
            except Exception:
                continue
        if not all_data:
            return pd.DataFrame()
        return pd.concat(all_data, axis=1).ffill().bfill()

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
                df = df.set_index('date').loc[start_date:end_date]
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
            bugun = pd.Timestamp.today()
            baslangic = bugun - pd.Timedelta(days=3)
            df_kod = self.crawler.fetch(start=baslangic.strftime("%Y-%m-%d"), end=bugun.strftime("%Y-%m-%d"))
            if df_kod is None or df_kod.empty:
                return pd.DataFrame()
            tum_kodlar = df_kod["code"].dropna().unique().tolist()
            df_list = []
            for fund in tum_kodlar[:100]:
                try:
                    df_fund = self.crawler.fetch(start=start_date.strftime("%Y-%m-%d"),
                                                 end=end_date.strftime("%Y-%m-%d"), name=fund,
                                                 columns=["code", "date", "price"])
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
# 5. PARAMETRELER
# =============================================
DEFAULT_UNIVERSE = {
    'hisse_bist': {'XU100': 'XU100.IS', 'THYAO': 'THYAO.IS', 'GARAN': 'GARAN.IS', 'AKBNK': 'AKBNK.IS', 'SISE': 'SISE.IS'},
    'hisse_abd': {'AAPL': 'AAPL', 'MSFT': 'MSFT', 'GOOGL': 'GOOGL', 'AMZN': 'AMZN', 'META': 'META', 'SPY': 'SPY'},
    'hisse_avrupa': {'SAP': 'SAP.DE', 'ASML': 'ASML.AS', 'NOVO': 'NOVO-B.CO', 'NESN': 'NESN.SW'},
    'kripto': {'BTC': 'BTC/USDT', 'ETH': 'ETH/USDT', 'BNB': 'BNB/USDT', 'SOL': 'SOL/USDT'},
    'emtia': {'Altın': 'GC=F', 'Gümüş': 'SI=F', 'Petrol': 'CL=F', 'Bakır': 'HG=F', 'Doğalgaz': 'NG=F'},
    'doviz': {'EUR/USD': 'EURUSD=X', 'GBP/USD': 'GBPUSD=X', 'USD/JPY': 'USDJPY=X',
              'GBPJPY': 'GBPJPY=X', 'USD/TRY': 'USDTRY=X', 'EUR/TRY': 'EURTRY=X'},
    'etf': {'SPY': 'SPY', 'QQQ': 'QQQ', 'VTI': 'VTI', 'BND': 'BND'}
}

ASSET_PARAMS = {
    'hisse_bist': {'komisyon_alis': 0.0020, 'komisyon_satis': 0.0020, 'valor': 2, 'max_position': 0.70},
    'hisse_abd': {'komisyon_alis': 0.0015, 'komisyon_satis': 0.0015, 'valor': 2, 'max_position': 0.70},
    'hisse_avrupa': {'komisyon_alis': 0.0015, 'komisyon_satis': 0.0015, 'valor': 2, 'max_position': 0.70},
    'kripto': {'komisyon_alis': 0.0010, 'komisyon_satis': 0.0010, 'valor': 0, 'max_position': 0.80},
    'emtia': {'komisyon_alis': 0.0010, 'komisyon_satis': 0.0010, 'valor': 1, 'max_position': 0.80},
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
# 6. STRATEJİ VE REJİM FİLTRELERİ
# =============================================
def calculate_momentum(df_prices, lookback=90):
    if len(df_prices) < lookback:
        return pd.Series(index=df_prices.columns, data=0.0)
    recent = df_prices.tail(lookback)
    return (recent.iloc[-1] / recent.iloc[0] - 1)

def calculate_value(df_prices, lookback=90):
    if len(df_prices) < lookback:
        return pd.Series(index=df_prices.columns, data=0.0)
    recent = df_prices.tail(lookback)
    mean_price = recent.mean()
    current = df_prices.iloc[-1]
    return (1 - (current / mean_price)).clip(0, 1)

def calculate_low_volatility(df_prices, lookback=90):
    if len(df_prices) < lookback:
        return pd.Series(index=df_prices.columns, data=0.0)
    returns = df_prices.tail(lookback).pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    return 1.0 / (volatility + 1e-6)

def get_strategy_scores(df_prices, strategy_type, lookback=90):
    if strategy_type == 'momentum':
        return calculate_momentum(df_prices, lookback)
    elif strategy_type == 'value':
        return calculate_value(df_prices, lookback)
    elif strategy_type == 'low_vol':
        return calculate_low_volatility(df_prices, lookback)
    elif strategy_type == 'blend':
        mom = calculate_momentum(df_prices, lookback)
        val = calculate_value(df_prices, lookback)
        return 0.6 * mom + 0.4 * val
    return calculate_momentum(df_prices, lookback)

def _check_trend_regime(df_benchmark, current_date, window=200):
    if df_benchmark is None or df_benchmark.empty:
        return False
    bench_series = df_benchmark.loc[:current_date]  # >>> DÜZELTİLDİ: look-ahead koruması
    if len(bench_series) < window:
        return False
    series = bench_series.iloc[:, 0] if isinstance(bench_series, pd.DataFrame) else bench_series
    sma_200 = series.tail(window).mean()
    return series.iloc[-1] < sma_200

def _apply_volatility_targeting(weights, df_prices_sub, target_volatility=0.25, window=30):
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
        scale_factor = max(target_volatility / realized_vol, 0.85)
        return {k: v * scale_factor for k, v in weights.items()}
    return weights

def get_ai_weights(funds, df_prices_sub=None, target_vol=0.25, is_bearish=False):
    base_weight = 1.0 / len(funds) if funds else 0
    weights = {f: base_weight for f in funds}
    if is_bearish:
        weights = {f: w * 0.5 for f, w in weights.items()}
    if df_prices_sub is not None and not df_prices_sub.empty:
        weights = _apply_volatility_targeting(weights, df_prices_sub, target_volatility=target_vol)
    return weights, "Ağırlıklar Rejim & Volatilite Filtreleriyle Hesaplandı."

# =============================================
# 7. FX DÖNÜŞÜMÜ (ORTAK YARDIMCI)
# =============================================
def apply_fx_conversion(df, asset_class):
    """Emtia/ABD/Avrupa varlıklarını TL'ye çevirir. Ortak fonksiyon."""
    df = df.copy()
    if asset_class in ['emtia', 'hisse_abd', 'etf']:
        try:
            usd_try_df = yf.download("USDTRY=X", start=df.index.min() - timedelta(days=5),
                                     end=df.index.max() + timedelta(days=2),
                                     progress=False, auto_adjust=True)['Close']
            if isinstance(usd_try_df, pd.DataFrame):
                usd_try_df = usd_try_df.iloc[:, 0]
            usd_try = usd_try_df.reindex(df.index).ffill().bfill()
            for col in df.columns:
                df[col] = df[col].astype(float) * usd_try.astype(float)
            df = df.ffill().bfill().dropna(how='all')
        except Exception:
            pass
    elif asset_class == 'hisse_avrupa':
        try:
            eur_try = yf.download("EURTRY=X", start=df.index.min(), end=df.index.max(),
                                  progress=False, auto_adjust=True)['Close']
            if isinstance(eur_try, pd.DataFrame):
                eur_try = eur_try.iloc[:, 0]
            df = df.mul(eur_try.reindex(df.index).ffill().bfill(), axis=0).ffill().bfill()
        except Exception:
            pass
    return df

# =============================================
# 8. BACKTEST MOTORU (STOP-LOSS EKLENDİ)
# =============================================
def run_backtest(df_prices, initial_capital=100000, stop_loss_pct=0.12, top_n=3, lookback=90,
                 max_position_pct=0.85, asset_class='hisse_bist', strategy_type='blend',
                 df_benchmark=None, target_vol=0.25, rebalance_freq_months=3):
    df = df_prices.sort_index().copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]

    if df.empty or len(df) < lookback:
        return pd.DataFrame(), [], {}

    # FX Dönüşümü
    df = apply_fx_conversion(df, asset_class)

    asset_params = ASSET_PARAMS.get(asset_class,
                                    {'komisyon_alis': 0.0005, 'komisyon_satis': 0.0005, 'valor': 0})
    ledger = PortfolioLedger(initial_capital=initial_capital)

    dates = df.index
    rebalance_dates = dates[dates.is_month_start][::rebalance_freq_months]
    if len(rebalance_dates) == 0:
        rebalance_dates = [dates[0]]

    history = []
    decision_log = []

    for i, current_date in enumerate(dates):
        ledger.process_settlements(current_date)

        row = df.loc[current_date]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        current_prices = row.to_dict()

        # BIST/TEFAS için TL mevduat faiz tahakkuku (%40)
        if asset_class in ['hisse_bist', 'tefas_fon']:
            ledger.cash += ledger.cash * (0.40 / 365.0)

        # >>> YENİ: STOP-LOSS KONTROLÜ (her gün çalışır)
        if stop_loss_pct and stop_loss_pct > 0:
            for asset, units in list(ledger.positions.items()):
                if units <= 1e-6:
                    continue
                price = current_prices.get(asset)
                if price is None or (isinstance(price, float) and np.isnan(price)):
                    continue
                if isinstance(price, dict):
                    price = next((v for v in price.values() if isinstance(v, (int, float)) and not np.isnan(v)), None)
                if price is None or price <= 0:
                    continue
                entry = ledger.entry_prices.get(asset, price)
                if price < entry * (1 - stop_loss_pct):
                    ledger.sell_asset(asset, units, price, asset_params, current_date)
                    decision_log.append({
                        'Tarih': current_date.strftime('%Y-%m-%d'),
                        'Seçilen Fonlar': asset,
                        'Ağırlıklar': '-',
                        'Strateji': 'STOP-LOSS',
                        'Ayı Rejimi': '-',
                        'Gerekçe': f'Zarar kes: giriş {entry:.2f}, fiyat {price:.2f}'
                    })

        toplam_deger = ledger.get_total_value(current_prices)
        pending_val = sum(item['amount'] for item in ledger.pending_cash_settlements)

        history.append({
            'Tarih': current_date,
            'Toplam_Varlik': toplam_deger,
            'Nakit': ledger.cash,
            'Portfoy_Degeri': toplam_deger - ledger.cash - pending_val,
            'Bloke_Nakit': pending_val
        })

        if current_date in rebalance_dates and len(dates) - i > 5:
            df_sub = df.loc[:current_date].dropna(axis=1, how='any')
            scores = get_strategy_scores(df_sub, strategy_type, lookback)
            top_funds = scores.nlargest(min(top_n, len(scores))).index.tolist()

            is_bearish = _check_trend_regime(df_benchmark, current_date)
            weights, reason = get_ai_weights(top_funds, df_sub[top_funds], target_vol, is_bearish)

            for asset, amount in list(ledger.positions.items()):
                if amount > 1e-6 and asset in current_prices:
                    ledger.sell_asset(asset, amount, current_prices[asset], asset_params, current_date)

            available_cash = ledger.cash * max_position_pct
            for fon, w in weights.items():
                if fon in current_prices:
                    price_val = current_prices[fon]
                    if isinstance(price_val, dict):
                        price_val = next((v for v in price_val.values()
                                          if isinstance(v, (int, float)) and not np.isnan(v)), 0.0)
                    if price_val > 0 and not np.isnan(price_val):
                        ledger.buy_asset(fon, available_cash * w, price_val, asset_params)

            decision_log.append({
                'Tarih': current_date.strftime('%Y-%m-%d'),
                'Seçilen Fonlar': ', '.join(weights.keys()),
                'Ağırlıklar': ', '.join([f"{f}: %{w * 100:.1f}" for f, w in weights.items()]),
                'Strateji': strategy_type,
                'Ayı Rejimi': "Evet" if is_bearish else "Hayır",
                'Gerekçe': reason
            })

    df_history = pd.DataFrame(history).set_index('Tarih')
    rf_rate = 0.035 if asset_class in ['emtia', 'hisse_avrupa', 'hisse_abd', 'etf', 'doviz', 'kripto'] else 0.40
    risk_metrics = PerformanceAnalytics.calculate_metrics(df_history, risk_free_rate=rf_rate)
    return df_history, decision_log, risk_metrics

# =============================================
# 9. CANLI SİNYAL ÜRETİCİ  (>>> YENİ)
# =============================================
def generate_live_signal(df_prices, df_benchmark, asset_class,
                         strategy_type='blend', lookback=90,
                         top_n=3, target_vol=0.18):
    """
    Bugünün piyasa verisine göre alınacak portföyü önerir.
    Dönüş: {'tarih', 'varliklar', 'agirliklar', 'risk_off', 'gerekce', 'skorlar'}
    """
    if df_prices is None or df_prices.empty:
        return None

    df = df_prices.sort_index().copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    df = apply_fx_conversion(df, asset_class)

    df = df.dropna(axis=1, how='any')
    if len(df) < lookback:
        return None

    scores = get_strategy_scores(df, strategy_type, lookback)
    n = min(top_n, len(scores.dropna()))
    top_funds = scores.dropna().nlargest(n).index.tolist()
    if not top_funds:
        return None

    scores_df = scores.dropna().nlargest(n).reset_index()
    scores_df.columns = ['Varlık', 'Skor']

    son_tarih = df.index[-1]
    is_bearish = _check_trend_regime(df_benchmark, son_tarih)
    weights, reason = get_ai_weights(top_funds, df[top_funds], target_vol, is_bearish)

    # Fiyat bilgisini de ekle (kaç TL'den alınacak)
    son_fiyatlar = df[top_funds].iloc[-1].to_dict()

    return {
        'tarih': son_tarih,
        'varliklar': top_funds,
        'agirliklar': weights,
        'fiyatlar': son_fiyatlar,
        'risk_off': bool(is_bearish),
        'gerekce': reason,
        'skorlar': scores_df
    }

# =============================================
# 10. STREAMLIT UI
# =============================================
st.set_page_config(page_title="Kişisel AI Portföy Sinyal Sistemi", layout="wide")
st.title("🏛️ Kişisel Portföy Sinyal & Backtest Sistemi")

ALL_ASSET_CLASSES = list(DEFAULT_UNIVERSE.keys()) + ['tefas_fon']

# ---- SIDEBAR ----
with st.sidebar:
    st.header("⚙️ Ayarlar")
    asset_class = st.selectbox("Varlık Sınıfı", ALL_ASSET_CLASSES, index=3)  # kripto default

    end_date = datetime.date.today()
    start_date = end_date - timedelta(days=365 * 2)

    force_refresh = st.checkbox("🔄 Cache'i yenile (güncel veri)", value=True)

    st.divider()
    st.subheader("📡 Telegram (Opsiyonel)")
    tg_token = st.text_input("Bot Token", type="password",
                             value=os.environ.get("TG_TOKEN", ""))
    tg_chat = st.text_input("Chat ID", value=os.environ.get("TG_CHAT", ""))

    st.divider()
    st.caption("⚠️ Bu araç yatırım tavsiyesi değildir. Kararlar size aittir.")

# ---- VERİ SAĞLAYICI ----
if asset_class == "tefas_fon":
    provider = TefasProvider()
elif asset_class == "kripto":
    provider = BinanceProvider()
    provider.set_symbols(DEFAULT_UNIVERSE.get(asset_class, {}))
else:
    provider = YahooProvider()
    provider.set_symbols(DEFAULT_UNIVERSE.get(asset_class, {}))

# Benchmark (BIST ise XU100)
df_bench = pd.DataFrame()
if asset_class in ['tefas_fon', 'hisse_bist', 'hisse_abd', 'etf', 'emtia', 'hisse_avrupa']:
    try:
        df_bench = yf.download("XU100.IS", start=start_date, end=end_date,
                               progress=False, auto_adjust=True)['Close']
        if isinstance(df_bench, pd.Series):
            df_bench = df_bench.to_frame("XU100")
    except Exception:
        df_bench = pd.DataFrame()

# ---- SEKMELER ----
tab_sinyal, tab_backtest, tab_gecmis = st.tabs(["📡 Bugünün Sinyali", "🧪 Backtest", "📜 Sinyal Geçmişi"])

# =============================================
# SEKME 1: CANLI SİNYAL
# =============================================
with tab_sinyal:
    st.subheader(f"Bugünkü Önerilen Portföy — {asset_class}")

    col1, col2, col3 = st.columns(3)
    with col1:
        sinyal_strateji = st.selectbox("Strateji", list(STRATEGY_TYPES.keys()), index=4, key="sig_strat")
    with col2:
        sinyal_topn = st.slider("Kaç varlık?", 1, 10, 3, key="sig_topn")
    with col3:
        sinyal_vol = st.slider("Hedef Volatilite", 0.05, 0.40, 0.18, 0.01, key="sig_vol")

    if st.button("🎯 Sinyal Üret", type="primary"):
        with st.spinner("Veri çekiliyor ve sinyal hesaplanıyor..."):
            df_prices = fon_verilerini_yukle(provider, start_date, end_date, asset_class,
                                             force_refresh=force_refresh)
            sinyal = generate_live_signal(
                df_prices=df_prices,
                df_benchmark=df_bench,
                asset_class=asset_class,
                strategy_type=STRATEGY_TYPES[sinyal_strateji],
                lookback=90,
                top_n=sinyal_topn,
                target_vol=sinyal_vol
            )

        if sinyal is None:
            st.error("❌ Yeterli veri yok. Cache'i yenilemeyi deneyin veya farklı varlık sınıfı seçin.")
        else:
            # Risk-off uyarısı
            if sinyal['risk_off']:
                st.warning("🚨 **AYI REJİMİ AKTİF** — Pozisyon büyüklükleri yarıya indirildi. "
                           "Defansif kalın, nakit oranını yüksek tutun.")
            else:
                st.success("✅ **BOĞA REJİMİ** — Piyasa 200 günlük ortalamanın üzerinde.")

            # Ana sinyal tablosu
            st.markdown(f"**📅 Sinyal Tarihi:** `{str(sinyal['tarih'])[:10]}`")
            st.markdown(f"**📝 Gerekçe:** {sinyal['gerekce']}")

            sinyal_df = pd.DataFrame({
                'Varlık': sinyal['varliklar'],
                'Skor': [f"{sinyal['skorlar'].set_index('Varlık').loc[v, 'Skor']:.4f}" for v in sinyal['varliklar']],
                'Öneri Ağırlık (%)': [f"%{sinyal['agirliklar'][v]*100:.1f}" for v in sinyal['varliklar']],
                'Son Fiyat': [f"{sinyal['fiyatlar'][v]:,.4f}" for v in sinyal['varliklar']]
            })
            st.dataframe(sinyal_df, use_container_width=True, hide_index=True)

            # Sermaye dağıtımı (100.000 TL örnek)
            st.markdown("### 💰 100.000 TL İçin Örnek Dağıtım")
            ornek = []
            for v in sinyal['varliklar']:
                tutar = 100000 * sinyal['agirliklar'][v]
                ornek.append({
                    'Varlık': v,
                    'TL Tutar': f"{tutar:,.2f} ₺",
                    'Adet (yaklaşık)': f"{tutar / sinyal['fiyatlar'][v]:,.4f}"
                })
            st.dataframe(pd.DataFrame(ornek), use_container_width=True, hide_index=True)

            # Kaydet + bildir
            c1, c2 = st.columns(2)
            with c1:
                if st.button("💾 Sinyali Kaydet"):
                    save_signal(asset_class, sinyal)
                    st.success("Sinyal veritabanına kaydedildi.")
            with c2:
                if st.button("📨 Telegram'a Gönder"):
                    msg = (f"📡 <b>Portföy Sinyali</b> ({asset_class})\n"
                           f"📅 {str(sinyal['tarih'])[:10]}\n"
                           f"🚦 Rejim: {'AYI (risk-off)' if sinyal['risk_off'] else 'BOĞA'}\n\n")
                    for v in sinyal['varliklar']:
                        msg += f"• {v}: %{sinyal['agirliklar'][v]*100:.1f}\n"
                    ok = send_telegram(msg, tg_token, tg_chat)
                    if ok:
                        st.success("Telegram'a gönderildi.")
                    else:
                        st.error("Gönderilemedi. Token/Chat ID'yi kontrol edin.")

# =============================================
# SEKME 2: BACKTEST
# =============================================
with tab_backtest:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        initial_capital = st.number_input("Başlangıç Sermayesi (TL)", value=100000, step=10000)
    with col2:
        strategy_type = st.selectbox("Strateji Türü", list(STRATEGY_TYPES.keys()), index=4, key="bt_strat")
    with col3:
        target_vol_bt = st.slider("Hedef Volatilite", 0.05, 0.40, 0.18, 0.01, key="bt_vol")
    with col4:
        stop_loss_pct = st.slider("Stop-Loss (%)", 0.0, 0.30, 0.12, 0.01, key="bt_sl",
                                  help="0 = devre dışı")

    if st.button("🚀 Backtest'i Çalıştır", type="primary"):
        with st.spinner("Hesaplanıyor..."):
            df_prices = fon_verilerini_yukle(provider, start_date, end_date, asset_class,
                                             force_refresh=force_refresh)
            df_result, log, risk_met = run_backtest(
                df_prices=df_prices,
                initial_capital=initial_capital,
                stop_loss_pct=stop_loss_pct,
                asset_class=asset_class,
                strategy_type=STRATEGY_TYPES[strategy_type],
                df_benchmark=df_bench,
                target_vol=target_vol_bt,
                lookback=90,
                rebalance_freq_months=3
            )

        if not df_result.empty and risk_met:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Toplam Getiri", f"%{risk_met.get('Toplam Getiri (%)', 0):.2f}")
            c2.metric("Sharpe", f"{risk_met.get('Sharpe Oranı', 0):.2f}")
            c3.metric("Sortino", f"{risk_met.get('Sortino Oranı', 0):.2f}")
            c4.metric("Calmar", f"{risk_met.get('Calmar Oranı', 0):.2f}")
            c5.metric("Max DD", f"%{risk_met.get('Max Drawdown (%)', 0):.2f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_result.index, y=df_result['Toplam_Varlik'],
                                     name='Toplam Portföy', line=dict(color='green', width=2.5)))
            fig.add_trace(go.Scatter(x=df_result.index, y=df_result['Nakit'],
                                     name='Nakit', line=dict(color='orange', dash='dash')))
            fig.update_layout(title="Portföy Büyümesi (TL)",
                              xaxis_title="Tarih", yaxis_title="TL")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📋 İşlem Günlüğü (Rebalance + Stop-Loss)")
            st.dataframe(pd.DataFrame(log), use_container_width=True)
        else:
            st.warning("Backtest sonuç üretemedi.")

# =============================================
# SEKME 3: SİNYAL GEÇMİŞİ
# =============================================
with tab_gecmis:
    st.subheader("📜 Kaydedilmiş Sinyaller")
    df_sig = load_signals(asset_class=asset_class, limit=100)
    if df_sig.empty:
        st.info("Henüz kayıtlı sinyal yok. 'Bugünün Sinyali' sekmesinden kaydedebilirsiniz.")
    else:
        df_sig = df_sig.rename(columns={
            'tarih': 'Sinyal Tarihi', 'asset_class': 'Varlık Sınıfı',
            'varliklar': 'Varlıklar', 'agirliklar': 'Ağırlıklar',
            'risk_off': 'Ayı Rejimi', 'gerekce': 'Gerekçe',
            'created_at': 'Kayıt Zamanı'
        })
        df_sig['Ayı Rejimi'] = df_sig['Ayı Rejimi'].apply(lambda x: 'Evet' if x else 'Hayır')
        st.dataframe(df_sig[['Sinyal Tarihi', 'Varlıklar', 'Ağırlıklar',
                              'Ayı Rejimi', 'Kayıt Zamanı']],
                     use_container_width=True, hide_index=True)

        csv = df_sig.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ CSV İndir", csv, "sinyaller.csv", "text/csv")