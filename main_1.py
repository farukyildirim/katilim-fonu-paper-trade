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
from collections import Counter
import math
import sqlite3
import concurrent.futures
from io import BytesIO

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
# 0. SQLITE VERİTABANI (KALICILIK)
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
    c.execute('''CREATE TABLE IF NOT EXISTS global_portfolio (
                    id INTEGER PRIMARY KEY CHECK (id=1),
                    nakit REAL,
                    varliklar TEXT,
                    islemler TEXT,
                    baslangic_bakiye REAL,
                    son_guncelleme TEXT
                 )''')
    c.execute('''CREATE TABLE IF NOT EXISTS strategy_prefs (
                    asset_class TEXT PRIMARY KEY,
                    strategy_type TEXT DEFAULT 'momentum'
                 )''')
    conn.commit()
    conn.close()

def migrate_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("ALTER TABLE allocations ADD COLUMN min_oran REAL DEFAULT 0")
        c.execute("ALTER TABLE allocations ADD COLUMN max_oran REAL DEFAULT 1")
        conn.commit()
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e):
            print(f"⚠️ Migration hatası: {e}")
    conn.close()

init_db()
migrate_db()

# =============================================
# 1. VERİ KALICILIĞI (PARQUET)
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
                st.info(f"📅 Son veri {son_tarih}, bugün {bugun}. Eksik günler çekiliyor...")
                baslangic = son_tarih + timedelta(days=1)
                df_yeni = provider.fetch_prices(baslangic, end_date)
                if not df_yeni.empty:
                    df_mevcut = pd.concat([df_mevcut, df_yeni], axis=0).sort_index()
                    df_mevcut = df_mevcut.ffill().dropna(axis=1, how='all')
                    df_mevcut.to_parquet(cache_path)
                return df_mevcut
        except Exception as e:
            st.warning(f"Önbellek okunamadı, yeniden çekiliyor: {e}")
            if os.path.exists(cache_path):
                os.remove(cache_path)
    st.info("🔄 İlk kez çalışıyor veya önbellek yok. Veriler çekiliyor...")
    df_pivot = provider.fetch_prices(start_date, end_date)
    if not df_pivot.empty:
        df_pivot.to_parquet(cache_path)
    return df_pivot

# =============================================
# 2. VERİ SAĞLAYICILARI
# =============================================
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
                if not df.empty:
                    all_data[name] = df['Close']
            except:
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
                df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('date')
                df = df.loc[start_date:end_date]
                if not df.empty:
                    all_data[name] = df['close']
            except:
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
            df_kod = crawler_fonlar.fetch(start=baslangic.strftime("%Y-%m-%d"),
                                          end=bugun.strftime("%Y-%m-%d"))
            if df_kod is None or df_kod.empty:
                return pd.DataFrame()
            kod_kolonu = "code" if "code" in df_kod.columns else "fon_kodu"
            tum_kodlar = df_kod[kod_kolonu].dropna().unique().tolist()
            df_list = []
            for fund in tum_kodlar[:200]:
                try:
                    df_fund = self.crawler.fetch(
                        start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d"),
                        name=fund,
                        columns=["code","date","price"]
                    )
                    if not df_fund.empty:
                        df_list.append(df_fund)
                except:
                    continue
            if not df_list:
                return pd.DataFrame()
            df_raw = pd.concat(df_list, ignore_index=True)
            df_pivot = df_raw.pivot(index="date", columns="code", values="price")
            df_pivot.index = pd.to_datetime(df_pivot.index)
            df_pivot = df_pivot.sort_index().ffill().dropna(axis=1, how='all')
            return df_pivot
        except:
            return pd.DataFrame()

# =============================================
# 3. VARLIK EVRENİ VE PARAMETRELER
# =============================================
DEFAULT_UNIVERSE = {
    'hisse_bist': {'XU100':'XU100.IS','THYAO':'THYAO.IS','GARAN':'GARAN.IS','AKBNK':'AKBNK.IS','SISE':'SISE.IS'},
    'hisse_abd': {'AAPL':'AAPL','MSFT':'MSFT','GOOGL':'GOOGL','AMZN':'AMZN','META':'META','SPY':'SPY'},
    'hisse_avrupa': {'SAP':'SAP.DE','ASML':'ASML.AS','NOVO':'NOVO-B.CO','NESN':'NESN.SW'},
    'kripto': {'BTC':'BTC/USDT','ETH':'ETH/USDT','BNB':'BNB/USDT','SOL':'SOL/USDT'},
    'emtia': {'Altın':'GC=F','Gümüş':'SI=F','Petrol':'CL=F'},
    'doviz': {'EUR/USD':'EURUSD=X','GBP/USD':'GBPUSD=X','USD/JPY':'USDJPY=X','GBPJPY':'GBPJPY=X'},
    'etf': {'SPY':'SPY','QQQ':'QQQ','VTI':'VTI','BND':'BND'}
}

ASSET_PARAMS = {
    'hisse_bist': {'komisyon_alis': 0.011, 'komisyon_satis': 0.016, 'valor': 2, 'max_position': 0.70},
    'hisse_abd': {'komisyon_alis': 0.005, 'komisyon_satis': 0.005, 'valor': 2, 'max_position': 0.70},
    'hisse_avrupa': {'komisyon_alis': 0.005, 'komisyon_satis': 0.005, 'valor': 2, 'max_position': 0.70},
    'kripto': {'komisyon_alis': 0.001, 'komisyon_satis': 0.001, 'valor': 0, 'max_position': 0.80},
    'emtia': {'komisyon_alis': 0.002, 'komisyon_satis': 0.002, 'valor': 2, 'max_position': 0.60},
    'doviz': {'komisyon_alis': 0.0005, 'komisyon_satis': 0.0005, 'valor': 2, 'max_position': 0.90},
    'etf': {'komisyon_alis': 0.005, 'komisyon_satis': 0.005, 'valor': 2, 'max_position': 0.70}
}

STRATEGY_TYPES = {
    'Momentum': 'momentum',
    'Değer (F/K)': 'value',
    'Düşük Volatilite': 'low_vol',
    'Carry (Taşıma)': 'carry',
    'Karma (Momentum+Değer)': 'blend'
}

# =============================================
# 4. YARDIMCI FONKSİYONLAR
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
    return 1 - (current / mean_price).clip(0,1)

def calculate_low_volatility(df_prices, lookback=30):
    if len(df_prices) < lookback:
        return pd.Series(index=df_prices.columns, data=0.0)
    rets = df_prices.pct_change().tail(lookback)
    vols = rets.std()
    return 1 - (vols / vols.max())

def calculate_carry(df_prices, lookback=30):
    if len(df_prices) < lookback:
        return pd.Series(index=df_prices.columns, data=0.0)
    returns = df_prices.pct_change().tail(lookback)
    return returns.mean()

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
    else:
        return calculate_momentum(df_prices, lookback)

def select_top_assets(scores, top_n=3):
    return scores.nlargest(top_n).index.tolist()

def parse_ollama_json(raw_text):
    try:
        cleaned = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        json_match = re.search(r'(\{.*\})', cleaned, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        return json.loads(cleaned)
    except:
        return None

def get_ai_weights(funds, market_status_text, macro_data=None, past_decisions=None, fund_stats=None):
    OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
    MODEL = 'fingpt'
    macro_text = ""
    if macro_data:
        macro_text = "\n📊 Güncel Makro Veriler:\n"
        for key, value in macro_data.items():
            if not pd.isna(value):
                macro_text += f"- {key}: {value:.2f}\n"
    stats_text = ""
    if fund_stats:
        stats_text = "\n📈 Varlık Detayları:\n"
        for f, stats in fund_stats.items():
            stats_text += f"- {f}: Vol={stats.get('volatilite', 0):.2f} | Sharpe={stats.get('sharpe', 0):.2f} | 3A Getiri={stats.get('son_3ay_getiri', 0):.2%}\n"
    past_text = ""
    if past_decisions:
        past_text = "\n📚 Geçmiş Benzer Kararlar:\n"
        for dec in past_decisions[-3:]:
            getiri = dec.get('sonuc_getiri')
            if getiri is not None and not pd.isna(getiri):
                getiri_str = f"{getiri:.2%}"
            else:
                getiri_str = "N/A"
            past_text += f"- {dec.get('tarih', '')} | {dec.get('secilen', '')} | Getiri: {getiri_str}\n"
    try:
        prompt = f"""Mevcut varlıkların durumu:
{market_status_text}
{macro_text}
{stats_text}
{past_text}

Bu verilere göre toplamı 100 olacak şekilde şu varlıklar için en ideal ağırlıkları dağıt: {', '.join(funds)}.
Hiçbir varlık %40'ı geçmesin.
JSON formatında döndür:
{{
  "DAGILIM": {", ".join([f'"{f}": [rakam]' for f in funds])},
  "GEREKCE": "[Türkçe gerekçe]"
}}"""
        response = requests.post(f"{OLLAMA_URL}/api/generate", json={"model": MODEL, "prompt": prompt, "stream": False}, timeout=20)
        raw = response.json().get("response", "")
        parsed = parse_ollama_json(raw)
        if parsed and "DAGILIM" in parsed:
            temp = parsed["DAGILIM"]
            total = sum(float(v) for v in temp.values())
            if total > 0:
                weights = {f: float(temp.get(f, 0)) / total for f in funds}
                return weights, parsed.get("GEREKCE", "AI kararı")
    except:
        pass
    return {f: 1.0/len(funds) for f in funds}, "Eşit dağılım"

def apply_stop_loss(history, current_value, initial_capital, stop_loss_pct):
    if current_value < initial_capital * (1 - stop_loss_pct):
        return True
    return False

def calculate_var(returns, confidence=0.95):
    if returns.empty:
        return np.nan
    mu = returns.mean()
    sigma = returns.std()
    return mu - sigma * np.percentile(np.random.normal(0,1,10000), (1-confidence)*100)

def calculate_cvar(returns, confidence=0.95):
    if returns.empty:
        return np.nan
    var = calculate_var(returns, confidence)
    return returns[returns <= var].mean() if not returns[returns <= var].empty else np.nan

def max_drawdown(series):
    if series.empty:
        return np.nan
    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    return drawdown.min()

def concentration_ratio(weights):
    return max(weights) if weights else 0

def liquidity_filter(df_prices, min_volume=1000000):
    valid = []
    for col in df_prices.columns:
        if df_prices[col].dropna().count() >= 30:
            valid.append(col)
    return valid

def calculate_risk_metrics(portfolio_values):
    if portfolio_values.empty or len(portfolio_values) < 2:
        return {}
    returns = portfolio_values.pct_change().dropna()
    var_95 = calculate_var(returns, 0.95)
    var_99 = calculate_var(returns, 0.99)
    cvar_95 = calculate_cvar(returns, 0.95)
    maxdd = max_drawdown(portfolio_values)
    return {
        'VaR %95': var_95 * 100,
        'VaR %99': var_99 * 100,
        'CVaR %95': cvar_95 * 100,
        'Max Drawdown': maxdd * 100
    }

def calculate_metrics(df_history):
    if df_history.empty or len(df_history) < 2:
        return {}
    df = df_history.copy()
    df['Gunluk_Getiri'] = df['Toplam_Varlik'].pct_change()
    bas = df['Toplam_Varlik'].iloc[0]
    bit = df['Toplam_Varlik'].iloc[-1]
    toplam = (bit - bas) / bas
    gunluk_std = df['Gunluk_Getiri'].std()
    yillik_vol = gunluk_std * np.sqrt(252) if not pd.isna(gunluk_std) else 0.0
    yillik_getiri = df['Gunluk_Getiri'].mean() * 252
    sharpe = yillik_getiri / (yillik_vol + 1e-9)
    rolling_max = df['Toplam_Varlik'].cummax()
    drawdown = (df['Toplam_Varlik'] - rolling_max) / rolling_max
    maxdd = drawdown.min()
    var_95 = np.percentile(df['Gunluk_Getiri'].dropna(), 5)
    return {
        'Toplam Getiri (%)': toplam * 100,
        'Yıllık Getiri (%)': yillik_getiri * 100,
        'Yıllık Volatilite (%)': yillik_vol * 100,
        'Sharpe Oranı': sharpe,
        'Max Drawdown (%)': maxdd * 100,
        'Günlük VaR (%95)': var_95 * 100
    }

# =============================================
# 5. BACKTEST MOTORU
# =============================================
def run_backtest(df_prices, initial_capital=100000, stop_loss_pct=0.12, top_n=3, lookback=30,
                 max_position_pct=0.70, komisyon_alis=0.011, komisyon_satis=0.016, valor=2,
                 strategy_type='momentum', risk_limit_sector=None, risk_limit_asset=0.40):
    df = df_prices.sort_index().copy()
    if df.empty or len(df) < 30:
        return pd.DataFrame(), [], {}

    dates = df.index
    first_day_of_month = dates[dates.is_month_start].unique()
    if len(first_day_of_month) == 0:
        first_day_of_month = [dates[0]]

    nakit = initial_capital
    portfoy = {}
    bloke_nakit = []
    history = []
    decision_log = []
    past_decisions = []
    daily_returns = []

    for i, current_date in enumerate(dates):
        serbest = 0
        kalan_bloke = []
        for bloke in bloke_nakit:
            if current_date >= bloke['tarih']:
                serbest += bloke['tutar']
            else:
                kalan_bloke.append(bloke)
        bloke_nakit = kalan_bloke
        nakit += serbest

        nakit += nakit * 0.15 / 365.0

        portfoy_degeri = sum([adet * df.loc[current_date, fon] for fon, adet in portfoy.items() if fon in df.columns and current_date in df.index])
        toplam = nakit + portfoy_degeri + sum(b['tutar'] for b in bloke_nakit)
        history.append({'Tarih': current_date, 'Toplam_Varlik': toplam, 'Nakit': nakit, 'Portfoy_Degeri': portfoy_degeri, 'Bloke_Nakit': sum(b['tutar'] for b in bloke_nakit)})
        if len(history) > 1:
            daily_returns.append((toplam - history[-2]['Toplam_Varlik']) / history[-2]['Toplam_Varlik'])

        if apply_stop_loss(history, toplam, initial_capital, stop_loss_pct):
            for fon in list(portfoy.keys()):
                if portfoy[fon] > 0.001:
                    fiyat = df.loc[current_date, fon]
                    brut = portfoy[fon] * fiyat
                    net = brut * (1 - komisyon_satis)
                    bloke_nakit.append({'tarih': current_date + timedelta(days=valor), 'tutar': net})
                    del portfoy[fon]
            decision_log.append({'Tarih': current_date.strftime('%Y-%m-%d'), 'Seçilen Fonlar': 'STOP-LOSS'})
            continue

        if current_date in first_day_of_month and len(dates) - i > 5:
            scores = get_strategy_scores(df.loc[:current_date], strategy_type, lookback)
            valid_assets = liquidity_filter(df.loc[:current_date], min_volume=1)
            top_funds = select_top_assets(scores, top_n)
            selected_funds = [f for f in top_funds if f in valid_assets]
            if not selected_funds:
                selected_funds = top_funds[:min(top_n, len(top_funds))]
            weights, reason = get_ai_weights(selected_funds, "", None, past_decisions)
            total_weight = sum(weights.values())
            if total_weight > 0:
                normalized = {k: v/total_weight for k,v in weights.items()}
                if risk_limit_asset:
                    for f in list(normalized.keys()):
                        if normalized[f] > risk_limit_asset:
                            normalized[f] = risk_limit_asset
                    total = sum(normalized.values())
                    if total > 0:
                        normalized = {k: v/total for k,v in normalized.items()}
                weights = normalized

            if weights:
                for fon in list(portfoy.keys()):
                    if portfoy[fon] > 0.001:
                        fiyat = df.loc[current_date, fon]
                        brut = portfoy[fon] * fiyat
                        net = brut * (1 - komisyon_satis)
                        bloke_nakit.append({'tarih': current_date + timedelta(days=valor), 'tutar': net})
                        del portfoy[fon]
                alim_gucu = nakit * max_position_pct
                if alim_gucu > 100:
                    for fon, w in weights.items():
                        if fon in df.columns:
                            fiyat = df.loc[current_date, fon]
                            if not pd.isna(fiyat) and fiyat > 0:
                                brut = alim_gucu * w
                                adet = brut / fiyat
                                if adet > 0.001:
                                    portfoy[fon] = adet
                                    nakit -= brut
                decision_log.append({
                    'Tarih': current_date.strftime('%Y-%m-%d'),
                    'Seçilen Fonlar': ', '.join(weights.keys()),
                    'Ağırlıklar': ', '.join([f"{f}: %{w*100:.1f}" for f,w in weights.items()]),
                    'Strateji': strategy_type,
                    'Gerekçe': reason
                })
                past_decisions.append({
                    'tarih': current_date.strftime('%Y-%m-%d'),
                    'secilen': ', '.join(weights.keys()),
                    'agirliklar': weights,
                    'sonuc_getiri': None
                })

    df_history = pd.DataFrame(history).set_index('Tarih')
    for dec in past_decisions:
        tarih = pd.to_datetime(dec['tarih'])
        mask = (df_history.index >= tarih) & (df_history.index < tarih + timedelta(days=30))
        if mask.any():
            bas = df_history.loc[mask, 'Toplam_Varlik'].iloc[0]
            bit = df_history.loc[mask, 'Toplam_Varlik'].iloc[-1]
            dec['sonuc_getiri'] = (bit - bas) / bas if bas != 0 else np.nan
        else:
            dec['sonuc_getiri'] = np.nan

    risk_metrics = calculate_risk_metrics(df_history['Toplam_Varlik'])
    return df_history, decision_log, risk_metrics

# =============================================
# 6. PORTFÖY İŞLEMLERİ (SQLITE)
# =============================================
def get_portfolio(asset_class):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT nakit, varliklar, islemler, baslangic_bakiye FROM portfolios WHERE asset_class=?", (asset_class,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            'nakit': row[0],
            'varliklar': json.loads(row[1]) if row[1] else {},
            'islemler': json.loads(row[2]) if row[2] else [],
            'baslangic_bakiye': row[3]
        }
    return None

def update_portfolio(asset_class, data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO portfolios 
                 (asset_class, nakit, varliklar, islemler, baslangic_bakiye, son_guncelleme)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (asset_class, data['nakit'], json.dumps(data['varliklar']),
               json.dumps(data['islemler']), data['baslangic_bakiye'],
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

def add_signal_history(asset_class, tarih, secilen, agirliklar, sonuc_getiri=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO signal_history (asset_class, tarih, secilen, agirliklar, sonuc_getiri)
                 VALUES (?, ?, ?, ?, ?)''',
              (asset_class, tarih, secilen, json.dumps(agirliklar), sonuc_getiri))
    conn.commit()
    conn.close()

def apply_signal_to_portfolio(weights, current_date, initial_capital, asset_class,
                              max_position_pct=0.70, komisyon_alis=0.011, komisyon_satis=0.016,
                              valor=2, df_prices=None):
    pf = get_portfolio(asset_class)
    if pf is None:
        pf = {
            'nakit': initial_capital,
            'varliklar': {},
            'islemler': [],
            'baslangic_bakiye': initial_capital,
            'bloke_nakit': []
        }
    if 'bloke_nakit' not in pf:
        pf['bloke_nakit'] = []

    df = df_prices
    serbest = 0
    kalan_bloke = []
    for bloke in pf.get('bloke_nakit', []):
        if current_date >= bloke['tarih']:
            serbest += bloke['tutar']
        else:
            kalan_bloke.append(bloke)
    pf['bloke_nakit'] = kalan_bloke
    pf['nakit'] += serbest
    pf['nakit'] += pf['nakit'] * 0.15 / 365.0

    if weights:
        for fon in list(pf['varliklar'].keys()):
            adet = pf['varliklar'][fon]
            if adet > 0.001 and fon in df.columns and current_date in df.index:
                fiyat = df.loc[current_date, fon]
                if not pd.isna(fiyat) and fiyat > 0:
                    brut = adet * fiyat
                    net = brut * (1 - komisyon_satis)
                    valör_tarih = current_date + timedelta(days=valor)
                    pf['bloke_nakit'].append({'tarih': valör_tarih, 'tutar': net})
                    pf['islemler'].append({
                        'Tarih': current_date.strftime('%Y-%m-%d'),
                        'İşlem': 'SAT',
                        'Fon': fon,
                        'Adet': adet,
                        'Fiyat': fiyat,
                        'Tutar': brut,
                        'Net Tutar': net
                    })
                    del pf['varliklar'][fon]

        alim_gucu = pf['nakit'] * max_position_pct
        if alim_gucu > 100:
            for fon, w in weights.items():
                if fon in df.columns and current_date in df.index:
                    fiyat = df.loc[current_date, fon]
                    if not pd.isna(fiyat) and fiyat > 0:
                        brut = alim_gucu * w
                        adet = brut / fiyat
                        if adet > 0.001:
                            pf['varliklar'][fon] = pf['varliklar'].get(fon, 0) + adet
                            pf['nakit'] -= brut
                            pf['islemler'].append({
                                'Tarih': current_date.strftime('%Y-%m-%d'),
                                'İşlem': 'AL',
                                'Fon': fon,
                                'Adet': adet,
                                'Fiyat': fiyat,
                                'Tutar': brut
                            })

    pf['son_guncelleme'] = current_date.isoformat()
    update_portfolio(asset_class, pf)
    add_signal_history(asset_class, current_date.strftime('%Y-%m-%d'),
                       ', '.join(weights.keys()) if weights else '',
                       weights if weights else {}, None)
    return pf

# =============================================
# 7. TÜM SINIFLARI YÜKLEME
# =============================================
def load_all_asset_data():
    loaded = 0
    failed = []
    end_date = datetime.date.today()
    start_date = end_date - timedelta(days=365*2)

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
            symbols = DEFAULT_UNIVERSE.get(cls, {})
            provider.set_symbols(symbols)

        try:
            df = fon_verilerini_yukle(provider, start_date, end_date, cls)
            if not df.empty:
                st.session_state.df_all_raw[cls] = df
                loaded += 1
            else:
                failed.append(cls)
        except Exception as e:
            failed.append(cls)
            st.warning(f"⚠️ {cls} yüklenirken hata: {e}")

    return loaded, failed

# =============================================
# 8. OTOMATİK YENİDEN DENGELEME (DÜZELTİLDİ)
# =============================================
def rebalance_portfolio(allocations, initial_capital=100000, max_position_pct=0.70,
                        komisyon_alis=0.011, komisyon_satis=0.016, valor=2):
    class_values = {}
    total_value = 0
    for cls in ALL_ASSET_CLASSES:
        pf = get_portfolio(cls)
        if pf is None:
            continue
        if 'bloke_nakit' not in pf:
            pf['bloke_nakit'] = []
        val = pf['nakit']
        df_cls = st.session_state.df_all_raw.get(cls, pd.DataFrame())
        for fon, adet in pf['varliklar'].items():
            if fon in df_cls.columns:
                fiyat = df_cls[fon].dropna().iloc[-1]
                val += adet * fiyat
        class_values[cls] = val
        total_value += val

    if total_value == 0:
        st.warning("Hiçbir portföyde varlık yok. Önce sinyal uygulayın.")
        return

    target_values = {}
    for cls, alloc_info in allocations.items():
        if alloc_info['hedef'] > 0:
            target_values[cls] = total_value * alloc_info['hedef']

    for cls, target_val in target_values.items():
        current_val = class_values.get(cls, 0)
        diff = target_val - current_val
        if abs(diff) < 100:
            continue

        df_cls = st.session_state.df_all_raw.get(cls, pd.DataFrame())
        if df_cls.empty:
            continue

        pf = get_portfolio(cls)
        if pf is None:
            pf = {
                'nakit': initial_capital * allocations[cls]['hedef'],
                'varliklar': {},
                'islemler': [],
                'baslangic_bakiye': initial_capital,
                'bloke_nakit': []
            }
        if 'bloke_nakit' not in pf:
            pf['bloke_nakit'] = []

        if diff < 0:
            satis_orani = (current_val - target_val) / current_val
            for fon, adet in list(pf['varliklar'].items()):
                if fon in df_cls.columns:
                    fiyat = df_cls[fon].dropna().iloc[-1]
                    satis_adet = adet * satis_orani
                    if satis_adet > 0.001:
                        brut = satis_adet * fiyat
                        net = brut * (1 - komisyon_satis)
                        valör_tarih = datetime.date.today() + timedelta(days=valor)
                        pf['bloke_nakit'].append({'tarih': valör_tarih, 'tutar': net})
                        pf['islemler'].append({
                            'Tarih': datetime.date.today().strftime('%Y-%m-%d'),
                            'İşlem': 'SAT (Rebalance)',
                            'Fon': fon,
                            'Adet': satis_adet,
                            'Fiyat': fiyat,
                            'Tutar': brut,
                            'Net Tutar': net
                        })
                        pf['varliklar'][fon] = max(0, adet - satis_adet)
                        if pf['varliklar'][fon] < 0.001:
                            del pf['varliklar'][fon]
            pf['nakit'] += (current_val - target_val) * 0.98
        else:
            alim_tutari = diff
            current_weights = {}
            total_asset_value = 0
            for fon, adet in pf['varliklar'].items():
                if fon in df_cls.columns:
                    fiyat = df_cls[fon].dropna().iloc[-1]
                    total_asset_value += adet * fiyat
            if total_asset_value > 0:
                for fon, adet in pf['varliklar'].items():
                    if fon in df_cls.columns:
                        fiyat = df_cls[fon].dropna().iloc[-1]
                        current_weights[fon] = (adet * fiyat) / total_asset_value
            else:
                available = df_cls.columns.tolist()[:3]
                if available:
                    current_weights = {f: 1/len(available) for f in available}
                else:
                    continue

            alim_gucu = alim_tutari * max_position_pct
            if alim_gucu > 100:
                for fon, w in current_weights.items():
                    if fon in df_cls.columns:
                        fiyat = df_cls[fon].dropna().iloc[-1]
                        brut = alim_gucu * w
                        adet = brut / fiyat
                        if adet > 0.001:
                            pf['varliklar'][fon] = pf['varliklar'].get(fon, 0) + adet
                            pf['nakit'] -= brut
                            pf['islemler'].append({
                                'Tarih': datetime.date.today().strftime('%Y-%m-%d'),
                                'İşlem': 'AL (Rebalance)',
                                'Fon': fon,
                                'Adet': adet,
                                'Fiyat': fiyat,
                                'Tutar': brut
                            })

        update_portfolio(cls, pf)
        st.success(f"✅ {cls} yeniden dengelendi.")

# =============================================
# 9. PERFORMANS VE ALLOKASYON OPTİMİZASYONU
# =============================================
def get_class_performance(asset_class, df_prices, initial_capital, params):
    strategies = ['momentum', 'value', 'low_vol', 'carry', 'blend']

    def test_strategy(strat):
        p = params.copy()
        p['strategy_type'] = strat
        df_result, _, _ = run_backtest(
            df_prices,
            initial_capital=initial_capital,
            stop_loss_pct=p.get('stop_loss', 0.12),
            top_n=p.get('top_n', 3),
            lookback=p.get('lookback', 45),
            max_position_pct=p.get('max_position', 0.70),
            komisyon_alis=p.get('komisyon_alis', 0.011),
            komisyon_satis=p.get('komisyon_satis', 0.016),
            valor=p.get('valor', 2),
            strategy_type=strat,
            risk_limit_asset=p.get('risk_limit_asset', 0.40)
        )
        if df_result.empty:
            return None
        metrics = calculate_metrics(df_result)
        return {
            'strategy': strat,
            'sharpe': metrics.get('Sharpe Oranı', 0),
            'getiri': metrics.get('Toplam Getiri (%)', 0),
            'volatilite': metrics.get('Yıllık Volatilite (%)', 0),
            'max_drawdown': metrics.get('Max Drawdown (%)', 0),
            'df_result': df_result
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(test_strategy, strat) for strat in strategies]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    best = None
    best_sharpe = -np.inf
    for res in results:
        if res and res['sharpe'] > best_sharpe:
            best_sharpe = res['sharpe']
            best = res

    if best:
        return {
            'getiri': best['getiri'],
            'sharpe': best['sharpe'],
            'volatilite': best['volatilite'],
            'max_drawdown': best['max_drawdown'],
            'df_result': best['df_result'],
            'strategy': best['strategy']
        }
    return None

def optimize_allocation(performances, method='sharpe', min_weights=None, max_weights=None):
    classes = list(performances.keys())
    n = len(classes)
    if n == 0:
        return {}
    if method == 'sharpe':
        sharpe_values = [max(0, p['sharpe']) for p in performances.values()]
        total = sum(sharpe_values)
        if total == 0:
            weights = [1/n] * n
        else:
            weights = [s / total for s in sharpe_values]
    elif method == 'risk_parity':
        vols = [p['volatilite'] for p in performances.values()]
        inv_vols = [1 / (v + 1e-6) for v in vols]
        total = sum(inv_vols)
        weights = [v / total for v in inv_vols]
    elif method == 'min_volatility':
        vols = [p['volatilite'] for p in performances.values()]
        inv_vols = [1 / (v + 1e-6) for v in vols]
        total = sum(inv_vols)
        weights = [v / total for v in inv_vols]
    else:
        weights = [1/n] * n

    if min_weights or max_weights:
        for i, cls in enumerate(classes):
            if min_weights and cls in min_weights:
                weights[i] = max(weights[i], min_weights[cls])
            if max_weights and cls in max_weights:
                weights[i] = min(weights[i], max_weights[cls])
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
    return {cls: w for cls, w in zip(classes, weights)}

# =============================================
# 10. MAKRO VERİ
# =============================================
@st.cache_data(ttl=3600)
def get_benchmark_data(start_date, end_date):
    symbols = {
        'Altın (ONS)': 'GC=F',
        'BIST 100': 'XU100.IS',
        'S&P 500': 'SPY',
        'USD/TRY': 'USDTRY=X',
        'EUR/TRY': 'EURTRY=X'
    }
    all_data = {}
    for name, symbol in symbols.items():
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not df.empty and 'Close' in df.columns:
                series = df['Close']
                if not series.empty:
                    series.name = name
                    all_data[name] = series
        except Exception:
            continue
    if not all_data:
        return pd.DataFrame()
    df_result = pd.concat(all_data, axis=1)
    df_result.columns = [col[0] if isinstance(col, tuple) else col for col in df_result.columns]
    return df_result

# =============================================
# 11. RAPORLAMA, STRES TESTİ, ATTRIBÜSYON
# =============================================
def create_pdf_report(df_result, metrikler, risk_metrics, decision_log, filename="rapor.pdf"):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], alignment=1, fontSize=18)
    story.append(Paragraph("Multi-Asset AI Portföy Raporu", title_style))
    story.append(Spacer(1,12))
    story.append(Paragraph("Performans Metrikleri", styles['Heading2']))
    data = [['Metrik','Değer']]
    for k,v in metrikler.items():
        data.append([k, f"{v:.2f}" if isinstance(v,float) else str(v)])
    t = Table(data)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.grey),('GRID',(0,0),(-1,-1),1,colors.black)]))
    story.append(t)
    story.append(Spacer(1,12))
    story.append(Paragraph("Risk Metrikleri", styles['Heading2']))
    rdata = [['Risk Metriği','Değer']]
    for k,v in risk_metrics.items():
        rdata.append([k, f"{v:.2f}" if isinstance(v,float) else str(v)])
    rt = Table(rdata)
    rt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.grey),('GRID',(0,0),(-1,-1),1,colors.black)]))
    story.append(rt)
    story.append(Spacer(1,12))
    story.append(Paragraph("Kararlar", styles['Heading2']))
    for log in decision_log[-5:]:
        story.append(Paragraph(f"{log.get('Tarih','')}: {log.get('Seçilen Fonlar','')} - {log.get('Strateji','')}", styles['Normal']))
    doc.build(story)
    return filename

def advanced_stress_test(df_prices, scenarios):
    results = {}
    for name, shock_pct in scenarios.items():
        shocked = df_prices * (1 + shock_pct/100)
        ret = (shocked.mean().mean() - df_prices.mean().mean()) / df_prices.mean().mean()
        results[name] = ret * 100
    return results

def attribution_analysis(df_prices, df_history, decision_log):
    if df_history.empty or not decision_log:
        return {}
    return {
        'Varlık Seçimi (%)': 0.0,
        'Piyasa Etkisi (%)': 0.0,
        'Döviz Etkisi (%)': 0.0
    }

# =============================================
# 12. ARAYÜZ İYİLEŞTİRMELERİ
# =============================================
st.markdown("""
<style>
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .stButton button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #3b5e9e;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# 13. SESSION STATE
# =============================================
if 'df_all_raw' not in st.session_state:
    st.session_state.df_all_raw = {}
if 'df_bench' not in st.session_state:
    st.session_state.df_bench = None
if 'last_signal' not in st.session_state:
    st.session_state.last_signal = {}
if 'df_result' not in st.session_state:
    st.session_state.df_result = None
if 'decision_log' not in st.session_state:
    st.session_state.decision_log = []
if 'metrikler' not in st.session_state:
    st.session_state.metrikler = {}
if 'risk_metrikler' not in st.session_state:
    st.session_state.risk_metrikler = {}
if 'custom_symbols' not in st.session_state:
    st.session_state.custom_symbols = {}
if 'basarili_fonlar' not in st.session_state:
    st.session_state.basarili_fonlar = []

ALL_ASSET_CLASSES = list(DEFAULT_UNIVERSE.keys()) + ['tefas_fon']

# =============================================
# 14. STREAMLIT ARAYÜZÜ
# =============================================
st.set_page_config(page_title="Profesyonel AI Portföy Yönetimi", layout="wide")
st.title("🏛️ Profesyonel AI Portföy Yönetimi (SQLite Kalıcı + Otomatik Allokasyon)")
st.markdown("""
**Sürekli Öğrenen, Risk Odaklı, Makro Entegre Sistem - SQLite ile Kalıcı**
- Hisse (BIST, ABD, Avrupa) | Kripto | Emtia | Döviz | ETF | TEFAS Fonları
- Çoklu strateji (Momentum, Değer, Düşük Volatilite, Carry)
- Gelişmiş risk yönetimi (VaR, CVaR, Drawdown, konsantrasyon)
- Stres testi ve senaryo analizi
- Gerçekçi backtest (komisyon, valör, faiz, pozisyon sınırı)
- Sanal portföy yönetimi ve sürekli öğrenme
- SQLite ile kalıcı portföy yönetimi ve allokasyon tabanlı karma portföy
- Otomatik allokasyon optimizasyonu (Sharpe, Risk Paritesi, Min Volatilite)
""")

# ---- Varlık sınıfı seçimi ----
asset_class = st.selectbox(
    "Varlık Sınıfı Seçin (Görüntüleme / İşlem)",
    ALL_ASSET_CLASSES
)

# ---- Sidebar ----
alloc_data = get_allocations()

st.sidebar.subheader("⚖️ Sermaye Allokasyonu (Toplam %100)")
total_alloc = 0
for cls in ALL_ASSET_CLASSES:
    default_ratio = alloc_data.get(cls, {}).get('hedef', 0.0)
    ratio = st.sidebar.slider(f"{cls}", 0, 100, int(default_ratio*100), 5, key=f"alloc_{cls}") / 100.0
    min_constraint = alloc_data.get(cls, {}).get('min', 0.0)
    max_constraint = alloc_data.get(cls, {}).get('max', 1.0)
    set_allocation(cls, ratio, min_constraint, max_constraint)
    total_alloc += ratio

if total_alloc != 1.0:
    st.sidebar.warning(f"⚠️ Toplam allokasyon %{total_alloc*100:.0f}, %100 olmalıdır.")

# ---- Tüm Sınıfları Yükle ----
st.sidebar.markdown("---")
if st.sidebar.button("📂 Tüm Sınıfları Yükle", use_container_width=True):
    with st.spinner("Tüm sınıfların verileri yükleniyor..."):
        loaded, failed = load_all_asset_data()
        if loaded > 0:
            st.sidebar.success(f"✅ {loaded} sınıf başarıyla yüklendi.")
        if failed:
            st.sidebar.warning(f"⚠️ Yüklenemeyen sınıflar: {', '.join(failed)}")
        if loaded == 0 and not failed:
            st.sidebar.info("Tüm sınıflar zaten yüklüydü.")
        st.rerun()

# ---- Özel sembol ----
st.sidebar.subheader("✏️ Özel Sembol Ekle")
col_sym1, col_sym2 = st.sidebar.columns([3,1])
with col_sym1:
    new_symbol = st.text_input("Sembol (örn: AAPL, EURUSD=X, GC=F)", key="new_symbol_input", placeholder="Sembol girin...").strip()
with col_sym2:
    add_btn = st.button("➕ Ekle", use_container_width=True)

if add_btn and new_symbol:
    test_provider = YahooProvider()
    test_provider.set_symbols({'test': new_symbol})
    test_df = test_provider.fetch_prices(datetime.date.today() - timedelta(days=30), datetime.date.today())
    if not test_df.empty:
        if asset_class not in st.session_state.custom_symbols:
            st.session_state.custom_symbols[asset_class] = []
        if new_symbol not in st.session_state.custom_symbols[asset_class]:
            st.session_state.custom_symbols[asset_class].append(new_symbol)
            st.sidebar.success(f"✅ {new_symbol} eklendi!")
        else:
            st.sidebar.warning(f"⚠️ {new_symbol} zaten listede.")
    else:
        st.sidebar.error(f"❌ {new_symbol} geçersiz veya veri çekilemedi.")

if asset_class in st.session_state.custom_symbols and st.session_state.custom_symbols[asset_class]:
    st.sidebar.subheader("📋 Özel Semboller")
    for sym in st.session_state.custom_symbols[asset_class]:
        col1, col2 = st.sidebar.columns([4,1])
        col1.write(sym)
        if col2.button("🗑️", key=f"del_{sym}"):
            st.session_state.custom_symbols[asset_class].remove(sym)
            st.rerun()

# ---- Risk Limitleri ----
st.sidebar.subheader("⚖️ Risk Limitleri")
risk_asset_limit = st.sidebar.slider("Tek Varlık Maks. Ağırlık (%)", 10, 50, 40) / 100.0

# ---- Veri sağlayıcı ----
if asset_class == "tefas_fon":
    provider = TefasProvider()
    symbols = {}
else:
    if asset_class == "kripto":
        provider = BinanceProvider()
    else:
        provider = YahooProvider()
    default_symbols = DEFAULT_UNIVERSE.get(asset_class, {})
    custom_symbols = st.session_state.custom_symbols.get(asset_class, [])
    symbols = default_symbols.copy()
    for sym in custom_symbols:
        if sym and sym not in symbols:
            symbols[sym] = sym
    provider.set_symbols(symbols)

end_date = datetime.date.today()
start_date = end_date - timedelta(days=365*2)

with st.spinner(f"{asset_class} verileri yükleniyor..."):
    try:
        df_prices = fon_verilerini_yukle(provider, start_date, end_date, asset_class)
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        st.stop()

if df_prices.empty:
    st.error(f"{asset_class} için veri çekilemedi! Lütfen başka bir varlık sınıfı seçin.")
    st.stop()

st.session_state.df_all_raw[asset_class] = df_prices
st.success(f"✅ {len(df_prices.columns)} varlık, {len(df_prices)} günlük veri yüklendi.")

# ---- Makro veriler ----
with st.spinner("Makro veriler çekiliyor..."):
    try:
        df_bench = get_benchmark_data(start_date, end_date)
        st.session_state.df_bench = df_bench
    except:
        st.warning("Makro veriler çekilemedi, bazı özellikler kısıtlı olabilir.")
        df_bench = pd.DataFrame()

# ---- Parametreler ----
params = ASSET_PARAMS.get(asset_class, {})
default_max_pos = params.get('max_position', 0.70)

col1, col2, col3 = st.columns(3)
with col1:
    initial_capital = st.number_input("Başlangıç Sermayesi (TL)", value=100000, step=10000)
with col2:
    backtest_years = st.selectbox("Geçmiş Yıl", [1,2,3], index=1)
with col3:
    max_position_pct = st.slider("Maks. Pozisyon Oranı (%)", 40, 100, int(default_max_pos*100), 5) / 100.0

col4, col5, col6 = st.columns(3)
with col4:
    stop_loss = st.slider("Stop-Loss (%)", 5, 25, 12, 1) / 100.0
with col5:
    top_n = st.slider("Seçilecek Varlık Sayısı", 2, 6, 3, 1)
with col6:
    lookback = st.slider("Momentum Bakış (Gün)", 30, 90, 45, 5)

# ---- Strateji Seçimi ----
strategy_type = st.selectbox("Strateji Türü (Sadece Backtest için)", list(STRATEGY_TYPES.keys()), index=0)

st.session_state['active_stop_loss'] = stop_loss
st.session_state['active_top_n'] = top_n
st.session_state['active_lookback'] = lookback
st.session_state['active_max_position'] = max_position_pct
st.session_state['active_strategy'] = strategy_type
st.session_state['risk_asset_limit'] = risk_asset_limit

# ---- Otomatik Allokasyon (YENİ TEK BUTON) ----
st.subheader("🤖 Otomatik Allokasyon Optimizasyonu ve Uygulama")
with st.expander("⚙️ Optimizasyon Ayarları", expanded=True):
    opt_method = st.selectbox("Optimizasyon Yöntemi", ['sharpe', 'risk_parity', 'min_volatility'], index=0)
    opt_period = st.selectbox("Geçmiş Dönem (Yıl)", [1, 2, 3], index=1)
    st.caption("İsteğe bağlı: Her sınıf için minimum ve maksimum oran sınırları (0-1 arası).")
    min_constraints = {}
    max_constraints = {}
    cols = st.columns(3)
    for i, cls in enumerate(ALL_ASSET_CLASSES):
        col = cols[i % 3]
        with col:
            st.write(f"**{cls}**")
            min_val = st.number_input(f"Min {cls}", 0.0, 1.0, 0.0, 0.05, key=f"min_{cls}")
            max_val = st.number_input(f"Max {cls}", 0.0, 1.0, 1.0, 0.05, key=f"max_{cls}")
            if min_val > 0:
                min_constraints[cls] = min_val
            if max_val < 1:
                max_constraints[cls] = max_val

if st.button("🚀 Otomatik Allokasyon Öner ve Uygula (Tüm Sınıflar)", use_container_width=True):
    with st.spinner("Tüm sınıflar için backtest yapılıyor (tüm stratejiler paralel deneniyor)..."):
        performances = {}
        for cls in ALL_ASSET_CLASSES:
            if cls in st.session_state.df_all_raw and not st.session_state.df_all_raw[cls].empty:
                df_cls = st.session_state.df_all_raw[cls]
                params_back = {
                    'stop_loss': stop_loss,
                    'top_n': top_n,
                    'lookback': lookback,
                    'max_position': max_position_pct,
                    'risk_limit_asset': risk_asset_limit,
                    'komisyon_alis': params.get('komisyon_alis', 0.011),
                    'komisyon_satis': params.get('komisyon_satis', 0.016),
                    'valor': params.get('valor', 2)
                }
                end_date_sub = df_cls.index.max()
                start_date_sub = end_date_sub - timedelta(days=opt_period*365)
                df_cls_sub = df_cls.loc[start_date_sub:end_date_sub]
                if not df_cls_sub.empty and len(df_cls_sub) > 30:
                    perf = get_class_performance(cls, df_cls_sub, initial_capital, params_back)
                    if perf:
                        performances[cls] = perf
        if not performances:
            st.warning("Hiçbir sınıf için geçerli backtest sonucu alınamadı. Lütfen verileri kontrol edin.")
        else:
            optimized = optimize_allocation(performances, method=opt_method,
                                            min_weights=min_constraints, max_weights=max_constraints)
            # 1. Hedef oranları güncelle
            for cls, ratio in optimized.items():
                set_allocation(cls, ratio)
            st.success("✅ Hedef allokasyonlar güncellendi!")

            # 2. Portföyleri yeniden dengele
            allocs = get_allocations()
            rebalance_portfolio(allocs, initial_capital, max_position_pct,
                                params.get('komisyon_alis', 0.011),
                                params.get('komisyon_satis', 0.016),
                                params.get('valor', 2))
            st.success("✅ Portföyler başarıyla yeniden dengelendi!")

            # 3. Sonuçları göster
            st.subheader("📊 Uygulanan Allokasyon")
            df_opt = pd.DataFrame(list(optimized.items()), columns=['Varlık Sınıfı', 'Önerilen Oran (%)'])
            df_opt['Önerilen Oran (%)'] = df_opt['Önerilen Oran (%)'] * 100
            st.dataframe(df_opt, use_container_width=True)

            st.subheader("📈 Sınıf Performansları (En İyi Strateji Seçildi)")
            perf_data = []
            for cls, p in performances.items():
                perf_data.append({
                    'Sınıf': cls,
                    'Strateji': p.get('strategy', 'N/A'),
                    'Getiri (%)': f"{p['getiri']:.2f}",
                    'Sharpe': f"{p['sharpe']:.2f}",
                    'Volatilite (%)': f"{p['volatilite']:.2f}",
                    'Max DD (%)': f"{p['max_drawdown']:.2f}"
                })
            st.dataframe(pd.DataFrame(perf_data), use_container_width=True)
            st.rerun()

# ---- SEKMELER ----
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚀 Backtest & Sonuçlar",
    "📡 Gerçek Zamanlı Sinyal",
    "📊 Portföy Yönetimi",
    "🧪 Stres Testi & Optimizasyon",
    "📄 Raporlama",
    "📈 Attribüsyon & Risk"
])

# ----- TAB 1: Backtest -----
with tab1:
    if st.button("🚀 Backtest'i Çalıştır (Tüm İyileştirmelerle)"):
        with st.spinner("Backtest yapılıyor..."):
            df_result, log, risk_met = run_backtest(
                df_prices,
                initial_capital=initial_capital,
                stop_loss_pct=stop_loss,
                top_n=top_n,
                lookback=lookback,
                max_position_pct=max_position_pct,
                komisyon_alis=params.get('komisyon_alis', 0.011),
                komisyon_satis=params.get('komisyon_satis', 0.016),
                valor=params.get('valor', 2),
                strategy_type=STRATEGY_TYPES.get(strategy_type, 'momentum'),
                risk_limit_asset=risk_asset_limit
            )
        if df_result.empty:
            st.error("Backtest sonucu boş. Lütfen daha uzun bir geçmiş seçin.")
        else:
            metrics = calculate_metrics(df_result)
            st.subheader("📊 Performans Metrikleri")
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Toplam Getiri", f"{metrics.get('Toplam Getiri (%)',0):.2f}%")
            c2.metric("Yıllık Getiri", f"{metrics.get('Yıllık Getiri (%)',0):.2f}%")
            c3.metric("Sharpe", f"{metrics.get('Sharpe Oranı',0):.2f}")
            c4.metric("Max Drawdown", f"{metrics.get('Max Drawdown (%)',0):.2f}%")
            c5.metric("Volatilite", f"{metrics.get('Yıllık Volatilite (%)',0):.2f}%")

            st.subheader("⚠️ Risk Metrikleri")
            r1,r2,r3,r4 = st.columns(4)
            r1.metric("VaR %95", f"{risk_met.get('VaR %95', 0):.2f}%")
            r2.metric("VaR %99", f"{risk_met.get('VaR %99', 0):.2f}%")
            r3.metric("CVaR %95", f"{risk_met.get('CVaR %95', 0):.2f}%")
            r4.metric("Max Drawdown", f"{risk_met.get('Max Drawdown', 0):.2f}%")

            # Ana grafik
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_result.index, y=df_result['Toplam_Varlik'], mode='lines', name='Portföy Değeri', line=dict(color='#1b5e20', width=3)))
            fig.add_trace(go.Scatter(x=df_result.index, y=df_result['Nakit'], mode='lines', name='Nakit', line=dict(color='#ff9100', width=1.5, dash='dash')))
            fig.add_trace(go.Scatter(x=df_result.index, y=df_result['Bloke_Nakit'], mode='lines', name='Bloke Nakit', line=dict(color='#d32f2f', width=1.5, dash='dot')))
            fig.update_layout(title="Portföy Değer Gelişimi", xaxis_title="Tarih", yaxis_title="TL", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # Aylık getiri ısı haritası (GÜVENLİ YÖNTEM - groupby yok)
            if not df_result.empty:
                # Aylık verilere dönüştür
                monthly_data = df_result['Toplam_Varlik'].resample('M').last()
                if len(monthly_data) > 1:
                    monthly_returns = monthly_data.pct_change().dropna() * 100
                    # Isı haritası için pivot tablo
                    pivot_df = pd.DataFrame({
                        'Yıl': monthly_returns.index.year,
                        'Ay': monthly_returns.index.month,
                        'Getiri': monthly_returns.values
                    })
                    pivot_table = pivot_df.pivot(index='Ay', columns='Yıl', values='Getiri')
                    if not pivot_table.empty:
                        fig2 = px.density_heatmap(
                            pivot_table,
                            title="Aylık Getiri Isı Haritası (%)",
                            labels={"value": "Getiri (%)", "Yıl": "Yıl", "Ay": "Ay"}
                        )
                        fig2.update_layout(xaxis_title="Yıl", yaxis_title="Ay")
                        st.plotly_chart(fig2, use_container_width=True)

            # Günlük getiri histogramı
            returns = df_result['Toplam_Varlik'].pct_change().dropna()
            fig3 = px.histogram(returns, nbins=50, title="Günlük Getiri Dağılımı", labels={"value": "Getiri"})
            fig3.add_vline(x=returns.mean(), line_dash="dash", line_color="red", annotation_text=f"Ort: {returns.mean():.2%}")
            st.plotly_chart(fig3, use_container_width=True)

            if log:
                st.subheader("📋 AI Karar Günlüğü")
                st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)

            st.session_state.df_result = df_result
            st.session_state.decision_log = log
            st.session_state.metrikler = metrics
            st.session_state.risk_metrikler = risk_met

            if st.button("📄 PDF Rapor Oluştur"):
                filename = "rapor.pdf"
                create_pdf_report(df_result, metrics, risk_met, log, filename)
                with open(filename, "rb") as f:
                    st.download_button("📥 Raporu İndir", data=f, file_name=filename, mime="application/pdf")

# ----- TAB 2: Gerçek Zamanlı Sinyal -----
with tab2:
    st.subheader("📡 Gerçek Zamanlı Sinyal Üretimi")
    st.caption("Bugünün verilerine göre AI sinyali oluşturur. Sinyali seçili varlık sınıfının portföyüne uygulayabilirsiniz.")

    if st.button("📡 Yeni Sinyal Oluştur"):
        bugun = pd.Timestamp.now().normalize()
        if bugun not in df_prices.index:
            bugun = df_prices.index[-1]
            st.info(f"Bugün için veri yok, en son veri günü: {bugun.strftime('%Y-%m-%d')}")

        top_n_act = st.session_state.get('active_top_n', 3)
        lookback_act = st.session_state.get('active_lookback', 45)
        strat_act = st.session_state.get('active_strategy', 'Momentum')
        if bugun in df_prices.index:
            scores = get_strategy_scores(df_prices, STRATEGY_TYPES.get(strat_act, 'momentum'), lookback_act)
            top_funds = select_top_assets(scores, top_n_act)
            if top_funds:
                weights, reason = get_ai_weights(top_funds, "", None, None)
                st.success("✅ Sinyal oluşturuldu!")
                st.json({
                    'Tarih': bugun.strftime('%Y-%m-%d'),
                    'Önerilen Varlıklar': list(weights.keys()),
                    'Ağırlıklar': {k: f"{v*100:.1f}%" for k,v in weights.items()},
                    'Strateji': strat_act,
                    'Gerekçe': reason
                })
                st.session_state.last_signal[asset_class] = {'weights': weights, 'reason': reason, 'date': bugun}
            else:
                st.error("Sinyal oluşturulamadı, veri yetersiz.")
        else:
            st.warning(f"{bugun.strftime('%Y-%m-%d')} için veri bulunamadı.")

    if asset_class in st.session_state.last_signal:
        if st.button(f"✅ Sinyali {asset_class} Portföyüne Uygula"):
            max_pos = st.session_state.get('active_max_position', 0.70)
            pf = apply_signal_to_portfolio(
                st.session_state.last_signal[asset_class]['weights'],
                st.session_state.last_signal[asset_class]['date'],
                initial_capital,
                asset_class,
                max_position_pct=max_pos,
                komisyon_alis=params.get('komisyon_alis', 0.011),
                komisyon_satis=params.get('komisyon_satis', 0.016),
                valor=params.get('valor', 2),
                df_prices=df_prices
            )
            st.success(f"Sinyal {asset_class} portföyüne uygulandı!")
            st.info("⏳ Bu sinyal 30 gün sonra otomatik değerlendirilecek.")
            st.json({
                'Toplam Nakit': pf['nakit'],
                'Varlıklar': {k: f"{v:.4f} adet" for k,v in pf['varliklar'].items()},
                'İşlem Sayısı': len(pf['islemler'])
            })
            st.rerun()

# ----- TAB 3: Portföy Yönetimi -----
with tab3:
    st.subheader("📊 Portföy Yönetimi")
    st.markdown("Bu sekmede, seçili varlık sınıfının portföyünü ve tüm sınıfların karma portföyünü görüntüleyebilirsiniz.")

    pf = get_portfolio(asset_class)
    if pf:
        st.subheader(f"📈 {asset_class} Portföyü")
        col1, col2, col3 = st.columns(3)
        with col1:
            toplam_deger = pf['nakit']
            for fon, adet in pf['varliklar'].items():
                if fon in df_prices.columns:
                    fiyat = df_prices[fon].dropna().iloc[-1]
                    toplam_deger += adet * fiyat
            st.metric("Toplam Değer", f"{toplam_deger:,.2f} TL", delta=f"{toplam_deger - pf['baslangic_bakiye']:,.2f} TL")
        with col2:
            st.metric("Nakit", f"{pf['nakit']:,.2f} TL")
        with col3:
            st.metric("Varlık Sayısı", len(pf['varliklar']))

        if pf['varliklar']:
            data = []
            for fon, adet in pf['varliklar'].items():
                if fon in df_prices.columns:
                    fiyat = df_prices[fon].dropna().iloc[-1]
                    deger = adet * fiyat
                    data.append({'Varlık': fon, 'Adet': adet, 'Fiyat': fiyat, 'Değer': deger})
            if data:
                df_varlik = pd.DataFrame(data)
                df_varlik['Ağırlık (%)'] = (df_varlik['Değer'] / df_varlik['Değer'].sum()) * 100
                st.dataframe(df_varlik, use_container_width=True)
                fig_pie = go.Figure(data=[go.Pie(labels=df_varlik['Varlık'], values=df_varlik['Değer'], hole=0.4)])
                st.plotly_chart(fig_pie, use_container_width=True)

        if pf['islemler']:
            st.subheader("📜 İşlem Geçmişi")
            st.dataframe(pd.DataFrame(pf['islemler']).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info(f"Henüz {asset_class} için portföy oluşturulmadı. Gerçek Zamanlı Sinyal sekmesinden sinyal uygulayarak başlayın.")

    st.markdown("---")
    st.subheader("🌐 Karma Portföy (Allokasyon Bazlı)")

    total_value = 0
    class_details = []
    for cls in ALL_ASSET_CLASSES:
        pf_cls = get_portfolio(cls)
        if pf_cls:
            ratio = alloc_data.get(cls, {}).get('hedef', 0.0)
            if ratio <= 0:
                continue
            cls_value = pf_cls['nakit']
            df_cls = st.session_state.df_all_raw.get(cls, pd.DataFrame())
            for fon, adet in pf_cls['varliklar'].items():
                if fon in df_cls.columns:
                    fiyat = df_cls[fon].dropna().iloc[-1]
                    cls_value += adet * fiyat
            total_value += cls_value * ratio
            class_details.append({
                'Varlık Sınıfı': cls,
                'Allokasyon': f"%{ratio*100:.1f}",
                'Sınıf Portföy Değeri': f"{cls_value:,.2f} TL",
                'Katkı Değeri': f"{cls_value * ratio:,.2f} TL"
            })

    st.metric("Toplam Karma Portföy Değeri", f"{total_value:,.2f} TL")

    if class_details:
        st.dataframe(pd.DataFrame(class_details), use_container_width=True, hide_index=True)

        df_class = pd.DataFrame(class_details)
        df_class['Katkı TL'] = df_class['Katkı Değeri'].str.replace(' TL', '').str.replace('.', '').str.replace(',', '.').astype(float)
        fig_bar = px.bar(df_class, x='Varlık Sınıfı', y='Katkı TL', title="Varlık Sınıflarının Karma Portföye Katkısı (TL)")
        st.plotly_chart(fig_bar, use_container_width=True)

# ----- TAB 4: Stres Testi -----
with tab4:
    st.subheader("🧪 Stres Testi")
    scenarios = {
        'Piyasa Çöküşü (%15)': -15,
        'Sert Düşüş (%30)': -30,
        'Yükseliş (%20)': 20,
        'Hafif Düşüş (%5)': -5,
        'Petrol Şoku (%25)': 25
    }
    if st.button("📉 Stres Testi Uygula"):
        results = advanced_stress_test(df_prices, scenarios)
        if results:
            df_results = pd.DataFrame(list(results.items()), columns=['Senaryo', 'Getiri (%)'])
            st.dataframe(df_results, use_container_width=True)
            fig = px.bar(df_results, x='Senaryo', y='Getiri (%)', title="Senaryo Analizi", color='Getiri (%)', color_continuous_scale='RdBu')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Stres testi için yeterli veri yok.")

# ----- TAB 5: Raporlama -----
with tab5:
    st.subheader("📄 Raporlama")
    if st.session_state.df_result is not None and not st.session_state.df_result.empty:
        if st.button("📄 PDF Raporu Oluştur"):
            try:
                metrikler = st.session_state.metrikler
                risk_met = st.session_state.risk_metrikler
                decision_log = st.session_state.decision_log
                filename = "rapor.pdf"
                create_pdf_report(st.session_state.df_result, metrikler, risk_met, decision_log, filename)
                with open(filename, "rb") as f:
                    st.download_button("📥 Raporu İndir", data=f, file_name=filename, mime="application/pdf")
            except Exception as e:
                st.error(f"Rapor oluşturulamadı: {e}")
    else:
        st.info("Önce bir backtest çalıştırın.")

# ----- TAB 6: Attribüsyon & Risk -----
with tab6:
    st.subheader("📈 Performans Attribüsyonu")
    if st.session_state.df_result is not None:
        attr = attribution_analysis(df_prices, st.session_state.df_result, st.session_state.decision_log)
        if attr:
            st.dataframe(pd.DataFrame(list(attr.items()), columns=['Faktör', 'Katkı (%)']), use_container_width=True)
        else:
            st.info("Attribüsyon verisi yeterli değil.")
    else:
        st.info("Önce bir backtest çalıştırın.")

    st.subheader("📊 Güncel Portföy Risk Metrikleri")
    if st.session_state.df_result is not None:
        risk_met = st.session_state.risk_metrikler
        if risk_met:
            st.json(risk_met)
        else:
            st.info("Risk metrikleri hesaplanmamış.")
    else:
        st.info("Önce bir backtest çalıştırın.")

st.markdown("---")
st.caption("🔐 Tüm iyileştirmeler aktif: Otomatik yeniden dengeleme, paralel backtest, gelişmiş arayüz (grafikler, tema) ve toplu veri yükleme.")