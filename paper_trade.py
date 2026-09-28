"""
Paper Trading Takip Scripti v2
===============================
Adetleri bir kez hesaplar, kalıcı saklar.
Sonraki çalıştırmalarda SADECE fiyat değişimini takip eder.
"""

import sqlite3
import pandas as pd
from datetime import date
import os
import sys

# >>> YENİ: Windows UTF-8 encoding düzeltmesi
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# =============================================
# KONFİGÜRASYON
# =============================================
DB_PATH = "tefas_data.db"
LOG_FILE = "paper_trade_log.csv"
UNITS_FILE = "paper_trade_units.csv"
TOTAL_CAPITAL = 100000

# Başlangıç tarihi (adetlerin hesaplanacağı tarih)
INIT_DATE = "2026-09-17"

# Portföy hedefleri (TL)
TARGETS = {
    'TI3': 40000,
    'CKL': 4330.48,
    'DLN': 4189.96,
    'ICH': 4098.45,
    'TVH': 2964.34,
    'VEL': 2824.10,
    'THE': 2796.20,
    'AZD': 2794.23,
    'FOA': 2621.14,
    'PPF': 23391,        # 15.000 + 8.391 (THF/DOH zararı)
    'TUA': 5000,
    'IPV': 5000,
}


def get_fund_prices(target_date=None):
    """DB'den fon fiyatlarını çeker."""
    if not os.path.exists(DB_PATH):
        print(f"❌ DB bulunamadı: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    result = {}

    for fund in TARGETS:
        for kind in ['YAT', 'EMK', 'BYF']:
            try:
                if target_date:
                    query = """
                        SELECT price, date FROM fund_info
                        WHERE kind=? AND fund_code=? AND date<=?
                        ORDER BY date DESC LIMIT 1
                    """
                    df = pd.read_sql(query, conn,
                                     params=(kind, fund, target_date))
                else:
                    query = """
                        SELECT price, date FROM fund_info
                        WHERE kind=? AND fund_code=?
                        ORDER BY date DESC LIMIT 1
                    """
                    df = pd.read_sql(query, conn, params=(kind, fund))

                if not df.empty and pd.notna(df['price'].iloc[0]) \
                        and df['price'].iloc[0] > 0:
                    result[fund] = {
                        'price': float(df['price'].iloc[0]),
                        'date': str(df['date'].iloc[0]),
                        'kind': kind,
                    }
                    break
            except Exception as e:
                print(f"⚠️  {fund}/{kind}: {e}")
                continue

    conn.close()
    return result


def load_or_create_units():
    """
    Adetleri yükler. Dosya yoksa ilk kez hesaplar.

    Returns:
        (units_dict, created_now: bool)
    """
    if os.path.exists(UNITS_FILE):
        # Var olan adetleri oku
        df = pd.read_csv(UNITS_FILE)
        units = dict(zip(df['Fon'], df['Adet']))
        return units, False
    else:
        # İlk kez hesapla
        print(f"📌 İlk çalıştırma — adetler hesaplanıyor ({INIT_DATE})")
        prices = get_fund_prices(target_date=INIT_DATE)
        units = {}
        for fund, target_tl in TARGETS.items():
            if fund in prices and prices[fund]['price'] > 0:
                units[fund] = target_tl / prices[fund]['price']
            else:
                units[fund] = 0
                print(f"⚠️  {fund}: fiyat bulunamadı")

        # Kaydet
        df_units = pd.DataFrame([
            {'Fon': f, 'Adet': u, 'Başlangıç Fiyat': prices.get(f, {}).get('price', 0),
             'Başlangıç Tarih': prices.get(f, {}).get('date', '')}
            for f, u in units.items()
        ])
        df_units.to_csv(UNITS_FILE, index=False)
        print(f"✅ Adetler kaydedildi: {UNITS_FILE}")
        return units, True


def current_value(units):
    """Portföyün güncel değerini hesaplar (bugünün fiyatlarıyla)."""
    prices = get_fund_prices()
    total = 0
    details = []

    for fund, unit in units.items():
        if fund in prices and unit > 0:
            value = unit * prices[fund]['price']
            total += value
            details.append({
                'Fon': fund,
                'Kind': prices[fund]['kind'],
                'Adet': round(unit, 4),
                'Fiyat': round(prices[fund]['price'], 6),
                'Değer': round(value, 2),
                'Tarih': prices[fund]['date'],
            })

    return total, details


def save_snapshot(total, details, units, note=""):
    """Anlık görüntüyü CSV'ye kaydeder."""
    today = date.today().isoformat()

    row = {
        'Tarih': today,
        'Toplam Değer': round(total, 2),
        'Getiri %': round(((total / TOTAL_CAPITAL) - 1) * 100, 2),
        'Getiri TL': round(total - TOTAL_CAPITAL, 2),
        'Not': note,
    }

    for fund, unit in units.items():
        row[f'{fund}_adet'] = round(unit, 4)

    for d in details:
        row[f'{d["Fon"]}_fiyat'] = d['Fiyat']
        row[f'{d["Fon"]}_deger'] = d['Değer']

    df_new = pd.DataFrame([row])

    if os.path.exists(LOG_FILE):
        try:
            df_old = pd.read_csv(LOG_FILE)
            # Aynı gün varsa güncelle
            df_old = df_old[df_old['Tarih'] != today]
            df = pd.concat([df_old, df_new], ignore_index=True)
        except Exception as e:
            print(f"⚠️  Eski CSV okunamadı: {e}")
            df = df_new
    else:
        df = df_new

    df.to_csv(LOG_FILE, index=False)
    return row


def print_report(units):
    """Detaylı rapor yazdırır."""
    print("═" * 72)
    print(f"  HİBRİT PORTFÖY — PAPER TRADING")
    print(f"  Başlangıç: {INIT_DATE}  |  Sermaye: {TOTAL_CAPITAL:,.2f} TL")
    print(f"  Bugün: {date.today().isoformat()}")
    print("═" * 72)

    total, details = current_value(units)

    print(f"\n{'Fon':6} {'Kind':6} {'Adet':>14} {'Fiyat':>12} {'Değer (TL)':>15}")
    print("─" * 72)

    for d in sorted(details, key=lambda x: -x['Değer']):
        print(f"{d['Fon']:6} {d['Kind']:6} {d['Adet']:>14,.2f} "
              f"{d['Fiyat']:>12.6f} {d['Değer']:>15,.2f}")

    print("─" * 72)
    print(f"{'TOPLAM':<40} {total:>30,.2f}")
    print("─" * 72)

    getiri_pct = ((total / TOTAL_CAPITAL) - 1) * 100
    getiri_tl = total - TOTAL_CAPITAL

    if getiri_tl > 0:
        icon = "📈"
    elif getiri_tl < 0:
        icon = "📉"
    else:
        icon = "➡️"

    print(f"\n  {icon}  Getiri: {getiri_pct:+.2f}%  ({getiri_tl:+,.2f} TL)")
    print("═" * 72)

    return total


if __name__ == '__main__':
    print("📊 Paper Trading v2\n")
    print(f"   DB: {os.path.abspath(DB_PATH)}")
    print(f"   Log: {os.path.abspath(LOG_FILE)}")
    print(f"   Units: {os.path.abspath(UNITS_FILE)}\n")

    # 1. Adetleri yükle veya oluştur
    units, created_now = load_or_create_units()

    print(f"\n💡 Portföy Adetleri ({len(units)} fon):")
    for fund, unit in units.items():
        print(f"   {fund:5} {unit:>14,.2f} adet")

    # 2. Güncel değeri hesapla
    total = print_report(units)

    # 3. CSV'ye kaydet
    _, details = current_value(units)
    save_row = save_snapshot(total, details, units)

    print(f"\n✅ Kaydedildi: {LOG_FILE}")
    print(f"   Tarih: {save_row['Tarih']}")
    print(f"   Değer: {save_row['Toplam Değer']:,.2f} TL")
    print(f"   Getiri: {save_row['Getiri %']:+.2f}%")

    # 4. Geçmiş
    if os.path.exists(LOG_FILE):
        try:
            df_hist = pd.read_csv(LOG_FILE)
            if len(df_hist) > 1:
                print(f"\n📈 Geçmiş ({len(df_hist)} kayıt):")
                for _, r in df_hist.iterrows():
                    print(f"   {r['Tarih']}: {r['Toplam Değer']:>12,.2f} TL  "
                          f"({r['Getiri %']:+.2f}%)")
        except Exception as e:
            print(f"⚠️  Geçmiş: {e}")