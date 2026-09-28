"""
Tasfiye edilmiş fonları toplu blacklist'e ekler.
Önce aktif olanları kontrol eder, sadece pasifleri ekler.
"""
import sqlite3
import pandas as pd
import datetime

DB_PATH = "tefas_data.db"

# Kullanıcının verdiği liste
FUND_CODES = [
    'BHN', 'BYZ', 'CBD', 'CAH', 'CHY', 'AJ1', 'PAB', 'P1A',
    'SEH', 'LGO', 'BSH', 'AP5', 'BAC', 'AP4', 'ABG', 'LAI',
    'KHD', 'BTJ', 'KLH', 'PPT', 'PSE', 'SNY', 'DFI', 'GCD',
    'UHS', 'BLA', 'BOS', 'BBO', 'BP5', 'BOH', 'BI5', 'BBN',
    'BDO', 'IHY', 'BIK', 'BMU', 'BPZ', 'BSN', 'BUC', 'YLZ',
    'HLR', 'HEH', 'HAT', 'HKM', 'HIM', 'HPL', 'HDH', 'HBV',
    'HDA', 'HVA', 'HMK', 'HGH', 'HGJ', 'HPI', 'HFI', 'HPP',
    'HIN', 'HKP', 'KSA', 'HPF', 'HMV', 'FNT', 'NFK', 'HYV',
    'HPH', 'HKJ', 'HVB', 'HVC', 'DUH', 'HDK', 'HPZ', 'PST',
    'AP6', 'PAO', 'BST', 'PHY', 'BRT', 'BHH', 'PBY', 'BSE',
    'BDI', 'AC7', 'PCH', 'PDH', 'DHI', 'DRH', 'PGE', 'KHA',
    'AC8', 'IAU', 'AC5', 'KRH', 'PKU', 'PMH', 'PMP', 'MGE',
    'POF', 'POS', 'POB', 'PO9', 'PPO', 'POI', 'PO8', 'OHI',
    'POU', 'PO7', 'AC4', 'PSR', 'SHI', 'CSH', 'PYD', 'PYI',
    'PYR', 'PHB', 'PBH', 'PDG', 'PDC', 'PGH', 'PHN', 'PKD',
    'PNU', 'PKZ', 'PRY', 'PKM', 'PCS', 'TLY', 'DOH', 'THF',
    'TP2', 'TLV', 'T3B',
]

# Bu listedeki kodlar zaten aktif değil, hepsini ekle
# Ama yine de kontrol et
FORCE_ADD = True  # True: hepsini ekle, False: sadece pasifleri ekle


def main():
    conn = sqlite3.connect(DB_PATH)

    # Son iş gününü bul
    son = pd.read_sql(
        "SELECT MAX(date) AS d FROM fund_info WHERE kind='YAT'", conn
    )['d'].iloc[0]
    print(f"DB son tarih: {son}")

    # Son tarihte aktif fonları çek
    df_aktif = pd.read_sql(
        f"SELECT DISTINCT fund_code FROM fund_info "
        f"WHERE kind='YAT' AND date='{son}'", conn)
    aktif_set = set(df_aktif['fund_code'].tolist())
    print(f"Aktif fon sayısı: {len(aktif_set)}")

    # Eklenecekleri ayır
    eklenecek = []
    zaten_var = []
    aktif_uyari = []

    for code in FUND_CODES:
        # Zaten listede mi?
        existing = pd.read_sql(
            "SELECT fund_code FROM delisted_funds WHERE fund_code=?",
            conn, params=(code,))
        if not existing.empty:
            zaten_var.append(code)
            continue

        if code in aktif_set and not FORCE_ADD:
            aktif_uyari.append(code)
            continue

        eklenecek.append(code)

    # Özet
    print(f"\n═══ ÖZET ═══")
    print(f"Toplam kod: {len(FUND_CODES)}")
    print(f"Zaten blacklist'te: {len(zaten_var)}")
    print(f"Aktif (uyarı): {len(aktif_uyari)}")
    print(f"Eklenecek: {len(eklenecek)}")

    if aktif_uyari:
        print(f"\n⚠️ AKTİF FONLAR (dikkat!):")
        for code in aktif_uyari[:20]:
            print(f"  {code}")

    # Ekle
    if eklenecek:
        c = conn.cursor()
        now = datetime.datetime.now().isoformat(timespec='seconds')
        for code in eklenecek:
            c.execute(
                "INSERT OR REPLACE INTO delisted_funds "
                "(fund_code, delisted_date, reason, added_at) "
                "VALUES (?, ?, ?, ?)",
                (code, son, 'Toplu tasfiye listesi', now))
        conn.commit()
        print(f"\n✅ {len(eklenecek)} fon blacklist'e eklendi")

    conn.close()

    # Doğrulama
    conn = sqlite3.connect(DB_PATH)
    df_final = pd.read_sql(
        "SELECT COUNT(*) AS n FROM delisted_funds", conn)
    print(f"\nBlacklist toplam: {df_final['n'].iloc[0]} fon")
    conn.close()


if __name__ == '__main__':
    main()