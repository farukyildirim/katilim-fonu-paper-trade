# fon_data_indir.py
"""
TEFAS'tan ZPG, GLS, IAT sukuk fonlarını indirir ve
Streamlit uygulamasına uygun formatta birleştirir.

Not: TEFAS maksimum 5 yıllık geçmiş veri sunar.
"""
import pandas as pd
from pytefas import Crawler

# Fon kodları
FONLAR = ["ZPG", "GLS", "IAT"]

# Tarih aralığı (TEFAS max 5 yıl geriye gider)
BASLANGIC = "2021-09-22"
BITIS = "2026-09-22"

tefas = Crawler()

# Her fon için veri çek
tum_veriler = {}

for fon in FONLAR:
    print(f"\n📥 {fon} indiriliyor...")
    try:
        df = tefas.fetch(
            start=BASLANGIC,
            end=BITIS,
            fund_code=fon,          # ✅ 'name' değil, 'fund_code'
            columns="info",         # fiyat, pay, büyüklük bilgisi
            kind="YAT"              # Yatırım fonu
        )

        if df is None or df.empty:
            print(f"❌ {fon} için veri boş.")
            continue

        # Tarih sütununu index yap
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # Sadece 'price' sütununu al (fiyat)
        if "price" in df.columns:
            seri = df["price"].astype(float)
        else:
            print(f"⚠️ {fon}: 'price' sütunu bulunamadı.")
            print(f"   Mevcut sütunlar: {df.columns.tolist()}")
            continue

        tum_veriler[fon] = seri
        print(f"   ✅ {fon}: {len(seri)} günlük fiyat alındı.")
        print(f"   İlk: {seri.index.min().date()} → {seri.iloc[0]:.6f}")
        print(f"   Son: {seri.index.max().date()} → {seri.iloc[-1]:.6f}")

    except Exception as e:
        print(f"❌ {fon} hatası: {e}")

# Birleştir
if not tum_veriler:
    print("\n❌ Hiçbir fon verisi alınamadı.")
    exit(1)

birlesik = pd.DataFrame(tum_veriler)
birlesik.index.name = "tarih"
birlesik = birlesik.sort_index().ffill()

# Kaydet
birlesik.to_csv("sukuk_birlesik.csv")
print(f"\n✅ Birleşik CSV kaydedildi: sukuk_birlesik.csv")
print(f"   Şekil: {birlesik.shape}")
print(f"   Tarih aralığı: {birlesik.index.min().date()} - {birlesik.index.max().date()}")
print(f"\nİlk 5 satır:")
print(birlesik.head())
print(f"\nSon 5 satır:")
print(birlesik.tail())