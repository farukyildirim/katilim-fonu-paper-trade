import borsapy as bp
t = bp.Ticker("TUPRS")

# Tüm metod ve özellikleri listele
print("=== TÜM METODLAR ===")
for m in dir(t):
    if not m.startswith("_"):
        try:
            val = getattr(t, m)
            tip = type(val).__name__
            if callable(val):
                print(f"  {m}()  →  metot")
            else:
                print(f"  {m}  →  {tip}")
        except Exception as e:
            print(f"  {m}  →  HATA: {e}")

# Temettü ile ilgili olabilecekleri dene
print("\n=== TEMETTÜ DENEMELERİ ===")
for attr in ["dividends", "temettu", "dividend_history", "temettu_gecmisi",
             "dividend", "cash_dividend", "kar_payi", "karPayi"]:
    if hasattr(t, attr):
        try:
            v = getattr(t, attr)
            if callable(v):
                v = v()
            print(f"✅ {attr} → {type(v).__name__}")
            print(f"   Örnek: {str(v)[:200]}")
        except Exception as e:
            print(f"❌ {attr} → {e}")
    else:
        print(f"— {attr} → yok")