from pytefas import Crawler
tefas = Crawler()
df = tefas.fetch("2025-09-24", "2026-09-24", kind="YAT", fund_code="KPC")
print(df[["date", "price"]].tail())