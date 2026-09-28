import email
from email.header import decode_header
import imaplib
import re
import sqlite3
import time
import numpy as np
import pandas as pd
import requests

# ==========================================
# 1. KONFİGÜRASYON VE AYARLAR
# ==========================================
IMAP_SERVER = "imap.gmail.com"
EMAIL_ACCOUNT = "vadilandb2@gmail.com"
APP_PASSWORD = "ldpt unpx mpjh qucm"  # Gmail / Outlook Uygulama Şifresi

TELEGRAM_BOT_TOKEN = "123456789:ABCdefGHIjklMNOpqrsTUVwxyZ"
TELEGRAM_CHAT_ID = "987654321"

DB_FILE = "opportunity_tracker.db"


# ==========================================
# 2. VERİTABANI YÖNETİMİ
# ==========================================
class OpportunityDB:

    def __init__(self, db_file=DB_FILE):
        self.db_file = db_file
        self.create_tables()

    def get_connection(self):
        # Multithreading veya döngülerde güvenli SQLite bağlantısı
        return sqlite3.connect(self.db_file)

    def create_tables(self):
        query = """
        CREATE TABLE IF NOT EXISTS listings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            listing_id TEXT UNIQUE,
            title TEXT,
            category TEXT,
            price REAL,
            link TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        with self.get_connection() as conn:
            conn.execute(query)
            conn.commit()

    def add_listing(self, listing_id, title, category, price, link):
        try:
            query = "INSERT INTO listings (listing_id, title, category, price, link) VALUES (?, ?, ?, ?, ?)"
            with self.get_connection() as conn:
                conn.execute(
                    query, (listing_id, title, category, price, link)
                )
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Mükerrer ilan (Zaten kaydedilmiş)

    def get_category_data(self, category):
        query = "SELECT price FROM listings WHERE category = ?"
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(category,))
        return df["price"].tolist()


# ==========================================
# 3. BÜTÇESİZ FIRSAT SKORLAMA MOTORU
# ==========================================
class OpportunityEngine:

    @staticmethod
    def evaluate(target_price, category_prices):
        """Girdi ve bütçeden bağımsız, kategorinin geçmiş verisine dayalı

        Z-Score ve iskonto hesaplaması.
        """
        if len(category_prices) < 3:
            return 50.0, "Yetersiz Kategori Verisi (Toplanıyor)", 0.0

        avg_price = np.mean(category_prices)
        std_price = (
            np.std(category_prices) if np.std(category_prices) > 0 else 1.0
        )

        z_score = (target_price - avg_price) / std_price
        discount_pct = ((avg_price - target_price) / avg_price) * 100

        score = 50.0
        score += discount_pct * 2.5  # Her %1 iskonto için +2.5 puan

        if z_score <= -1.5:
            score += 15.0  # Fiyat ortalamanın belirgin şekilde altında

        score = min(max(score, 0.0), 100.0)

        if score >= 75:
            recommendation = "🔥 YÜKSEK FIRSAT (Anlık Aksiyon Alınmalı)"
        elif 60 <= score < 75:
            recommendation = "👀 MUKAYESELİ UCUZ (Takip Edilebilir)"
        else:
            recommendation = "➡️ STANDART PİYASA FİYATI"

        return round(score, 1), recommendation, round(discount_pct, 1)


# ==========================================
# 4. TELEGRAM BİLDİRİM SERVİSİ
# ==========================================
def send_telegram_alert(
    title, price, category, score, recommendation, discount, link
):
    message = (
        f"{recommendation}\n\n"
        f"📌 **İlan:** {title}\n"
        f"🏷 **Kategori:** {category}\n"
        f"💰 **Fiyat:** {price:,.0f} TL\n"
        f"📊 **Kategori İskontosu:** %{discount}\n"
        f"⭐️ **Fırsat Skoru:** {score} / 100\n\n"
        f"🔗 [İlana Git]({link})"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": False,
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"[-] Telegram bildirimi gönderilemedi: {e}")


def decode_mime_header(header_value):
    """E-posta başlıklarını güvenli şekilde decode eder."""
    if not header_value:
        return ""
    decoded_fragments = decode_header(header_value)
    text = ""
    for fragment, encoding in decoded_fragments:
        if isinstance(fragment, bytes):
            text += fragment.decode(encoding if encoding else "utf-8", errors="ignore")
        else:
            text += fragment
    return text


# ==========================================
# 5. E-POSTA IMAP DİNLEYİCİ
# ==========================================
def process_emails(db):
    mail = None
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ACCOUNT, APP_PASSWORD)
        mail.select("inbox")

        # Okunmamış Sahibinden e-postalarını ara
        status, messages = mail.search(None, '(UNSEEN FROM "sahibinden.com")')
        email_ids = messages[0].split()

        if not email_ids:
            print("[✓] Yeni ilan e-postası yok.")
            return

        print(f"[+] {len(email_ids)} adet yeni bildirim işleniyor...")

        for e_id in email_ids:
            res, msg_data = mail.fetch(e_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])

                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/html":
                                body = part.get_payload(decode=True).decode(
                                    "utf-8", errors="ignore"
                                )
                    else:
                        body = msg.get_payload(decode=True).decode(
                            "utf-8", errors="ignore"
                        )

                    # Link ve Fiyat Ayıklama (Regex)
                    link_match = re.search(
                        r"https://[^\s\"]*sahibinden\.com/ilan/[^\s\"]*", body
                    )
                    price_match = re.search(r"([\d\.]+)\s*TL", body)

                    if link_match and price_match:
                        link = link_match.group(0)
                        # Binlik ayırıcı noktaları kaldırıp float'a çevirme
                        price = float(price_match.group(1).replace(".", ""))

                        # URL'den İlan ID Çıkarma
                        id_match = re.search(r"-(\d+)/detay", link)
                        listing_id = (
                            id_match.group(1) if id_match else str(hash(link))
                        )

                        subject = decode_mime_header(msg["Subject"])

                        # Kategori tespiti
                        category = "Genel_Arama"
                        if "passat" in subject.lower():
                            category = "VW_Passat_1.6TDI"

                        # DÜZELTME: Önce mevcut verileri çek (Yeni ilan ortalamayı bozmasın)
                        category_prices = db.get_category_data(category)

                        # Veritabanına Ekle
                        is_new = db.add_listing(
                            listing_id, subject, category, price, link
                        )

                        if is_new:
                            # Fırsat Motoruyla Analiz Et
                            score, rec, discount = OpportunityEngine.evaluate(
                                price, category_prices
                            )

                            print(
                                f"[+] İşlendi: {subject} | Fiyat: {price:,.0f} TL | Skor: {score}"
                            )

                            # Fırsat Skoru 75 ve Üzeriyse Telegram Bildirimi Gönder
                            if score >= 75:
                                send_telegram_alert(
                                    subject,
                                    price,
                                    category,
                                    score,
                                    rec,
                                    discount,
                                    link,
                                )

    except Exception as e:
        print(f"[-] IMAP/İşleme Hatası: {e}")
    finally:
        if mail:
            try:
                mail.logout()
            except Exception:
                pass


# ==========================================
# 6. ANA ÇALIŞTIRMA DÖNGÜSÜ
# ==========================================
if __name__ == "__main__":
    db = OpportunityDB()
    print("[🚀] Fırsat Tespit Motoru Başlatıldı. E-postalar dinleniyor...")

    while True:
        try:
            process_emails(db)
        except Exception as e:
            print(f"[-] Beklenmeyen Ana Döngü Hatası: {e}")

        # Her 3 dakikada bir tarama yap
        time.sleep(180)