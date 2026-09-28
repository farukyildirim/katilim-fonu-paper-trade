import email
from email.header import decode_header
import imaplib
import re
import sqlite3
import time
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import threading
import json
import os
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================================
# KONFİGÜRASYON (JSON)
# ==========================================
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "IMAP_SERVER": "imap.gmail.com",
    "EMAIL_ACCOUNT": "vadilandb2@gmail.com",
    "APP_PASSWORD": "ldpt unpx mpjh qucm",
    "TELEGRAM_BOT_TOKEN": "123456789:ABCdefGHIjklMNOpqrsTUVwxyZ",
    "TELEGRAM_CHAT_ID": "987654321",
    "SCORE_THRESHOLD": 75,
    "SCAN_INTERVAL": 180,  # saniye
    "DB_FILE": "opportunity_tracker.db",
    "SCRAPING_INTERVAL": 3600,  # 1 saat
    "CATEGORY_URLS": [
        "https://www.sahibinden.com/kategori/otomobil",
        "https://www.sahibinden.com/otomobil/dizel,dizel-hafif-hibrit/otomatik/ikinci-el?a5_max=2023&a4_max=150000&sorting=price_asc&a5_min=2017&price_max=1600000"
    ]
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    else:
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

config = load_config()

# ==========================================
# VERİTABANI (Gelişmiş)
# ==========================================
class OpportunityDB:
    def __init__(self, db_file=None):
        self.db_file = db_file or config["DB_FILE"]
        self.create_tables()

    def get_connection(self):
        return sqlite3.connect(self.db_file, check_same_thread=False)

    def create_tables(self):
        with self.get_connection() as conn:
            # Ana tablo (mevcut alanlar + yeniler)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS listings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    listing_id TEXT UNIQUE,
                    title TEXT,
                    category TEXT,
                    subcategory TEXT,
                    price REAL,
                    link TEXT,
                    year INTEGER,
                    km INTEGER,
                    city TEXT,
                    listing_date TEXT,
                    last_seen_price REAL,
                    price_change REAL,
                    status TEXT,
                    days_on_site INTEGER,
                    score REAL,
                    recommendation TEXT,
                    discount REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Yeni sütunlar (eski veritabanı ile uyumluluk için)
            for col, col_type in [
                ("subcategory", "TEXT"),
                ("year", "INTEGER"),
                ("km", "INTEGER"),
                ("city", "TEXT"),
                ("listing_date", "TEXT"),
                ("last_seen_price", "REAL"),
                ("price_change", "REAL"),
                ("status", "TEXT"),
                ("days_on_site", "INTEGER"),
                ("updated_at", "TIMESTAMP")
            ]:
                try:
                    conn.execute(f"ALTER TABLE listings ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass

            # Kategori istatistikleri tablosu
            conn.execute("""
                CREATE TABLE IF NOT EXISTS category_stats (
                    category TEXT PRIMARY KEY,
                    avg_price REAL,
                    avg_km REAL,
                    avg_year REAL,
                    avg_days_to_sell REAL,
                    total_listings INTEGER,
                    active_listings INTEGER,
                    new_per_day REAL,
                    demand_score REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def add_listing(self, listing_data):
        # listing_data dict içinde gerekli alanlar olmalı
        try:
            query = """
                INSERT INTO listings (
                    listing_id, title, category, subcategory, price, link,
                    year, km, city, listing_date, status, last_seen_price,
                    price_change, days_on_site, score, recommendation, discount
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            with self.get_connection() as conn:
                conn.execute(query, (
                    listing_data['listing_id'],
                    listing_data['title'],
                    listing_data['category'],
                    listing_data.get('subcategory', ''),
                    listing_data['price'],
                    listing_data['link'],
                    listing_data.get('year', 0),
                    listing_data.get('km', 0),
                    listing_data.get('city', ''),
                    listing_data.get('listing_date', ''),
                    listing_data.get('status', 'active'),
                    listing_data['price'],
                    0.0,
                    0,
                    None, None, None
                ))
                conn.commit()
            # Skoru hesaplayıp güncelle
            self.update_score(listing_data['listing_id'])
            return True
        except sqlite3.IntegrityError:
            # Zaten varsa güncelle
            self.update_listing(listing_data)
            return False

    def update_listing(self, listing_data):
        query = """
            UPDATE listings SET
                price = ?, last_seen_price = ?, price_change = ?,
                status = ?, days_on_site = ?, updated_at = CURRENT_TIMESTAMP
            WHERE listing_id = ?
        """
        with self.get_connection() as conn:
            conn.execute(query, (
                listing_data['price'],
                listing_data['last_seen_price'],
                listing_data['price_change'],
                listing_data['status'],
                listing_data['days_on_site'],
                listing_data['listing_id']
            ))
            conn.commit()
        self.update_score(listing_data['listing_id'])

    def update_score(self, listing_id):
        # Skoru yeniden hesapla ve güncelle
        listing = self.get_listing(listing_id)
        if not listing:
            return
        stats = self.get_category_stats(listing['category'])
        score, rec, discount = self.calculate_score(listing, stats)
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE listings SET score = ?, recommendation = ?, discount = ? WHERE listing_id = ?",
                (score, rec, discount, listing_id)
            )
            conn.commit()

    def get_listing(self, listing_id):
        query = "SELECT * FROM listings WHERE listing_id = ?"
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(listing_id,))
        return df.to_dict('records')[0] if len(df) > 0 else None

    def get_category_stats(self, category):
        query = "SELECT * FROM category_stats WHERE category = ?"
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(category,))
        return df.to_dict('records')[0] if len(df) > 0 else None

    def update_category_stats(self, category):
        # Kategoriye ait tüm ilanlardan istatistik hesapla
        query = """
            SELECT 
                AVG(price) as avg_price,
                AVG(km) as avg_km,
                AVG(year) as avg_year,
                AVG(days_on_site) as avg_days,
                COUNT(*) as total,
                SUM(CASE WHEN status='active' THEN 1 ELSE 0 END) as active,
                COUNT(*) / (JULIANDAY('now') - JULIANDAY(MIN(created_at))) as new_per_day
            FROM listings WHERE category = ?
        """
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(category,))
        if df.empty:
            return
        row = df.iloc[0]
        # Talep skoru: ortalama satış süresine göre (ters orantılı)
        demand_score = 0
        if row['avg_days'] and row['avg_days'] > 0:
            demand_score = max(0, 100 - row['avg_days'] * 2)
        # Veritabanına kaydet
        upsert = """
            INSERT OR REPLACE INTO category_stats
            (category, avg_price, avg_km, avg_year, avg_days_to_sell,
             total_listings, active_listings, new_per_day, demand_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """
        with self.get_connection() as conn:
            conn.execute(upsert, (
                category,
                row['avg_price'] or 0,
                row['avg_km'] or 0,
                row['avg_year'] or 0,
                row['avg_days'] or 0,
                row['total'] or 0,
                row['active'] or 0,
                row['new_per_day'] or 0,
                demand_score
            ))
            conn.commit()

    def calculate_score(self, listing, stats):
        if not stats or stats.get('total_listings', 0) < 3:
            return 50.0, "Yetersiz veri", 0.0

        # 1. Fiyat avantajı (kategori ortalamasına göre)
        avg_price = stats['avg_price']
        price_ratio = listing['price'] / avg_price if avg_price > 0 else 1
        price_score = max(0, min(100, (1 - price_ratio) * 100))

        # 2. Talep skoru
        demand_score = stats.get('demand_score', 50)

        # 3. Fiyat düşüş trendi
        trend_score = 0
        if listing.get('price_change', 0) < 0:
            trend_score = min(20, abs(listing['price_change']) * 2)

        # 4. Rekabet yoğunluğu (çok ilan varsa alıcı avantajlı)
        comp_score = min(20, stats['total_listings'] * 0.02)

        # Toplam skor (ağırlıklı)
        total = demand_score * 0.4 + price_score * 0.3 + trend_score + comp_score * 0.1
        score = min(100, max(0, total))

        # Öneri metni
        if score >= 75:
            rec = "🔥 YÜKSEK FIRSAT"
        elif score >= 60:
            rec = "👀 MUKAYESELİ UCUZ"
        else:
            rec = "➡️ STANDART PİYASA"

        discount = ((avg_price - listing['price']) / avg_price * 100) if avg_price > 0 else 0
        return round(score, 1), rec, round(discount, 1)

    def get_all_listings(self, category=None, min_score=None, sort_by='score'):
        query = "SELECT * FROM listings WHERE 1=1"
        params = []
        if category:
            query += " AND category = ?"
            params.append(category)
        if min_score is not None:
            query += " AND (score >= ? OR score IS NULL)"
            params.append(min_score)
        query += f" ORDER BY {sort_by} DESC"
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df.to_dict('records')

    def get_total_count(self):
        with self.get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM listings")
            return cur.fetchone()[0]

    def get_high_score_count(self, threshold=None):
        if threshold is None:
            threshold = config.get("SCORE_THRESHOLD", 75)
        with self.get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM listings WHERE score >= ?", (threshold,))
            return cur.fetchone()[0]

    def get_category_counts(self):
        with self.get_connection() as conn:
            df = pd.read_sql_query("SELECT category, COUNT(*) as cnt FROM listings GROUP BY category", conn)
        return dict(zip(df['category'], df['cnt']))

    def get_all_categories(self):
        with self.get_connection() as conn:
            cur = conn.execute("SELECT DISTINCT category FROM listings")
            return [row[0] for row in cur.fetchall()]

    def get_category_stats_all(self):
        with self.get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM category_stats", conn)
        return df.to_dict('records')

# ==========================================
# SCRAPING MODÜLÜ (Sahibinden)
# ==========================================
class SahibindenScraper:
    HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    @staticmethod
    def scrape_category(url, db):
        """Verilen kategori URL'sindeki ilanları tarar ve veritabanına ekler."""
        try:
            response = requests.get(url, headers=SahibindenScraper.HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            items = soup.select('a[href*="/ilan/"]')
            if not items:
                print(f"Uyarı: {url} için hiç ilan bulunamadı.")
                return 0

            count = 0
            for item in items[:30]:  # Her seferinde 30 ilan ile sınırlı tutalım
                link = item.get('href')
                if not link.startswith('http'):
                    link = 'https://www.sahibinden.com' + link
                # İlan detaylarını çek
                details = SahibindenScraper.get_listing_details(link)
                if details:
                    # Kategori adını URL'den çıkar
                    category = url.split('/')[-1].replace('-', ' ').title()
                    details['category'] = category
                    details['link'] = link
                    details['listing_id'] = link.split('/')[-2]  # URL'den ID çıkarma
                    db.add_listing(details)
                    count += 1
                    time.sleep(1)  # Saygılı scraping
            return count
        except Exception as e:
            print(f"Scraping hatası ({url}): {e}")
            return 0

    @staticmethod
    def get_listing_details(link):
        """İlan detay sayfasından yıl, km, şehir vb. bilgileri alır."""
        try:
            response = requests.get(link, headers=SahibindenScraper.HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Fiyat
            price_elem = soup.find('span', class_='price')
            if not price_elem:
                return None
            price_text = price_elem.text.replace('.', '').replace(' TL', '').strip()
            price = float(price_text) if price_text else 0

            # Başlık
            title_elem = soup.find('h1', class_='item-title')
            title = title_elem.text.strip() if title_elem else ''

            # Detaylar (yıl, km, şehir) - Sahibinden'in yapısına göre değişir
            # Örnek: "İlan Tarihi", "Kilometre", "Model Yılı" gibi alanlar
            details = {}
            info_items = soup.find_all('li', class_='info-row')
            for item in info_items:
                label = item.find('span', class_='label')
                value = item.find('span', class_='value')
                if label and value:
                    key = label.text.strip().lower()
                    val = value.text.strip()
                    if 'yıl' in key:
                        details['year'] = int(val) if val.isdigit() else 0
                    elif 'km' in key or 'kilometre' in key:
                        details['km'] = int(val.replace('.', '')) if val else 0
                    elif 'şehir' in key or 'il' in key:
                        details['city'] = val
            # Varsayılanlar
            details.setdefault('year', 0)
            details.setdefault('km', 0)
            details.setdefault('city', '')
            details['price'] = price
            details['title'] = title
            details['listing_date'] = datetime.now().strftime('%Y-%m-%d')
            details['status'] = 'active'
            details['last_seen_price'] = price
            details['price_change'] = 0.0
            details['days_on_site'] = 0
            return details
        except Exception as e:
            print(f"Detay hatası ({link}): {e}")
            return None

# ==========================================
# E-POSTA TARAMA (Mevcut - IMAP)
# ==========================================
def process_emails(db, callback=None):
    mail = None
    new_count = 0
    try:
        mail = imaplib.IMAP4_SSL(config["IMAP_SERVER"])
        mail.login(config["EMAIL_ACCOUNT"], config["APP_PASSWORD"])
        mail.select("inbox")
        status, messages = mail.search(None, '(UNSEEN FROM "sahibinden.com")')
        email_ids = messages[0].split()
        if not email_ids:
            if callback:
                callback(0, "Yeni ilan e-postası yok.")
            return
        if callback:
            callback(0, f"{len(email_ids)} adet yeni bildirim işleniyor...")
        for e_id in email_ids:
            res, msg_data = mail.fetch(e_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/html":
                                body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                    else:
                        body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
                    link_match = re.search(r"https://[^\s\"]*sahibinden\.com/ilan/[^\s\"]*", body)
                    price_match = re.search(r"([\d\.]+)\s*TL", body)
                    if link_match and price_match:
                        link = link_match.group(0)
                        price = float(price_match.group(1).replace(".", ""))
                        id_match = re.search(r"-(\d+)/detay", link)
                        listing_id = id_match.group(1) if id_match else str(hash(link))
                        subject = decode_mime_header(msg["Subject"])
                        # Kategori tahmini
                        category = "Genel_Arama"
                        if "passat" in subject.lower():
                            category = "Volkswagen Passat"
                        # Detayları scrape et
                        details = SahibindenScraper.get_listing_details(link)
                        if details:
                            details['listing_id'] = listing_id
                            details['category'] = category
                            details['link'] = link
                            details['title'] = subject
                            is_new = db.add_listing(details)
                            if is_new:
                                new_count += 1
        if callback:
            callback(new_count, f"E-posta taraması tamamlandı. {new_count} yeni ilan eklendi.")
    except Exception as e:
        if callback:
            callback(0, f"Hata: {e}")
    finally:
        if mail:
            try:
                mail.logout()
            except:
                pass

# ==========================================
# TELEGRAM BİLDİRİM (Aynı)
# ==========================================
def send_telegram_alert(title, price, category, score, recommendation, discount, link):
    message = (
        f"{recommendation}\n\n"
        f"📌 **İlan:** {title}\n"
        f"🏷 **Kategori:** {category}\n"
        f"💰 **Fiyat:** {price:,.0f} TL\n"
        f"📊 **İskonto:** %{discount}\n"
        f"⭐️ **Skor:** {score} / 100\n\n"
        f"🔗 [İlana Git]({link})"
    )
    url = f"https://api.telegram.org/bot{config['TELEGRAM_BOT_TOKEN']}/sendMessage"
    payload = {
        "chat_id": config["TELEGRAM_CHAT_ID"],
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": False,
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"Telegram hatası: {e}")

def decode_mime_header(header_value):
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
# TKINTER ARAYÜZÜ (Gelişmiş)
# ==========================================
class OpportunityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fırsat Tespit Motoru - Sahibinden Analiz")
        self.root.geometry("1400x800")
        self.db = OpportunityDB()
        self.scanning = False
        self.scan_thread = None
        self.after_id = None

        # Menü
        menubar = tk.Menu(root)
        root.config(menu=menubar)
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ayarlar", menu=settings_menu)
        settings_menu.add_command(label="Konfigürasyon", command=self.open_settings)
        settings_menu.add_command(label="Manuel Tara (E-posta)", command=self.manual_scan)
        settings_menu.add_command(label="Scrape Kategoriler", command=self.scrape_categories)

        # Ana panel (Notebook - sekmeler)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Sekme 1: Dashboard
        self.dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.dashboard_frame, text="📊 Dashboard")

        # Sekme 2: İlan Listesi
        self.listings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.listings_frame, text="📋 İlanlar")

        # Sekme 3: Kategori İstatistikleri
        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="📈 Kategori Metrikleri")

        # --- Dashboard ---
        top_frame = ttk.Frame(self.dashboard_frame)
        top_frame.pack(fill=tk.X, pady=5)

        self.total_label = ttk.Label(top_frame, text="Toplam İlan: 0", font=("Arial", 12))
        self.total_label.pack(side=tk.LEFT, padx=10)

        self.high_label = ttk.Label(top_frame, text="Yüksek Skor (>75): 0", font=("Arial", 12))
        self.high_label.pack(side=tk.LEFT, padx=10)

        self.last_scan_label = ttk.Label(top_frame, text="Son Tarama: -", font=("Arial", 12))
        self.last_scan_label.pack(side=tk.LEFT, padx=10)

        self.status_label = ttk.Label(top_frame, text="Durum: Beklemede", font=("Arial", 12), foreground="blue")
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Grafik alanı
        self.figure, self.ax = plt.subplots(figsize=(8, 4), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.dashboard_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
        self.update_chart()

        # --- İlan Listesi ---
        filter_frame = ttk.Frame(self.listings_frame)
        filter_frame.pack(fill=tk.X, pady=5)

        ttk.Label(filter_frame, text="Kategori:").pack(side=tk.LEFT, padx=5)
        self.cat_var = tk.StringVar(value="Tümü")
        self.cat_combo = ttk.Combobox(filter_frame, textvariable=self.cat_var, width=25)
        self.cat_combo.pack(side=tk.LEFT, padx=5)
        self.cat_combo.bind("<<ComboboxSelected>>", self.apply_filter)

        ttk.Label(filter_frame, text="Min Skor:").pack(side=tk.LEFT, padx=5)
        self.score_var = tk.StringVar(value="0")
        self.score_entry = ttk.Entry(filter_frame, textvariable=self.score_var, width=8)
        self.score_entry.pack(side=tk.LEFT, padx=5)
        self.score_entry.bind("<KeyRelease>", self.apply_filter)

        self.filter_btn = ttk.Button(filter_frame, text="Filtrele", command=self.apply_filter)
        self.filter_btn.pack(side=tk.LEFT, padx=10)

        self.refresh_btn = ttk.Button(filter_frame, text="Yenile", command=self.refresh_list)
        self.refresh_btn.pack(side=tk.LEFT, padx=5)

        # Treeview
        tree_frame = ttk.Frame(self.listings_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        columns = ("ID", "Başlık", "Kategori", "Fiyat", "Yıl", "Km", "Skor", "Öneri", "Link")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20)
        for col in columns:
            self.tree.heading(col, text=col)
            if col == "Başlık":
                self.tree.column(col, width=250)
            elif col == "Link":
                self.tree.column(col, width=150)
            elif col == "Öneri":
                self.tree.column(col, width=150)
            else:
                self.tree.column(col, width=100)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Kategori İstatistikleri ---
        stats_tree_frame = ttk.Frame(self.stats_frame)
        stats_tree_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        stats_cols = ("Kategori", "Ort. Fiyat", "Ort. Km", "Ort. Yıl", "Ort. Satış Günü", "Aktif İlan", "Talep Skoru")
        self.stats_tree = ttk.Treeview(stats_tree_frame, columns=stats_cols, show="headings", height=15)
        for col in stats_cols:
            self.stats_tree.heading(col, text=col)
            self.stats_tree.column(col, width=120)
        self.stats_tree.pack(fill=tk.BOTH, expand=True)

        # Log
        log_frame = ttk.Frame(root)
        log_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=4, state='disabled')
        self.log_text.pack(fill=tk.X)

        # İlk yükleme
        self.refresh_list()
        self.update_stats()
        self.update_category_list()
        self.update_stats_tree()
        self.start_background_scanner()
        self.start_scraping_scheduler()

    # ====== Arayüz metodları ======
    def update_stats(self):
        total = self.db.get_total_count()
        high = self.db.get_high_score_count()
        self.total_label.config(text=f"Toplam İlan: {total}")
        self.high_label.config(text=f"Yüksek Skor (>75): {high}")

    def update_category_list(self):
        cats = self.db.get_all_categories()
        items = ["Tümü"] + cats
        self.cat_combo['values'] = items
        if self.cat_var.get() not in items:
            self.cat_var.set("Tümü")

    def refresh_list(self):
        category = self.cat_var.get()
        if category == "Tümü":
            category = None
        try:
            min_score = float(self.score_var.get()) if self.score_var.get().strip() else None
        except ValueError:
            min_score = None
        records = self.db.get_all_listings(category=category, min_score=min_score)
        self.tree.delete(*self.tree.get_children())
        for rec in records:
            values = (
                rec['listing_id'],
                rec['title'][:60],
                rec['category'],
                f"{rec['price']:,.0f} TL",
                rec['year'] or '-',
                f"{rec['km']:,}" if rec['km'] else '-',
                rec['score'] if rec['score'] is not None else "?",
                rec['recommendation'][:30] if rec['recommendation'] else "-",
                rec['link'][:40] + "..."
            )
            self.tree.insert("", tk.END, values=values)
        self.update_stats()

    def apply_filter(self, event=None):
        self.refresh_list()

    def update_stats_tree(self):
        stats = self.db.get_category_stats_all()
        self.stats_tree.delete(*self.stats_tree.get_children())
        for row in stats:
            values = (
                row['category'],
                f"{row['avg_price']:,.0f} TL" if row['avg_price'] else '-',
                f"{row['avg_km']:,.0f}" if row['avg_km'] else '-',
                f"{row['avg_year']:.0f}" if row['avg_year'] else '-',
                f"{row['avg_days_to_sell']:.1f}" if row['avg_days_to_sell'] else '-',
                row['active_listings'] or 0,
                f"{row['demand_score']:.1f}" if row['demand_score'] else '-'
            )
            self.stats_tree.insert("", tk.END, values=values)

    def update_chart(self):
        """Skor dağılımı grafiği"""
        self.ax.clear()
        records = self.db.get_all_listings()
        scores = [r['score'] for r in records if r['score'] is not None]
        if scores:
            self.ax.hist(scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
            self.ax.set_title('Fırsat Skoru Dağılımı')
            self.ax.set_xlabel('Skor')
            self.ax.set_ylabel('İlan Sayısı')
        else:
            self.ax.text(0.5, 0.5, 'Henüz veri yok', ha='center', va='center', fontsize=14)
        self.canvas.draw()

    def log_message(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def set_status(self, msg, color="blue"):
        self.status_label.config(text=f"Durum: {msg}", foreground=color)

    # ====== Tarama ve Scraping İşlemleri ======
    def manual_scan(self):
        if self.scanning:
            messagebox.showinfo("Bilgi", "Tarama zaten çalışıyor.")
            return
        self.log_message("Manuel e-posta taraması başlatıldı...")
        self.set_status("Taranıyor (e-posta)...", "orange")
        self.scanning = True
        self.scan_thread = threading.Thread(target=self.scan_task, daemon=True)
        self.scan_thread.start()

    def scan_task(self):
        def callback(new_count, msg):
            self.root.after(0, lambda: self.scan_finished(new_count, msg))
        process_emails(self.db, callback)

    def scan_finished(self, new_count, msg):
        self.scanning = False
        self.log_message(msg)
        self.set_status("Beklemede", "blue")
        self.last_scan_label.config(text=f"Son Tarama: {datetime.now().strftime('%H:%M:%S')}")
        self.refresh_list()
        self.update_category_list()
        self.update_stats_tree()
        self.update_chart()

    def scrape_categories(self):
        if self.scanning:
            messagebox.showinfo("Bilgi", "Tarama zaten çalışıyor.")
            return
        self.log_message("Kategori scraping başlatıldı...")
        self.set_status("Scraping yapılıyor...", "orange")
        self.scanning = True
        self.scan_thread = threading.Thread(target=self.scrape_task, daemon=True)
        self.scan_thread.start()

    def scrape_task(self):
        def callback(count, msg):
            self.root.after(0, lambda: self.scrape_finished(count, msg))

        urls = config.get("CATEGORY_URLS", [])
        total = 0
        for url in urls:
            self.log_message(f"Taranıyor: {url}")
            count = SahibindenScraper.scrape_category(url, self.db)
            total += count
            time.sleep(2)
        # Kategori istatistiklerini güncelle
        for url in urls:
            category = url.split('/')[-1].replace('-', ' ').title()
            self.db.update_category_stats(category)
        callback(total, f"Scraping tamamlandı. {total} yeni ilan eklendi.")

    def scrape_finished(self, count, msg):
        self.scanning = False
        self.log_message(msg)
        self.set_status("Beklemede", "blue")
        self.last_scan_label.config(text=f"Son Scrape: {datetime.now().strftime('%H:%M:%S')}")
        self.refresh_list()
        self.update_category_list()
        self.update_stats_tree()
        self.update_chart()

    # ====== Arka Plan Zamanlayıcılar ======
    def start_background_scanner(self):
        interval = config.get("SCAN_INTERVAL", 180)
        self.schedule_scan(interval)

    def schedule_scan(self, interval):
        if self.after_id:
            self.root.after_cancel(self.after_id)
        if not self.scanning:
            self.scanning = True
            self.scan_thread = threading.Thread(target=self.scan_task, daemon=True)
            self.scan_thread.start()
        self.after_id = self.root.after(interval * 1000, lambda: self.schedule_scan(interval))

    def start_scraping_scheduler(self):
        interval = config.get("SCRAPING_INTERVAL", 3600)
        self.schedule_scrape(interval)

    def schedule_scrape(self, interval):
        if not self.scanning:
            self.scraping_thread = threading.Thread(target=self.scrape_task, daemon=True)
            self.scraping_thread.start()
        self.root.after(interval * 1000, lambda: self.schedule_scrape(interval))

    # ====== Ayarlar Penceresi ======
    def open_settings(self):
        settings_win = tk.Toplevel(self.root)
        settings_win.title("Ayarlar")
        settings_win.geometry("600x500")
        settings_win.transient(self.root)
        settings_win.grab_set()

        ttk.Label(settings_win, text="Telegram Bot Token:").pack(pady=5)
        token_var = tk.StringVar(value=config.get("TELEGRAM_BOT_TOKEN", ""))
        ttk.Entry(settings_win, textvariable=token_var, width=50).pack(pady=5)

        ttk.Label(settings_win, text="Telegram Chat ID:").pack(pady=5)
        chat_var = tk.StringVar(value=config.get("TELEGRAM_CHAT_ID", ""))
        ttk.Entry(settings_win, textvariable=chat_var, width=50).pack(pady=5)

        ttk.Label(settings_win, text="Skor Eşiği (bildirim için):").pack(pady=5)
        score_thr_var = tk.StringVar(value=str(config.get("SCORE_THRESHOLD", 75)))
        ttk.Entry(settings_win, textvariable=score_thr_var, width=10).pack(pady=5)

        ttk.Label(settings_win, text="E-posta Tarama Aralığı (sn):").pack(pady=5)
        scan_int_var = tk.StringVar(value=str(config.get("SCAN_INTERVAL", 180)))
        ttk.Entry(settings_win, textvariable=scan_int_var, width=10).pack(pady=5)

        ttk.Label(settings_win, text="Scraping Aralığı (sn):").pack(pady=5)
        scrape_int_var = tk.StringVar(value=str(config.get("SCRAPING_INTERVAL", 3600)))
        ttk.Entry(settings_win, textvariable=scrape_int_var, width=10).pack(pady=5)

        ttk.Label(settings_win, text="Kategori URL'leri (virgülle ayır):").pack(pady=5)
        url_var = tk.StringVar(value=",".join(config.get("CATEGORY_URLS", [])))
        ttk.Entry(settings_win, textvariable=url_var, width=70).pack(pady=5)

        def save_settings():
            try:
                config["TELEGRAM_BOT_TOKEN"] = token_var.get().strip()
                config["TELEGRAM_CHAT_ID"] = chat_var.get().strip()
                config["SCORE_THRESHOLD"] = float(score_thr_var.get())
                config["SCAN_INTERVAL"] = int(scan_int_var.get())
                config["SCRAPING_INTERVAL"] = int(scrape_int_var.get())
                config["CATEGORY_URLS"] = [u.strip() for u in url_var.get().split(",") if u.strip()]
                save_config(config)
                messagebox.showinfo("Başarılı", "Ayarlar kaydedildi.")
                settings_win.destroy()
            except ValueError:
                messagebox.showerror("Hata", "Sayısal değerleri doğru girin.")

        ttk.Button(settings_win, text="Kaydet", command=save_settings).pack(pady=20)

# ==========================================
# ANA ÇALIŞTIRMA
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = OpportunityApp(root)
    root.mainloop()