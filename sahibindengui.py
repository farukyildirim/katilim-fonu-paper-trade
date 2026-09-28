import email
from email.header import decode_header
import imaplib
import re
import sqlite3
import time
import numpy as np
import pandas as pd
import requests
import threading
import json
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# ==========================================
# KONFİGÜRASYON (JSON'dan okunur)
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
    "DB_FILE": "opportunity_tracker.db"
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
# VERİTABANI YÖNETİMİ (geliştirilmiş)
# ==========================================
class OpportunityDB:
    def __init__(self, db_file=None):
        self.db_file = db_file or config["DB_FILE"]
        self.create_tables()

    def get_connection(self):
        return sqlite3.connect(self.db_file, check_same_thread=False)

    def create_tables(self):
        with self.get_connection() as conn:
            # Ana tablo
            conn.execute("""
                CREATE TABLE IF NOT EXISTS listings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    listing_id TEXT UNIQUE,
                    title TEXT,
                    category TEXT,
                    price REAL,
                    link TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Yeni sütunları ekle (eğer yoksa)
            for col, col_type in [("score", "REAL"), ("recommendation", "TEXT"), ("discount", "REAL")]:
                try:
                    conn.execute(f"ALTER TABLE listings ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass  # zaten var
            conn.commit()

    def add_listing(self, listing_id, title, category, price, link):
        category_prices = self.get_category_data(category)
        score, rec, discount = OpportunityEngine.evaluate(price, category_prices)
        try:
            query = """
                INSERT INTO listings 
                (listing_id, title, category, price, link, score, recommendation, discount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            with self.get_connection() as conn:
                conn.execute(query, (listing_id, title, category, price, link, score, rec, discount))
                conn.commit()
            return True, score, rec, discount
        except sqlite3.IntegrityError:
            return False, None, None, None

    def get_category_data(self, category):
        query = "SELECT price FROM listings WHERE category = ?"
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(category,))
        return df["price"].tolist()

    def get_all_listings(self, category=None, min_score=None):
        query = "SELECT * FROM listings WHERE 1=1"
        params = []
        if category:
            query += " AND category = ?"
            params.append(category)
        if min_score is not None:
            # Eski kayıtlarda score null olabilir, onları da göster
            query += " AND (score >= ? OR score IS NULL)"
            params.append(min_score)
        query += " ORDER BY created_at DESC"
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

    def get_latest_listings(self, limit=5):
        query = "SELECT * FROM listings ORDER BY created_at DESC LIMIT ?"
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(limit,))
        return df.to_dict('records')

# ==========================================
# FIRSAT MOTORU (aynı)
# ==========================================
class OpportunityEngine:
    @staticmethod
    def evaluate(target_price, category_prices):
        if len(category_prices) < 3:
            return 50.0, "Yetersiz Kategori Verisi (Toplanıyor)", 0.0
        avg_price = np.mean(category_prices)
        std_price = np.std(category_prices) if np.std(category_prices) > 0 else 1.0
        z_score = (target_price - avg_price) / std_price
        discount_pct = ((avg_price - target_price) / avg_price) * 100
        score = 50.0
        score += discount_pct * 2.5
        if z_score <= -1.5:
            score += 15.0
        score = min(max(score, 0.0), 100.0)
        if score >= 75:
            recommendation = "🔥 YÜKSEK FIRSAT (Anlık Aksiyon Alınmalı)"
        elif 60 <= score < 75:
            recommendation = "👀 MUKAYESELİ UCUZ (Takip Edilebilir)"
        else:
            recommendation = "➡️ STANDART PİYASA FİYATI"
        return round(score, 1), recommendation, round(discount_pct, 1)

# ==========================================
# TELEGRAM BİLDİRİM
# ==========================================
def send_telegram_alert(title, price, category, score, recommendation, discount, link):
    message = (
        f"{recommendation}\n\n"
        f"📌 **İlan:** {title}\n"
        f"🏷 **Kategori:** {category}\n"
        f"💰 **Fiyat:** {price:,.0f} TL\n"
        f"📊 **Kategori İskontosu:** %{discount}\n"
        f"⭐️ **Fırsat Skoru:** {score} / 100\n\n"
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
        print(f"[-] Telegram bildirimi gönderilemedi: {e}")

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
# E-POSTA TARAMA (IMAP)
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
                        category = "Genel_Arama"
                        if "passat" in subject.lower():
                            category = "VW_Passat_1.6TDI"
                        is_new, score, rec, discount = db.add_listing(listing_id, subject, category, price, link)
                        if is_new:
                            new_count += 1
                            if score >= config.get("SCORE_THRESHOLD", 75):
                                send_telegram_alert(subject, price, category, score, rec, discount, link)
        if callback:
            callback(new_count, f"Tarama tamamlandı. {new_count} yeni ilan eklendi.")
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
# TKINTER ARAYÜZÜ
# ==========================================
class OpportunityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fırsat Tespit Motoru - Tkinter")
        self.root.geometry("1200x700")
        self.db = OpportunityDB()
        self.scanning = False
        self.scan_thread = None
        self.after_id = None

        # Menü çubuğu
        menubar = tk.Menu(root)
        root.config(menu=menubar)
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ayarlar", menu=settings_menu)
        settings_menu.add_command(label="Konfigürasyon", command=self.open_settings)
        settings_menu.add_command(label="Manuel Tara", command=self.manual_scan)

        # Ana frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Üst kartlar
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=5)

        self.total_label = ttk.Label(top_frame, text="Toplam İlan: 0", font=("Arial", 12))
        self.total_label.pack(side=tk.LEFT, padx=10)

        self.high_label = ttk.Label(top_frame, text="Yüksek Skor (>75): 0", font=("Arial", 12))
        self.high_label.pack(side=tk.LEFT, padx=10)

        self.last_scan_label = ttk.Label(top_frame, text="Son Tarama: -", font=("Arial", 12))
        self.last_scan_label.pack(side=tk.LEFT, padx=10)

        self.status_label = ttk.Label(top_frame, text="Durum: Beklemede", font=("Arial", 12), foreground="blue")
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Filtreler
        filter_frame = ttk.Frame(main_frame)
        filter_frame.pack(fill=tk.X, pady=5)

        ttk.Label(filter_frame, text="Kategori:").pack(side=tk.LEFT, padx=5)
        self.cat_var = tk.StringVar(value="Tümü")
        self.cat_combo = ttk.Combobox(filter_frame, textvariable=self.cat_var, width=20)
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

        # Treeview (ilan listesi)
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        columns = ("ID", "Başlık", "Kategori", "Fiyat", "Skor", "Öneri", "Link")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20)
        for col in columns:
            self.tree.heading(col, text=col)
            if col == "Başlık":
                self.tree.column(col, width=300)
            elif col == "Link":
                self.tree.column(col, width=200)
            elif col == "Öneri":
                self.tree.column(col, width=200)
            else:
                self.tree.column(col, width=100)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Alt bilgi / log alanı
        log_frame = ttk.Frame(main_frame)
        log_frame.pack(fill=tk.X, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=4, state='disabled')
        self.log_text.pack(fill=tk.X)

        # İlk verileri yükle
        self.refresh_list()
        self.update_stats()
        self.update_category_list()
        # Arka plan tarama thread'ini başlat
        self.start_background_scanner()

    def update_stats(self):
        total = self.db.get_total_count()
        high = self.db.get_high_score_count()
        self.total_label.config(text=f"Toplam İlan: {total}")
        self.high_label.config(text=f"Yüksek Skor (>75): {high}")

    def update_category_list(self):
        cats = self.db.get_category_counts()
        items = ["Tümü"] + list(cats.keys())
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
                rec['score'] if rec['score'] is not None else "?",
                rec['recommendation'][:30] if rec['recommendation'] else "-",
                rec['link'][:50] + "..."
            )
            self.tree.insert("", tk.END, values=values)
        self.update_stats()

    def apply_filter(self, event=None):
        self.refresh_list()

    def log_message(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def set_status(self, msg, color="blue"):
        self.status_label.config(text=f"Durum: {msg}", foreground=color)

    def manual_scan(self):
        if self.scanning:
            messagebox.showinfo("Bilgi", "Tarama zaten çalışıyor.")
            return
        self.log_message("Manuel tarama başlatıldı...")
        self.set_status("Taranıyor...", "orange")
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

    def open_settings(self):
        settings_win = tk.Toplevel(self.root)
        settings_win.title("Ayarlar")
        settings_win.geometry("500x400")
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

        ttk.Label(settings_win, text="Tarama Aralığı (saniye):").pack(pady=5)
        interval_var = tk.StringVar(value=str(config.get("SCAN_INTERVAL", 180)))
        ttk.Entry(settings_win, textvariable=interval_var, width=10).pack(pady=5)

        def save_settings():
            try:
                new_thr = float(score_thr_var.get())
                new_interval = int(interval_var.get())
                if new_interval < 30:
                    messagebox.showerror("Hata", "Tarama aralığı en az 30 saniye olmalı.")
                    return
                config["TELEGRAM_BOT_TOKEN"] = token_var.get().strip()
                config["TELEGRAM_CHAT_ID"] = chat_var.get().strip()
                config["SCORE_THRESHOLD"] = new_thr
                config["SCAN_INTERVAL"] = new_interval
                save_config(config)
                self.schedule_scan(new_interval)
                messagebox.showinfo("Başarılı", "Ayarlar kaydedildi.")
                settings_win.destroy()
            except ValueError:
                messagebox.showerror("Hata", "Skor ve aralık sayısal olmalı.")

        ttk.Button(settings_win, text="Kaydet", command=save_settings).pack(pady=20)

# ==========================================
# ANA ÇALIŞTIRMA
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = OpportunityApp(root)
    root.mainloop()