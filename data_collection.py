import psutil
import sqlite3
import time
from datetime import datetime, timezone

DB_NAME = "metrics.db"

def create_table():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            cpu_percent REAL,
            ram_percent REAL,
            disk_read INTEGER,
            disk_write INTEGER,
            net_sent INTEGER,
            net_recv INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def collect_and_store():
    try:
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent

        disk_io = psutil.disk_io_counters()
        disk_read = disk_io.read_bytes if disk_io else 0
        disk_write = disk_io.write_bytes if disk_io else 0

        net_io = psutil.net_io_counters()
        net_sent = net_io.bytes_sent if net_io else 0
        net_recv = net_io.bytes_recv if net_io else 0

        timestamp = datetime.now(timezone.utc).isoformat()
    except Exception as e:
        print(f"Error collecting metrics: {e}")
        return None    
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO system_metrics (timestamp, cpu_percent, ram_percent, disk_read, disk_write, net_sent, net_recv)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, cpu, ram, disk_read, disk_write, net_sent, net_recv))
    conn.commit()
    conn.close()

    return timestamp

def get_last_10_metrics():
    """Get the last 10 metric entries from the database"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT cpu_percent, ram_percent, disk_read, disk_write, net_sent, net_recv
        FROM system_metrics 
        ORDER BY timestamp DESC 
        LIMIT 10
    ''')
    rows = cursor.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    create_table()
    while True:
        timestamp = collect_and_store()
        print(f"[{timestamp}] Metrics collected.")
        time.sleep(60)
