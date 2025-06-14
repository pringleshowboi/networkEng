import sqlite3
import random
from datetime import datetime, timezone
import numpy as np

DB_NAME = "metrics.db"

def simulate_metrics():
    cpu = random.uniform(5, 100)
    ram = random.uniform(5, 100)
    disk_read = random.randint(1000, 100000)
    disk_write = random.randint(1000, 100000)
    net_sent = random.randint(500, 100000)
    net_recv = random.randint(500, 100000)
    
    # More sophisticated labeling logic
    # High CPU + RAM = shutdown candidate
    # Very high disk activity = potential issue
    # Network spikes might indicate problems
    
    shutdown_score = 0
    if cpu > 85 and ram > 85:
        shutdown_score += 0.6
    elif cpu > 70 or ram > 70:
        shutdown_score += 0.3
    
    if disk_read > 80000 or disk_write > 80000:
        shutdown_score += 0.2
    
    if net_sent > 80000 or net_recv > 80000:
        shutdown_score += 0.1
    
    # Add some randomness to make it more realistic
    shutdown_score += random.uniform(-0.1, 0.1)
    
    label = 1 if shutdown_score > 0.5 else 0
    
    return (datetime.now(timezone.utc).isoformat(), cpu, ram, disk_read, disk_write, net_sent, net_recv, label)

def create_labeled_table():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS labeled_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            cpu_percent REAL,
            ram_percent REAL,
            disk_read INTEGER,
            disk_write INTEGER,
            net_sent INTEGER,
            net_recv INTEGER,
            label INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def populate_labeled_data(n=500):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    for _ in range(n):
        row = simulate_metrics()
        cursor.execute('''
            INSERT INTO labeled_metrics (timestamp, cpu_percent, ram_percent, disk_read, disk_write, net_sent, net_recv, label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', row)
    conn.commit()
    conn.close()

def get_training_data():
    """Get all labeled data for training"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT cpu_percent, ram_percent, disk_read, disk_write, net_sent, net_recv, label
        FROM labeled_metrics
    ''')
    data = cursor.fetchall()
    conn.close()
    return data

def get_class_distribution():
    """Check the balance of your dataset"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT label, COUNT(*) FROM labeled_metrics GROUP BY label')
    results = cursor.fetchall()
    conn.close()
    return dict(results)

if __name__ == "__main__":
    create_labeled_table()
    populate_labeled_data(500)
    distribution = get_class_distribution()
    print("✅ 500 rows of labeled training data inserted.")
    print(f"📊 Class distribution: {distribution}")
