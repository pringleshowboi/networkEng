import sqlite3
import json
from datetime import datetime
import pandas as pd
from redis_cache import get_last_5_decisions

def get_metrics_dataframe():
    """Return metrics as a pandas DataFrame for easy PowerBI consumption"""
    conn = sqlite3.connect("metrics.db")
    try:
        # Get metrics with proper datetime conversion
        df = pd.read_sql_query("""
            SELECT 
                id,
                datetime(timestamp) as timestamp,
                cpu_percent,
                ram_percent,
                disk_read,
                disk_write,
                net_sent,
                net_recv
            FROM system_metrics 
            ORDER BY timestamp DESC
            LIMIT 1000
        """, conn)
        
        # Convert bytes to MB for better readability
        for col in ['disk_read', 'disk_write', 'net_sent', 'net_recv']:
            df[col] = df[col] / (1024 * 1024)  # Convert to MB
            
        return df
    
    finally:
        conn.close()

def get_decisions_dataframe():
    """Return decisions as a pandas DataFrame"""
    try:
        decisions = get_last_5_decisions()
        if decisions:
            df = pd.DataFrame(decisions)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"Error retrieving decisions: {e}")
        return pd.DataFrame()

def save_data_for_powerbi():
    """Save data to CSV files for PowerBI import"""
    try:
        # Get and save metrics
        metrics_df = get_metrics_dataframe()
        metrics_df.to_csv('powerbi_metrics.csv', index=False)
        
        # Get and save decisions
        decisions_df = get_decisions_dataframe()
        if not decisions_df.empty:
            decisions_df.to_csv('powerbi_decisions.csv', index=False)
        
        print("Data successfully saved for PowerBI:")
        print(f"- Metrics: {len(metrics_df)} records (powerbi_metrics.csv)")
        print(f"- Decisions: {len(decisions_df)} records (powerbi_decisions.csv)")
        
        return True
    except Exception as e:
        print(f"Error saving data for PowerBI: {e}")
        return False

if __name__ == "__main__":
    # When run directly, show sample data and save to CSV
    print("=== System Metrics Sample ===")
    print(get_metrics_dataframe().head())
    
    print("\n=== Recent Decisions ===")
    print(get_decisions_dataframe())
    
    # Save data for PowerBI
    save_data_for_powerbi()

import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        fn = sys.argv[1]
        if fn == "get_metrics_dataframe":
            print(get_metrics_dataframe().to_json(orient="records"))
        elif fn == "get_decisions_dataframe":
            print(get_decisions_dataframe().to_json(orient="records"))
        elif fn == "save_data_for_powerbi":
            print(json.dumps({"success": save_data_for_powerbi()}))
        else:
            print(json.dumps({"error": "Unknown function"}))