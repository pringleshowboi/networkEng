import threading
import time
import subprocess
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_monitor.log'),
        logging.StreamHandler()
    ]
)

def run_data_collection():
    """Run data collection in a separate thread"""
    from data_collection import create_table, collect_and_store
    create_table()
    while True:
        try:
            timestamp = collect_and_store()
            logging.info(f"Metrics collected at {timestamp}")
            time.sleep(60)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(f"Data collection error: {e}", exc_info=True)
            time.sleep(10)

def run_model_predictions():
    """Run model predictions periodically"""
    while True:
        try:
            time.sleep(120)  # Wait 2 minutes before first prediction
            logging.info("Running model prediction...")
            result = subprocess.run(
                [sys.executable, "metric_modeller.py"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logging.error(f"Prediction failed: {result.stderr}")
            else:
                logging.info(f"Prediction completed: {result.stdout}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(f"Model prediction error: {e}", exc_info=True)
            time.sleep(30)

def system_status_monitor():
    """Monitor and log system resource availability"""
    import psutil
    while True:
        try:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            logging.info(f"System Status - CPU: {cpu}%, Memory: {mem}%")
            time.sleep(300)  # Check every 5 minutes
        except Exception as e:
            logging.error(f"System monitor error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    logging.info("Starting system monitoring...")
    
    # Start all threads
    threads = [
        threading.Thread(target=run_data_collection, daemon=True),
        threading.Thread(target=run_model_predictions, daemon=True),
        threading.Thread(target=system_status_monitor, daemon=True)
    ]
    
    for thread in threads:
        thread.start()
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down system monitoring...")
    finally:
        logging.info("System monitoring stopped.")