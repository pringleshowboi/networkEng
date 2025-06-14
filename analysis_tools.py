import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def analyze_dataset():
    """Analyze the generated training data"""
    conn = sqlite3.connect("metrics.db")
    
    # Load data
    df = pd.read_sql_query('''
        SELECT cpu_percent, ram_percent, disk_read, disk_write, 
               net_sent, net_recv, label 
        FROM labeled_metrics
    ''', conn)
    conn.close()
    
    print("📊 Dataset Analysis:")
    print(f"Total samples: {len(df)}")
    print(f"Features: {df.columns[:-1].tolist()}")
    print(f"Class distribution:")
    print(df['label'].value_counts())
    print(f"Class balance: {df['label'].value_counts(normalize=True)}")
    
    # Correlation analysis
    print("\n🔗 Feature correlations with label:")
    correlations = df.corr()['label'].drop('label').sort_values(key=abs, ascending=False)
    for feature, corr in correlations.items():
        print(f"{feature}: {corr:.3f}")
    
    # Basic statistics
    print("\n📈 Feature statistics:")
    print(df.describe())
    
    return df

def visualize_data():
    """Create visualizations of the dataset"""
    df = analyze_dataset()
    
    # Set up the plotting
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    features = ['cpu_percent', 'ram_percent', 'disk_read', 'disk_write', 'net_sent', 'net_recv']
    
    for i, feature in enumerate(features):
        row, col = i // 3, i % 3
        
        # Box plot by label
        df.boxplot(column=feature, by='label', ax=axes[row, col])
        axes[row, col].set_title(f'{feature} by Label')
        axes[row, col].set_xlabel('Label (0=Keep, 1=Shutdown)')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Visualization saved as 'dataset_analysis.png'")

if __name__ == "__main__":
    analyze_dataset()