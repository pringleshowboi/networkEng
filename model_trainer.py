import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from data_generator import get_training_data
import pickle
import matplotlib.pyplot as plt

class MetricsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ImprovedAgentModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, dropout_rate=0.3):
        super(ImprovedAgentModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.out = nn.Linear(hidden_size // 4, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
    
    def forward(self, x):
        x = torch.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return torch.sigmoid(self.out(x))

def plot_training_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.close()

def train_model():
    print("📚 Loading training data...")
    data = get_training_data()
    
    if len(data) < 50:
        print("❌ Not enough training data. Generate more samples first.")
        return None, None
    
    # Prepare data
    X = np.array([[row[i] for i in range(6)] for row in data])  # features
    y = np.array([row[6] for row in data])  # labels
    
    # Data validation
    if np.isnan(X).any() or np.isinf(X).any():
        print("❌ Dataset contains NaN or infinite values")
        return None, None
    
    print(f"📊 Dataset: {len(X)} samples, {np.sum(y)} positive labels ({np.mean(y):.1%})")
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create datasets and loaders
    train_dataset = MetricsDataset(X_train, y_train)
    test_dataset = MetricsDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedAgentModel().to(device)
    
    # Handle class imbalance
    pos_weight = torch.tensor([len(y_train)/sum(y_train) - 1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    # Training loop
    print("🏋️ Training model...")
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(100):
        # Training
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                val_loss += criterion(outputs, batch_y).item()
        
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        # Early stopping and model saving
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': 6,
                'hidden_size': 64,
                'dropout_rate': 0.3
            }, 'best_model.pth')
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if patience_counter > 20:
            print("Early stopping triggered")
            break
    
    # Plot training curve
    plot_training_curve(train_losses, val_losses)
    
    # Load best model and evaluate
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Full evaluation
    test_probs = []
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(batch_y.cpu().numpy())
    
    # Print results
    print("\n📊 Training Results:")
    print(classification_report(test_labels, test_preds, 
                              target_names=['Keep Running', 'Shutdown']))
    print(f"AUC Score: {roc_auc_score(test_labels, test_probs):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("✅ Model and scaler saved!")
    print(f"📈 Training curve saved as 'training_curve.png'")
    return model, scaler

if __name__ == "__main__":
    train_model()