import torch
import torch.nn as nn
import torch.nn.functional as F
from data_collection import get_last_10_metrics
from redis_cache import cache_decision
import numpy as np
import time
from model_trainer import ImprovedAgentModel  # Import the trained model class
import pickle

class AgentModel(nn.Module):
    """Fallback model if ImprovedAgentModel not available"""
    def __init__(self, input_size=6, hidden_size=32, output_size=1):
        super(AgentModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.out(x))

def load_scaler():
    """Load the scaler used during training"""
    try:
        with open('scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Warning: No scaler found. Using default normalization.")
        return None

def prepare_input(metrics, scaler):
    """Prepare input tensor matching training format"""
    # Convert to numpy array
    metrics_array = np.array(metrics, dtype=np.float32)
    
    # Use most recent metric (or aggregate as needed)
    latest_metric = metrics_array[0]  # Using most recent observation
    
    # Reshape for scaler (1 sample, n features)
    if scaler:
        features = scaler.transform(latest_metric.reshape(1, -1))
    else:
        # Fallback normalization (should match training)
        features = latest_metric / np.array([100, 100, 1e6, 1e6, 1e6, 1e6])  # Rough normalization
    
    return torch.FloatTensor(features)

if __name__ == "__main__":
    try:
        # Step 1: get latest metrics
        metrics = get_last_10_metrics()
        
        if len(metrics) < 1:
            print("Not enough data for prediction yet.")
        else:
            # Step 2: load model and scaler
            scaler = load_scaler()
            
            try:
                model = ImprovedAgentModel()
                model.load_state_dict(torch.load('best_model.pth'))
                print("✅ Loaded trained ImprovedAgentModel")
            except:
                print("⚠️ Using fallback AgentModel")
                model = AgentModel()
            
            model.eval()
            
            # Prepare input tensor
            input_tensor = prepare_input(metrics, scaler)
            
            # Make prediction
            with torch.no_grad():
                prediction = model(input_tensor)
                decision = "Shutdown" if prediction.item() > 0.5 else "Keep running"
                confidence = prediction.item() if prediction.item() > 0.5 else 1 - prediction.item()
                
                print(f"🧠 Decision: {decision} | Confidence: {confidence:.2%} | Score: {prediction.item():.4f}")
                
                # Cache decision
                cache_decision({
                    "decision": decision,
                    "confidence": confidence,
                    "score": prediction.item(),
                    "timestamp": time.time()
                })
                
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")