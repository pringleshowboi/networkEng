import torch
import pickle
import numpy as np
from data_collection import get_last_10_metrics
from redis_cache import cache_decision
import time

class TrainedPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        try:
            # Load trained model
            from model_trainer import ImprovedAgentModel
            self.model = ImprovedAgentModel()
            self.model.load_state_dict(torch.load('best_model.pth'))
            self.model.eval()
            
            # Load scaler
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            print("✅ Trained model loaded successfully!")
        except FileNotFoundError:
            print("❌ No trained model found. Train the model first.")
            self.model = None
            self.scaler = None
    
    def predict(self):
        if self.model is None or self.scaler is None:
            print("❌ Model not loaded. Cannot make predictions.")
            return
        
        # Get recent metrics
        metrics = get_last_10_metrics()
        
        if len(metrics) < 1:
            print("❌ No metrics data available.")
            return
        
        # Use most recent metric for prediction
        latest_metric = metrics[0]
        features = np.array([latest_metric]).reshape(1, -1)
        
        # Normalize features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(features_scaled)
            prediction = self.model(input_tensor).item()
            
            decision = "Shutdown" if prediction > 0.5 else "Keep running"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            print(f"🧠 AI Decision: {decision}")
            print(f"📊 Confidence: {confidence:.2%}")
            print(f"🔢 Raw score: {prediction:.4f}")
            
            # Cache decision
            cache_decision({
                "decision": decision,
                "confidence": confidence,
                "raw_score": prediction,
                "timestamp": time.time()
            })
            
            return decision, confidence, prediction

if __name__ == "__main__":
    predictor = TrainedPredictor()
    predictor.predict()
