import requests
import json
import time
from typing import List, Dict, Optional

class ProcessManagerAPI:
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_processes(self) -> Dict:
        """Get list of running processes"""
        try:
            response = self.session.get(f"{self.base_url}/processes")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error getting processes: {e}")
            return {}
    
    def search_processes(self, query: str) -> Dict:
        """Search for processes by name or PID"""
        try:
            response = self.session.get(f"{self.base_url}/processes/search/{query}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error searching processes: {e}")
            return {}
    
    def monitor_process(self, pid: int, name: str, condition: str = "ai_decision") -> bool:
        """Add process to monitoring list"""
        try:
            data = {"pid": pid, "name": name, "shutdownCondition": condition}
            response = self.session.post(f"{self.base_url}/processes/monitor", json=data)
            response.raise_for_status()
            print(f"✅ Process {name} (PID: {pid}) added to monitoring")
            return True
        except requests.RequestException as e:
            print(f"Error monitoring process: {e}")
            return False
    
    def kill_process(self, pid: int, force: bool = False, reason: str = "AI decision") -> bool:
        """Kill a specific process"""
        try:
            data = {"force": force, "reason": reason}
            response = self.session.post(f"{self.base_url}/processes/kill/{pid}", json=data)
            response.raise_for_status()
            print(f"✅ Process {pid} terminated: {reason}")
            return True
        except requests.RequestException as e:
            print(f"Error killing process: {e}")
            return False
    
    def shutdown_system(self, delay: int = 60, reason: str = "AI decision") -> bool:
        """Shutdown the system"""
        try:
            data = {"delay": delay, "reason": reason}
            response = self.session.post(f"{self.base_url}/system/shutdown", json=data)
            response.raise_for_status()
            print(f"✅ System shutdown initiated: {reason}")
            return True
        except requests.RequestException as e:
            print(f"Error shutting down system: {e}")
            return False
    
    def get_system_status(self) -> Dict:
        """Get system status and monitoring info"""
        try:
            response = self.session.get(f"{self.base_url}/status")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error getting status: {e}")
            return {}

# Enhanced AI predictor with process management
class EnhancedAIPredictor:
    def __init__(self, api_url: str = "http://localhost:3000"):
        self.api = ProcessManagerAPI(api_url)
        # Load your trained model here
        
    def analyze_and_act(self):
        """Main AI decision loop with process management"""
        # Get AI prediction (from your existing model)
        decision, confidence, raw_score = self.make_prediction()
        
        if decision == "Shutdown" and confidence > 0.8:
            # Get high-resource processes
            processes = self.api.get_processes()
            if processes and 'processes' in processes:
                high_resource_processes = self.find_resource_intensive_processes(
                    processes['processes']
                )
                
                if high_resource_processes:
                    print(f"🎯 Found {len(high_resource_processes)} resource-intensive processes")
                    
                    # Kill problematic processes first
                    for proc in high_resource_processes[:3]:  # Top 3
                        self.api.kill_process(
                            proc['pid'], 
                            reason=f"High resource usage - AI confidence: {confidence:.2%}"
                        )
                        time.sleep(2)  # Wait between kills
                    
                    # Wait and re-evaluate
                    time.sleep(30)
                    new_decision, new_confidence, _ = self.make_prediction()
                    
                    if new_decision == "Shutdown" and new_confidence > 0.9:
                        # Still need shutdown
                        self.api.shutdown_system(
                            delay=120, 
                            reason=f"System still stressed after process cleanup - AI confidence: {new_confidence:.2%}"
                        )
                else:
                    # No specific processes to blame, shutdown system
                    self.api.shutdown_system(
                        delay=60,
                        reason=f"General system stress - AI confidence: {confidence:.2%}"
                    )
    
    def find_resource_intensive_processes(self, processes: List[Dict]) -> List[Dict]:
        """Find processes using high resources"""
        # Sort by memory usage (you might need to parse memory strings)
        resource_intensive = []
        
        for proc in processes:
            # Skip system processes
            if proc['pid'] < 100:  # Usually system processes
                continue
                
            # You can add more sophisticated filtering here
            # based on CPU, memory, or specific process names
            if 'chrome' in proc['name'].lower() or 'firefox' in proc['name'].lower():
                resource_intensive.append(proc)
            
        return resource_intensive[:5]  # Top 5
    
    def make_prediction(self):
        """Your existing AI prediction logic"""
        # This would integrate with your trained model
        # For now, returning mock values
        return "Keep running", 0.3, 0.3

if __name__ == "__main__":
    # Example usage
    api = ProcessManagerAPI()
    
    # Get and display processes
    processes = api.get_processes()
    print(f"Found {processes.get('count', 0)} processes")
    
    # Search for specific processes
    chrome_processes = api.search_processes("chrome")
    print(f"Chrome processes: {chrome_processes.get('count', 0)}")