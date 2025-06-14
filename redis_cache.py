import redis
import json
import time

try:
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    # Test connection
    r.ping()
except redis.ConnectionError:
    print("Warning: Redis connection failed. Using mock cache.")
    r = None

class MockRedis:
    """Mock Redis for when Redis is not available"""
    def __init__(self):
        self.data = {}
    
    def lpush(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].insert(0, value)
    
    def ltrim(self, key, start, end):
        if key in self.data:
            self.data[key] = self.data[key][start:end+1]
    
    def lrange(self, key, start, end):
        if key in self.data:
            return self.data[key][start:end+1] if end >= 0 else self.data[key][start:]
        return []

if r is None:
    r = MockRedis()

def cache_decision(decision):
    key = "agent:recent_decisions"
    decision['timestamp'] = time.time()
    # push decision to head of list
    r.lpush(key, json.dumps(decision))
    # trim list to max 5
    r.ltrim(key, 0, 4)

def get_last_5_decisions():
    key = "agent:recent_decisions"
    raw = r.lrange(key, 0, 4)
    return [json.loads(x) for x in raw]
