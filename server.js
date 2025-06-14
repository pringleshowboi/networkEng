const express = require('express');
const axios = require('axios');
const sqlite3 = require('sqlite3').verbose();
const redis = require('redis');

const app = express();
const port = 3000;

// SQLite connection
const db = new sqlite3.Database('./metrics.db', (err) => {
  if (err) {
    console.error("SQLite connection error:", err.message);
  } else {
    console.log("✅ Connected to SQLite database.");
  }
});

// Redis client (v4+)
const redisClient = redis.createClient();

redisClient.on('error', (err) => console.error('❌ Redis error:', err));

(async () => {
  try {
    await redisClient.connect();
    console.log('✅ Redis connected');
  } catch (err) {
    console.error('Redis connection failed:', err);
  }
})();

// Basic health check
app.get('/', (req, res) => {
  res.send('Monitoring Server is running 🚀');
});

// 🟩 Viewer.py integrations (served by FastAPI on port 8001)
app.get('/viewer/metrics', async (req, res) => {
  try {
    const response = await axios.get('http://localhost:8001/viewer/metrics');
    res.json(response.data);
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ error: 'Failed to fetch metrics from viewer.py' });
  }
});

app.get('/viewer/decisions', async (req, res) => {
  try {
    const response = await axios.get('http://localhost:8001/viewer/decisions');
    res.json(response.data);
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ error: 'Failed to fetch decisions from viewer.py' });
  }
});

// 🟨 Trigger CSV saving for PowerBI
app.post('/viewer/save', async (req, res) => {
  try {
    const response = await axios.post('http://localhost:8001/viewer/save');
    res.json(response.data);
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ error: 'Failed to trigger save from viewer.py' });
  }
});

// 🟥 Raw system metrics from SQLite
app.get('/system/metrics', (req, res) => {
  db.all(`SELECT * FROM system_metrics ORDER BY timestamp DESC LIMIT 1000`, [], (err, rows) => {
    if (err) {
      console.error(err.message);
      return res.status(500).json({ error: 'Failed to fetch metrics' });
    }
    res.json(rows);
  });
});

// 🟥 Recent AI decisions from Redis (async/await for Redis v4)
app.get('/system/decisions', async (req, res) => {
  try {
    const replies = await redisClient.lRange('agent:recent_decisions', 0, 4);
    const decisions = replies.map(r => {
      try {
        return JSON.parse(r);
      } catch (e) {
        return { error: 'Invalid JSON', raw: r };
      }
    });
    res.json(decisions);
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ error: 'Failed to fetch decisions from Redis' });
  }
});

// Start server
app.listen(port, () => {
  console.log(`🟢 Server is running at http://localhost:${port}`);
});
