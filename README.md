# Monitoring Server with Express, SQLite, Redis, and FastAPI Integration

## Overview  
This project is a monitoring server application built with Node.js and Express. It integrates multiple data sources and provides APIs to fetch and manage system monitoring metrics and AI decisions.

## Key Components

- **Express Server:**  
  Hosts the backend API on port 3000, serving various endpoints for metrics and decision data.

- **SQLite Database:**  
  Stores raw system metrics in a local `metrics.db` file. The server queries recent system metrics (up to 1000 latest records) and exposes them via an API endpoint.

- **Redis Client:**  
  Connects to a Redis instance (local or remote) to retrieve recent AI decision logs stored as a list. Provides an endpoint to fetch and parse these decisions.

- **FastAPI Integration:**  
  Acts as a proxy to a Python FastAPI service running on port 8001 (`viewer.py`). Fetches metrics, decisions, and triggers CSV export saves from this service, exposing these functionalities through the Express API.

- **Error Handling & Health Checks:**  
  Includes basic error handling for database and Redis operations and provides a simple health check endpoint (`/`) to confirm the server is running.

## Running the Server

1. Ensure Redis is running locally or configure the Redis connection accordingly.
2. Ensure the SQLite database (`metrics.db`) exists and contains the necessary tables.
3. Start the FastAPI server on port 8001 (`viewer.py`).
4. Run this Express server:

   ```bash
   node server.js
##Sources & References
Book: https://www.google.co.za/books/edition/Data_Science_and_Network_Engineering/_irhEAAAQBAJ?hl=en&gbpv=1&printsec=frontcover

YouTube Tutorial Video:
https://www.youtube.com/watch?v=6kWQ7QkUR_I

