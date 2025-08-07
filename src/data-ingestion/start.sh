#!/bin/bash
# Startup script for Data Ingestion

echo "Starting Data Ingestion..."

# Start metrics server in background
python metrics_server.py &
METRICS_PID=$!
echo "Data Ingestion metrics server started (PID: $METRICS_PID)"

# Cleanup function
cleanup() {
    echo "Stopping ingestion services..."
    kill $METRICS_PID 2>/dev/null
    exit 0
}

# Capture signals for cleanup
trap cleanup SIGTERM SIGINT

# Wait for services to initialize
sleep 5

# Start main application
echo "Starting main Data Ingestion..."
python data_ingestion.py &
MAIN_PID=$!

# Wait for main application
wait $MAIN_PID
