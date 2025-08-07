#!/bin/bash
# Startup script for ML Processor

echo "Starting ML Processor..."

# Start metrics server in background
python metrics_server.py &
METRICS_PID=$!
echo "Metrics server started (PID: $METRICS_PID)"

# Cleanup function
cleanup() {
    echo "Stopping services..."
    kill $METRICS_PID 2>/dev/null
    exit 0
}

# Capture signals for cleanup
trap cleanup SIGTERM SIGINT

# Wait for services to initialize
sleep 5

# Start main application
echo "Starting main application..."
python ml_pipeline.py &
MAIN_PID=$!

# Wait for main application
wait $MAIN_PID
