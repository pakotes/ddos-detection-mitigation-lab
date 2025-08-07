#!/bin/bash
# Startup script for BGP Controller

echo "Starting BGP Controller..."

# Start metrics server in background
python metrics_server.py &
METRICS_PID=$!
echo "BGP metrics server started (PID: $METRICS_PID)"

# Cleanup function
cleanup() {
    echo "Stopping BGP services..."
    kill $METRICS_PID 2>/dev/null
    exit 0
}

# Capture signals for cleanup
trap cleanup SIGTERM SIGINT

# Wait for services to initialize
sleep 5

# Start main application
echo "Starting main BGP Controller..."
python bgp_controller.py &
MAIN_PID=$!

# Wait for main application
wait $MAIN_PID
