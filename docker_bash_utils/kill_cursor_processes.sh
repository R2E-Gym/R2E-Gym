#!/bin/bash

# Script to kill Cursor IDE processes that are consuming resources
# This will help reduce latency and free up system resources

echo "Starting Cursor process cleanup at $(date)"
echo "========================================"

# Get all cursor processes
CURSOR_PIDS=$(ps aux | grep -E "cursor|\.cursor-server" | grep -v grep | awk '{print $2}')

if [ -z "$CURSOR_PIDS" ]; then
    echo "No Cursor processes found to kill."
    exit 0
fi

echo "Found Cursor processes with PIDs: $CURSOR_PIDS"
echo ""

# Kill processes gracefully first (SIGTERM)
echo "Attempting graceful shutdown (SIGTERM)..."
for pid in $CURSOR_PIDS; do
    if kill -0 $pid 2>/dev/null; then
        echo "Sending SIGTERM to PID $pid"
        kill -TERM $pid
    fi
done

# Wait a few seconds for graceful shutdown
echo "Waiting 5 seconds for graceful shutdown..."
sleep 5

# Check which processes are still running and force kill them
echo ""
echo "Checking for remaining processes..."
REMAINING_PIDS=$(ps aux | grep -E "cursor|\.cursor-server" | grep -v grep | awk '{print $2}')

if [ ! -z "$REMAINING_PIDS" ]; then
    echo "Force killing remaining processes with PIDs: $REMAINING_PIDS"
    for pid in $REMAINING_PIDS; do
        if kill -0 $pid 2>/dev/null; then
            echo "Force killing PID $pid"
            kill -KILL $pid
        fi
    done
    sleep 2
fi

# Final check
echo ""
echo "Final verification..."
FINAL_CHECK=$(ps aux | grep -E "cursor|\.cursor-server" | grep -v grep | awk '{print $2}')

if [ -z "$FINAL_CHECK" ]; then
    echo "SUCCESS: All Cursor processes have been terminated!"
else
    echo "WARNING: Some processes may still be running: $FINAL_CHECK"
fi

echo ""
echo "Cleanup completed at $(date)"
echo "You may need to restart Cursor IDE to reconnect."
