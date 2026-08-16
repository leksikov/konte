#!/bin/bash
# Restart the Konte API server

echo "Stopping existing servers..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
sleep 1

echo "Starting API server on port 8000..."
konte serve --port 8000 &

echo ""
echo "Servers started:"
echo "  API: http://localhost:8000"
