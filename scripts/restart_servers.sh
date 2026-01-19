#!/bin/bash
# Restart Konte API and UI servers

echo "Stopping existing servers..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:7860 | xargs kill -9 2>/dev/null
sleep 1

echo "Starting API server on port 8000..."
konte serve --port 8000 &

echo "Starting UI server on port 7860..."
konte ui --port 7860 &

echo ""
echo "Servers started:"
echo "  API: http://localhost:8000"
echo "  UI:  http://localhost:7860"
