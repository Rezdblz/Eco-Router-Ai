#!/bin/sh
set -e

if [ "$1" = "build" ]; then
    ollama serve &
    SERVER_PID=$!

    echo "Waiting for Ollama..."

    until ollama list >/dev/null 2>&1; do
        sleep 1
    done

    ollama pull qwen2.5:0.5b
    ollama create eco-router -f /app/Modelfile

    kill $SERVER_PID
    wait $SERVER_PID || true

    exit 0
fi

echo "Starting Ollama..."

ollama serve &
SERVER_PID=$!

until ollama list >/dev/null 2>&1; do
    sleep 1
done

echo "Starting Python..."

exec python -m app.main