#!/bin/bash
set -e

echo "=== GridForecast Pro Setup ==="

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Initialize .env
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    # Generate random secret key
    SEC_KEY=$(openssl rand -hex 32)
    python3 -c "import sys; file_path = '.env'; content = open(file_path).read().replace('SECRET_KEY=generate-a-secure-key-here', 'SECRET_KEY=\'$SEC_KEY\''); open(file_path, 'w').write(content)"
    echo ".env initialized with new SECRET_KEY."
fi

echo "Building and starting containers..."
docker compose build
docker compose up -d

echo "=== Setup Complete! ==="
echo "Access the dashboard at: http://localhost:3000"
echo "API documentation at: http://localhost:8000/docs"
