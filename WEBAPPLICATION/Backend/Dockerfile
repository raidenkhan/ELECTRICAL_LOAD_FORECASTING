FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install CPU-only PyTorch first to avoid downloading 2.5GB+ CUDA version
RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app ./app
COPY ./scripts ./scripts
COPY ./migrations ./migrations
COPY ./alembic.ini .

# Copy models (trained checkpoints)
COPY ./models ./models

# Create data directory
RUN mkdir -p /app/data

# Make startup script executable
RUN chmod +x /app/scripts/startup.sh

# Expose port
EXPOSE 8000

# Run startup script
CMD ["/app/scripts/startup.sh"]
