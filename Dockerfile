FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project BUT restructure it
COPY app/ ./app/
COPY data/ ./data/
COPY *.txt ./
COPY *.yaml ./

# Set Python path to find modules
ENV PYTHONPATH=/app

# Verify the structure
RUN ls -la && ls -la app/

EXPOSE 8000

# Use 1 worker to avoid multiprocessing issues
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]