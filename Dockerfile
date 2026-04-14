FROM python:3.10-slim

WORKDIR /app

# System deps (important for chromadb + numpy)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache optimization)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Env
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create required dirs
RUN mkdir -p /app/data/chroma_db
RUN mkdir -p /app/data/docs_txt
RUN mkdir -p /app/data/raw

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]