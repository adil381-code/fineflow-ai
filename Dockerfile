FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
RUN mkdir -p /app/data/chroma_db /app/data/docs_txt /app/data/raw
EXPOSE 8000
# State lives in MySQL → multiple workers are safe. Chroma is read-only at runtime.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
