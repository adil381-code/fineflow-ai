FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

# Create data directory
RUN mkdir -p /app/data

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]