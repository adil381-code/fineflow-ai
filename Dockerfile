FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]