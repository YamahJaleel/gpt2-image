FROM python:3.10-slim

# Create a writable cache directory
RUN mkdir -p /app/cache && chmod -R 777 /app/cache

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 10000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
