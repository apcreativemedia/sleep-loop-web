FROM python:3.11-slim

# Install ffmpeg (required for audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway will set $PORT. Use long timeout because 8h renders can take 10-20min on CPU.
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} \
    --workers 1 \
    --threads 4 \
    --timeout 1800 \
    --access-logfile - \
    app:app
