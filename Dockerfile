FROM python:3.11-slim

WORKDIR /app

ENV PORT=8000

# Install system deps
RUN apt-get update && apt-get install -y curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies before copying source (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

COPY --chown=appuser:appuser app ./app

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production Gunicorn
CMD ["sh", "-c", "gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 1 --timeout 120 --graceful-timeout 30 --bind 0.0.0.0:${PORT} --log-level info"]
