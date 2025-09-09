# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files & enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1         PYTHONUNBUFFERED=1         PIP_NO_CACHE_DIR=1         HF_HOME=/app/.cache/huggingface

WORKDIR /app

# System deps (git for HF, curl for healthcheck, libgomp for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement pins first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy app source
COPY . /app

# Create non-root user
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Healthcheck hits the FastAPI docs endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -fsS http://localhost:8000/docs >/dev/null || exit 1

# Default environment (override in compose/.env)
ENV HOST=0.0.0.0 PORT=8000 MODEL_NAME=toddie314/toddric-1_5b-merged-v1

# Mount points for persistence (optional)
VOLUME ["/app/models", "/app/store", "/app/.cache"]

# Start the server (expects FastAPI app exposed as `app` in app.py)
CMD ["uvicorn", "app_toddric:app", "--host", "0.0.0.0", "--port", "8000"]
