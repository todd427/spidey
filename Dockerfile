# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf

# System deps (build tools only if your reqs need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
      gosu ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# App user (uid/gid 1000 for host bind-mount friendliness)
RUN groupadd -g 1000 appuser && useradd -m -u 1000 -g 1000 appuser

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -U pip wheel && \
    pip install -r requirements.txt

# Copy app
COPY . /app

# Entrypoint will fix cache perms & drop privileges
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app_toddric:app", "--host", "0.0.0.0", "--port", "8000"]

