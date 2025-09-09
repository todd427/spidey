# Dockerizing spidey (drop-in starter)

This folder ships a **drop-in Docker setup** for spidey with:

- `Dockerfile` (Python 3.12 slim, non-root, healthcheck)
- `docker-compose.yml` (ports, volumes, env file, optional GPU support)
- `.dockerignore`
- `.env` (editable)
- `requirements.txt` (CPU by default; swap to CUDA if you want GPU)
- `app.py` (tiny FastAPI stub in case you want to test quickly)

## 1) Place these files in your repo root

Your repo should look roughly like:
```
spidey/
  app.py                # your real app entrypoint exposing FastAPI `app`
  requirements.txt
  Dockerfile
  docker-compose.yml
  .dockerignore
  .env
  models/               # (created on first run)
  store/                # (created on first run)
  cache/                # (created on first run)
```

If your entrypoint isn’t `app.py` with `app = FastAPI(...)`, edit the last line in `Dockerfile`:
```
CMD ["uvicorn", "path.to.your.module:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 2) Build and run (CPU)
```bash
docker compose build
docker compose up
```
Visit http://localhost:8000

## 3) Enable GPU (optional)
1. Install NVIDIA Container Toolkit on the host.
2. Uncomment `# gpus: all` in `docker-compose.yml`.
3. Replace the torch wheel in `requirements.txt` with the CUDA build, e.g. (example for CUDA 12.x):
   ```
   torch --index-url https://download.pytorch.org/whl/cu121
   ```
4. Rebuild:
   ```bash
   docker compose build --no-cache
   docker compose up
   ```

> Tip: you can also run directly:
> ```bash
> docker run --rm -it -p 8000:8000 --gpus all --env-file .env -v $PWD/models:/app/models -v $PWD/store:/app/store -v $PWD/cache:/app/.cache spidey:latest
> ```

## 4) Persist model + cache
The compose file mounts local folders so your HF cache and models persist between container runs:
- `./models` → `/app/models`
- `./store`  → `/app/store`
- `./cache`  → `/app/.cache`

## 5) Config via `.env`
Set the model you want:
```env
MODEL_NAME=toddie314/toddric-1_5b-merged-v1
```
For private/gated repos, set `HUGGINGFACE_HUB_TOKEN` as well.

## 6) Healthcheck
The Dockerfile healthcheck pings `/docs`. If you don’t expose docs, change it to `/healthz`.

## 7) Cloudflare Tunnel
Once running locally on port 8000, tunnel it:
```bash
cloudflared tunnel run <your-tunnel-id>
# map spidey.foxxelabs.com -> http://localhost:8000
```

## 8) Common tweaks
- Change exposed port: set `PORT=8100` in `.env` and map `- "8100:8000"` in compose if you prefer host 8100 → container 8000.
- If you use a different app file/module, update the `CMD` in `Dockerfile` accordingly.
- Add extra system libs by editing the `apt-get install` line.

---
**Goal:** Same container on Lava, Daisy, and production → same behavior. No more “works on my machine.”
