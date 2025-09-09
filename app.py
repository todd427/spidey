# Minimal FastAPI app stub for spidey (adjust to your actual app structure)
from fastapi import FastAPI
import os

app = FastAPI(title="spidey")

@app.get("/healthz")
def health():
    return {"ok": True}

@app.get("/config")
def config():
    return {
        "MODEL_NAME": os.getenv("MODEL_NAME", "unset"),
        "HOST": os.getenv("HOST", "0.0.0.0"),
        "PORT": os.getenv("PORT", "8000"),
    }
