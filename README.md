# 🕷️ spidey

**spidey** is the reference build for running and serving **Toddric** and related models on the web.  
It’s designed around a simple principle: **It Just Works**.  

Whether you’re deploying locally, tunneling with Cloudflare, or hosting on Hugging Face Spaces, spidey handles model download, setup, and serving out-of-the-box.

---

## ✨ Features

- 🔌 **Plug-and-play**: automatically downloads and caches target models if missing  
- 🌐 **Web UI**: simple, clean chat interface for interacting with Toddric  
- 🚇 **Cloudflare Tunnel support**: serve securely from anywhere  
- 📦 **Docker-ready**: consistent builds across environments  
- ⚡ **vLLM / HuggingFace Transformers** backend support  
- 🧩 Extendable with tools (RAG, LoRA training, Suchi, etc.)

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/toddie314/spidey.git
cd spidey
```

(Optional) create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Run locally
```bash
python app.py
```
By default, spidey will:
- check for the target model (`toddric-1_5b-merged-v1`, etc.)  
- download it from Hugging Face if missing  
- launch the web UI at `http://localhost:8000`

### 2. With Docker
```bash
docker-compose up --build
```

### 3. With Cloudflare Tunnel
If you’ve set up a Cloudflare tunnel:
```bash
cloudflared tunnel run <tunnel-id>
```

Now your instance is securely accessible via `https://spidey.foxxelabs.com`.

---

## ⚙️ Configuration

Environment variables:

| Variable          | Description                              | Default |
|-------------------|------------------------------------------|---------|
| `MODEL_NAME`      | Hugging Face model ID to load            | `toddie314/toddric-1_5b-merged-v1` |
| `HOST`            | Host interface for the web server        | `0.0.0.0` |
| `PORT`            | Port to serve the app on                 | `8000` |

---

## 🧪 Development

- **Models directory**: all models are cached under `./models`  
- **RAG store**: lives in `./store` (ignored in Git)  
- **Extensions**: add tools (e.g. Suchi) in `tools/`

---

## 🗺️ Roadmap

- [ ] Add support for LoRA hot-swap  
- [ ] Integrate RAG by default  
- [ ] Build tool registry (`suchi`, etc.)  
- [ ] Package a one-line install script

---

## 🤝 Contributing

Pull requests are welcome. Please open an issue first for discussion of major changes.

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

### 🌐 Project links
- Website: [foxxelabs.com](https://foxxelabs.com)  
- Hugging Face: [toddie314](https://huggingface.co/toddie314)  
- Related repos: `toddric`, `twilio`, `training`, `pics`, etc.
