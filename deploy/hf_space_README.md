# Deploy the OSS Assistant on Hugging Face Spaces

The free CPU tier (2 vCPU / 16GB RAM) is enough for Qwen2.5-0.5B-Instruct.

## 1. Create a Space

1. Go to https://huggingface.co/new-space
2. Name: `oss-vs-frontier-assistant`
3. SDK: **Streamlit**
4. Hardware: **CPU basic** (free)
5. Visibility: Public

## 2. Push the code

```bash
git remote add space https://huggingface.co/spaces/<your-username>/oss-vs-frontier-assistant
git push space main
```

## 3. Required files at the repo root

Hugging Face needs the Streamlit entrypoint at the root. Add a tiny shim:

```python
# app.py  (at repo root — HF Spaces looks for this by default)
import runpy, sys
sys.argv = ["streamlit", "run", "app/frontend/streamlit_app.py"]
runpy.run_module("streamlit.web.cli", run_name="__main__")
```

Or set `app_file: app/frontend/streamlit_app.py` in the Space README front-matter:

```yaml
---
title: OSS vs Frontier Assistant
emoji: 🤖
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.40.0
app_file: app/frontend/streamlit_app.py
pinned: false
---
```

## 4. Secrets

In **Space Settings → Variables and secrets**, add this one key:

| Key | Value | Notes |
|---|---|---|
| `GROQ_API_KEY` | (from https://console.groq.com/keys) | Free tier, no card. Powers Frontier (GPT-OSS-120B). |

> Optional: the backend also auto-detects `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`,
> or `OPENROUTER_API_KEY` if you'd rather use a different provider — see
> `deploy/.env.example` for the full priority chain.

The OSS backend needs no secret — Qwen2.5-0.5B downloads on first run.

## 5. First-boot cost

- Model download: ~1.0 GB (Qwen2.5-0.5B-Instruct weights), cached after first boot.
- Cold start: ~45s (downloading) → ~10s (warm boot from cache).
- Idle: free (Spaces sleep automatically).

## 6. Live demo URL

`https://huggingface.co/spaces/<your-username>/oss-vs-frontier-assistant`
