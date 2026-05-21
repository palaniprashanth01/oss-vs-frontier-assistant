"""
Deploy the OSS assistant on Modal as a serverless web endpoint.

    pip install modal
    modal token new
    modal deploy deploy/modal_app.py

Free tier on Modal includes $30/mo compute credit — plenty for a demo.
"""

from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.45",
        "torch",
        "accelerate",
        "sentencepiece",
        "streamlit",
        "openai",
    )
    .add_local_dir(".", remote_path="/app")
)

app = modal.App("oss-vs-frontier-assistant", image=image)


@app.function(
    cpu=2.0,
    memory=4096,
    timeout=600,
    secrets=[
        # Mounts the environment secrets. The frontend and backend automatically
        # auto-detect DEEPSEEK_API_KEY or OPENROUTER_API_KEY from the environment.
        # Ensure your Modal Secret contains either of these keys!
        modal.Secret.from_name("openrouter-key")
    ],
)
@modal.asgi_app()
def streamlit_app():
    """Serve the Streamlit UI as an ASGI app."""
    import subprocess
    import threading
    import time
    from fastapi import FastAPI
    from starlette.responses import RedirectResponse

    def _run_streamlit():
        subprocess.run(
            [
                "streamlit", "run", "/app/app/frontend/streamlit_app.py",
                "--server.port", "8000",
                "--server.headless", "true",
                "--server.address", "0.0.0.0",
                "--browser.gatherUsageStats", "false",
            ]
        )

    threading.Thread(target=_run_streamlit, daemon=True).start()
    time.sleep(5)  # wait for Streamlit to bind

    web = FastAPI()

    @web.get("/")
    def index():
        return RedirectResponse(url="http://localhost:8000")

    return web


@app.function(cpu=1.0, memory=2048, timeout=120)
def chat_oss(message: str) -> str:
    """Standalone /chat endpoint, useful for load tests and pipelines."""
    import sys
    sys.path.insert(0, "/app")
    from app.backend import OSSAssistant
    return OSSAssistant().chat(message)["reply"]


@app.local_entrypoint()
def main():
    print("Deployed. Streamlit UI at the modal app URL.")
