FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    PORT=8501

WORKDIR /app

# System deps for tokenizers / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download Qwen weights at build time (optional — uncomment to skip cold start)
# RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
#     AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); \
#     AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"

EXPOSE 8501

CMD ["streamlit", "run", "app/frontend/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false"]
