FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_TOKEN=${HF_TOKEN}

RUN apt-get update && apt-get install -y python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app/requirements.txt .

RUN pip install --no-cache-dir torch==2.3.0+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
        git+https://github.com/huggingface/transformers.git@main && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
