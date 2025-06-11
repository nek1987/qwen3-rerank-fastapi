# Qwen3-Reranker-4B • FastAPI micro-service

Cross-encoder reranker with OpenAI-compatible `/v1/rerank` endpoint.

## Quick start (locally)

```bash
# 1 – clone & set HF token
export HF_TOKEN=hf_********************************

# 2 – build
docker compose build

# 3 – run
docker compose up
