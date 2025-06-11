import os, logging, torch, transformers
from typing import List

from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)

logging.basicConfig(level=logging.INFO)
logging.info("🤗 Transformers %s", transformers.__version__)

# ───────── Auth middleware ──────────────────────────────────────────
API_KEY = os.getenv("RERANK_API_KEY")          # задайте в compose/.env

async def verify_request(request: Request) -> None:
    if API_KEY is None:
        return                                  # защита отключена
    header = request.headers.get("authorization")
    if not (header and header.lower().startswith("bearer ")):
        raise HTTPException(status_code=401, detail="Missing token")
    token = header.split(None, 1)[1]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Bad token")

# ───────── Load model ───────────────────────────────────────────────
MODEL_ID = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
MAX_LEN  = 8192

tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    config=cfg,
    torch_dtype=torch.bfloat16,     # CPU? → уберите аргумент
    device_map="auto",
).eval()
logging.info("Model loaded OK")

# ───────── FastAPI schema & route ───────────────────────────────────
class RerankReq(BaseModel):
    query: str
    documents: List[str]
    top_n: int = Field(5, ge=1)

class DocScore(BaseModel):
    index: int
    relevance_score: float

app = FastAPI(title="Qwen3-Reranker-0.6B")

@app.post("/v1/rerank", response_model=dict, dependencies=[Depends(verify_request)])
@torch.no_grad()
def rerank(req: RerankReq):
    pairs = [f"{req.query} </s> {d}" for d in req.documents]
    inputs = tok(
        pairs,
        padding=True,
        truncation="longest_first",
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(model.device)

    logits  = model(**inputs).logits.squeeze(-1)   # shape [B]
    scores  = torch.sigmoid(logits)                # 0-1
    ordering = scores.argsort(descending=True)[: req.top_n]

    return {
        "object": "rerank_result",
        "data": [
            {"index": int(i), "relevance_score": round(float(scores[i]), 6)}
            for i in ordering
        ],
    }
