from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import torch, logging
import transformers

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)
logging.info(f"🤗 Transformers {transformers.__version__}")

logging.basicConfig(level=logging.INFO)

MODEL_ID = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"   # seq-cls чек-пойнт
MAX_LEN  = 8192                                      # хватит для query+doc

# ── загрузка
tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    config=cfg,
    torch_dtype=torch.bfloat16,     # A40 / 3090 → BF16
    device_map="auto",
).eval()
logging.info("Model loaded OK")

# ── FastAPI
app = FastAPI(title="Qwen3-Reranker-0.6B")

class RerankReq(BaseModel):
    query: str = Field(..., example="largest mobile operator uzbekistan")
    documents: List[str] = Field(..., min_items=1, example=["Beeline ...", "Ucell ..."])
    top_n: int = Field(5, ge=1, le=100)

class DocScore(BaseModel):
    index: int
    relevance_score: float

@app.post("/v1/rerank", response_model=dict)
@torch.no_grad()
def rerank(req: RerankReq):
    pairs = [f"{req.query} </s> {doc}" for doc in req.documents]
    inputs = tok(
        pairs,
        padding=True,
        truncation="longest_first",
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(model.device)

    logits = model(**inputs).logits.squeeze(-1)       # [B]
    scores = torch.sigmoid(logits)                    # 0-1 relevance
    order  = scores.argsort(descending=True)[: req.top_n]

    return {
        "object": "rerank_result",
        "data": [
            {"index": int(i), "relevance_score": round(float(scores[i]), 6)}
            for i in order
        ],
    }
