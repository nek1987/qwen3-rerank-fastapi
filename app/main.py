from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, Field
import torch, math, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-Reranker-4B")
INSTRUCT = (
    "Given a web search query, retrieve relevant passages "
    "that answer the query."
)

# --- Prompt templates --------------------------------------------------------
SYS = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query "
    "and the Instruct provided. Answer strictly with \"yes\" or \"no\"."
    "<|im_end|>\n"
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

MAX_LEN = 8192   # n_ctx модели (можно меньше для экономии VRAM)

# --- Load model --------------------------------------------------------------
print(f"Loading {MODEL_ID} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left", use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

cfg = AutoConfig.from_pretrained(
    MODEL_ID,
    trust_remote_code=True      # ← критичный параметр
)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    config=cfg,                 # обязательно тот же config
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

#YES_ID  = tokenizer.convert_tokens_to_ids("yes")
#NO_ID   = tokenizer.convert_tokens_to_ids("no")
cls_head = torch.nn.Sigmoid() 

def build_pair(query: str, doc: str) -> str:
    return f"{SYS}<Instruct>: {INSTRUCT}\n<Query>: {query}\n<Document>: {doc}{SUFFIX}"

@torch.no_grad()
def score_pairs(pairs: List[str]) -> List[float]:
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation="longest_first",
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(model.device)

    logits = model(**inputs).logits.squeeze(-1)          # last token
    probs = torch.sigmoid(logits)
    return probs.cpu().float().tolist()

# --- FastAPI schema ----------------------------------------------------------
class RerankRequest(BaseModel):
    query: str = Field(..., description="User search query")
    documents: List[str] = Field(..., description="List of candidate passages")
    top_n: int = Field(5, ge=1, description="Return N best documents")

class DocScore(BaseModel):
    index: int
    relevance_score: float

class RerankResponse(BaseModel):
    object: str = "rerank_result"
    data: List[DocScore]

# --- API ---------------------------------------------------------------------
app = FastAPI(title="Qwen3-4B Cross-Encoder Reranker")

@app.post("/v1/rerank", response_model=RerankResponse, tags=["rerank"])
def rerank(req: RerankRequest):
    pairs = [build_pair(req.query, d) for d in req.documents]
    scores = score_pairs(pairs)

    ranked = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )[: min(req.top_n, len(scores))]

    return {
        "data": [
            {"index": idx, "relevance_score": round(score, 6)}
            for idx, score in ranked
        ]
    }
