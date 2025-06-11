from fastapi import Request, HTTPException

EXPECTED = os.getenv("RERANK_API_KEY")

async def check_key(request: Request):
    auth = request.headers.get("authorization")
    if not (auth and auth.lower().startswith("bearer ")):
        raise HTTPException(status_code=401, detail="Missing token")
    token = auth.split(None, 1)[1]
    if token != EXPECTED:
        raise HTTPException(status_code=403, detail="Bad token")