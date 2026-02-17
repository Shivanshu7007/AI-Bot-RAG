from fastapi import FastAPI, Depends
from app.routes import ingest, ask
from app.core.security import verify_api_key
from app.core.config import settings

app = FastAPI(
    title="Enterprise RAG API",
    docs_url=None if settings.ENV == "production" else "/docs",
    redoc_url=None if settings.ENV == "production" else "/redoc",
    openapi_url=None if settings.ENV == "production" else "/openapi.json"
)

# 🔐 Protect all API routes
app.include_router(
    ingest.router,
    dependencies=[Depends(verify_api_key)]
)

app.include_router(
    ask.router,
    dependencies=[Depends(verify_api_key)]
)

@app.get("/health")
def health():
    return {"status": "ok"}
