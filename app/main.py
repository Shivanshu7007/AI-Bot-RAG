from fastapi import FastAPI, Depends
from app.routes import ask
from app.routes import ingest   # 👈 ADD THIS
from app.core.security import verify_api_key
from app.core.config import settings

app = FastAPI(
    title="Enterprise RAG API",
    docs_url=None if settings.ENV == "production" else "/docs",
    redoc_url=None if settings.ENV == "production" else "/redoc",
    openapi_url=None if settings.ENV == "production" else "/openapi.json"
)

# 🔹 ASK ROUTE
app.include_router(
    ask.router,
    dependencies=[Depends(verify_api_key)]
)

# 🔹 INGEST ROUTE  👈 ADD THIS
app.include_router(
    ingest.router
)

@app.get("/health")
def health():
    return {"status": "ok"}
