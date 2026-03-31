import logging
import requests
from fastapi import FastAPI, Depends
from app.routes import ask
from app.routes import ingest
from app.routes import delete_collection
from app.core.security import verify_api_key
from app.core.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise RAG API",
    docs_url=None if settings.ENV == "production" else "/docs",
    redoc_url=None if settings.ENV == "production" else "/redoc",
    openapi_url=None if settings.ENV == "production" else "/openapi.json"
)

# Both routes require a valid API key
app.include_router(
    ask.router,
    dependencies=[Depends(verify_api_key)]
)

app.include_router(
    ingest.router,
    dependencies=[Depends(verify_api_key)]
)

app.include_router(
    delete_collection.router,
    dependencies=[Depends(verify_api_key)]
)


@app.get("/health")
def health():
    status = {"status": "ok", "qdrant": "unknown"}

    try:
        r = requests.get(
            f"{settings.QDRANT_URL}/readyz",
            headers={"api-key": settings.QDRANT_API_KEY} if settings.QDRANT_API_KEY else {},
            timeout=5
        )
        status["qdrant"] = "ok" if r.status_code == 200 else "degraded"
    except Exception as e:
        logger.warning(f"Health check - Qdrant unreachable: {e}")
        status["qdrant"] = "unavailable"

    return status
