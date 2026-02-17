import uuid

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.services.embedding import get_embeddings_batch
from app.services.qdrant import ensure_collection, upsert_points
from app.utils.chunker import chunk_text
from app.core.security import verify_api_key

router = APIRouter()


class IngestRequest(BaseModel):
    product_id: int
    text: str


def generate_chunk_id(product_id: int, chunk: str) -> str:
    """
    Generate deterministic UUID per product + chunk.
    Fully compatible with Qdrant.
    """
    normalized = chunk.strip().lower()
    base = f"{product_id}:{normalized}"

    # UUID5 = deterministic + valid UUID
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))


@router.post("/ingest")
def ingest(
    request: IngestRequest,
    _: str = Depends(verify_api_key)
):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    collection = f"product_{request.product_id}"

    ensure_collection(collection)

    chunks = chunk_text(request.text)

    if not chunks:
        raise HTTPException(status_code=400, detail="No valid chunks generated")

    vectors = get_embeddings_batch(chunks)

    if len(vectors) != len(chunks):
        raise HTTPException(status_code=500, detail="Embedding mismatch error")

    points = []

    for chunk, vector in zip(chunks, vectors):
        chunk_id = generate_chunk_id(request.product_id, chunk)

        points.append({
            "id": chunk_id,
            "vector": vector,
            "payload": {
                "text": chunk
            }
        })

    res = upsert_points(collection, points)

    if res.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Qdrant error: {res.text}"
        )

    return {
        "stored_chunks": len(points),
        "deduplicated": True,
        "collection": collection
    }
