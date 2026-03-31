import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.embedding import get_embeddings_batch
from app.services.qdrant import ensure_collection, upsert_points
from app.utils.chunker import chunk_text

router = APIRouter()


class IngestRequest(BaseModel):
    product_id: int
    text: str


def generate_chunk_id(product_id: int, chunk: str) -> str:
    """Generate deterministic UUID per product+chunk for idempotent upserts."""
    normalized = chunk.strip().lower()
    base = f"{product_id}:{normalized}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))


@router.post("/ingest")
def ingest(request: IngestRequest):
    try:

        # -----------------------------
        # 🔹 Validate Input
        # -----------------------------
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text")

        collection = f"product_{request.product_id}"

        # -----------------------------
        # 🔹 VERY IMPORTANT FIX
        # Ensure collection exists BEFORE upsert
        # -----------------------------
        ensure_collection(collection)

        # -----------------------------
        # 🔹 Chunk Text
        # -----------------------------
        chunks = chunk_text(request.text)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No valid chunks generated"
            )

        # -----------------------------
        # 🔹 Generate Embeddings (Batch)
        # -----------------------------
        vectors = get_embeddings_batch(chunks)

        if len(vectors) != len(chunks):
            raise HTTPException(
                status_code=500,
                detail="Embedding mismatch error"
            )

        # -----------------------------
        # 🔹 Build Points
        # -----------------------------
        points = []

        for chunk, vector in zip(chunks, vectors):

            if not vector:
                continue

            chunk_id = generate_chunk_id(
                request.product_id,
                chunk
            )

            points.append({
                "id": chunk_id,
                "vector": vector,
                "payload": {
                    "text": chunk
                }
            })

        if not points:
            raise HTTPException(
                status_code=500,
                detail="No valid embeddings generated"
            )

        # -----------------------------
        # 🔹 Upsert to Qdrant Cloud
        # -----------------------------
        res = upsert_points(collection, points)

        if res.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Qdrant error: {res.text}"
            )

        # -----------------------------
        # 🔹 Success Response
        # -----------------------------
        return {
            "collection": collection,
            "stored_chunks": len(points),
            "deduplicated": True
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingest failed: {str(e)}"
        )
