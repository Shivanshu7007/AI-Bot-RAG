import logging
from fastapi import APIRouter, HTTPException
from app.services.qdrant import delete_collection

logger = logging.getLogger(__name__)

router = APIRouter()


@router.delete("/collection/{product_id}")
def remove_collection(product_id: int):
    collection = f"product_{product_id}"

    try:
        deleted = delete_collection(collection)

        if deleted:
            logger.info(f"[DeleteCollection] Deleted Qdrant collection: {collection}")
            return {"deleted": True, "collection": collection}
        else:
            # Collection didn't exist — treat as success
            logger.info(f"[DeleteCollection] Collection not found (already gone): {collection}")
            return {"deleted": False, "collection": collection, "reason": "not_found"}

    except Exception as e:
        logger.error(f"[DeleteCollection] Failed to delete {collection}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")
