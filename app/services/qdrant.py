import requests
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

HEADERS = {
    "Content-Type": "application/json"
}

if settings.QDRANT_API_KEY:
    HEADERS["api-key"] = settings.QDRANT_API_KEY


def ensure_collection(name: str):

    try:
        r = requests.get(
            f"{settings.QDRANT_URL}/collections/{name}",
            headers=HEADERS,
            timeout=10
        )

        if r.status_code == 200:
            return

        payload = {
            "vectors": {
                "size": 384,
                "distance": "Cosine"
            }
        }

        create = requests.put(
            f"{settings.QDRANT_URL}/collections/{name}",
            json=payload,
            headers=HEADERS,
            timeout=10
        )

        if create.status_code not in [200, 201]:
            logger.error(f"Collection creation failed: {create.text}")
            raise Exception("Collection creation failed")

    except Exception as e:
        logger.error(f"Qdrant ensure_collection error: {e}")
        raise


def upsert_points(name: str, points):

    try:
        r = requests.put(
            f"{settings.QDRANT_URL}/collections/{name}/points?wait=true",
            json={"points": points},
            headers=HEADERS,
            timeout=30
        )

        if r.status_code != 200:
            logger.error(f"Upsert failed: {r.text}")
            raise Exception("Upsert failed")

        return r

    except Exception as e:
        logger.error(f"Qdrant upsert error: {e}")
        raise


def delete_collection(name: str) -> bool:
    """Delete a Qdrant collection. Returns True if deleted, False if not found."""
    try:
        r = requests.delete(
            f"{settings.QDRANT_URL}/collections/{name}",
            headers=HEADERS,
            timeout=10
        )

        if r.status_code == 200:
            return True
        elif r.status_code == 404:
            return False
        else:
            logger.error(f"Qdrant delete_collection unexpected status {r.status_code}: {r.text}")
            raise Exception(f"Qdrant returned {r.status_code}: {r.text}")

    except Exception as e:
        logger.error(f"Qdrant delete_collection error: {e}")
        raise


def search(name: str, vector):

    payload = {
        "vector": vector,
        "limit": 8,
        "with_payload": True
    }

    r = requests.post(
        f"{settings.QDRANT_URL}/collections/{name}/points/search",
        json=payload,
        headers=HEADERS,
        timeout=20
    )

    if r.status_code != 200:
        logger.error(f"Search failed: {r.text}")
        raise Exception("Search failed")

    return r
