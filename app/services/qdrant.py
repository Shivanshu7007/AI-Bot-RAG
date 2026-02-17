import requests
from app.core.config import settings

def ensure_collection(name: str):
    r = requests.get(f"{settings.QDRANT_URL}/collections/{name}", timeout=5)
    if r.status_code == 200:
        return

    payload = {
        "vectors": {
            "size": 384,
            "distance": "Cosine"
        }
    }

    requests.put(
        f"{settings.QDRANT_URL}/collections/{name}",
        json=payload,
        timeout=5
    )

def upsert_points(name: str, points):
    return requests.put(
        f"{settings.QDRANT_URL}/collections/{name}/points",
        json={"points": points},
        timeout=30
    )

def search(name: str, vector):
    payload = {
        "vector": vector,
        "limit": 8,
        "with_payload": True
    }

    return requests.post(
        f"{settings.QDRANT_URL}/collections/{name}/points/search",
        json=payload,
        timeout=10
    )
