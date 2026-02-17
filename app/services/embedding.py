import hashlib
import json
import redis
import logging
from typing import List
from sentence_transformers import SentenceTransformer

# ---------------- Logging ----------------
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# ---------------- Redis Connection Pool ----------------
redis_pool = redis.ConnectionPool(
    host="redis",
    port=6379,
    decode_responses=True,
    max_connections=10
)

r = redis.Redis(connection_pool=redis_pool)

# ---------------- Load Model Once ----------------
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    trust_remote_code=False
)

CACHE_TTL = 86400  # 24 hours

# ---------------- Single Embedding ----------------
def get_embedding(text: str) -> List[float]:
    key = hashlib.sha256(text.encode()).hexdigest()

    try:
        cached = r.get(key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Redis read failed: {e}")

    vector = model.encode(
        text,
        normalize_embeddings=True
    ).tolist()

    try:
        r.set(key, json.dumps(vector), ex=CACHE_TTL)
    except Exception as e:
        logger.warning(f"Redis write failed: {e}")

    return vector


# ---------------- Batch Embeddings (Optimized) ----------------
def get_embeddings_batch(texts: List[str]) -> List[List[float]]:

    keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]

    cached_results = []
    missing_indices = []
    missing_texts = []

    # Try fetching from Redis
    try:
        cached_values = r.mget(keys)
    except Exception as e:
        logger.warning(f"Redis batch read failed: {e}")
        cached_values = [None] * len(keys)

    for i, value in enumerate(cached_values):
        if value:
            cached_results.append(json.loads(value))
        else:
            cached_results.append(None)
            missing_indices.append(i)
            missing_texts.append(texts[i])

    # Compute missing embeddings in batch
    if missing_texts:
        new_vectors = model.encode(
            missing_texts,
            batch_size=32,
            normalize_embeddings=True
        ).tolist()

        for idx, vector in zip(missing_indices, new_vectors):
            cached_results[idx] = vector
            try:
                r.set(keys[idx], json.dumps(vector), ex=CACHE_TTL)
            except Exception as e:
                logger.warning(f"Redis batch write failed: {e}")

    return cached_results
