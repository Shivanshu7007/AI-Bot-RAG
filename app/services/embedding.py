import os
import hashlib
import json
import redis
import logging
from typing import List
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

# ---------------- Logging ----------------
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# ---------------- HuggingFace Login ----------------
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    try:
        login(token=hf_token)
        logger.info("HuggingFace authenticated")
    except Exception as e:
        logger.warning(f"HF login failed: {e}")
else:
    logger.warning("HF_TOKEN not set, running anonymous")

# ---------------- Redis ----------------
_redis_client: redis.Redis | None = None

try:
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    _redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
    _redis_client.ping()
    logger.info("Redis connected")
except Exception as e:
    _redis_client = None
    logger.warning(f"Redis not available at startup, running without cache: {e}")


def _get_redis() -> redis.Redis | None:
    """Return the Redis client if reachable, otherwise None. Checks on each call."""
    if _redis_client is None:
        return None
    try:
        _redis_client.ping()
        return _redis_client
    except Exception as e:
        logger.warning(f"Redis ping failed, skipping cache: {e}")
        return None


# ---------------- Load Model ----------------
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    trust_remote_code=False
)

CACHE_TTL = 86400


# ---------------- Single Embedding ----------------
def get_embedding(text: str) -> List[float]:

    key = hashlib.sha256(text.encode()).hexdigest()
    r = _get_redis()

    if r:
        try:
            cached = r.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis cache read failed: {e}")

    vector = model.encode(text, normalize_embeddings=True).tolist()

    if r:
        try:
            r.set(key, json.dumps(vector), ex=CACHE_TTL)
        except Exception as e:
            logger.warning(f"Redis cache write failed: {e}")

    return vector


# ---------------- Batch Embeddings ----------------
def get_embeddings_batch(texts: List[str]) -> List[List[float]]:

    keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    r = _get_redis()

    cached_results: List[List[float] | None] = [None] * len(texts)
    missing_indices: List[int] = []
    missing_texts: List[str] = []

    if r:
        try:
            cached_values = r.mget(keys)
        except Exception as e:
            logger.warning(f"Redis batch read failed: {e}")
            cached_values = [None] * len(keys)
    else:
        cached_values = [None] * len(keys)

    for i, value in enumerate(cached_values):
        if value:
            cached_results[i] = json.loads(value)
        else:
            missing_indices.append(i)
            missing_texts.append(texts[i])

    if missing_texts:
        new_vectors = model.encode(
            missing_texts,
            batch_size=32,
            normalize_embeddings=True
        ).tolist()

        r2 = _get_redis()
        for idx, vector in zip(missing_indices, new_vectors):
            cached_results[idx] = vector
            if r2:
                try:
                    r2.set(keys[idx], json.dumps(vector), ex=CACHE_TTL)
                except Exception as e:
                    logger.warning(f"Redis batch write failed for key {keys[idx]}: {e}")

    return cached_results  # type: ignore[return-value]
