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
        logger.info("✅ HuggingFace authenticated")
    except Exception as e:
        logger.warning(f"HF login failed: {e}")
else:
    logger.warning("⚠️ HF_TOKEN not set, running anonymous")

# ---------------- Redis ----------------
# ---------------- Redis ----------------
try:
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", 6379))

    r = redis.Redis(
        host=redis_host,
        port=redis_port,
        decode_responses=True
    )
    r.ping()
    REDIS_AVAILABLE = True
    logger.info("✅ Redis connected")

except Exception:
    r = None
    REDIS_AVAILABLE = False
    logger.warning("⚠️ Redis not available, running without cache")

# ---------------- Load Model ----------------
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    trust_remote_code=False
)

CACHE_TTL = 86400

# ---------------- Single Embedding ----------------
def get_embedding(text: str) -> List[float]:

    key = hashlib.sha256(text.encode()).hexdigest()

    if r:
        try:
            cached = r.get(key)
            if cached:
                return json.loads(cached)
        except:
            pass

    vector = model.encode(
        text,
        normalize_embeddings=True
    ).tolist()

    if r:
        try:
            r.set(key, json.dumps(vector), ex=CACHE_TTL)
        except:
            pass

    return vector


# ==============================
# 🔹 BATCH EMBEDDINGS (OPTIMIZED)
# ==============================

def get_embeddings_batch(texts: List[str]) -> List[List[float]]:

    keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]

    cached_results = []
    missing_indices = []
    missing_texts = []

    # Try Redis batch fetch
    if REDIS_AVAILABLE:
        try:
            cached_values = r.mget(keys)
        except Exception as e:
            logger.warning(f"Redis batch read failed: {e}")
            cached_values = [None] * len(keys)
    else:
        cached_values = [None] * len(keys)

    for i, value in enumerate(cached_values):
        if value:
            cached_results.append(json.loads(value))
        else:
            cached_results.append(None)
            missing_indices.append(i)
            missing_texts.append(texts[i])

    # Compute missing in one batch
    if missing_texts:
        new_vectors = model.encode(
            missing_texts,
            batch_size=32,
            normalize_embeddings=True
        ).tolist()

        for idx, vector in zip(missing_indices, new_vectors):
            cached_results[idx] = vector

            if REDIS_AVAILABLE:
                try:
                    r.set(keys[idx], json.dumps(vector), ex=CACHE_TTL)
                except Exception as e:
                    logger.warning(f"Redis batch write failed: {e}")

    return cached_results
