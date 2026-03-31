import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    ENV = os.getenv("ENV", "development")

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1"
    )

    MODEL_NAME = os.getenv(
        "MODEL_NAME",
        "qwen/qwen3-next-80b-a3b-instruct"
    )

    SERVICE_API_KEY = os.getenv("SERVICE_API_KEY")

    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    # RAG tuning — configurable via env vars
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.30"))
    MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "2500"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))


settings = Settings()
