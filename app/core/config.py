import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    ENV = os.getenv("ENV", "development")

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    SERVICE_API_KEY = os.getenv("SERVICE_API_KEY")

    QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

    # ✅ ADD THIS
    MODEL_NAME = os.getenv(
        "MODEL_NAME",
        "qwen/qwen3-next-80b-a3b-instruct"
    )

    # OpenRouter base URL
    OPENROUTER_BASE_URL = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1"
    )

    SIMILARITY_THRESHOLD = 0.15
    MAX_CONTEXT_LENGTH = 4000
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150


settings = Settings()
