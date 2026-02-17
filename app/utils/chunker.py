from app.core.config import settings

def chunk_text(text: str):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + settings.CHUNK_SIZE
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += settings.CHUNK_SIZE - settings.CHUNK_OVERLAP

    return chunks
