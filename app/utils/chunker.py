import re
from app.core.config import settings


# Split on sentence-ending punctuation followed by whitespace.
# Keeps the delimiter with the preceding sentence via a lookbehind.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def chunk_text(text: str):
    """
    Split text into overlapping chunks that respect sentence boundaries.

    Strategy:
    1. Split the full text into sentences using punctuation as delimiters.
    2. Accumulate sentences into a chunk until the word budget is full.
    3. When the budget overflows, emit the current chunk and start the next
       one by retaining the last `overlap_sentences` sentences as context.
    """
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunk_size = settings.CHUNK_SIZE       # target words per chunk
    chunk_overlap = settings.CHUNK_OVERLAP  # overlap words (approximated as sentences)

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        if current_word_count + word_count > chunk_size and current_sentences:
            # Emit the accumulated chunk
            chunks.append(" ".join(current_sentences))

            # Keep trailing sentences whose total word count ≤ chunk_overlap
            overlap: list[str] = []
            overlap_words = 0
            for s in reversed(current_sentences):
                s_words = len(s.split())
                if overlap_words + s_words > chunk_overlap:
                    break
                overlap.insert(0, s)
                overlap_words += s_words

            current_sentences = overlap
            current_word_count = overlap_words

        current_sentences.append(sentence)
        current_word_count += word_count

    # Emit any remaining sentences
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks
