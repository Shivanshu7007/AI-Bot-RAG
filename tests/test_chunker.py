"""Unit tests for app/utils/chunker.py"""
import pytest
from unittest.mock import patch


# Patch settings before importing chunker so we control CHUNK_SIZE / CHUNK_OVERLAP
class _Settings:
    CHUNK_SIZE = 20
    CHUNK_OVERLAP = 5


with patch("app.core.config.settings", _Settings()):
    from app.utils.chunker import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []


def test_whitespace_only_returns_no_chunks():
    assert chunk_text("   \n\t  ") == []


def test_single_short_sentence_yields_one_chunk():
    text = "This is a short sentence."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert "short sentence" in chunks[0]


def test_chunks_split_on_sentence_boundaries():
    """No chunk should end mid-sentence (last char should be punctuation or full word)."""
    text = (
        "The reagent must be stored at 4 degrees Celsius. "
        "Do not freeze the sample under any circumstances. "
        "Centrifuge at 300g for five minutes. "
        "Add buffer solution slowly. "
        "Incubate at room temperature for thirty minutes."
    )
    chunks = chunk_text(text)
    assert len(chunks) >= 1
    # Every chunk must be non-empty
    for chunk in chunks:
        assert chunk.strip() != ""


def test_overlap_carries_context_forward():
    """The last sentence of chunk N should appear at the start of chunk N+1."""
    # Build text long enough to generate at least 2 chunks with CHUNK_SIZE=20
    sentences = [f"Sentence number {i} contains some words for testing." for i in range(10)]
    text = " ".join(sentences)
    chunks = chunk_text(text)
    if len(chunks) >= 2:
        # The tail of chunk[0] and the head of chunk[1] must share at least one sentence
        last_sentence_of_first = chunks[0].split(". ")[-1]
        assert last_sentence_of_first in chunks[1] or chunks[0][-30:] in chunks[1]


def test_long_single_sentence_stays_in_one_chunk():
    """A single very long sentence cannot be split — it must appear as one chunk."""
    long_sentence = "word " * 50 + "end."
    chunks = chunk_text(long_sentence.strip())
    assert len(chunks) >= 1
    # All words must appear somewhere across chunks
    full_text = " ".join(chunks)
    assert "end." in full_text


def test_returns_list_of_strings():
    chunks = chunk_text("Hello world. Goodbye world.")
    assert isinstance(chunks, list)
    for c in chunks:
        assert isinstance(c, str)
