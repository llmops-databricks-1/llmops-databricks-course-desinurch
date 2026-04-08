"""Chunking utilities for processing long text content."""

import re
from typing import Any


def fixed_size_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Create fixed-size chunks with overlap.

    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def sentence_chunking(text: str, max_sentences: int = 5) -> list[str]:
    """Create chunks based on sentence boundaries.

    Args:
        text: Text to chunk
        max_sentences: Maximum sentences per chunk

    Returns:
        List of text chunks
    """
    if not text:
        return [text]

    # Simple sentence splitter (matches ., !, ? followed by whitespace)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) <= max_sentences:
        return [text]

    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) >= max_sentences:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # Add remaining sentences
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def smart_chunking(
    text: str, max_chars: int = 500, strategy: str = "sentence"
) -> list[str]:
    """Smart chunking that tries to preserve semantic boundaries.

    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        strategy: Chunking strategy ('sentence' or 'fixed')

    Returns:
        List of text chunks
    """
    if not text or len(text) <= max_chars:
        return [text]

    if strategy == "sentence":
        # Try sentence-based first
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence would exceed max_chars
            if current_length + sentence_length > max_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add remaining sentences
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
    else:
        # Fall back to fixed-size
        return fixed_size_chunking(text, chunk_size=max_chars, overlap=50)


def estimate_tokens(text: str) -> int:
    """Estimate number of tokens in text.

    Uses the rule of thumb: ~4 characters = 1 token for English text.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return len(text) // 4


def should_chunk(text: str, max_tokens: int = 512) -> bool:
    """Determine if text needs chunking based on token limit.

    Args:
        text: Text to check
        max_tokens: Maximum tokens allowed

    Returns:
        True if text should be chunked
    """
    estimated_tokens = estimate_tokens(text)
    return estimated_tokens > max_tokens


def chunk_with_metadata(
    text: str,
    metadata: dict[str, Any],
    chunk_size: int = 500,
    strategy: str = "sentence",
) -> list[dict[str, Any]]:
    """Chunk text and preserve metadata for each chunk.

    Args:
        text: Text to chunk
        metadata: Metadata to attach to each chunk
        chunk_size: Maximum characters per chunk
        strategy: Chunking strategy

    Returns:
        List of dictionaries with 'text' and metadata
    """
    chunks = smart_chunking(text, max_chars=chunk_size, strategy=strategy)

    result = []
    for i, chunk in enumerate(chunks):
        chunk_data = metadata.copy()
        chunk_data.update(
            {
                "text": chunk,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "chunk_length": len(chunk),
                "estimated_tokens": estimate_tokens(chunk),
            }
        )
        result.append(chunk_data)

    return result


def preprocess_text(text: str) -> str:
    """Clean and preprocess text before chunking.

    Args:
        text: Text to preprocess

    Returns:
        Cleaned text
    """
    if not text:
        return text

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Fix common issues
    text = text.replace("\n\n\n", "\n\n")  # Multiple newlines
    text = text.replace("- ", "-")  # Hyphenation issues

    return text
