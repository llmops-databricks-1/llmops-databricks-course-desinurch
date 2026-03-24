"""Query classification utilities."""

from typing import Any, Literal


def classify_query_category(
    query_text: str,
) -> Literal[
    "code_generation",
    "code_debugging",
    "explanation",
    "translation",
    "summarization",
    "writing",
    "analysis",
    "comparison",
    "general_qa",
]:
    """Classify query into categories using heuristics.

    Args:
        query_text: Query text to classify

    Returns:
        Category name

    Categories:
        - code_generation: Writing new code
        - code_debugging: Fixing errors
        - explanation: Explaining concepts
        - translation: Language translation
        - summarization: Text summarization
        - writing: Content creation
        - analysis: Data/content analysis
        - comparison: Comparing options
        - general_qa: General questions
    """
    query_lower = query_text.lower()

    if any(
        word in query_lower
        for word in ["write", "create", "generate", "function", "class"]
    ):
        return "code_generation"
    elif any(word in query_lower for word in ["fix", "debug", "error", "bug"]):
        return "code_debugging"
    elif any(word in query_lower for word in ["explain", "what is", "how does"]):
        return "explanation"
    elif any(word in query_lower for word in ["translate", "translation"]):
        return "translation"
    elif any(word in query_lower for word in ["summarize", "summary"]):
        return "summarization"
    elif any(
        word in query_lower for word in ["write", "draft", "compose", "email", "letter"]
    ):
        return "writing"
    elif any(word in query_lower for word in ["analyze", "analysis", "sentiment"]):
        return "analysis"
    elif any(word in query_lower for word in ["compare", "difference", "vs", "versus"]):
        return "comparison"
    else:
        return "general_qa"


def estimate_complexity(input_tokens: int, output_tokens: int) -> str:
    """Estimate query complexity based on token counts.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Complexity level: "low", "medium", or "high"
    """
    total = input_tokens + output_tokens
    if total < 200:
        return "low"
    elif total < 800:
        return "medium"
    else:
        return "high"


def extract_query_features(query_text: str) -> dict[str, Any]:
    """Extract features from query text for optimization.

    Args:
        query_text: Query text to analyze

    Returns:
        Dictionary of extracted features
    """
    return {
        "token_count": len(query_text.split()),
        "char_count": len(query_text),
        "has_code_block": "```" in query_text,
        "has_url": "http://" in query_text or "https://" in query_text,
        "question_words": sum(
            1
            for word in ["what", "why", "how", "when", "where", "who", "which"]
            if word in query_text.lower()
        ),
        "category": classify_query_category(query_text),
    }
