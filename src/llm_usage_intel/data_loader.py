"""Data loading and enrichment utilities."""

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


def get_request_logs_schema() -> StructType:
    """Get the schema for LLM request logs.

    Returns:
        PySpark StructType schema
    """
    return StructType(
        [
            # Original fields
            StructField("timestamp", TimestampType(), False),
            StructField("user_id", StringType(), False),
            StructField("cost", DoubleType(), False),
            StructField("model", StringType(), False),
            StructField("user_agent", StringType(), True),
            StructField("input_tokens", IntegerType(), False),
            StructField("output_tokens", IntegerType(), False),
            # Extended fields
            StructField("query_text", StringType(), True),
            StructField("response_text", StringType(), True),
            StructField("query_category", StringType(), True),
            StructField("latency_ms", IntegerType(), True),
            StructField("status", StringType(), True),
            StructField("quality_score", DoubleType(), True),
            # Derived fields
            StructField("total_tokens", IntegerType(), True),
            StructField("cost_per_1k_tokens", DoubleType(), True),
        ]
    )


def load_existing_logs(
    spark: SparkSession,
    table_path: str | None = None,
    data: list[dict[str, Any]] | None = None,
) -> DataFrame:
    """Load existing LLM request logs.

    Args:
        spark: SparkSession
        table_path: Path to Delta table (optional)
        data: List of log dictionaries (for testing)

    Returns:
        PySpark DataFrame with request logs
    """
    if table_path:
        return spark.read.format("delta").load(table_path)
    elif data:
        schema = get_request_logs_schema()
        return spark.createDataFrame(data, schema=schema)
    else:
        msg = "Either table_path or data must be provided"
        raise ValueError(msg)


def generate_synthetic_queries(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic but realistic LLM queries.

    Args:
        n_samples: Number of queries to generate

    Returns:
        DataFrame with query_text, query_category, and complexity
    """
    import random

    # Query templates by category
    templates = {
        "code_generation": [
            "Write a {language} function to {task}",
            "Create a {language} script that {task}",
            "Generate code to {task} in {language}",
            "How do I implement {task} in {language}?",
        ],
        "code_debugging": [
            "Fix this {language} error: {error}",
            "Debug this code: {error}",
            "Why am I getting '{error}' in {language}?",
            "Help me resolve this {language} issue: {error}",
        ],
        "explanation": [
            "Explain {concept} in simple terms",
            "What is {concept}?",
            "How does {concept} work?",
            "Can you break down {concept} for me?",
        ],
        "translation": [
            "Translate '{text}' to {language}",
            "How do you say '{text}' in {language}?",
            "Convert this to {language}: {text}",
        ],
        "summarization": [
            "Summarize this text: {text}",
            "Give me a brief summary of {text}",
            "TL;DR: {text}",
            "What are the key points in {text}?",
        ],
        "writing": [
            "Write a {type} about {topic}",
            "Draft a {type} for {topic}",
            "Compose a {type} regarding {topic}",
            "Help me write a {type} on {topic}",
        ],
        "analysis": [
            "Analyze this {subject}: {content}",
            "What insights can you provide on {subject}?",
            "Evaluate this {subject}: {content}",
        ],
        "comparison": [
            "Compare {item1} and {item2}",
            "What's the difference between {item1} and {item2}?",
            "{item1} vs {item2}: which is better?",
        ],
    }

    # Fill values
    fill_values = {
        "language": ["Python", "JavaScript", "Java", "Go", "Rust", "TypeScript"],
        "task": [
            "sort a list",
            "validate an email",
            "parse JSON",
            "connect to a database",
            "handle errors",
            "read a file",
        ],
        "error": [
            "TypeError: 'NoneType' object is not subscriptable",
            "IndexError: list index out of range",
            "KeyError: 'user_id'",
            "AttributeError: 'str' object has no attribute 'append'",
        ],
        "concept": [
            "machine learning",
            "blockchain",
            "REST APIs",
            "recursion",
            "async/await",
            "Docker containers",
        ],
        "text": [
            "Hello, how are you?",
            "Thank you very much",
            "I need help with this",
            "Where is the nearest restaurant?",
        ],
        "type": [
            "email",
            "blog post",
            "documentation",
            "README file",
            "cover letter",
            "product description",
        ],
        "topic": [
            "Python best practices",
            "remote work",
            "climate change",
            "software architecture",
            "data privacy",
        ],
        "subject": ["customer review", "code snippet", "business plan", "dataset"],
        "content": ["sample text here", "example content", "data to analyze"],
        "item1": ["React", "Python", "MySQL", "AWS", "REST"],
        "item2": ["Vue", "JavaScript", "PostgreSQL", "Azure", "GraphQL"],
    }

    queries = []
    categories = list(templates.keys())

    for _ in range(n_samples):
        category = random.choice(categories)
        template = random.choice(templates[category])

        # Fill template with random values
        query_text = template
        for key in fill_values:
            if f"{{{key}}}" in query_text:
                query_text = query_text.replace(f"{{{key}}}", random.choice(fill_values[key]))

        # Estimate complexity based on category
        complexity_map = {
            "code_generation": "medium",
            "code_debugging": "medium",
            "explanation": "high",
            "translation": "low",
            "summarization": "medium",
            "writing": "medium",
            "analysis": "high",
            "comparison": "medium",
        }

        queries.append(
            {
                "query_text": query_text,
                "query_category": category,
                "complexity": complexity_map[category],
                "source": "synthetic",
            }
        )

    return pd.DataFrame(queries)


def enrich_logs_with_queries(
    existing_logs: pd.DataFrame, query_dataset: pd.DataFrame
) -> pd.DataFrame:
    """Enrich existing logs with query content.

    Args:
        existing_logs: DataFrame with timestamp, user_id, cost, model, tokens
        query_dataset: DataFrame with query_text, query_category

    Returns:
        Combined DataFrame with all fields
    """
    import random

    # Sample queries to match existing data size
    n_records = len(existing_logs)
    sampled_queries = query_dataset.sample(n=n_records, replace=True).reset_index(
        drop=True
    )

    # Merge datasets
    combined = existing_logs.copy()
    combined["query_text"] = sampled_queries["query_text"].values
    combined["query_category"] = sampled_queries.get(
        "query_category", ["unknown"] * n_records
    ).values

    # Generate synthetic response text (placeholder)
    combined["response_text"] = combined["query_text"].apply(
        lambda x: f"[Response to: {x[:50]}...]"
    )

    # Simulate latency based on tokens and model
    combined["latency_ms"] = combined.apply(
        lambda row: int(
            (row["input_tokens"] + row["output_tokens"])
            * (2 if "gpt-4" in row["model"] else 1)
            * random.uniform(0.8, 1.2)
        ),
        axis=1,
    )

    # Simulate status (most successful)
    combined["status"] = combined.apply(
        lambda x: "error" if random.random() < 0.05 else "success",
        axis=1,
    )

    # Simulate quality scores
    def simulate_quality_score(model: str, category: str) -> float | None:
        """Simulate quality scores based on model and category."""
        base_score = 4.5 if "gpt-4" in model.lower() else 4.0

        # Some categories perform better with certain models
        if category in ["code_generation", "code_debugging"] and "gpt-3.5" in model.lower():
            base_score += 0.2

        # Add random variation
        return min(5.0, max(1.0, base_score + random.uniform(-0.5, 0.5)))

    combined["quality_score"] = combined.apply(
        lambda row: (
            simulate_quality_score(row["model"], row["query_category"])
            if row["status"] == "success"
            else None
        ),
        axis=1,
    )

    # Add derived fields
    combined["total_tokens"] = combined["input_tokens"] + combined["output_tokens"]
    combined["cost_per_1k_tokens"] = (
        combined["cost"] / combined["total_tokens"]
    ) * 1000

    return combined


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Validate the combined dataset.

    Args:
        df: DataFrame to validate

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "total_records": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "unique_users": df["user_id"].nunique(),
        "unique_models": df["model"].nunique(),
        "date_range": (df["timestamp"].min(), df["timestamp"].max()),
        "total_cost": df["cost"].sum(),
        "avg_cost_per_request": df["cost"].mean(),
        "category_distribution": df["query_category"].value_counts().to_dict()
        if "query_category" in df.columns
        else {},
        "model_distribution": df["model"].value_counts().to_dict(),
        "status_distribution": df["status"].value_counts().to_dict()
        if "status" in df.columns
        else {},
    }

    return validation_results
