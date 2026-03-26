"""Vector search utilities for query clustering and similarity analysis."""

import re
from typing import Any

import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import ArrayType, DoubleType


def _normalize_index_name(index_name: str) -> str:
    """Convert an index name to Databricks-compatible format.

    Supports Unity Catalog full names: catalog.schema.index_name.
    Each name segment may contain only alphanumerics and underscores.
    """

    parts = [part for part in index_name.split(".") if part]
    normalized_parts: list[str] = []
    for part in parts:
        normalized_part = re.sub(r"[^0-9A-Za-z_]", "_", part)
        normalized_part = re.sub(r"_+", "_", normalized_part).strip("_")
        if normalized_part:
            normalized_parts.append(normalized_part)

    normalized = ".".join(normalized_parts)
    if not normalized:
        msg = "Index name is empty after normalization"
        raise ValueError(msg)
    return normalized


def _resolve_existing_index_name(
    vsc: VectorSearchClient,
    endpoint_name: str,
    requested_index_name: str,
) -> str:
    """Resolve a requested index name to an existing index on the endpoint."""

    normalized = _normalize_index_name(requested_index_name)
    if "." in normalized:
        return normalized

    # Backward compatibility for old style: catalog__schema__index_name
    if "__" in requested_index_name:
        raw_parts = requested_index_name.split("__", maxsplit=2)
        if len(raw_parts) == 3:
            maybe_full = ".".join(raw_parts)
            normalized_maybe_full = _normalize_index_name(maybe_full)
            if "." in normalized_maybe_full:
                return normalized_maybe_full

    try:
        payload = vsc.list_indexes(endpoint_name)
    except Exception:
        return normalized

    indexes = payload.get("vector_indexes", []) if isinstance(payload, dict) else []
    existing_names = [idx.get("name") for idx in indexes if isinstance(idx, dict)]
    existing_names = [name for name in existing_names if isinstance(name, str)]

    exact_suffix_matches = [
        name for name in existing_names if name.endswith(f".{normalized}")
    ]
    if exact_suffix_matches:
        return exact_suffix_matches[0]

    compact_requested = normalized.replace("__", "_")
    for name in existing_names:
        compact_name = name.replace(".", "_")
        if compact_name.endswith(compact_requested):
            return name

    return normalized


def generate_embeddings(
    df: DataFrame,
    text_column: str,
    embedding_endpoint: str,
    output_column: str = "embedding",
) -> DataFrame:
    """Generate embeddings for text data using Databricks embedding endpoint.

    Args:
        df: DataFrame with text data
        text_column: Name of column containing text
        embedding_endpoint: Databricks embedding endpoint name
        output_column: Name for output embedding column

    Returns:
        DataFrame with embeddings added
    """
    workspace_client = WorkspaceClient()
    token = workspace_client.tokens.create(lifetime_seconds=1200).token_value
    serving_base_url = f"{workspace_client.config.host.rstrip('/')}/serving-endpoints"

    @pandas_udf(ArrayType(DoubleType()))
    def embed_text(texts: pd.Series) -> pd.Series:
        """UDF to generate embeddings for a batch of texts."""
        from openai import OpenAI

        # Lazily initialize per worker process to avoid pickling client internals.
        if not hasattr(embed_text, "_client"):
            embed_text._client = OpenAI(
                api_key=token,
                base_url=serving_base_url,
            )

        client = embed_text._client
        embeddings = []
        batch_size = 100  # Process in batches to avoid rate limits

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size].tolist()
            try:
                response = client.embeddings.create(
                    model=embedding_endpoint, input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error generating embeddings: {e}")
                embeddings.extend([[0.0] * 1024] * len(batch))

        return pd.Series(embeddings)

    return df.withColumn(output_column, embed_text(col(text_column)))


def create_vector_search_index(
    spark: SparkSession,
    source_table: str,
    index_name: str,
    embedding_column: str,
    vector_search_endpoint: str,
    text_columns: list[str] | None = None,
) -> str:
    """Create a Databricks Vector Search index.

    Args:
        spark: SparkSession
        source_table: Fully qualified source table name
        index_name: Name for the vector search index
        embedding_column: Name of embedding column
        vector_search_endpoint: Vector search endpoint name
        text_columns: List of text columns to include in index
    """
    from databricks.vector_search.client import VectorSearchClient

    normalized_index_name = _normalize_index_name(index_name)

    # If caller passed only short index name, expand to full UC name.
    if "." not in normalized_index_name and source_table.count(".") >= 2:
        source_parts = source_table.split(".")
        normalized_index_name = (
            f"{source_parts[0]}.{source_parts[1]}.{normalized_index_name}"
        )

    vsc = VectorSearchClient()

    try:
        vsc.get_endpoint(vector_search_endpoint)
    except Exception:
        print(f"Creating vector search endpoint: {vector_search_endpoint}")
        vsc.create_endpoint(name=vector_search_endpoint)

    try:
        vsc.create_delta_sync_index(
            endpoint_name=vector_search_endpoint,
            source_table_name=source_table,
            index_name=normalized_index_name,
            pipeline_type="TRIGGERED",
            primary_key="row_id",
            embedding_dimension=1024,  # For databricks-gte-large-en
            embedding_vector_column=embedding_column,
            columns=text_columns,
        )
        print(f"Created vector search index: {normalized_index_name}")
    except Exception as e:
        print(f"Index creation error (may already exist): {e}")

    return normalized_index_name


def search_similar_queries(
    vector_search_endpoint: str,
    index_name: str,
    query_text: str,
    embedding_endpoint: str,
    num_results: int = 10,
) -> pd.DataFrame:
    """Search for similar queries using vector search.

    Args:
        vector_search_endpoint: Vector search endpoint name
        index_name: Vector search index name
        query_text: Query text to search for
        embedding_endpoint: Embedding endpoint name
        num_results: Number of results to return

    Returns:
        DataFrame with similar queries and scores
    """
    from openai import OpenAI

    w = WorkspaceClient()
    token = w.tokens.create(lifetime_seconds=1200).token_value
    client = OpenAI(
        api_key=token, base_url=f"{w.config.host.rstrip('/')}/serving-endpoints"
    )

    vsc = VectorSearchClient()
    normalized_index_name = _resolve_existing_index_name(
        vsc,
        endpoint_name=vector_search_endpoint,
        requested_index_name=index_name,
    )

    response = client.embeddings.create(model=embedding_endpoint, input=[query_text])
    query_embedding = response.data[0].embedding

    results = vsc.get_index(
        endpoint_name=vector_search_endpoint,
        index_name=normalized_index_name,
    ).similarity_search(
        query_vector=query_embedding,
        columns=["query_text", "query_category", "cost"],
        num_results=num_results,
    )

    if results and "data_array" in results:
        return pd.DataFrame(results["data_array"])
    else:
        return pd.DataFrame()


def cluster_queries_kmeans(
    embeddings_df: pd.DataFrame,
    embedding_column: str = "embedding",
    n_clusters: int = 10,
) -> pd.DataFrame:
    """Cluster queries using K-means on embeddings.

    Args:
        embeddings_df: DataFrame with embeddings
        embedding_column: Name of embedding column
        n_clusters: Number of clusters

    Returns:
        DataFrame with cluster assignments
    """
    import numpy as np
    from sklearn.cluster import KMeans

    # Convert embeddings to numpy array
    embeddings = np.array(embeddings_df[embedding_column].tolist())

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)

    # Add cluster assignments
    result_df = embeddings_df.copy()
    result_df["cluster_id"] = clusters

    return result_df


def analyze_clusters(
    clustered_df: pd.DataFrame, include_columns: list[str] | None = None
) -> dict[int, dict[str, Any]]:
    """Analyze characteristics of each cluster.

    Args:
        clustered_df: DataFrame with cluster assignments
        include_columns: Additional columns to aggregate

    Returns:
        Dictionary mapping cluster_id to cluster statistics
    """
    if include_columns is None:
        include_columns = ["cost", "total_tokens", "latency_ms", "quality_score"]

    cluster_analysis = {}

    for cluster_id in sorted(clustered_df["cluster_id"].unique()):
        cluster_data = clustered_df[clustered_df["cluster_id"] == cluster_id]

        analysis = {
            "cluster_id": int(cluster_id),
            "size": len(cluster_data),
            "percentage": (len(cluster_data) / len(clustered_df)) * 100,
        }

        # Category distribution
        if "query_category" in cluster_data.columns:
            analysis["categories"] = cluster_data["query_category"].value_counts().to_dict()
            analysis["dominant_category"] = cluster_data["query_category"].mode()[0]

        # Model distribution
        if "model" in cluster_data.columns:
            analysis["models"] = cluster_data["model"].value_counts().to_dict()
            analysis["dominant_model"] = cluster_data["model"].mode()[0]

        # Numeric aggregations
        for col in include_columns:
            if col in cluster_data.columns:
                analysis[f"{col}_mean"] = cluster_data[col].mean()
                analysis[f"{col}_median"] = cluster_data[col].median()
                analysis[f"{col}_std"] = cluster_data[col].std()

        # Sample queries
        if "query_text" in cluster_data.columns:
            analysis["sample_queries"] = cluster_data["query_text"].head(3).tolist()

        cluster_analysis[cluster_id] = analysis

    return cluster_analysis


def find_cluster_optimization_opportunities(
    cluster_analysis: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Identify optimization opportunities within clusters.

    Args:
        cluster_analysis: Output from analyze_clusters()

    Returns:
        List of optimization recommendations per cluster
    """
    opportunities = []

    for cluster_id, analysis in cluster_analysis.items():
        if "models" not in analysis or "cost_mean" not in analysis:
            continue

        # Check if multiple models serve this cluster
        models = analysis["models"]
        if len(models) > 1:
            # Find if there's cost variation across models
            opportunity = {
                "cluster_id": cluster_id,
                "size": analysis["size"],
                "dominant_category": analysis.get("dominant_category", "unknown"),
                "current_models": models,
                "avg_cost": analysis["cost_mean"],
                "avg_quality": analysis.get("quality_score_mean", None),
            }

            # Simple heuristic: if cluster is large and has cost variation
            if analysis["size"] > 50: # Arbitrary threshold for "large" cluster
                opportunity["recommendation"] = (
                    f"Analyze model performance in this cluster of {analysis['size']} queries. "
                    f"Multiple models ({len(models)}) are being used."
                )
                opportunities.append(opportunity)

    return opportunities


def calculate_embedding_similarity(
    embedding1: list[float], embedding2: list[float]
) -> float:
    """Calculate cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score (0-1)
    """
    import numpy as np

    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def visualize_clusters_2d(
    embeddings_df: pd.DataFrame, embedding_column: str = "embedding"
) -> tuple[pd.DataFrame, Any]:
    """Reduce embeddings to 2D using UMAP for visualization.

    Args:
        embeddings_df: DataFrame with embeddings and cluster assignments
        embedding_column: Name of embedding column

    Returns:
        Tuple of (DataFrame with 2D coordinates, UMAP model)
    """
    import numpy as np
    from umap import UMAP

    embeddings = np.array(embeddings_df[embedding_column].tolist())

    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords_2d = reducer.fit_transform(embeddings)

    result_df = embeddings_df.copy()
    result_df["x"] = coords_2d[:, 0]
    result_df["y"] = coords_2d[:, 1]

    return result_df, reducer

def delete_vector_search_endpoint(vector_search_endpoint: str) -> None:
    """Delete a Databricks Vector Search endpoint.

    Args:
        vector_search_endpoint: Name of the vector search endpoint to delete
    """
    vsc = VectorSearchClient()
    try:
        vsc.delete_endpoint(vector_search_endpoint)
        print(f"Deleted vector search endpoint: {vector_search_endpoint}")
    except Exception as e:
        print(f"Error deleting endpoint (may not exist): {e}")
