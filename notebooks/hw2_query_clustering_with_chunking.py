# Databricks notebook source
"""
Homework 2: Query Clustering with Vector Search (with Chunking)

This notebook demonstrates:
1. Loading request logs from Homework 1
2. Chunking long queries (from Lecture 2.3)
3. Generating embeddings for queries
4. Creating vector search index
5. Clustering similar queries
6. Analyzing cluster characteristics
7. Identifying optimization opportunities per cluster

Uses the llm_usage_intel package for shared utilities.
"""

import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

from llm_usage_intel.chunking import (
    chunk_with_metadata,
    estimate_tokens,
    preprocess_text,
    should_chunk,
    smart_chunking,
)
from llm_usage_intel.config import get_env, load_config
from llm_usage_intel.vector_search import (
    analyze_clusters,
    cluster_queries_kmeans,
    create_vector_search_index,
    delete_vector_search_endpoint,
    find_cluster_optimization_opportunities,
    generate_embeddings,
    search_similar_queries,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Load Data from Homework 1

# COMMAND ----------

# Create Spark session
spark = SparkSession.builder.getOrCreate()

# Load config
env = get_env()
cfg = load_config("../project_config.yml", env)

logger.info(f"Environment: {env}")
logger.info(f"Loading data from: {cfg.full_request_logs_table}")

# COMMAND ----------

# Load request logs from Homework 1
df_logs = spark.table(cfg.full_request_logs_table)

logger.info(f"Loaded {df_logs.count()} request logs")
logger.info("Schema:")
df_logs.printSchema()

# Show sample
logger.info("\nSample queries:")
df_logs.select("query_text", "query_category", "model", "cost").show(5, truncate=50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Analyze Query Length and Identify Long Queries
# MAGIC
# MAGIC embedding models have token limits (typically 512 tokens).
# MAGIC identify queries that need chunking.

# COMMAND ----------

# Convert to pandas for analysis
df_logs_pd = df_logs.toPandas()

# Preprocess text
logger.info("Preprocessing query text...")
df_logs_pd["query_text_clean"] = df_logs_pd["query_text"].apply(preprocess_text)

# Estimate tokens
df_logs_pd["estimated_tokens"] = df_logs_pd["query_text_clean"].apply(estimate_tokens)

# Identify queries that need chunking
df_logs_pd["needs_chunking"] = df_logs_pd["query_text_clean"].apply(
    lambda x: should_chunk(x, max_tokens=512)
)

logger.info("Query Length Analysis:")
logger.info(f"  Total queries: {len(df_logs_pd)}")
logger.info(f"  Avg tokens: {df_logs_pd['estimated_tokens'].mean():.0f}")
logger.info(f"  Max tokens: {df_logs_pd['estimated_tokens'].max()}")
logger.info(f"  Queries needing chunking: {df_logs_pd['needs_chunking'].sum()}")
logger.info(
    f"  Percentage: {(df_logs_pd['needs_chunking'].sum() / len(df_logs_pd)) * 100:.1f}%"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Apply Chunking Strategy
# MAGIC
# MAGIC Following Lecture 2.3's guidance:
# MAGIC - Use **sentence-based chunking** to preserve semantic boundaries
# MAGIC - Max chunk size: 500 characters (~125 tokens)
# MAGIC - Preserve metadata for each chunk

# COMMAND ----------

# Chunk long queries
logger.info("Chunking long queries...")

all_chunks = []
for idx, row in df_logs_pd.iterrows():
    query_text = row["query_text_clean"]

    if row["needs_chunking"]:
        # Chunk with metadata
        metadata = {
            "original_row_id": idx,
            "user_id": row["user_id"],
            "model": row["model"],
            "cost": row["cost"],
            "query_category": row["query_category"],
            "quality_score": row["quality_score"],
            "total_tokens": row["total_tokens"],
            "latency_ms": row["latency_ms"],
            "is_chunked": True,
        }

        chunks = chunk_with_metadata(
            query_text, metadata, chunk_size=500, strategy="sentence"
        )

        all_chunks.extend(chunks)
    else:
        # No chunking needed - create single chunk
        chunk_data = {
            "text": query_text,
            "original_row_id": idx,
            "user_id": row["user_id"],
            "model": row["model"],
            "cost": row["cost"],
            "query_category": row["query_category"],
            "quality_score": row["quality_score"],
            "total_tokens": row["total_tokens"],
            "latency_ms": row["latency_ms"],
            "chunk_id": 0,
            "total_chunks": 1,
            "chunk_length": len(query_text),
            "estimated_tokens": row["estimated_tokens"],
            "is_chunked": False,
        }
        all_chunks.append(chunk_data)

logger.info(f"Created {len(all_chunks)} chunks from {len(df_logs_pd)} queries")

# COMMAND ----------

df_chunks = pd.DataFrame(all_chunks)

# Rename 'text' to 'query_text' for consistency
df_chunks = df_chunks.rename(columns={"text": "query_text"})

logger.info("\nChunk Statistics:")
logger.info(f"  Total chunks: {len(df_chunks)}")
logger.info(f"  Original queries: {len(df_logs_pd)}")
logger.info(f"  Expansion ratio: {len(df_chunks) / len(df_logs_pd):.2f}x")
logger.info(f"  Avg chunk size: {df_chunks['chunk_length'].mean():.0f} chars")
logger.info(f"  Avg tokens per chunk: {df_chunks['estimated_tokens'].mean():.0f}")

# Show chunked queries
chunked_queries = df_chunks[df_chunks["is_chunked"]]
if len(chunked_queries) > 0:
    logger.info("\nExample of chunked query:")
    example = chunked_queries.iloc[0]
    logger.info(f"  Original row: {example['original_row_id']}")
    logger.info(f"  Chunk {example['chunk_id'] + 1}/{example['total_chunks']}")
    logger.info(f"  Text: {example['query_text'][:150]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Generate Embeddings for Chunks
# MAGIC
# MAGIC Now that queries are properly chunked, we can safely generate embeddings.

# COMMAND ----------

# Convert to Spark DataFrame
df_chunks_spark = spark.createDataFrame(df_chunks)

# Add unique row ID for vector search
df_chunks_spark = df_chunks_spark.withColumn("row_id", monotonically_increasing_id())

# Generate embeddings (this may take a few minutes)
logger.info(f"Generating embeddings using: {cfg.embedding_endpoint}")
logger.info("This may take a few minutes...")

df_embeddings = generate_embeddings(
    df_chunks_spark,
    text_column="query_text",
    embedding_endpoint=cfg.embedding_endpoint,
    output_column="embedding",
)

logger.info("Embeddings generated!")
logger.info(f"Embedding dimension: {len(df_embeddings.first().embedding)}")

# COMMAND ----------

# Save embeddings to Delta table
df_embeddings.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(cfg.full_query_embeddings_table)

logger.info(f"Saved embeddings to: {cfg.full_query_embeddings_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Aggregate Chunks Back to Original Queries
# MAGIC
# MAGIC For clustering, we'll use the first chunk of each original query
# MAGIC (or the only chunk if it wasn't chunked).

# COMMAND ----------

# Load embeddings back
df_embeddings_pd = df_embeddings.toPandas()

# For queries that were chunked, keep only the first chunk for clustering
df_for_clustering = df_embeddings_pd[df_embeddings_pd["chunk_id"] == 0].copy()

logger.info(f"Queries for clustering: {len(df_for_clustering)}")
logger.info(
    f"Original queries: {len(df_for_clustering[~df_for_clustering['is_chunked']])}"
)
logger.info(
    "First chunks of long queries:"
    f" {len(df_for_clustering[df_for_clustering['is_chunked']])}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Test Vector Search with Chunked Queries
# MAGIC
# MAGIC Before clustering, let's test vector search to find similar queries.
# MAGIC This demonstrates how chunking improves search for long queries.

# COMMAND ----------

# Note: This creates a Databricks Vector Search index
# If the endpoint doesn't exist, it will be created
try:
    create_vector_search_index(
        spark=spark,
        source_table=cfg.full_query_embeddings_table,
        index_name=f"{cfg.full_query_embeddings_table}_idx",
        embedding_column="embedding",
        vector_search_endpoint=cfg.vector_search_endpoint,
        text_columns=["query_text", "query_category", "model"],
    )
    logger.info("Vector search index created/updated")
except Exception as e:
    logger.warning(f"Vector search index creation skipped: {e}")
    logger.info("Continuing with clustering...")

# COMMAND ----------

# Test vector search with a long query
test_long_query = """
Write a comprehensive Python function that processes a large dataset,
handles errors gracefully, logs progress to a file, and supports
parallel processing with configurable batch sizes. The function should
also implement retry logic with exponential backoff for failed operations.
"""

logger.info("\nTesting Vector Search with Long Query:")
logger.info("=" * 80)
logger.info(f"Query: {test_long_query.strip()[:100]}...")
logger.info(
    f"Query length: {len(test_long_query)} chars\n"
    f"(~{estimate_tokens(test_long_query)} tokens)"
)

# Check if this would need chunking
if should_chunk(test_long_query):
    logger.info("✓ This query would be chunked (>512 tokens)")
    chunks = smart_chunking(test_long_query, max_chars=500)
    logger.info(f"  Would create {len(chunks)} chunks")
    logger.info(f"  Chunk 1: {chunks[0][:80]}...")
else:
    logger.info("✗ This query doesn't need chunking")

# COMMAND ----------

# Search for similar queries (using the full long query)
try:
    similar_queries = search_similar_queries(
        vector_search_endpoint=cfg.vector_search_endpoint,
        index_name=f"{cfg.full_query_embeddings_table}_idx",
        query_text=test_long_query.strip(),
        embedding_endpoint=cfg.embedding_endpoint,
        num_results=5,
    )

    if not similar_queries.empty:
        logger.info("\nTop 5 Similar Queries Found:")
        logger.info("-" * 80)
        for idx, row in similar_queries.iterrows():
            is_chunked = row.get("is_chunked", False)
            chunk_info = (
                f" [Chunk {row.get('chunk_id', 0) + 1}/{row.get('total_chunks', 1)}]"
                if is_chunked
                else ""
            )
            logger.info(f"\n{idx + 1}. [{row.get('query_category', 'N/A')}]{chunk_info}")
            logger.info(f"   Query: {row['query_text'][:100]}...")
            logger.info(
                f"   Model: {row.get('model', 'N/A')}, Cost: ${row.get('cost', 0):.6f}"
            )
            logger.info(f"   Score: {row.get('score', 0):.3f}")
    else:
        logger.info("No similar queries found (vector search may not be ready yet)")
except Exception as e:
    logger.warning(f"Vector search test skipped: {e}")
    logger.info("This is expected if vector search index is not yet synced")

# COMMAND ----------

# Also test with a short, specific query
test_short_query = "Fix Python TypeError"

logger.info(f"\nTesting with Short Query: '{test_short_query}'")
logger.info(
    f"Query length: {len(test_short_query)} chars\n"
    f"(~{estimate_tokens(test_short_query)} tokens)"
)

try:
    similar_short = search_similar_queries(
        vector_search_endpoint=cfg.vector_search_endpoint,
        index_name=f"{cfg.full_query_embeddings_table}_idx",
        query_text=test_short_query,
        embedding_endpoint=cfg.embedding_endpoint,
        num_results=3,
    )

    if not similar_short.empty:
        logger.info("Top 3 Similar Queries:")
        for idx, row in similar_short.iterrows():
            logger.info(
                f"{idx + 1}. {row['query_text'][:80]}...\n"
                f"   (score: {row.get('score', 0):.3f})"
            )
except Exception:
    logger.info("Vector search test skipped (continuing with clustering)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Cluster Queries Using K-Means

# COMMAND ----------

logger.info(f"Clustering {len(df_for_clustering)} queries...")

# Perform clustering
n_clusters = 10
df_clustered = cluster_queries_kmeans(
    df_for_clustering, embedding_column="embedding", n_clusters=n_clusters
)

logger.info(f"Created {n_clusters} clusters")
logger.info("\nCluster distribution:")
logger.info(df_clustered["cluster_id"].value_counts().sort_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Analyze Cluster Characteristics

# COMMAND ----------

# Analyze each cluster
cluster_analysis = analyze_clusters(
    df_clustered,
    include_columns=["cost", "total_tokens", "latency_ms", "quality_score"],
)

logger.info("Cluster Analysis:")
logger.info("=" * 80)

for cluster_id, analysis in cluster_analysis.items():
    logger.info(f"\nCluster {cluster_id}:")
    logger.info(f"  Size: {analysis['size']} queries ({analysis['percentage']:.1f}%)")
    logger.info(f"  Dominant Category: {analysis.get('dominant_category', 'N/A')}")
    logger.info(f"  Dominant Model: {analysis.get('dominant_model', 'N/A')}")
    logger.info(f"  Avg Cost: ${analysis.get('cost_mean', 0):.6f}")
    logger.info(f"  Avg Quality: {analysis.get('quality_score_mean', 0):.2f}")
    logger.info(f"  Avg Tokens: {analysis.get('total_tokens_mean', 0):.0f}")

    # Check if this cluster has chunked queries
    cluster_data = df_clustered[df_clustered["cluster_id"] == cluster_id]
    chunked_count = (cluster_data["is_chunked"]).sum()
    if chunked_count > 0:
        logger.info(f"  Long queries (chunked): {chunked_count}")

    if "sample_queries" in analysis:
        logger.info("  Sample queries:")
        for i, query in enumerate(analysis["sample_queries"][:2], 1):
            logger.info(f"    {i}. {query[:80]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Identify Optimization Opportunities per Cluster

# COMMAND ----------

# Find optimization opportunities
cluster_opportunities = find_cluster_optimization_opportunities(cluster_analysis)

logger.info(f"\nFound {len(cluster_opportunities)} cluster optimization opportunities:")
logger.info("=" * 80)

for i, opp in enumerate(cluster_opportunities, 1):
    logger.info(f"\n{i}. Cluster {opp['cluster_id']} - {opp['dominant_category']}")
    logger.info(f"   Size: {opp['size']} queries")
    logger.info(f"   Current models: {opp['current_models']}")
    logger.info(f"   Avg cost: ${opp['avg_cost']:.6f}")
    if opp["avg_quality"]:
        logger.info(f"   Avg quality: {opp['avg_quality']:.2f}")
    logger.info(f"   Recommendation: {opp['recommendation']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Save Results

# COMMAND ----------

# Save clustered data
df_clustered_spark = spark.createDataFrame(df_clustered)

clustered_table = f"{cfg.catalog}.{cfg.schema}.llm_query_clusters"
df_clustered_spark.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(clustered_table)

logger.info(f"Saved cluster assignments to: {clustered_table}")

# COMMAND ----------

# Save cluster summary
cluster_summary = []
for cluster_id, analysis in cluster_analysis.items():
    cluster_summary.append(
        {
            "cluster_id": cluster_id,
            "size": analysis["size"],
            "dominant_category": analysis.get("dominant_category", "unknown"),
            "dominant_model": analysis.get("dominant_model", "unknown"),
            "avg_cost": analysis.get("cost_mean", 0),
            "total_cost": analysis.get("cost_mean", 0) * analysis["size"],
            "avg_quality": analysis.get("quality_score_mean", 0),
            "avg_tokens": analysis.get("total_tokens_mean", 0),
        }
    )

cluster_summary_df = pd.DataFrame(cluster_summary)
cluster_summary_spark = spark.createDataFrame(cluster_summary_df)

cluster_summary_table = f"{cfg.catalog}.{cfg.schema}.llm_cluster_summary"
cluster_summary_spark.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(cluster_summary_table)

logger.info(f"Saved cluster summary to: {cluster_summary_table}")

delete_vector_search_endpoint(cfg.vector_search_endpoint)
