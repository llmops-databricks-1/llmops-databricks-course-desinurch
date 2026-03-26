# Databricks notebook source
"""
Homework 2: Query Clustering with Vector Search

This notebook demonstrates:
1. Loading request logs from Homework 1
2. Generating embeddings for queries
3. Creating vector search index
4. Clustering similar queries
5. Analyzing cluster characteristics
6. Identifying optimization opportunities per cluster

Uses the llm_usage_intel package for shared utilities.
"""

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

from llm_usage_intel.config import get_env, load_config
from llm_usage_intel.vector_search import (
    analyze_clusters,
    cluster_queries_kmeans,
    create_vector_search_index,
    delete_vector_search_endpoint,
    find_cluster_optimization_opportunities,
    generate_embeddings,
    search_similar_queries,
    visualize_clusters_2d,
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
# MAGIC ## 2. Generate Embeddings for Queries
# MAGIC
# MAGIC We'll use the Databricks embedding endpoint to generate embeddings for each query.
# MAGIC This allows us to find semantically similar queries.

# COMMAND ----------

# Add unique row ID for vector search
df_with_id = df_logs.withColumn("row_id", monotonically_increasing_id())

# Generate embeddings (this may take a few minutes)
logger.info(f"Generating embeddings using: {cfg.embedding_endpoint}")
logger.info("This may take a few minutes...")

df_embeddings = generate_embeddings(
    df_with_id,
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

# Create vector search index from embeddings table
query_index_name = f"{cfg.catalog}.{cfg.schema}.{cfg.query_embeddings_table}_index"
query_index_name = create_vector_search_index(
    spark=spark,
    source_table=cfg.full_query_embeddings_table,
    index_name=query_index_name,
    embedding_column="embedding",
    vector_search_endpoint=cfg.vector_search_endpoint,
    text_columns=["query_text", "query_category", "cost", "row_id"],
)
logger.info(f"Using vector index: {query_index_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test Vector Search: Find Similar Queries

# COMMAND ----------

# Example: Find queries similar to a code debugging query
test_query = "Fix this Python error: TypeError in list comprehension"

logger.info(f"\nSearching for queries similar to: '{test_query}'")
logger.info("-" * 80)

# Recompute in this cell to avoid stale notebook variable values from previous runs.
query_index_name = f"{cfg.catalog}.{cfg.schema}.{cfg.query_embeddings_table}_index"
logger.info(f"Searching index: {query_index_name}")

similar_queries = search_similar_queries(
    vector_search_endpoint=cfg.vector_search_endpoint,
    index_name=query_index_name,
    query_text=test_query,
    embedding_endpoint=cfg.embedding_endpoint,
    num_results=5,
)

if not similar_queries.empty:
    logger.info("Top 5 similar queries:")
    for idx, row in similar_queries.iterrows():
        logger.info(
            f"{idx + 1}. [{row.get('query_category', 'N/A')}] {row['query_text']}"
        )
        logger.info(
            f"   Cost: ${row.get('cost', 0):.6f}, Score: {row.get('score', 0):.3f}"
        )
else:
    logger.info(
        "Note: Vector search index may not be ready yet. Proceeding with clustering..."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Cluster Queries Using K-Means
# MAGIC
# MAGIC Group similar queries together to understand usage patterns.

# COMMAND ----------

df_embeddings_pd = df_embeddings.toPandas()

logger.info(f"Clustering {len(df_embeddings_pd)} queries...")

n_clusters = 10
df_clustered = cluster_queries_kmeans(
    df_embeddings_pd, embedding_column="embedding", n_clusters=n_clusters
)

logger.info(f"Created {n_clusters} clusters")
logger.info("\nCluster distribution:")
logger.info(df_clustered["cluster_id"].value_counts().sort_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Analyze Cluster Characteristics

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

    if "sample_queries" in analysis:
        logger.info("  Sample queries:")
        for i, query in enumerate(analysis["sample_queries"][:2], 1):
            logger.info(f"    {i}. {query[:80]}...")

    if "categories" in analysis and len(analysis["categories"]) > 1:
        logger.info(f"  Categories: {analysis['categories']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Identify Optimization Opportunities per Cluster

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
# MAGIC ## 7. Compare Cluster Costs
# MAGIC
# MAGIC Identify which clusters are most expensive and could benefit from optimization.

# COMMAND ----------

# Create cluster cost summary
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
        }
    )

cluster_summary_df = pd.DataFrame(cluster_summary)
cluster_summary_df = cluster_summary_df.sort_values("total_cost", ascending=False)

logger.info("\nCluster Cost Summary (sorted by total cost):")
logger.info(cluster_summary_df.to_string(index=False))

# COMMAND ----------

# Calculate potential savings by cluster
logger.info("\n" + "=" * 80)
logger.info("Potential Optimization by Cluster:")
logger.info("=" * 80)

total_current_cost = cluster_summary_df["total_cost"].sum()
logger.info(f"Current total cost: ${total_current_cost:.4f}")

# Simulate optimization: use cheapest model with acceptable quality
# Assume 30% savings possible by using optimized routing
estimated_savings = 0
for _, cluster in cluster_summary_df.iterrows():
    if cluster["avg_cost"] > 0.01:
        cluster_savings = cluster["total_cost"] * 0.3
        estimated_savings += cluster_savings
        logger.info(
            f"Cluster {cluster['cluster_id']} ({cluster['dominant_category']}): "
            f"Potential savings ${cluster_savings:.4f}"
        )

logger.info(f"\nTotal estimated savings: ${estimated_savings:.4f}")
logger.info(f"Savings rate: {(estimated_savings / total_current_cost) * 100:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Visualize Clusters (2D Projection)
# MAGIC
# MAGIC Use UMAP to reduce embeddings to 2D for visualization.

# COMMAND ----------

# Reduce to 2D for visualization
logger.info("Reducing embeddings to 2D for visualization...")
df_viz, _ = visualize_clusters_2d(df_clustered, embedding_column="embedding")

logger.info("2D projection complete!")
logger.info(f"X range: [{df_viz['x'].min():.2f}, {df_viz['x'].max():.2f}]")
logger.info(f"Y range: [{df_viz['y'].min():.2f}, {df_viz['y'].max():.2f}]")

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Color by cluster
scatter1 = axes[0].scatter(
    df_viz["x"],
    df_viz["y"],
    c=df_viz["cluster_id"],
    cmap="tab10",
    alpha=0.6,
    s=30,
)
axes[0].set_title("Query Clusters (by K-means)", fontsize=14)
axes[0].set_xlabel("UMAP Dimension 1")
axes[0].set_ylabel("UMAP Dimension 2")
plt.colorbar(scatter1, ax=axes[0], label="Cluster ID")

# Plot 2: Color by cost
scatter2 = axes[1].scatter(
    df_viz["x"],
    df_viz["y"],
    c=df_viz["cost"],
    cmap="RdYlGn_r",  # Red = expensive, Green = cheap
    alpha=0.6,
    s=30,
)
axes[1].set_title("Query Cost Distribution", fontsize=14)
axes[1].set_xlabel("UMAP Dimension 1")
axes[1].set_ylabel("UMAP Dimension 2")
plt.colorbar(scatter2, ax=axes[1], label="Cost ($)")

plt.tight_layout()
# display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Save Cluster Assignments

# COMMAND ----------

# Save clustered data back to Delta
df_clustered_spark = spark.createDataFrame(df_clustered)

clustered_table = f"{cfg.catalog}.{cfg.schema}.llm_query_clusters"
df_clustered_spark.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(clustered_table)

logger.info(f"Saved cluster assignments to: {clustered_table}")

# COMMAND ----------

# Save cluster analysis summary
cluster_summary_spark = spark.createDataFrame(cluster_summary_df)

cluster_summary_table = f"{cfg.catalog}.{cfg.schema}.llm_cluster_summary"
cluster_summary_spark.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(cluster_summary_table)

logger.info(f"Saved cluster summary to: {cluster_summary_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Query Pattern Discovery
# MAGIC
# MAGIC Let's examine specific patterns within clusters.

# COMMAND ----------

# Find the most common query patterns per cluster
logger.info("\nQuery Pattern Analysis:")
logger.info("=" * 80)

for cluster_id in sorted(df_clustered["cluster_id"].unique()):
    cluster_data = df_clustered[df_clustered["cluster_id"] == cluster_id]

    logger.info(f"\nCluster {cluster_id}:")

    # Most common category
    if "query_category" in cluster_data.columns:
        top_category = cluster_data["query_category"].mode()[0]
        category_pct = (
            (cluster_data["query_category"] == top_category).sum() / len(cluster_data)
        ) * 100
        logger.info(f"  Primary use case: {top_category} ({category_pct:.1f}%)")

    # Most used model
    if "model" in cluster_data.columns:
        top_model = cluster_data["model"].mode()[0]
        model_pct = ((cluster_data["model"] == top_model).sum() / len(cluster_data)) * 100
        logger.info(f"  Primary model: {top_model} ({model_pct:.1f}%)")

    # Cost efficiency
    avg_cost = cluster_data["cost"].mean()
    avg_quality = cluster_data["quality_score"].mean()
    efficiency = avg_quality / avg_cost if avg_cost > 0 else 0
    logger.info(f"  Efficiency: {efficiency:.0f} quality points per $")

    # Show representative query
    representative = cluster_data.nlargest(1, "quality_score")
    if len(representative) > 0:
        query = representative.iloc[0]["query_text"]
        logger.info(f"  Representative query: {query[:100]}...")

delete_vector_search_endpoint(cfg.vector_search_endpoint)
