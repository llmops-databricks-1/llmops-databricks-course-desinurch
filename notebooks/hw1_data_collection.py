# Databricks notebook source
"""
Homework 1: LLM Usage Data Collection and Analysis

This notebook demonstrates:
1. Loading existing LLM request logs
2. Enriching with query content
3. Data validation and quality checks
4. Initial cost analysis

Uses the llm_usage_intel package for shared utilities.
"""

import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession

from llm_usage_intel.config import get_env, load_config
from llm_usage_intel.cost_analyzer import (
    calculate_cost_metrics,
    calculate_model_efficiency,
    identify_optimization_opportunities,
)
from llm_usage_intel.data_loader import (
    enrich_logs_with_queries,
    generate_synthetic_queries,
    get_request_logs_schema,
    validate_dataset,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# Create Spark session
spark = SparkSession.builder.getOrCreate()

# Load config
env = get_env()
cfg = load_config("../project_config.yml", env)

logger.info(f"Environment: {env}")
logger.info(f"Catalog: {cfg.catalog}")
logger.info(f"Schema: {cfg.schema}")

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.full_schema_name}")
logger.info(f"Schema {cfg.full_schema_name} ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Existing Data

# COMMAND ----------

TABLE_NAME = "logs_20260201"
logs_df = spark.table(f"{cfg.full_schema_name}.{TABLE_NAME}")
logs_df = logs_df.select(
    "data_timestamp",
    "data_user_id",
    "data_cost",
    "data_model",
    "data_user_agent",
    "data_input_tokens",
    "data_output_tokens",
)
df_existing = logs_df.toPandas()
df_existing.rename(
    columns={
        "data_timestamp": "timestamp",
        "data_user_id": "user_id",
        "data_cost": "cost",
        "data_model": "model",
        "data_user_agent": "user_agent",
        "data_input_tokens": "input_tokens",
        "data_output_tokens": "output_tokens",
    },
    inplace=True,
)
df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"])
logger.info(f"Loaded {len(df_existing)} existing records")
logger.info(
    f"Date range: {df_existing['timestamp'].min()} to {df_existing['timestamp'].max()}"
)
logger.info(f"Total cost: ${df_existing['cost'].sum():.4f}")
logger.info(f"Models: {df_existing['model'].unique().tolist()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Generate Query Content
# MAGIC
# MAGIC Use synthetic queries to enrich existing data.

# COMMAND ----------

queries_df = generate_synthetic_queries(n_samples=len(df_existing))

logger.info(f"Generated {len(queries_df)} queries")
logger.info("Category distribution:")
logger.info(queries_df["query_category"].value_counts())
logger.info("\nSample queries:")
for _, row in queries_df.head(5).iterrows():
    logger.info(f"[{row['query_category']}] {row['query_text']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Enrich Existing Data with Queries

# COMMAND ----------

combined_df = enrich_logs_with_queries(df_existing, queries_df)

logger.info(f"Created combined dataset with {len(combined_df)} records")
logger.info(f"Columns: {combined_df.columns.tolist()}")
logger.info("\nSample enriched records:")
logger.info(
    combined_df[
        ["timestamp", "user_id", "model", "query_category", "cost", "quality_score"]
    ].head()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Data Validation

# COMMAND ----------

validation_results = validate_dataset(combined_df)

logger.info("Dataset Validation Results:")
logger.info("-" * 80)
for key, value in validation_results.items():
    if key not in ["missing_values", "category_distribution", "model_distribution"]:
        logger.info(f"{key}: {value}")

logger.info("\nCategory Distribution:")
for cat, count in validation_results["category_distribution"].items():
    logger.info(f"  {cat}: {count}")

logger.info("\nModel Distribution:")
for model, count in validation_results["model_distribution"].items():
    logger.info(f"  {model}: {count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Cost Analysis

# COMMAND ----------

cost_metrics = calculate_cost_metrics(combined_df)

logger.info("Cost Metrics:")
logger.info("-" * 80)
logger.info(f"Total Cost: ${cost_metrics['total_cost']:.4f}")
logger.info(f"Avg Cost per Request: ${cost_metrics['avg_cost_per_request']:.6f}")
logger.info(f"Median Cost per Request: ${cost_metrics['median_cost_per_request']:.6f}")
logger.info(f"Total Tokens: {cost_metrics['total_tokens']:,}")
logger.info(f"Avg Tokens per Request: {cost_metrics['avg_tokens_per_request']:.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cost by Model

# COMMAND ----------

if "cost_by_model" in cost_metrics:
    model_costs = pd.DataFrame(cost_metrics["cost_by_model"])
    model_costs.columns = ["total_cost", "avg_cost", "request_count"]
    model_costs = model_costs.sort_values("total_cost", ascending=False)
    logger.info("\nCost by Model:")
    logger.info(model_costs)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cost by Category

# COMMAND ----------

if "cost_by_category" in cost_metrics:
    category_costs = pd.DataFrame(cost_metrics["cost_by_category"])
    category_costs.columns = ["total_cost", "avg_cost", "request_count"]
    category_costs = category_costs.sort_values("total_cost", ascending=False)
    logger.info("\nCost by Category:")
    logger.info(category_costs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Model Efficiency Analysis

# COMMAND ----------

efficiency_df = calculate_model_efficiency(combined_df)

logger.info("\nModel Efficiency Ranking:")
logger.info("-" * 80)
logger.info(
    efficiency_df[
        [
            "model",
            "cost_mean",
            "quality_score_mean",
            "latency_ms_mean",
            "quality_per_dollar",
            "value_score",
        ]
    ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Identify Optimization Opportunities

# COMMAND ----------

opportunities = identify_optimization_opportunities(combined_df)

logger.info(f"\nFound {len(opportunities)} optimization opportunities:")
logger.info("-" * 80)

for i, opp in enumerate(opportunities[:5], 1):  # Show top 5
    logger.info(f"\n{i}. Category: {opp['category']}")
    logger.info(
        f" Current: {opp['current_model']} (${opp['current_cost']:.6f},"
        f" quality: {opp['current_quality']:.2f})"
    )
    logger.info(
        f" Recommended: {opp['recommended_model']} (${opp['recommended_cost']:.6f},"
        f" quality: {opp['recommended_quality']:.2f})"
    )
    logger.info(f"   Cost Reduction: {opp['cost_reduction_pct']:.1f}%")
    logger.info(f"   Quality Impact: {opp['quality_impact']:+.2f}")
    logger.info(f"   Affected Requests: {opp['affected_requests']}")
    logger.info(f"   Potential Savings: ${opp['potential_savings']:.4f}")

total_savings = sum(opp["potential_savings"] for opp in opportunities)
logger.info(f"\nTotal Potential Savings: ${total_savings:.4f}")
logger.info(f"Savings Rate: {(total_savings / cost_metrics['total_cost']) * 100:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Save to UC

# COMMAND ----------

schema = get_request_logs_schema()
spark_df = spark.createDataFrame(combined_df, schema=schema)

# Save request logs
spark_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(cfg.full_request_logs_table)

logger.info(f"Saved to: {cfg.full_request_logs_table}")

# COMMAND ----------

if opportunities:
    opportunities_df = pd.DataFrame(opportunities)
    spark_opportunities = spark.createDataFrame(opportunities_df)

    spark_opportunities.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(cfg.full_optimization_insights_table)

    logger.info(f"Saved optimization insights to: {cfg.full_optimization_insights_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Verify Saved Data

# COMMAND ----------

# verify
saved_df = spark.table(cfg.full_request_logs_table)

logger.info("Verification:")
logger.info(f"Records in table: {saved_df.count()}")
logger.info("Schema:")
saved_df.printSchema()

logger.info("Sample records:")
saved_df.select(
    "timestamp", "user_id", "model", "query_category", "cost", "quality_score"
).show(5)
