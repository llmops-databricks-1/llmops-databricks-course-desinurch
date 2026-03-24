# Databricks notebook source

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from logs_curator.cost_scraper import scrape_llm_model_costs

# COMMAND ----------
# Load logs table from Unity Catalog

spark = SparkSession.builder.getOrCreate()

TABLE_PATH = "llmops_dev.logs.logs_20260201"

logs_df = spark.table(TABLE_PATH)
print(f"Table loaded: {TABLE_PATH}")
print(f"Rows: {logs_df.count()}")
logs_df.printSchema()

# COMMAND ----------
# Validate expected columns

required_cols = {"data_model", "data_input_tokens", "data_output_tokens"}
missing_cols = sorted(required_cols - set(logs_df.columns))

if missing_cols:
    raise ValueError(
        "Missing required columns in Unity Catalog table: " + ", ".join(missing_cols)
    )

clean_df = logs_df.select(
    F.col("data_model").cast("string").alias("data_model"),
    F.coalesce(F.col("data_input_tokens"), F.lit(0))
    .cast("double")
    .alias("data_input_tokens"),
    F.coalesce(F.col("data_output_tokens"), F.lit(0))
    .cast("double")
    .alias("data_output_tokens"),
).filter(F.col("data_model").isNotNull())

print("Sample records:")
clean_df.show(10, truncate=False)

# COMMAND ----------
# Online pricing reference table (USD per 1M tokens)
# By default we fetch live public prices, then fall back to static values.

pricing_schema = StructType(
    [
        StructField("model_match", StringType(), False),
        StructField("provider", StringType(), False),
        StructField("input_usd_per_1m", DoubleType(), False),
        StructField("output_usd_per_1m", DoubleType(), False),
    ]
)


def _provider_from_model_id(model_id: str) -> str:
    model_id_lower = model_id.lower()
    if any(token in model_id_lower for token in ("openai", "gpt-", "/o1", "/o3")):
        return "openai"
    if any(
        token in model_id_lower
        for token in (
            "anthropic",
            "claude",
            "llama",
            "mistral",
            "cohere",
            "ai21",
            "amazon",
            "bedrock",
        )
    ):
        return "bedrock"
    return "unknown"


def _model_match_from_model_id(model_id: str) -> str:
    if "/" in model_id:
        return model_id.split("/", maxsplit=1)[1].lower()
    return model_id.lower()


fallback_pricing_rows = [
    ("gpt-4.1", "openai", 2.00, 8.00),
    ("gpt-4o", "openai", 5.00, 15.00),
    ("gpt-4o-mini", "openai", 0.15, 0.60),
    ("o1", "openai", 15.00, 60.00),
    ("o3", "openai", 10.00, 40.00),
    ("claude-3.5-sonnet", "bedrock", 3.00, 15.00),
    ("claude-3.7-sonnet", "bedrock", 3.00, 15.00),
    ("claude-3-haiku", "bedrock", 0.25, 1.25),
    ("llama", "bedrock", 0.60, 0.80),
    ("mistral", "bedrock", 2.00, 6.00),
]

pricing_rows: list[tuple[str, str, float, float]] = []
try:
    scraped_costs = scrape_llm_model_costs(timeout=20.0)
    pricing_rows = [
        (
            _model_match_from_model_id(cost.model_id),
            _provider_from_model_id(cost.model_id),
            float(cost.input_cost_per_1m_tokens or 0.0),
            float(cost.output_cost_per_1m_tokens or 0.0),
        )
        for cost in scraped_costs
        if cost.input_cost_per_1m_tokens is not None
        and cost.output_cost_per_1m_tokens is not None
    ]
except Exception as error:
    print(f"Could not fetch live pricing, falling back to static table: {error}")

if not pricing_rows:
    print("Using fallback static pricing table.")
    pricing_rows = fallback_pricing_rows
else:
    print(f"Loaded {len(pricing_rows)} pricing rows from online source.")

pricing_df = spark.createDataFrame(pricing_rows, schema=pricing_schema)

pricing_df.show(truncate=False)

# COMMAND ----------
# Normalize model/provider and estimate request-level costs

normalized_df = clean_df.withColumn(
    "model_lower", F.lower(F.col("data_model"))
).withColumn(
    "provider",
    F.when(
        F.col("model_lower").rlike("openai|gpt-|\\bo1\\b|\\bo3\\b"),
        F.lit("openai"),
    )
    .when(
        F.col("model_lower").rlike(
            "bedrock|anthropic|claude|llama|mistral|cohere|ai21|amazon\\."
        ),
        F.lit("bedrock"),
    )
    .otherwise(F.lit("unknown")),
)

# Join pricing by fuzzy model pattern match.
joined_df = (
    normalized_df.alias("l")
    .join(
        pricing_df.alias("p"),
        (
            (F.col("l.provider") == F.col("p.provider"))
            & (F.expr("l.model_lower LIKE concat('%', lower(p.model_match), '%')"))
        ),
        "left",
    )
    .select(
        F.col("l.data_model").alias("data_model"),
        F.col("l.model_lower").alias("model_lower"),
        F.col("l.provider").alias("provider"),
        F.col("l.data_input_tokens").alias("data_input_tokens"),
        F.col("l.data_output_tokens").alias("data_output_tokens"),
        F.col("p.model_match").alias("pricing_model_match"),
        F.col("p.input_usd_per_1m").alias("input_usd_per_1m"),
        F.col("p.output_usd_per_1m").alias("output_usd_per_1m"),
    )
    .withColumn(
        "input_cost_usd",
        (F.col("data_input_tokens") / F.lit(1_000_000)) * F.col("input_usd_per_1m"),
    )
    .withColumn(
        "output_cost_usd",
        (F.col("data_output_tokens") / F.lit(1_000_000)) * F.col("output_usd_per_1m"),
    )
    .withColumn(
        "total_cost_usd",
        F.coalesce(F.col("input_cost_usd"), F.lit(0.0))
        + F.coalesce(F.col("output_cost_usd"), F.lit(0.0)),
    )
)

print("Requests without a pricing match (update pricing_rows for these models):")
(
    joined_df.filter(
        F.col("input_usd_per_1m").isNull() | F.col("output_usd_per_1m").isNull()
    )
    .groupBy("provider", "data_model")
    .count()
    .orderBy(F.desc("count"))
    .show(100, truncate=False)
)

# COMMAND ----------
# Overall analytics

overall_df = joined_df.agg(
    F.count("*").alias("requests"),
    F.sum("data_input_tokens").alias("input_tokens"),
    F.sum("data_output_tokens").alias("output_tokens"),
    F.sum("total_cost_usd").alias("estimated_total_cost_usd"),
).withColumn(
    "estimated_cost_per_1k_tokens_usd",
    (F.col("estimated_total_cost_usd") * F.lit(1000.0))
    / (F.col("input_tokens") + F.col("output_tokens")),
)

print("Overall cost analytics:")
overall_df.show(truncate=False)

# COMMAND ----------
# Provider comparison (Bedrock vs OpenAI)

provider_df = (
    joined_df.groupBy("provider")
    .agg(
        F.count("*").alias("requests"),
        F.sum("data_input_tokens").alias("input_tokens"),
        F.sum("data_output_tokens").alias("output_tokens"),
        F.sum("total_cost_usd").alias("estimated_total_cost_usd"),
    )
    .withColumn("total_tokens", F.col("input_tokens") + F.col("output_tokens"))
    .withColumn(
        "estimated_cost_per_1k_tokens_usd",
        (F.col("estimated_total_cost_usd") * F.lit(1000.0)) / F.col("total_tokens"),
    )
    .orderBy(F.desc("estimated_total_cost_usd"))
)

print("Provider comparison:")
provider_df.show(truncate=False)

# COMMAND ----------
# Model-level analytics

model_df = (
    joined_df.groupBy("provider", "data_model")
    .agg(
        F.count("*").alias("requests"),
        F.sum("data_input_tokens").alias("input_tokens"),
        F.sum("data_output_tokens").alias("output_tokens"),
        F.avg("input_usd_per_1m").alias("input_usd_per_1m"),
        F.avg("output_usd_per_1m").alias("output_usd_per_1m"),
        F.sum("total_cost_usd").alias("estimated_total_cost_usd"),
    )
    .withColumn("total_tokens", F.col("input_tokens") + F.col("output_tokens"))
    .withColumn(
        "estimated_cost_per_1k_tokens_usd",
        (F.col("estimated_total_cost_usd") * F.lit(1000.0)) / F.col("total_tokens"),
    )
    .orderBy(F.desc("estimated_total_cost_usd"))
)

print("Top model costs:")
model_df.show(50, truncate=False)
