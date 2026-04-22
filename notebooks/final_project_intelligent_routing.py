# Databricks notebook source
"""
Final Project: Intelligent LLM Routing System with Budget Management

This notebook demonstrates a production-ready intelligent routing system that:
1. Routes LLM requests based on query patterns
2. Enforces monthly budget limit (e.g., $200K)
3. Dynamically adjusts routing based on budget utilization
4. Includes full MLflow tracing
5. Ready for production deployment

Architecture:
- Budget Manager: Tracks spending, enforces limits
- Intelligent Router: Makes routing decisions
- Routing Agent: Production wrapper with tracing
- Vector Search: Finds similar historical queries
- Cluster Analysis: Pattern-based optimization
"""

import mlflow
from loguru import logger
from pyspark.sql import SparkSession

from llm_usage_intel.budget_manager import BudgetManager
from llm_usage_intel.config import get_env, load_config
from llm_usage_intel.intelligent_router import IntelligentRouter
from llm_usage_intel.routing_agent import RoutingAgent

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# Create Spark session
spark = SparkSession.builder.getOrCreate()

# Load config
env = get_env()
cfg = load_config("../project_config.yml", env)

# Set MLflow experiment
mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/llm-routing-system")

logger.info(f"Environment: {env}")
logger.info(f"Catalog: {cfg.catalog}")
logger.info(f"Schema: {cfg.schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Cluster Analysis

# COMMAND ----------

cluster_summary_table = f"{cfg.catalog}.{cfg.schema}.llm_cluster_summary"
cluster_summary_df = spark.table(cluster_summary_table)

logger.info(f"Loaded {cluster_summary_df.count()} clusters from HW2")

# Convert to dictionary for router
cluster_analysis = {}
for row in cluster_summary_df.collect():
    cluster_analysis[row.cluster_id] = {
        "cluster_id": row.cluster_id,
        "size": row.size,
        "dominant_category": row.dominant_category,
        "dominant_model": row.dominant_model,
        "cost_mean": row.avg_cost,
        "quality_score_mean": row.avg_quality,
    }

logger.info(f"Cluster analysis loaded: {len(cluster_analysis)} clusters")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Initialize Budget Manager
# MAGIC
# MAGIC Set monthly budget to $200,000

# COMMAND ----------

# Initialize budget manager with $200K monthly budget
budget_manager = BudgetManager(
    monthly_budget_usd=200_000,
    catalog=cfg.catalog,
    schema=cfg.schema,
)

# Create budget tracking table if it doesn't exist
logger.info("Initializing budget tracking table...")
budget_manager.initialize_budget_table(spark)
logger.info(f"✓ Budget tracking table ready: {budget_manager.budget_table}")

# Get current budget status
budget_status = budget_manager.get_budget_status(spark)

logger.info("Budget Status:")
logger.info(f"  Monthly Budget: ${budget_status['monthly_budget']:,.2f}")
logger.info(f"  Current Spend: ${budget_status['current_spend']:,.2f}")
logger.info(f"  Remaining: ${budget_status['remaining_budget']:,.2f}")
logger.info(f"  Utilization: {budget_status['utilization_pct']:.1f}%")
logger.info(f"  Status: {budget_status['status']}")
logger.info(f"  Daily Avg: ${budget_status['daily_avg_spend']:,.2f}")
logger.info(f"  Projected Monthly: ${budget_status['projected_monthly_spend']:,.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Initialize Intelligent Router

# COMMAND ----------

# Initialize router with cluster analysis and budget manager
router = IntelligentRouter(
    budget_manager=budget_manager,
    cluster_analysis=cluster_analysis,
    embedding_endpoint=cfg.embedding_endpoint,
    vector_search_index=cfg.full_query_embeddings_table + "_idx",
)

logger.info("✓ Intelligent Router initialized")
logger.info(f"  Model tiers configured: {list(router.model_tiers.keys())}")
logger.info(f"  Cluster patterns loaded: {len(cluster_analysis)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Initialize Routing Agent with Tracing

# COMMAND ----------

agent = RoutingAgent(
    router=router,
    budget_manager=budget_manager,
    spark=spark,
)

logger.info("✓ Routing Agent initialized with MLflow tracing")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Test Routing System - Various Scenarios

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scenario 1: Standard Code Generation Query

# COMMAND ----------

test_query_1 = "Write a Python function to validate email addresses using regex"

result_1 = agent.predict(
    query=test_query_1,
    user_id="user_test_001",
    user_tier="standard",
    quality_threshold=4.0,
)

logger.info("\n" + "=" * 80)
logger.info("Scenario 1: Standard Code Generation")
logger.info("=" * 80)
logger.info(f"Query: {test_query_1}")
logger.info(f"\nRouting Decision:")
logger.info(f"  Model: {result_1['routing']['model']}")
logger.info(f"  Estimated Cost: ${result_1['routing']['estimated_cost']:.6f}")
logger.info(f"  Confidence: {result_1['routing']['confidence']:.2f}")
logger.info(f"  Reasoning: {result_1['routing']['reasoning']}")
logger.info(f"\nBudget Status:")
logger.info(f"  Remaining: ${result_1['budget']['remaining']:,.2f}")
logger.info(f"  Utilization: {result_1['budget']['utilization_pct']:.1f}%")
logger.info(f"  Tier: {result_1['budget']['tier']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scenario 2: Premium User - Complex Query

# COMMAND ----------

test_query_2 = """
Explain the architectural differences between microservices and monolithic 
applications, including pros/cons, when to use each, and migration strategies.
"""

result_2 = agent.predict(
    query=test_query_2,
    user_id="user_premium_001",
    user_tier="premium",
    quality_threshold=4.5,
)

logger.info("\n" + "=" * 80)
logger.info("Scenario 2: Premium User - Complex Explanation")
logger.info("=" * 80)
logger.info(f"Query: {test_query_2.strip()[:100]}...")
logger.info(f"\nRouting Decision:")
logger.info(f"  Model: {result_2['routing']['model']}")
logger.info(f"  Estimated Cost: ${result_2['routing']['estimated_cost']:.6f}")
logger.info(f"  Confidence: {result_2['routing']['confidence']:.2f}")
logger.info(f"  Reasoning: {result_2['routing']['reasoning']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scenario 3: Long Query (>512 tokens)

# COMMAND ----------

test_query_3 = """
Write a comprehensive Python class that implements a production-ready API client
with the following features: automatic retry logic with exponential backoff,
rate limiting, request/response logging, authentication handling, timeout management,
connection pooling, error handling with custom exceptions, and support for both
sync and async operations. Include detailed docstrings and type hints.
The class should follow SOLID principles and include proper testing utilities.
"""

result_3 = agent.predict(
    query=test_query_3,
    user_id="user_test_002",
    user_tier="standard",
)

logger.info("\n" + "=" * 80)
logger.info("Scenario 3: Long Query (>512 tokens)")
logger.info("=" * 80)
logger.info(f"Query length: {len(test_query_3)} chars, ~{len(test_query_3) // 4} tokens")
logger.info(f"\nRouting Decision:")
logger.info(f"  Model: {result_3['routing']['model']}")
logger.info(f"  Estimated Cost: ${result_3['routing']['estimated_cost']:.6f}")
logger.info(f"  Rule: {result_3['routing']['rule']}")
logger.info(f"  Reasoning: {result_3['routing']['reasoning']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scenario 4: Budget at 95% - Economy Mode

# COMMAND ----------

# Simulate high budget utilization
# In production, this would happen naturally as budget is consumed

logger.info("\n" + "=" * 80)
logger.info("Scenario 4: Budget Near Limit (Simulated)")
logger.info("=" * 80)

# Get recommended tier
budget_tier = budget_manager.get_recommended_model_tier(spark)
logger.info(f"Current budget tier: {budget_tier}")

test_query_4 = "Debug this Python error: KeyError in dictionary access"

result_4 = agent.predict(
    query=test_query_4,
    user_id="user_test_003",
    user_tier="standard",
)

logger.info(f"\nQuery: {test_query_4}")
logger.info(f"Model selected: {result_4['routing']['model']}")
logger.info(f"Reasoning: {result_4['routing']['reasoning']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Compare Routing Strategies

# COMMAND ----------

import pandas as pd

test_queries = [
    "Write Python code to read CSV files",
    "Explain machine learning in simple terms",
    "Debug TypeError in list comprehension",
    "Translate this text to Spanish",
    "Summarize this research paper",
]

comparison_results = []

for query in test_queries:
    # Intelligent routing
    result = agent.predict(
        query=query,
        user_id="comparison_test",
        user_tier="standard",
    )
    
    # Naive routing (always GPT-4)
    naive_cost = router.estimate_cost("gpt-4", len(query) // 4 * 4, 500)
    
    comparison_results.append({
        "query": query[:50] + "...",
        "intelligent_model": result['routing']['model'],
        "intelligent_cost": result['routing']['estimated_cost'],
        "naive_model": "gpt-4",
        "naive_cost": naive_cost,
        "savings": naive_cost - result['routing']['estimated_cost'],
        "savings_pct": ((naive_cost - result['routing']['estimated_cost']) / naive_cost) * 100,
    })

comparison_df = pd.DataFrame(comparison_results)

logger.info("\n" + "=" * 80)
logger.info("Routing Strategy Comparison")
logger.info("=" * 80)
logger.info(comparison_df.to_string(index=False))

logger.info(f"\nTotal Naive Cost: ${comparison_df['naive_cost'].sum():.6f}")
logger.info(f"Total Intelligent Cost: ${comparison_df['intelligent_cost'].sum():.6f}")
logger.info(f"Total Savings: ${comparison_df['savings'].sum():.6f}")
logger.info(f"Average Savings: {comparison_df['savings_pct'].mean():.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Budget Impact Analysis

# COMMAND ----------

# Calculate monthly impact at scale
requests_per_day = 10_000
days_per_month = 30
total_monthly_requests = requests_per_day * days_per_month

# Extrapolate from comparison
avg_naive_cost = comparison_df['naive_cost'].mean()
avg_intelligent_cost = comparison_df['intelligent_cost'].mean()

naive_monthly_cost = avg_naive_cost * total_monthly_requests
intelligent_monthly_cost = avg_intelligent_cost * total_monthly_requests
monthly_savings = naive_monthly_cost - intelligent_monthly_cost

logger.info("\n" + "=" * 80)
logger.info("Monthly Budget Impact (Projected)")
logger.info("=" * 80)
logger.info(f"Assumptions:")
logger.info(f"  Requests per day: {requests_per_day:,}")
logger.info(f"  Total monthly requests: {total_monthly_requests:,}")
logger.info(f"\nNaive Routing (Always GPT-4):")
logger.info(f"  Avg cost per request: ${avg_naive_cost:.6f}")
logger.info(f"  Monthly cost: ${naive_monthly_cost:,.2f}")
logger.info(f"\nIntelligent Routing:")
logger.info(f"  Avg cost per request: ${avg_intelligent_cost:.6f}")
logger.info(f"  Monthly cost: ${intelligent_monthly_cost:,.2f}")
logger.info(f"\nImpact:")
logger.info(f"  Monthly savings: ${monthly_savings:,.2f}")
logger.info(f"  Savings rate: {(monthly_savings / naive_monthly_cost) * 100:.1f}%")
logger.info(f"  Requests within budget: {intelligent_monthly_cost < 200_000}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Get Spending Report

# COMMAND ----------

spending_report = agent.get_spending_report()

logger.info("\n" + "=" * 80)
logger.info("Current Spending Report")
logger.info("=" * 80)
logger.info(f"\nBudget Overview:")
logger.info(f"  Monthly Budget: ${spending_report['budget']['monthly_budget']:,.2f}")
logger.info(f"  Current Spend: ${spending_report['budget']['current_spend']:,.2f}")
logger.info(f"  Remaining: ${spending_report['budget']['remaining_budget']:,.2f}")
logger.info(f"  Utilization: {spending_report['budget']['utilization_pct']:.1f}%")

if spending_report['top_users']:
    logger.info(f"\nTop 5 Spending Users:")
    for i, user in enumerate(spending_report['top_users'][:5], 1):
        logger.info(f"  {i}. {user['user_id']}: ${user['total_spend']:.4f} ({user['request_count']} requests)")

if spending_report['top_models']:
    logger.info(f"\nTop Models by Spend:")
    for i, model in enumerate(spending_report['top_models'][:5], 1):
        logger.info(f"  {i}. {model['model']}: ${model['total_spend']:.4f} ({model['request_count']} requests)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. View Traces in MLflow
# MAGIC

# COMMAND ----------

logger.info("\n" + "=" * 80)
logger.info("MLflow Tracing")
logger.info("=" * 80)
logger.info("✓ All routing decisions are traced in MLflow")
logger.info(f"✓ Experiment: /Users/.../llm-routing-system")
logger.info("✓ Each trace includes:")
logger.info("  - Budget check span")
logger.info("  - Routing decision span")
logger.info("  - Budget recording span")
logger.info("  - Metadata: request_id, user_id, costs, confidence")
