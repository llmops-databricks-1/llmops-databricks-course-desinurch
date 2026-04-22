"""Budget management for LLM request routing."""

from datetime import datetime
from typing import Any

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


class BudgetManager:
    """Manages monthly budget and enforces spending limits."""

    def __init__(
        self,
        monthly_budget_usd: float = 200_000,
        catalog: str = "llmops_dev",
        schema: str = "arxiv",
    ):
        """Initialize budget manager.

        Args:
            monthly_budget_usd: Total monthly budget in USD
            catalog: Unity Catalog name
            schema: Schema name for budget tracking tables
        """
        self.monthly_budget = monthly_budget_usd
        self.catalog = catalog
        self.schema = schema
        self.budget_table = f"{catalog}.{schema}.llm_budget_tracking"

    def initialize_budget_table(self, spark: SparkSession) -> None:
        """Create budget tracking table if it doesn't exist.

        Args:
            spark: SparkSession
        """
        # Ensure schema exists
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.catalog}.{self.schema}")

        # Create budget tracking table
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.budget_table} (
                timestamp TIMESTAMP,
                month STRING,
                cost DOUBLE,
                model STRING,
                user_id STRING,
                request_id STRING,
                session_id STRING,
                query_category STRING,
                estimated_tokens INT,
                budget_tier STRING,
                routing_rule STRING
            ) USING DELTA
            PARTITIONED BY (month)
        """)

    def get_current_month_spend(self, spark: SparkSession) -> float:
        """Get total spend for current month.

        Args:
            spark: SparkSession

        Returns:
            Total spend in USD for current month
        """
        current_month = datetime.now().strftime("%Y-%m")

        try:
            df = spark.table(self.budget_table)
            current_spend = (
                df.filter(df.month == current_month).agg({"cost": "sum"}).collect()[0][0]
            )
            return float(current_spend) if current_spend else 0.0
        except Exception:
            # Table doesn't exist yet
            return 0.0

    def get_remaining_budget(self, spark: SparkSession) -> float:
        """Get remaining budget for current month.

        Args:
            spark: SparkSession

        Returns:
            Remaining budget in USD
        """
        current_spend = self.get_current_month_spend(spark)
        return self.monthly_budget - current_spend

    def get_budget_utilization(self, spark: SparkSession) -> float:
        """Get budget utilization percentage.

        Args:
            spark: SparkSession

        Returns:
            Utilization percentage (0-100+)
        """
        current_spend = self.get_current_month_spend(spark)
        return (current_spend / self.monthly_budget) * 100

    def is_budget_exceeded(self, spark: SparkSession) -> bool:
        """Check if monthly budget is exceeded.

        Args:
            spark: SparkSession

        Returns:
            True if budget exceeded
        """
        return self.get_remaining_budget(spark) <= 0

    def get_budget_status(self, spark: SparkSession) -> dict[str, Any]:
        """Get comprehensive budget status.

        Args:
            spark: SparkSession

        Returns:
            Dictionary with budget information
        """
        current_spend = self.get_current_month_spend(spark)
        remaining = self.get_remaining_budget(spark)
        utilization = self.get_budget_utilization(spark)

        # Calculate daily average and projection
        current_day = datetime.now().day
        daily_avg = current_spend / current_day if current_day > 0 else 0
        days_in_month = 30  # Simplified
        projected_spend = daily_avg * days_in_month

        # Determine status
        if utilization >= 100:
            status = "EXCEEDED"
            severity = "critical"
        elif utilization >= 90:
            status = "WARNING"
            severity = "high"
        elif utilization >= 75:
            status = "ALERT"
            severity = "medium"
        else:
            status = "OK"
            severity = "low"

        return {
            "month": datetime.now().strftime("%Y-%m"),
            "monthly_budget": self.monthly_budget,
            "current_spend": current_spend,
            "remaining_budget": remaining,
            "utilization_pct": utilization,
            "daily_avg_spend": daily_avg,
            "projected_monthly_spend": projected_spend,
            "projected_over_budget": projected_spend > self.monthly_budget,
            "status": status,
            "severity": severity,
            "days_elapsed": current_day,
            "days_remaining": days_in_month - current_day,
        }

    def record_request(
        self,
        spark: SparkSession,
        cost: float,
        model: str,
        user_id: str,
        **metadata: object,
    ) -> None:
        """Record a request for budget tracking.

        Args:
            spark: SparkSession
            cost: Cost in USD
            model: Model used
            user_id: User ID
            **metadata: Additional metadata to track
        """
        current_month = datetime.now().strftime("%Y-%m")

        record = {
            "timestamp": datetime.now(),
            "month": current_month,
            "cost": float(cost),
            "model": model,
            "user_id": user_id,
            "request_id": metadata.get("request_id"),
            "session_id": metadata.get("session_id"),
            "query_category": metadata.get("query_category"),
            "estimated_tokens": (
                int(metadata["estimated_tokens"])
                if metadata.get("estimated_tokens") is not None
                else None
            ),
            "budget_tier": metadata.get("budget_tier"),
            "routing_rule": metadata.get("routing_rule"),
        }

        # Append to budget tracking table
        budget_record_schema = StructType(
            [
                StructField("timestamp", TimestampType(), True),
                StructField("month", StringType(), True),
                StructField("cost", DoubleType(), True),
                StructField("model", StringType(), True),
                StructField("user_id", StringType(), True),
                StructField("request_id", StringType(), True),
                StructField("session_id", StringType(), True),
                StructField("query_category", StringType(), True),
                StructField("estimated_tokens", IntegerType(), True),
                StructField("budget_tier", StringType(), True),
                StructField("routing_rule", StringType(), True),
            ]
        )
        df = spark.createDataFrame([record], schema=budget_record_schema)
        df.write.format("delta").mode("append").saveAsTable(self.budget_table)

    def get_spending_by_user(self, spark: SparkSession, top_n: int = 10) -> pd.DataFrame:
        """Get top spending users for current month.

        Args:
            spark: SparkSession
            top_n: Number of top users to return

        Returns:
            DataFrame with user spending
        """
        current_month = datetime.now().strftime("%Y-%m")

        try:
            df = spark.table(self.budget_table)
            user_spend = (
                df.filter(df.month == current_month)
                .groupBy("user_id")
                .agg({"cost": "sum", "user_id": "count"})
                .withColumnRenamed("sum(cost)", "total_spend")
                .withColumnRenamed("count(user_id)", "request_count")
                .orderBy("total_spend", ascending=False)
                .limit(top_n)
            )
            return user_spend.toPandas()
        except Exception:
            return pd.DataFrame()

    def get_spending_by_model(self, spark: SparkSession, top_n: int = 10) -> pd.DataFrame:
        """Get spending breakdown by model for current month.

        Args:
            spark: SparkSession
            top_n: Number of models to return

        Returns:
            DataFrame with model spending
        """
        current_month = datetime.now().strftime("%Y-%m")

        try:
            df = spark.table(self.budget_table)
            model_spend = (
                df.filter(df.month == current_month)
                .groupBy("model")
                .agg({"cost": "sum", "model": "count"})
                .withColumnRenamed("sum(cost)", "total_spend")
                .withColumnRenamed("count(model)", "request_count")
                .orderBy("total_spend", ascending=False)
                .limit(top_n)
            )
            return model_spend.toPandas()
        except Exception:
            return pd.DataFrame()

    def should_throttle(self, spark: SparkSession, threshold_pct: float = 95) -> bool:
        """Determine if requests should be throttled due to budget.

        Args:
            spark: SparkSession
            threshold_pct: Budget utilization threshold for throttling

        Returns:
            True if should throttle
        """
        utilization = self.get_budget_utilization(spark)
        return utilization >= threshold_pct

    def get_recommended_model_tier(self, spark: SparkSession) -> str:
        """Get recommended model tier based on budget status.

        Args:
            spark: SparkSession

        Returns:
            Model tier: "premium", "standard", "economy", "critical"
        """
        utilization = self.get_budget_utilization(spark)

        if utilization >= 100:
            return "critical"  # Only cheapest models
        elif utilization >= 90:
            return "economy"  # Prefer cheap models
        elif utilization >= 75:
            return "standard"  # Balanced approach
        else:
            return "premium"  # Can use best models

    def estimate_requests_remaining(
        self, spark: SparkSession, avg_cost_per_request: float
    ) -> int:
        """Estimate how many requests can be made with remaining budget.

        Args:
            spark: SparkSession
            avg_cost_per_request: Average cost per request

        Returns:
            Estimated number of requests possible
        """
        remaining = self.get_remaining_budget(spark)
        if avg_cost_per_request <= 0:
            return 0
        return int(remaining / avg_cost_per_request)
