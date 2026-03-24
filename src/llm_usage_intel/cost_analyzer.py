"""Cost analysis and optimization utilities."""

from typing import Any

import pandas as pd


def calculate_cost_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate cost metrics from request logs.

    Args:
        df: DataFrame with request logs

    Returns:
        Dictionary with cost metrics
    """
    metrics = {
        "total_cost": df["cost"].sum(),
        "avg_cost_per_request": df["cost"].mean(),
        "median_cost_per_request": df["cost"].median(),
        "cost_std": df["cost"].std(),
        "total_tokens": df["total_tokens"].sum() if "total_tokens" in df.columns else 0,
        "avg_tokens_per_request": df["total_tokens"].mean()
        if "total_tokens" in df.columns
        else 0,
    }

    # Cost by model
    if "model" in df.columns:
        metrics["cost_by_model"] = df.groupby("model")["cost"].agg(
            ["sum", "mean", "count"]
        ).to_dict()

    # Cost by category
    if "query_category" in df.columns:
        metrics["cost_by_category"] = df.groupby("query_category")["cost"].agg(
            ["sum", "mean", "count"]
        ).to_dict()

    # Cost by user
    if "user_id" in df.columns:
        metrics["cost_by_user"] = df.groupby("user_id")["cost"].agg(
            ["sum", "mean", "count"]
        ).to_dict()

    return metrics


def identify_optimization_opportunities(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Identify cost optimization opportunities.

    Args:
        df: DataFrame with request logs including quality_score

    Returns:
        List of optimization recommendations
    """
    opportunities = []

    # Only analyze successful requests with quality scores
    df_success = df[
        (df["status"] == "success") & (df["quality_score"].notna())
    ].copy()

    if len(df_success) == 0:
        return opportunities

    # Group by category and analyze model usage
    for category in df_success["query_category"].unique():
        category_df = df_success[df_success["query_category"] == category]

        # Get model performance in this category
        model_stats = (
            category_df.groupby("model")
            .agg(
                {
                    "cost": ["mean", "sum", "count"],
                    "quality_score": "mean",
                    "latency_ms": "mean",
                }
            )
            .reset_index()
        )

        # Flatten column names
        model_stats.columns = [
            "_".join(col).strip("_") if col[1] else col[0]
            for col in model_stats.columns.values
        ]

        # Find opportunities where cheaper model has similar quality
        models = model_stats.to_dict("records")
        for i, model_a in enumerate(models):
            for model_b in models[i + 1 :]:
                # Check if one model is cheaper but similar quality
                cost_diff = abs(model_a["cost_mean"] - model_b["cost_mean"])
                quality_diff = abs(
                    model_a["quality_score_mean"] - model_b["quality_score_mean"]
                )

                # If quality difference < 0.3 but cost difference > 50%
                if quality_diff < 0.3 and cost_diff > 0.005:
                    cheaper_model = (
                        model_a
                        if model_a["cost_mean"] < model_b["cost_mean"]
                        else model_b
                    )
                    expensive_model = (
                        model_b
                        if model_a["cost_mean"] < model_b["cost_mean"]
                        else model_a
                    )

                    potential_savings = (
                        expensive_model["cost_sum"] - cheaper_model["cost_mean"]
                        * expensive_model["cost_count"]
                    )

                    opportunities.append(
                        {
                            "category": category,
                            "current_model": expensive_model["model"],
                            "recommended_model": cheaper_model["model"],
                            "current_cost": expensive_model["cost_mean"],
                            "recommended_cost": cheaper_model["cost_mean"],
                            "cost_reduction_pct": (
                                (expensive_model["cost_mean"] - cheaper_model["cost_mean"])
                                / expensive_model["cost_mean"]
                            )
                            * 100,
                            "current_quality": expensive_model["quality_score_mean"],
                            "recommended_quality": cheaper_model["quality_score_mean"],
                            "quality_impact": (
                                cheaper_model["quality_score_mean"]
                                - expensive_model["quality_score_mean"]
                            ),
                            "affected_requests": expensive_model["cost_count"],
                            "potential_savings": potential_savings,
                        }
                    )

    # Sort by potential savings
    opportunities.sort(key=lambda x: x["potential_savings"], reverse=True)

    return opportunities


def calculate_model_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate efficiency metrics for each model.

    Args:
        df: DataFrame with request logs

    Returns:
        DataFrame with efficiency metrics per model
    """
    df_success = df[
        (df["status"] == "success") & (df["quality_score"].notna())
    ].copy()

    if len(df_success) == 0:
        return pd.DataFrame()

    efficiency = (
        df_success.groupby("model")
        .agg(
            {
                "cost": ["mean", "sum", "count"],
                "quality_score": ["mean", "std"],
                "latency_ms": ["mean", "std"],
                "total_tokens": "mean",
            }
        )
        .reset_index()
    )

    # Flatten columns
    efficiency.columns = [
        "_".join(col).strip("_") if col[1] else col[0]
        for col in efficiency.columns.values
    ]

    # Calculate efficiency score: quality / cost
    efficiency["quality_per_dollar"] = (
        efficiency["quality_score_mean"] / efficiency["cost_mean"]
    )

    # Calculate value score: (quality * speed) / cost
    # Lower latency is better, so invert it
    avg_latency = efficiency["latency_ms_mean"].mean()
    efficiency["speed_factor"] = avg_latency / efficiency["latency_ms_mean"]
    efficiency["value_score"] = (
        efficiency["quality_score_mean"] * efficiency["speed_factor"]
    ) / efficiency["cost_mean"]

    return efficiency.sort_values("value_score", ascending=False)
