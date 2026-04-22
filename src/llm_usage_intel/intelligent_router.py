"""Intelligent routing system with budget awareness."""

import pandas as pd
from pyspark.sql import SparkSession

from llm_usage_intel.budget_manager import BudgetManager
from llm_usage_intel.classifier import classify_query_category, extract_query_features
from llm_usage_intel.vector_search import search_similar_queries


class IntelligentRouter:
    """Routes LLM requests based on query patterns, budget, and optimization rules."""

    def __init__(
        self,
        budget_manager: BudgetManager,
        cluster_analysis: dict,
        embedding_endpoint: str,
        vector_search_index: str,
    ):
        """Initialize intelligent router.

        Args:
            budget_manager: BudgetManager instance
            cluster_analysis: Cluster analysis from HW2
            embedding_endpoint: Databricks embedding endpoint
            vector_search_index: Vector search index name
        """
        self.budget_manager = budget_manager
        self.cluster_analysis = cluster_analysis
        self.embedding_endpoint = embedding_endpoint
        self.vector_search_index = vector_search_index

        # Model tiers by cost (from cheapest to most expensive)
        self.model_tiers = {
            "critical": [
                "gpt-3.5-turbo",  # Cheapest
                "claude-3-haiku",
            ],
            "economy": [
                "gpt-3.5-turbo",
                "claude-3-haiku",
                "gpt-4o-mini",
            ],
            "standard": [
                "gpt-4o-mini",
                "gpt-3.5-turbo",
                "claude-3-sonnet",
                "gpt-4",
            ],
            "premium": [
                "gpt-4",
                "claude-3-opus",
                "gpt-4-turbo",
                "claude-3-sonnet",
            ],
        }

        # Model costs (per 1M tokens) - update with actual pricing
        self.model_costs = {
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4": {"input": 30.00, "output": 60.00},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
            "claude-3-sonnet": {"input": 3.00, "output": 15.00},
            "claude-3-opus": {"input": 15.00, "output": 75.00},
        }

    def estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int = 500
    ) -> float:
        """Estimate cost for a request.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Estimated output tokens

        Returns:
            Estimated cost in USD
        """
        if model not in self.model_costs:
            # Default to mid-range estimate
            return 0.015

        costs = self.model_costs[model]
        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (output_tokens / 1_000_000) * costs["output"]
        return input_cost + output_cost

    def route(
        self,
        query: str,
        user_id: str,
        spark: SparkSession,
        user_tier: str = "standard",
        quality_threshold: float = 4.0,
    ) -> dict:
        """Route query to optimal model considering budget and quality.

        Args:
            query: User query
            user_id: User identifier
            spark: SparkSession for budget lookup
            user_tier: User tier (premium, standard, free)
            quality_threshold: Minimum acceptable quality score

        Returns:
            Routing decision with model, cost, reasoning
        """
        # 1. Extract query features
        features = extract_query_features(query)
        category = features["category"]
        estimated_tokens = features["token_count"] * 4  # Rough estimate

        # 2. Get budget status
        budget_status = self.budget_manager.get_budget_status(spark)
        budget_tier = self.budget_manager.get_recommended_model_tier(spark)

        # 3. Search for similar historical queries
        try:
            similar_queries = search_similar_queries(
                index_name=self.vector_search_index,
                query_text=query,
                embedding_endpoint=self.embedding_endpoint,
                num_results=5,
            )
            avg_similarity = (
                similar_queries["score"].mean() if len(similar_queries) > 0 else 0.0
            )
        except Exception:
            similar_queries = pd.DataFrame()
            avg_similarity = 0.0

        # 4. Get cluster-based recommendation
        cluster_recommendation = self._get_cluster_recommendation(
            category, quality_threshold
        )

        # 5. Apply routing logic with budget consideration
        routing_decision = self._make_routing_decision(
            query=query,
            features=features,
            category=category,
            user_tier=user_tier,
            budget_tier=budget_tier,
            budget_status=budget_status,
            cluster_recommendation=cluster_recommendation,
            similar_queries=similar_queries,
            avg_similarity=avg_similarity,
            quality_threshold=quality_threshold,
        )

        # 6. Validate budget
        estimated_cost = self.estimate_cost(
            routing_decision["model"], estimated_tokens
        )

        # If budget critical and cost too high, force cheaper model
        if budget_tier == "critical" and estimated_cost > 0.01:
            routing_decision.update(
                {
                    "model": self.model_tiers["critical"][0],
                    "reasoning": f"BUDGET CRITICAL: Forced to cheapest model. {routing_decision['reasoning']}",
                    "budget_override": True,
                }
            )
            estimated_cost = self.estimate_cost(
                routing_decision["model"], estimated_tokens
            )

        # Add cost and budget info
        routing_decision.update(
            {
                "estimated_cost": estimated_cost,
                "estimated_tokens": estimated_tokens,
                "budget_remaining": budget_status["remaining_budget"],
                "budget_utilization": budget_status["utilization_pct"],
                "budget_tier": budget_tier,
                "budget_status": budget_status["status"],
            }
        )

        return routing_decision

    def _get_cluster_recommendation(
        self, category: str, quality_threshold: float
    ) -> dict:
        """Get recommendation from cluster analysis.

        Args:
            category: Query category
            quality_threshold: Minimum quality threshold

        Returns:
            Dictionary with recommended model and stats
        """
        # Find clusters with this category
        matching_clusters = []
        for cluster_id, analysis in self.cluster_analysis.items():
            if (
                analysis.get("dominant_category") == category
                or category in analysis.get("categories", {})
            ):
                matching_clusters.append(
                    {
                        "cluster_id": cluster_id,
                        "avg_cost": analysis.get("cost_mean", 0),
                        "avg_quality": analysis.get("quality_score_mean", 0),
                        "dominant_model": analysis.get("dominant_model", "gpt-4"),
                        "size": analysis.get("size", 0),
                    }
                )

        if not matching_clusters:
            # Default recommendation
            return {
                "recommended_model": "gpt-4",
                "confidence": 0.0,
                "cluster_id": None,
            }

        # Sort by quality (descending) then cost (ascending)
        matching_clusters.sort(
            key=lambda x: (-x["avg_quality"], x["avg_cost"])
        )

        # Find best cluster meeting quality threshold
        for cluster in matching_clusters:
            if cluster["avg_quality"] >= quality_threshold:
                return {
                    "recommended_model": cluster["dominant_model"],
                    "confidence": 0.8,
                    "cluster_id": cluster["cluster_id"],
                    "avg_cost": cluster["avg_cost"],
                    "avg_quality": cluster["avg_quality"],
                }

        # If no cluster meets threshold, use best quality one
        best = matching_clusters[0]
        return {
            "recommended_model": best["dominant_model"],
            "confidence": 0.5,
            "cluster_id": best["cluster_id"],
            "avg_cost": best["avg_cost"],
            "avg_quality": best["avg_quality"],
        }

    def _make_routing_decision(
        self,
        query: str,
        features: dict,
        category: str,
        user_tier: str,
        budget_tier: str,
        budget_status: dict,
        cluster_recommendation: dict,
        similar_queries: pd.DataFrame,
        avg_similarity: float,
        quality_threshold: float,
    ) -> dict:
        """Make final routing decision with all context.

        Args:
            query: Query text
            features: Extracted features
            category: Query category
            user_tier: User tier
            budget_tier: Budget-based tier
            budget_status: Budget status dict
            cluster_recommendation: Cluster-based recommendation
            similar_queries: Similar historical queries
            avg_similarity: Average similarity score
            quality_threshold: Min quality threshold

        Returns:
            Routing decision dict
        """
        reasoning_parts = []

        # Rule 1: Budget Critical - override everything
        if budget_tier == "critical":
            model = self.model_tiers["critical"][0]
            reasoning_parts.append(
                f"Budget CRITICAL ({budget_status['utilization_pct']:.1f}% used)"
            )
            return {
                "model": model,
                "reasoning": " | ".join(reasoning_parts),
                "confidence": 1.0,
                "rule": "budget_critical",
            }

        # Rule 2: Premium user + budget OK = best quality
        if user_tier == "premium" and budget_tier in ["premium", "standard"]:
            available_models = self.model_tiers["premium"]
            model = available_models[0]
            reasoning_parts.append("Premium user: optimizing for quality")
            return {
                "model": model,
                "reasoning": " | ".join(reasoning_parts),
                "confidence": 0.9,
                "rule": "premium_user",
            }

        # Rule 3: Long query (>512 tokens) = model with larger context
        if features["token_count"] > 512:
            if budget_tier in ["premium", "standard"]:
                model = "gpt-4-turbo"  # Has 128k context
            else:
                model = "gpt-3.5-turbo"  # 16k context, cheaper
            reasoning_parts.append(
                f"Long query ({features['token_count']} tokens): using extended context model"
            )
            return {
                "model": model,
                "reasoning": " | ".join(reasoning_parts),
                "confidence": 0.85,
                "rule": "long_query",
            }

        # Rule 4: High similarity + good cluster match = use cluster recommendation
        if (
            avg_similarity > 0.75
            and cluster_recommendation["confidence"] > 0.7
        ):
            model = cluster_recommendation["recommended_model"]

            # But check if model fits budget tier
            if budget_tier == "economy" and model in ["gpt-4", "claude-3-opus"]:
                # Downgrade to cheaper alternative
                model = "gpt-3.5-turbo"
                reasoning_parts.append(
                    f"Budget economy mode: downgraded from {cluster_recommendation['recommended_model']}"
                )
            else:
                reasoning_parts.append(
                    f"Cluster {cluster_recommendation['cluster_id']} pattern (similarity: {avg_similarity:.2f})"
                )

            return {
                "model": model,
                "reasoning": " | ".join(reasoning_parts),
                "confidence": cluster_recommendation["confidence"] * avg_similarity,
                "rule": "cluster_match",
                "cluster_id": cluster_recommendation["cluster_id"],
            }

        # Rule 5: Category-based defaults with budget awareness
        category_defaults = {
            "code_generation": {
                "premium": "gpt-4",
                "standard": "gpt-4o-mini",
                "economy": "gpt-3.5-turbo",
                "critical": "gpt-3.5-turbo",
            },
            "code_debugging": {
                "premium": "gpt-4",
                "standard": "gpt-3.5-turbo",
                "economy": "gpt-3.5-turbo",
                "critical": "gpt-3.5-turbo",
            },
            "explanation": {
                "premium": "claude-3-opus",
                "standard": "claude-3-sonnet",
                "economy": "claude-3-haiku",
                "critical": "claude-3-haiku",
            },
            "writing": {
                "premium": "claude-3-opus",
                "standard": "claude-3-sonnet",
                "economy": "gpt-3.5-turbo",
                "critical": "gpt-3.5-turbo",
            },
        }

        if category in category_defaults:
            model = category_defaults[category].get(
                budget_tier, "gpt-3.5-turbo"
            )
            reasoning_parts.append(
                f"Category: {category} | Budget tier: {budget_tier}"
            )
        else:
            # Default fallback
            model = self.model_tiers[budget_tier][0]
            reasoning_parts.append(
                f"Default for budget tier: {budget_tier}"
            )

        return {
            "model": model,
            "reasoning": " | ".join(reasoning_parts),
            "confidence": 0.6,
            "rule": "category_default",
        }
