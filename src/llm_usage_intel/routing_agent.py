"""Production routing agent with MLflow tracing."""

from datetime import datetime
from typing import Any
from uuid import uuid4

import mlflow
from mlflow.entities import SpanType
from pyspark.sql import SparkSession

from llm_usage_intel.budget_manager import BudgetManager
from llm_usage_intel.intelligent_router import IntelligentRouter


class RoutingAgent:
    """Production agent for intelligent LLM request routing with budget awareness."""

    def __init__(
        self,
        router: IntelligentRouter,
        budget_manager: BudgetManager,
        spark: SparkSession,
    ):
        """Initialize routing agent.

        Args:
            router: IntelligentRouter instance
            budget_manager: BudgetManager instance
            spark: SparkSession
        """
        self.router = router
        self.budget_manager = budget_manager
        self.spark = spark

    @mlflow.trace(span_type=SpanType.AGENT, name="routing_agent")
    def predict(
        self,
        query: str,
        user_id: str,
        user_tier: str = "standard",
        quality_threshold: float = 4.0,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Route request to optimal model with budget awareness.

        Args:
            query: User query
            user_id: User identifier
            user_tier: User tier (premium/standard/free)
            quality_threshold: Minimum quality threshold
            session_id: Optional session ID for tracking

        Returns:
            Routing decision with model, cost, reasoning, metadata
        """
        request_id = str(uuid4())
        timestamp = datetime.now()

        # Add trace metadata
        mlflow.set_tag("request_id", request_id)
        mlflow.set_tag("user_id", user_id)
        mlflow.set_tag("user_tier", user_tier)
        mlflow.set_tag("session_id", session_id or "none")

        # Span 1: Check budget status
        with mlflow.start_span("check_budget", span_type=SpanType.TOOL) as span:
            budget_status = self.budget_manager.get_budget_status(self.spark)
            span.set_attribute("budget_remaining", budget_status["remaining_budget"])
            span.set_attribute("budget_utilization", budget_status["utilization_pct"])
            span.set_attribute("budget_status", budget_status["status"])
            span.set_attribute("budget_tier", budget_status.get("severity", "unknown"))

            # Use metric instead of param because this value changes per request.
            mlflow.log_metric("budget_remaining_usd", budget_status["remaining_budget"])
            mlflow.log_metric("budget_utilization_pct", budget_status["utilization_pct"])

        # Span 2: Route request
        with mlflow.start_span("route_request", span_type=SpanType.CHAIN) as span:
            routing_decision = self.router.route(
                query=query,
                user_id=user_id,
                spark=self.spark,
                user_tier=user_tier,
                quality_threshold=quality_threshold,
            )

            span.set_attribute("selected_model", routing_decision["model"])
            span.set_attribute("estimated_cost", routing_decision["estimated_cost"])
            span.set_attribute("confidence", routing_decision["confidence"])
            span.set_attribute("rule", routing_decision["rule"])

            # Tags are mutable and safe across repeated requests in one run.
            mlflow.set_tag("selected_model", routing_decision["model"])
            mlflow.log_metric("routing_confidence", routing_decision["confidence"])
            mlflow.log_metric("estimated_cost_usd", routing_decision["estimated_cost"])

        # Span 3: Record for budget tracking
        with mlflow.start_span("record_budget", span_type=SpanType.TOOL) as span:
            self.budget_manager.record_request(
                spark=self.spark,
                cost=routing_decision["estimated_cost"],
                model=routing_decision["model"],
                user_id=user_id,
                request_id=request_id,
                session_id=session_id,
                query_category=routing_decision.get("category", "unknown"),
                estimated_tokens=routing_decision["estimated_tokens"],
                budget_tier=routing_decision["budget_tier"],
                routing_rule=routing_decision["rule"],
            )
            span.set_attribute("recorded", True)

        # Build response
        response = {
            "request_id": request_id,
            "timestamp": timestamp.isoformat(),
            "session_id": session_id,
            "user_id": user_id,
            "query": query,
            "routing": {
                "model": routing_decision["model"],
                "reasoning": routing_decision["reasoning"],
                "confidence": routing_decision["confidence"],
                "rule": routing_decision["rule"],
                "estimated_cost": routing_decision["estimated_cost"],
                "estimated_tokens": routing_decision["estimated_tokens"],
            },
            "budget": {
                "remaining": routing_decision["budget_remaining"],
                "utilization_pct": routing_decision["budget_utilization"],
                "status": routing_decision["budget_status"],
                "tier": routing_decision["budget_tier"],
            },
            "metadata": {
                "cluster_id": routing_decision.get("cluster_id"),
                "budget_override": routing_decision.get("budget_override", False),
            },
        }

        # Log response to MLflow
        mlflow.log_dict(response, f"routing_response_{request_id}.json")

        return response

    @mlflow.trace(span_type=SpanType.AGENT, name="get_budget_status")
    def get_budget_status(self) -> dict[str, Any]:
        """Get current budget status.

        Returns:
            Budget status dictionary
        """
        return self.budget_manager.get_budget_status(self.spark)

    @mlflow.trace(span_type=SpanType.AGENT, name="get_spending_report")
    def get_spending_report(self) -> dict[str, Any]:
        """Get comprehensive spending report.

        Returns:
            Spending report with user and model breakdowns
        """
        budget_status = self.budget_manager.get_budget_status(self.spark)
        user_spending = self.budget_manager.get_spending_by_user(self.spark, top_n=10)
        model_spending = self.budget_manager.get_spending_by_model(self.spark, top_n=10)

        return {
            "budget": budget_status,
            "top_users": (
                user_spending.to_dict("records") if not user_spending.empty else []
            ),
            "top_models": (
                model_spending.to_dict("records") if not model_spending.empty else []
            ),
        }
