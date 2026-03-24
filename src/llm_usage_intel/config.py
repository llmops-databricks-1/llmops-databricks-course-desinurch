"""Configuration management for LLM Usage Intelligence."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class LLMUsageConfig(BaseModel):
    """Configuration for LLM usage analytics."""

    catalog: str = Field(..., description="Unity Catalog name")
    db_schema: str = Field(..., description="Schema name", alias="schema")
    volume: str = Field(..., description="Volume name for data storage")
    
    # Endpoints
    llm_endpoint: str = Field(..., description="LLM endpoint for analysis")
    embedding_endpoint: str = Field(..., description="Embedding endpoint for vector search")
    vector_search_endpoint: str = Field(..., description="Vector search endpoint name")
    warehouse_id: str = Field(..., description="Warehouse ID")
    
    # Table names
    request_logs_table: str = Field(
        default="llm_request_logs",
        description="Table name for LLM request logs"
    )
    query_embeddings_table: str = Field(
        default="llm_query_embeddings",
        description="Table name for query embeddings"
    )
    optimization_insights_table: str = Field(
        default="llm_optimization_insights",
        description="Table name for optimization insights"
    )
    
    model_config = {"populate_by_name": True}

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "LLMUsageConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file
            env: Environment name (dev, acc, prd)

        Returns:
            LLMUsageConfig instance
        """
        if env not in ["prd", "acc", "dev"]:
            msg = f"Invalid environment: {env}. Expected 'prd', 'acc', or 'dev'"
            raise ValueError(msg)

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        if env not in config_data:
            msg = f"Environment '{env}' not found in config file"
            raise ValueError(msg)

        return cls(**config_data[env])

    @property
    def schema(self) -> str:
        """Alias for db_schema for backward compatibility."""
        return self.db_schema

    @property
    def full_schema_name(self) -> str:
        """Get fully qualified schema name."""
        return f"{self.catalog}.{self.db_schema}"

    @property
    def full_volume_path(self) -> str:
        """Get fully qualified volume path."""
        return f"{self.catalog}.{self.schema}.{self.volume}"

    @property
    def full_request_logs_table(self) -> str:
        """Get fully qualified request logs table name."""
        return f"{self.catalog}.{self.schema}.{self.request_logs_table}"

    @property
    def full_query_embeddings_table(self) -> str:
        """Get fully qualified query embeddings table name."""
        return f"{self.catalog}.{self.schema}.{self.query_embeddings_table}"

    @property
    def full_optimization_insights_table(self) -> str:
        """Get fully qualified optimization insights table name."""
        return f"{self.catalog}.{self.schema}.{self.optimization_insights_table}"


def load_config(
    config_path: str = "project_config.yml", env: str = "dev"
) -> LLMUsageConfig:
    """Load project configuration.

    Args:
        config_path: Path to configuration file
        env: Environment name

    Returns:
        LLMUsageConfig instance
    """
    # Handle relative paths from notebooks
    if not Path(config_path).is_absolute():
        # Try to find config in parent directories
        current = Path.cwd()
        for _ in range(3):  # Search up to 3 levels
            candidate = current / config_path
            if candidate.exists():
                config_path = str(candidate)
                break
            current = current.parent

    return LLMUsageConfig.from_yaml(config_path, env)


def get_env() -> str:
    """Get current environment from environment variable or default to 'dev'.

    Returns:
        Environment name (dev, acc, or prd)
    """
    return os.getenv("ENV", "dev")
