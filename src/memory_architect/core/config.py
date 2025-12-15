import os
from typing import List, Literal, Any
from pathlib import Path
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

# --- Policy Schema Definitions ---

class RetentionPolicy(BaseModel):
    """Defines how long memories persist based on their classification."""
    policy_name: str
    default_ttl_hours: int
    pruning_threshold_score: float = Field(..., ge=0.0, le=100.0, description="Score below which memory is pruned")
    
class PrivacyConfig(BaseModel):
    """Controls PII redaction behavior."""
    pii_entities: List[str] = Field(default_factory=list)
    redaction_mode: Literal["mask", "replace", "hash"] = "mask"

class ModelConfig(BaseModel):
    """LLM and Embedding model paths."""
    llm_model_path: str
    embedding_model: str = "all-MiniLM-L6-v2"
    context_window_limit: int = 4096

# --- Master Configuration ---

class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """
    Custom source to load settings from a policy.yaml file.
    This enables separation of code and policy configuration.
    """
    def get_field_value(self, field: Field, field_name: str) -> tuple[Any, str, bool]:
        encoding = self.config.get('env_file_encoding')
        file_content_json = yaml.safe_load(Path('config/policy.yaml').read_text(encoding))
        field_value = file_content_json.get(field_name)
        return field_value, field_name, False

    def prepare_field_value(self, field_name: str, field: Field, value: Any, value_is_complex: bool) -> Any:
        return value

    def __call__(self) -> dict[str, Any]:
        encoding = self.config.get('env_file_encoding')
        return yaml.safe_load(Path('config/policy.yaml').read_text(encoding))

class SystemSettings(BaseSettings):
    """
    Master configuration class loaded from environment variables 
    and YAML policy files. Validates types at startup.
    """
    environment: str = "development"
    log_level: str = "INFO"
    
    # Nested Configurations
    models: ModelConfig
    retention: RetentionPolicy
    privacy: PrivacyConfig
    
    # ChromaDB Configuration
    chroma_persist_path: str = "./data/chroma_db"
    
    model_config = SettingsConfigDict(
        env_prefix="MEM_ARCH_",
        env_nested_delimiter="__",
        extra="ignore"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple:
        """
        Injects the YAML source into the Pydantic loading chain.
        """
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls), 
            file_secret_settings,
        )
