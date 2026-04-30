from enum import Enum
from functools import lru_cache

from pydantic import Field
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    development = "development"
    production = "production"
    staging = "staging"


class Settings(BaseSettings):
    "Settings loaded from environment variables"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: SecretStr = Field(alias="DATABASE_URL")
    db_pool_min: int = 2
    db_pool_max: int = 10

    # AI provider — set AI_PROVIDER=openai or AI_PROVIDER=anthropic
    ai_provider: str = Field(default="openai", alias="AI_PROVIDER")

    # Control model — set CONTROL_MODEL=freeform or CONTROL_MODEL=graph
    control_model: str = Field(default="freeform", alias="CONTROL_MODEL")

    # OpenAI — gpt-4.1 | gpt-4.1-mini | gpt-4.1-nano
    openai_api_key: SecretStr = Field(alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")

    # Anthropic — claude-opus-4-6 | claude-sonnet-4-6 | claude-haiku-4-5-20251001
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-6", alias="ANTHROPIC_MODEL")

    # Voice (always OpenAI)
    whisper_model: str = "whisper-1"
    tts_model: str = "tts-1"
    tts_voice: str = "nova"

    # Auth — shared JWT secret with AuthKit (HS256)
    jwt_secret: str = Field(alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")

    # App
    environment: Environment = Field(alias="ENVIRONMENT")

    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, alias="RATE_LIMIT_PER_MINUTE")

    @property
    def is_dev(self) -> bool:
        return self.environment == "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Convenience alias
settings = get_settings()
