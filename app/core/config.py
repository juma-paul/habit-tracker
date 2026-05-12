from enum import StrEnum
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
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

    # OpenAI — gpt-4.1 | gpt-4.1-mini | gpt-4.1-nano
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")

    # Anthropic — claude-opus-4-6 | claude-sonnet-4-6 | claude-haiku-4-5-20251001
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-6", alias="ANTHROPIC_MODEL")

    # STT provider — whisper (OpenAI) | groq | elevenlabs
    # Groq: same Whisper model, ~10x faster, ~9x cheaper than OpenAI
    # ElevenLabs: Scribe v1, consolidates to one vendor
    stt_provider: str = Field(default="whisper", alias="STT_PROVIDER")
    whisper_model: str = Field(default="whisper-1", alias="WHISPER_MODEL")
    groq_api_key: SecretStr | None = Field(default=None, alias="GROQ_API_KEY")

    # Voice TTS — ElevenLabs
    # Voice IDs: Rachel=21m00Tcm4TlvDq8ikWAM, George=JBFqnCBsd6RMkjVDRZzb
    # Browse more at elevenlabs.io/voice-library
    elevenlabs_api_key: SecretStr = Field(alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field(
        default="21m00Tcm4TlvDq8ikWAM", alias="ELEVENLABS_VOICE_ID"
    )

    # Auth — shared JWT secret with AuthKit (HS256)
    jwt_secret: str = Field(alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")

    # App
    environment: Environment = Field(alias="ENVIRONMENT")

    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, alias="RATE_LIMIT_PER_MINUTE")

    # CORS — set to your frontend URL(s) in production
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000"], alias="ALLOWED_ORIGINS"
    )

    @property
    def is_dev(self) -> bool:
        return self.environment == "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Convenience alias
settings = get_settings()
