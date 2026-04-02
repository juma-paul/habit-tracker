from enum import Enum
from functools import lru_cache

from pydantic import Field
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict



class Environment(str, Enum):
    development = "development"
    production = "production"
    staging = "staging"

environment: Environment = Field(alias="ENVIRONMENT")

class Settings(BaseSettings):
    "Settings loaded from environment variables"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database 
    database_url: SecretStr = Field(alias="DATABASE_URL")
    db_pool_min: int = 2
    db_pool_max: int = 10

    # OpenAI
    openai_api_key: SecretStr = Field(alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4o-mini"
    whisper_model: str = "whisper-1"
    tts_model: str = "tts-1"
    tts_voice: str = "nova"

    # App
    environment: Environment = Field(alias="ENVIRONMENT")

    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, alias="RATE_LIMIT_PER_MINUTE")

    @property
    def is_dev(self) -> bool:
        return self.environment == 'development'
    
@lru_cache
def get_settings() -> Settings:
    return Settings()

# Convenience alias
settings = get_settings()