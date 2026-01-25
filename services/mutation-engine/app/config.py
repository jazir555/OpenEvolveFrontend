from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MUTATION_")

    host: str = "0.0.0.0"
    port: int = 8002
    max_concurrency: int = 10


settings = Settings()
