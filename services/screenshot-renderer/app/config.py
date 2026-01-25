from __future__ import annotations

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCREENSHOT_")

    host: str = "0.0.0.0"
    port: int = 8001
    max_concurrency: int = 10
    max_browsers: int = 2
    headless: bool = True
    chromium_executable: str | None = None
    browser_args: List[str] = Field(
        default_factory=lambda: [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
        ]
    )
    block_resource_types: List[str] = Field(
        default_factory=lambda: ["media", "font"]
    )
    block_url_patterns: List[str] = Field(
        default_factory=lambda: [
            "doubleclick",
            "googlesyndication",
            "adservice",
            "google-analytics",
            "segment.io",
            "facebook.net",
            "hotjar",
        ]
    )


settings = Settings()
