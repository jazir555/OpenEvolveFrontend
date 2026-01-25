from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class Viewport(BaseModel):
    width: int = Field(default=1920, ge=320)
    height: int = Field(default=1080, ge=240)
    device_scale_factor: float = Field(default=1.0, ge=0.5, le=3.0)


class RenderRequest(BaseModel):
    html: str = Field(..., min_length=1)
    viewport: Viewport = Field(default_factory=Viewport)
    wait_for_selector: Optional[str] = None
    wait_for_timeout_ms: Optional[int] = Field(default=None, ge=0)
    wait_for_network_idle: bool = True
    block_resources: bool = True
    extra_wait_ms: Optional[int] = Field(default=None, ge=0)
    retries: int = Field(default=1, ge=0, le=5)


class RenderResponse(BaseModel):
    image_base64: str
    mime_type: str
    width: int
    height: int
    duration_ms: int


class RenderBatchRequest(BaseModel):
    items: List[RenderRequest]
    max_concurrency: Optional[int] = Field(default=None, ge=1, le=50)


class RenderBatchResponse(BaseModel):
    results: List[RenderResponse]
    total: int
    completed: int
