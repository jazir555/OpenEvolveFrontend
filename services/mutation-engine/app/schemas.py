from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DesignInput(BaseModel):
    html: str = Field(..., min_length=1)
    css: Optional[str] = None
    metadata: Optional[Dict[str, object]] = None


class MutationRequest(BaseModel):
    design: DesignInput
    mutation_types: Optional[List[str]] = None
    constraints: Optional[Dict[str, object]] = None


class MutationResult(BaseModel):
    html: str
    css: Optional[str] = None
    changes: List[str]


class MutationBatchRequest(BaseModel):
    items: List[MutationRequest]
    max_concurrency: Optional[int] = Field(default=None, ge=1, le=50)


class MutationBatchResponse(BaseModel):
    results: List[MutationResult]
