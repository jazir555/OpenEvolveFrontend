from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .config import settings
from .mutation_engine import mutation_engine
from .schemas import MutationBatchRequest, MutationBatchResponse, MutationRequest, MutationResult

app = FastAPI(title="Mutation Engine", version="1.0.0")


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/mutate", response_model=MutationResult)
async def mutate(request: MutationRequest) -> MutationResult:
    try:
        return mutation_engine.mutate(request)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/mutate/batch", response_model=MutationBatchResponse)
async def mutate_batch(request: MutationBatchRequest) -> MutationBatchResponse:
    try:
        concurrency = request.max_concurrency or settings.max_concurrency
        results = await mutation_engine.mutate_batch(request.items, concurrency)
        return MutationBatchResponse(results=results)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": str(exc)})
