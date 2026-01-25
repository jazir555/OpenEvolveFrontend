from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .browser_pool import browser_pool
from .config import settings
from .renderer import render_batch, render_single
from .schemas import RenderBatchRequest, RenderBatchResponse, RenderRequest, RenderResponse

app = FastAPI(title="Screenshot Renderer", version="1.0.0")


@app.on_event("startup")
async def startup_event() -> None:
    await browser_pool.init()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await browser_pool.close()


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/render", response_model=RenderResponse)
async def render(request: RenderRequest) -> RenderResponse:
    try:
        return await render_single(request)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/render/batch", response_model=RenderBatchResponse)
async def render_batch_endpoint(request: RenderBatchRequest) -> RenderBatchResponse:
    try:
        concurrency = request.max_concurrency or settings.max_concurrency
        results = await render_batch(request.items, concurrency)
        total = len(results)
        return RenderBatchResponse(results=results, total=total, completed=total)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": str(exc)})
