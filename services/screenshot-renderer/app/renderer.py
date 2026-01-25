from __future__ import annotations

import asyncio
import base64
import time
from typing import Iterable

from pyppeteer.page import Page
from pyppeteer.browser import Browser

from .browser_pool import browser_pool
from .config import settings
from .schemas import RenderRequest, RenderResponse


async def _handle_request(req, block_types: Iterable[str]) -> None:
    url = req.url.lower()
    resource_type = req.resourceType
    should_block = resource_type in block_types or any(
        pattern in url for pattern in settings.block_url_patterns
    )
    if should_block:
        await req.abort()
        return
    await req.continue_()


async def _configure_page(page: Page, request: RenderRequest) -> None:
    await page.setViewport(
        {
            "width": request.viewport.width,
            "height": request.viewport.height,
            "deviceScaleFactor": request.viewport.device_scale_factor,
        }
    )

    if request.block_resources:
        await page.setRequestInterception(True)

        def on_request(req) -> None:
            asyncio.ensure_future(
                _handle_request(req, set(settings.block_resource_types))
            )

        page.on("request", on_request)


async def _render_with_browser(browser: Browser, request: RenderRequest) -> RenderResponse:
    start_time = time.monotonic()
    page: Page | None = None

    try:
        page = await browser.newPage()
        await _configure_page(page, request)

        wait_until = ["domcontentloaded"]
        if request.wait_for_network_idle:
            wait_until.append("networkidle0")

        await page.setContent(request.html, waitUntil=wait_until)

        if request.wait_for_selector:
            await page.waitForSelector(request.wait_for_selector)

        if request.wait_for_timeout_ms is not None:
            await page.waitFor(request.wait_for_timeout_ms)

        if request.extra_wait_ms:
            await page.waitFor(request.extra_wait_ms)

        image_bytes = await page.screenshot({"type": "png", "fullPage": True})
        duration_ms = int((time.monotonic() - start_time) * 1000)
        encoded = base64.b64encode(image_bytes).decode("ascii")

        return RenderResponse(
            image_base64=encoded,
            mime_type="image/png",
            width=request.viewport.width,
            height=request.viewport.height,
            duration_ms=duration_ms,
        )
    finally:
        if page is not None:
            await page.close()


async def render_single(request: RenderRequest) -> RenderResponse:
    last_error: Exception | None = None
    attempts = max(1, request.retries + 1)

    for _ in range(attempts):
        browser = await browser_pool.acquire()
        try:
            return await _render_with_browser(browser, request)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        finally:
            await browser_pool.release(browser)

    raise last_error or RuntimeError("Failed to render HTML")


async def render_batch(requests: list[RenderRequest], max_concurrency: int) -> list[RenderResponse]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_one(item: RenderRequest) -> RenderResponse:
        async with semaphore:
            return await render_single(item)

    return await asyncio.gather(*[run_one(item) for item in requests])
