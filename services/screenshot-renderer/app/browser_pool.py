from __future__ import annotations

import asyncio
from pyppeteer import launch
from pyppeteer.browser import Browser

from .config import settings


class BrowserPool:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Browser] = asyncio.Queue()
        self._browsers: list[Browser] = []
        self._lock = asyncio.Lock()
        self._initialized = False

    async def init(self) -> None:
        async with self._lock:
            if self._initialized:
                return
            for _ in range(settings.max_browsers):
                browser = await launch(
                    headless=settings.headless,
                    args=settings.browser_args,
                    executablePath=settings.chromium_executable,
                )
                self._browsers.append(browser)
                await self._queue.put(browser)
            self._initialized = True

    async def acquire(self) -> Browser:
        await self.init()
        return await self._queue.get()

    async def release(self, browser: Browser) -> None:
        await self._queue.put(browser)

    async def close(self) -> None:
        for browser in self._browsers:
            await browser.close()
        self._browsers.clear()
        self._initialized = False


browser_pool = BrowserPool()
