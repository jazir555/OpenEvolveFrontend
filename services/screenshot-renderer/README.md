# Screenshot Renderer Service

FastAPI service that renders HTML into PNG screenshots using a reusable Chromium pool.

## Run locally

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

## Environment

- `SCREENSHOT_MAX_CONCURRENCY` (default: 10)
- `SCREENSHOT_MAX_BROWSERS` (default: 2)
- `SCREENSHOT_CHROMIUM_EXECUTABLE` (optional path override)
