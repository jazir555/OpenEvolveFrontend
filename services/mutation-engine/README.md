# Mutation Engine Service

FastAPI service that applies OpenEvolve-inspired mutations to HTML/CSS.

## Run locally

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8002
```
