"""
CORS Configuration Middleware
"""
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from typing import List

load_dotenv()


def get_cors_origins() -> List[str]:
    """Get allowed CORS origins from environment"""
    origins_str = os.getenv("CORS_ORIGINS", '["http://localhost:3000"]')
    try:
        import json
        return json.loads(origins_str)
    except:
        return ["http://localhost:3000"]


def setup_cors(app):
    """
    Setup CORS middleware for the FastAPI application

    Args:
        app: FastAPI application instance
    """
    origins = get_cors_origins()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=os.getenv("CORS_ALLOW_CREDENTIALS", "True").lower() == "true",
        allow_methods=os.getenv("CORS_ALLOW_METHODS", ["*"]),
        allow_headers=os.getenv("CORS_ALLOW_HEADERS", ["*"]),
    )

    return app
