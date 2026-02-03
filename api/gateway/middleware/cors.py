"""
CORS Configuration Middleware
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Cors
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None

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
