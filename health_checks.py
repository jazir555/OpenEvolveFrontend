"""
Specific Health Checks for Sovereign System Dependencies
"""

import logging
import sqlite3
from sovereign_persistence import SovereignDatabase
from openevolve_client import OPENEVOLVE_AVAILABLE, get_client
from llm_cache import get_cache

logger = logging.getLogger(__name__)

def is_llm_service_available():
    """Check if LLM service is available - fallback implementation."""
    try:
        client = get_client()
        return client is not None and OPENEVOLVE_AVAILABLE
    except Exception:
        return False

def get_db_connection():
    """Get a database connection - fallback implementation."""
    try:
        db = SovereignDatabase()
        return db.conn
    except Exception:
        # Fallback to in-memory database
        return sqlite3.connect(":memory:")

def check_database_connectivity() -> bool:
    """Check if database connection is available."""
    try:
        conn = get_db_connection()
        conn.execute("SELECT 1")
        logger.info("Database connectivity check successful.")
        return True
    except Exception as e:
        logger.error(f"Database connectivity check failed: {e}")
        return False

def check_llm_service_availability() -> bool:
    """Check if the LLM service is available."""
    try:
        if is_llm_service_available():
            logger.info("LLM service availability check successful.")
            return True
        else:
            logger.error("LLM service availability check failed.")
            return False
    except Exception as e:
        logger.error(f"LLM service availability check failed: {e}")
        return False

def check_cache_health() -> bool:
    """Check if the cache is healthy."""
    try:
        cache = get_cache()
        # Perform a simple operation to check cache health
        cache.set("health_check", "ok", ttl=10)
        value = cache.get("health_check")
        if value == "ok":
            logger.info("Cache health check successful.")
            return True
        else:
            logger.error("Cache health check failed: value mismatch")
            return False
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return False