"""
Database module for OpenEvolve API

Provides SQLite-based persistent storage for teams, gauntlets, and settings.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict, List
from contextlib import contextmanager
import structlog

logger = structlog.get_logger()

# Database path
DB_PATH = Path(__file__).parent / "data" / "openevolve.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_db():
    """Get database connection context manager."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Initialize database with required tables."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Teams table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                members TEXT NOT NULL,  -- JSON array
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                user_id TEXT DEFAULT 'anonymous',
                tenant_id TEXT DEFAULT 'default'
            )
        """)
        
        # Gauntlets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gauntlets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                rounds TEXT NOT NULL,  -- JSON array
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                user_id TEXT DEFAULT 'anonymous',
                tenant_id TEXT DEFAULT 'default'
            )
        """)
        
        # Settings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        logger.info("database_initialized", db_path=str(DB_PATH))


# Team storage operations

def save_team(team_id: str, team_data: Dict[str, Any]) -> None:
    """Save or update a team."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO teams (
                id, name, description, members, created_at, updated_at, user_id, tenant_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            team_id,
            team_data.get("name", ""),
            team_data.get("description", ""),
            json.dumps(team_data.get("members", [])),
            team_data.get("created_at", datetime.utcnow().isoformat()),
            datetime.utcnow().isoformat(),
            team_data.get("user_id", "anonymous"),
            team_data.get("tenant_id", "default")
        ))
        conn.commit()
        logger.debug("team_saved", team_id=team_id)


def get_team(team_id: str) -> Optional[Dict[str, Any]]:
    """Get a team by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM teams WHERE id = ?", (team_id,))
        row = cursor.fetchone()
        if row:
            return _row_to_team(dict(row))
        return None


def get_all_teams() -> List[Dict[str, Any]]:
    """Get all teams."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM teams ORDER BY created_at DESC")
        return [_row_to_team(dict(row)) for row in cursor.fetchall()]


def delete_team(team_id: str) -> bool:
    """Delete a team."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM teams WHERE id = ?", (team_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.debug("team_deleted", team_id=team_id)
        return deleted


def _row_to_team(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert database row to team dict."""
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "members": json.loads(row["members"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "user_id": row["user_id"],
        "tenant_id": row["tenant_id"]
    }


# Gauntlet storage operations

def save_gauntlet(gauntlet_id: str, gauntlet_data: Dict[str, Any]) -> None:
    """Save or update a gauntlet."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO gauntlets (
                id, name, description, rounds, created_at, updated_at, user_id, tenant_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            gauntlet_id,
            gauntlet_data.get("name", ""),
            gauntlet_data.get("description", ""),
            json.dumps(gauntlet_data.get("rounds", [])),
            gauntlet_data.get("created_at", datetime.utcnow().isoformat()),
            datetime.utcnow().isoformat(),
            gauntlet_data.get("user_id", "anonymous"),
            gauntlet_data.get("tenant_id", "default")
        ))
        conn.commit()
        logger.debug("gauntlet_saved", gauntlet_id=gauntlet_id)


def get_gauntlet(gauntlet_id: str) -> Optional[Dict[str, Any]]:
    """Get a gauntlet by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM gauntlets WHERE id = ?", (gauntlet_id,))
        row = cursor.fetchone()
        if row:
            return _row_to_gauntlet(dict(row))
        return None


def get_all_gauntlets() -> List[Dict[str, Any]]:
    """Get all gauntlets."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM gauntlets ORDER BY created_at DESC")
        return [_row_to_gauntlet(dict(row)) for row in cursor.fetchall()]


def delete_gauntlet(gauntlet_id: str) -> bool:
    """Delete a gauntlet."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM gauntlets WHERE id = ?", (gauntlet_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.debug("gauntlet_deleted", gauntlet_id=gauntlet_id)
        return deleted


def _row_to_gauntlet(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert database row to gauntlet dict."""
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "rounds": json.loads(row["rounds"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "user_id": row["user_id"],
        "tenant_id": row["tenant_id"]
    }


# Settings storage operations

def get_setting(key: str, default: Any = None) -> Any:
    """Get a setting value."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            try:
                return json.loads(row["value"])
            except json.JSONDecodeError:
                return row["value"]
        return default


def set_setting(key: str, value: Any) -> None:
    """Set a setting value."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO settings (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (
            key,
            json.dumps(value),
            datetime.utcnow().isoformat()
        ))
        conn.commit()
        logger.debug("setting_saved", key=key)


# Initialize on module load
init_db()
