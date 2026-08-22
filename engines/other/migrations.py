"""
Sovereign-Grade Problem Decomposition System - Database Migrations

Maps schema version numbers to the list of SQL statements required to reach
that version. The base schema (version 0) is created by
``SovereignDatabase.init_database``; additional migrations can be appended here
as the schema evolves.
"""

from typing import Dict, List

MIGRATIONS: Dict[int, List[str]] = {}
