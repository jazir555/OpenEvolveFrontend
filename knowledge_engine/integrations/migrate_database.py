"""
Database Migration Tool for Mathematical Knowledge Integration

Handles:
- Schema creation and updates
- Data migration between versions
- Backup and restore
- Validation and repair

Usage:
    python migrate_database.py --init
    python migrate_database.py --migrate --version 2
    python migrate_database.py --backup --file backup.sql
    python migrate_database.py --restore --file backup.sql

Author: OpenEvolve
Created: 2026-01-31
"""

import argparse
import asyncio
import json
import sqlite3
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DatabaseMigration:
    """Database migration manager."""
    
    CURRENT_VERSION = 1
    
    def __init__(self, db_url: str = "sqlite:///math_knowledge.db"):
        self.db_url = db_url
        self.db_path = db_url.replace("sqlite:///", "")
        self.migrations: Dict[int, List[str]] = {
            1: self._get_v1_schema()
        }
    
    def _get_v1_schema(self) -> List[str]:
        """Get version 1 schema."""
        return [
            """
            CREATE TABLE IF NOT EXISTS z3_knowledge_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_type VARCHAR(50) NOT NULL,
                record_hash VARCHAR(64) UNIQUE,
                content TEXT,
                features JSON,
                metadata JSON,
                source_problem VARCHAR(500),
                problem_domain VARCHAR(100),
                confidence FLOAT DEFAULT 1.0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_record_type ON z3_knowledge_records(record_type)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_record_hash ON z3_knowledge_records(record_hash)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_source_problem ON z3_knowledge_records(source_problem)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_problem_domain ON z3_knowledge_records(problem_domain)
            """,
            """
            CREATE TABLE IF NOT EXISTS z3_proof_patterns_db (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                knowledge_record_id INTEGER,
                pattern_signature VARCHAR(255),
                tactic_sequence JSON,
                proof_tree_structure JSON,
                applicable_domains JSON,
                proof_depth INTEGER,
                branching_factor FLOAT,
                effectiveness_score FLOAT DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                average_solving_time FLOAT,
                FOREIGN KEY (knowledge_record_id) REFERENCES z3_knowledge_records(id)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_pattern_signature ON z3_proof_patterns_db(pattern_signature)
            """,
            """
            CREATE TABLE IF NOT EXISTS z3_solver_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id VARCHAR(100) UNIQUE,
                problem_hash VARCHAR(64),
                problem_statement TEXT,
                problem_type VARCHAR(50),
                result_status VARCHAR(20),
                solving_time_ms INTEGER,
                memory_usage_mb FLOAT,
                constraints_used JSON,
                tactics_used JSON,
                strategy_id VARCHAR(100),
                model_assignments JSON,
                proof_steps JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_exec_problem_hash ON z3_solver_executions(problem_hash)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_exec_status ON z3_solver_executions(result_status)
            """,
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            INSERT OR IGNORE INTO schema_version (version) VALUES (1)
            """
        ]
    
    async def init_database(self):
        """Initialize database with current schema."""
        print(f"Initializing database at {self.db_path}...")
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for sql in self.migrations[self.CURRENT_VERSION]:
                cursor.execute(sql)
            
            conn.commit()
            print(f"[OK] Database initialized (version {self.CURRENT_VERSION})")
            
        except Exception as e:
            conn.rollback()
            print(f"[FAIL] Error initializing database: {e}")
            raise
        finally:
            conn.close()
    
    async def get_current_version(self) -> int:
        """Get current database version."""
        if not Path(self.db_path).exists():
            return 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
            row = cursor.fetchone()
            return row[0] if row else 0
        except sqlite3.OperationalError:
            return 0
        finally:
            conn.close()
    
    async def migrate(self, target_version: Optional[int] = None):
        """Migrate database to target version."""
        current = await self.get_current_version()
        target = target_version or self.CURRENT_VERSION
        
        print(f"Current version: {current}")
        print(f"Target version: {target}")
        
        if current == target:
            print("Database is already at target version")
            return
        
        if current > target:
            print("Cannot downgrade database")
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for version in range(current + 1, target + 1):
                if version in self.migrations:
                    print(f"Applying migration {version}...")
                    for sql in self.migrations[version]:
                        cursor.execute(sql)
                    conn.commit()
                    print(f"[OK] Migrated to version {version}")
                else:
                    print(f"[FAIL] No migration found for version {version}")
            
        except Exception as e:
            conn.rollback()
            print(f"[FAIL] Migration failed: {e}")
            raise
        finally:
            conn.close()
    
    async def backup(self, backup_path: str):
        """Backup database."""
        if not Path(self.db_path).exists():
            print(f"Database not found at {self.db_path}")
            return
        
        print(f"Creating backup at {backup_path}...")
        
        # For SQLite, we can just copy the file
        if self.db_url.startswith("sqlite:///"):
            shutil.copy2(self.db_path, backup_path)
            print(f"[OK] Backup created: {backup_path}")
        else:
            print("Backup only supported for SQLite databases")
    
    async def restore(self, backup_path: str):
        """Restore database from backup."""
        if not Path(backup_path).exists():
            print(f"Backup not found at {backup_path}")
            return
        
        print(f"Restoring from {backup_path}...")
        
        # Create backup of current database first
        if Path(self.db_path).exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_backup = f"{self.db_path}.{timestamp}.bak"
            shutil.copy2(self.db_path, current_backup)
            print(f"Current database backed up to {current_backup}")
        
        # Restore
        shutil.copy2(backup_path, self.db_path)
        print(f"[OK] Database restored from {backup_path}")
    
    async def validate(self) -> bool:
        """Validate database integrity."""
        print("Validating database...")
        
        if not Path(self.db_path).exists():
            print(f"[FAIL] Database not found at {self.db_path}")
            return False
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check integrity
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            
            if result[0] != "ok":
                print(f"[FAIL] Integrity check failed: {result[0]}")
                return False
            
            # Check required tables exist
            required_tables = [
                'z3_knowledge_records',
                'z3_proof_patterns_db',
                'z3_solver_executions',
                'schema_version'
            ]
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row[0] for row in cursor.fetchall()}
            
            missing = set(required_tables) - existing_tables
            if missing:
                print(f"[FAIL] Missing tables: {missing}")
                return False
            
            # Get statistics
            stats = {}
            for table in required_tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
            
            print("[OK] Database validation passed")
            print("\nStatistics:")
            for table, count in stats.items():
                print(f"  {table}: {count} rows")
            
            return True
            
        except Exception as e:
            print(f"[FAIL] Validation error: {e}")
            return False
        finally:
            conn.close()
    
    async def reset(self):
        """Reset database (delete all data)."""
        confirm = input("Are you sure you want to reset the database? This will DELETE ALL DATA. (yes/no): ")
        
        if confirm.lower() != "yes":
            print("Cancelled")
            return
        
        print("Resetting database...")
        
        if Path(self.db_path).exists():
            Path(self.db_path).unlink()
        
        await self.init_database()
        print("[OK] Database reset complete")
    
    async def export(self, export_path: str):
        """Export database to JSON."""
        print(f"Exporting database to {export_path}...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "version": await self.get_current_version(),
                "tables": {}
            }
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                cursor.execute(f"SELECT * FROM {table}")
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                export_data["tables"][table] = {
                    "columns": columns,
                    "rows": [dict(zip(columns, row)) for row in rows]
                }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"[OK] Exported to {export_path}")
            
        except Exception as e:
            print(f"[FAIL] Export failed: {e}")
            raise
        finally:
            conn.close()
    
    async def import_data(self, import_path: str):
        """Import database from JSON."""
        print(f"Importing from {import_path}...")
        
        with open(import_path, 'r') as f:
            data = json.load(f)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for table, table_data in data.get("tables", {}).items():
                if table == "schema_version":
                    continue  # Skip schema version
                
                columns = table_data["columns"]
                rows = table_data["rows"]
                
                if not rows:
                    continue
                
                # Clear existing data
                cursor.execute(f"DELETE FROM {table}")
                
                # Insert new data
                placeholders = ", ".join(["?"] * len(columns))
                sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
                
                for row in rows:
                    values = [row.get(col) for col in columns]
                    cursor.execute(sql, values)
            
            conn.commit()
            print(f"[OK] Imported from {import_path}")
            
        except Exception as e:
            conn.rollback()
            print(f"[FAIL] Import failed: {e}")
            raise
        finally:
            conn.close()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Database migration tool")
    parser.add_argument("--db", default="sqlite:///math_knowledge.db",
                       help="Database URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Init command
    subparsers.add_parser("init", help="Initialize database")
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate database")
    migrate_parser.add_argument("--version", type=int, help="Target version")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Backup database")
    backup_parser.add_argument("--file", required=True, help="Backup file path")
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore database")
    restore_parser.add_argument("--file", required=True, help="Backup file path")
    
    # Validate command
    subparsers.add_parser("validate", help="Validate database")
    
    # Reset command
    subparsers.add_parser("reset", help="Reset database")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export to JSON")
    export_parser.add_argument("--file", required=True, help="Export file path")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import from JSON")
    import_parser.add_argument("--file", required=True, help="Import file path")
    
    # Version command
    subparsers.add_parser("version", help="Show version")
    
    args = parser.parse_args()
    
    migrator = DatabaseMigration(args.db)
    
    if args.command == "init":
        await migrator.init_database()
    
    elif args.command == "migrate":
        await migrator.migrate(args.version)
    
    elif args.command == "backup":
        await migrator.backup(args.file)
    
    elif args.command == "restore":
        await migrator.restore(args.file)
    
    elif args.command == "validate":
        valid = await migrator.validate()
        return 0 if valid else 1
    
    elif args.command == "reset":
        await migrator.reset()
    
    elif args.command == "export":
        await migrator.export(args.file)
    
    elif args.command == "import":
        await migrator.import_data(args.file)
    
    elif args.command == "version":
        version = await migrator.get_current_version()
        print(f"Database version: {version}")
        print(f"Current schema version: {DatabaseMigration.CURRENT_VERSION}")
    
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    asyncio.run(main())
