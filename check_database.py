"""Quick check of the mappings database."""
import sqlite3
import os

db_path = "hephaestus_workflow_mappings.db"

if os.path.exists(db_path):
    print(f"Database file exists: {db_path}")
    print(f"File size: {os.path.getsize(db_path)} bytes")
    print()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("Database Schema:")
    print("=" * 80)
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()
    for table in tables:
        print(table[0])
        print()

    print("Indexes:")
    print("=" * 80)
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name")
    indexes = cursor.fetchall()
    for idx in indexes:
        print(idx[0])
        print()

    print("Current Data:")
    print("=" * 80)
    cursor.execute("SELECT COUNT(*) FROM workflow_ticket_mappings")
    count = cursor.fetchone()[0]
    print(f"Total mappings: {count}")

    if count > 0:
        cursor.execute("SELECT workflow_id, ticket_id, ticket_status, created_at, updated_at FROM workflow_ticket_mappings LIMIT 5")
        rows = cursor.fetchall()
        print("\nRecent mappings:")
        for row in rows:
            workflow_id, ticket_id, ticket_status, created_at, updated_at = row
            print(f"  {workflow_id} -> {ticket_id} ({ticket_status})")

    conn.close()
else:
    print(f"Database file not found: {db_path}")
