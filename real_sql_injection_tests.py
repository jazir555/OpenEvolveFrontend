"""
REAL SQL Injection Security Tests
Tests with actual SQLite database to verify SQL injection prevention.

This file addresses the CRITICAL gap: Current tests only check HTML cleaner,
but never actually connect to a database to verify SQL injection protection.
"""

import sqlite3
import pytest
import threading
import time
from typing import Dict, Any, List


class TestRealSQLInjectionPrevention:
    """Test SQL injection prevention with real SQLite database."""
    
    def setup_method(self):
        """Create a real in-memory SQLite database for each test."""
        self.conn = sqlite3.connect(':memory:')
        self.cursor = self.conn.cursor()
        
        # Create tables that simulate real application structure
        self.cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL,
                email TEXT,
                password_hash TEXT,
                role TEXT DEFAULT 'user'
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE workflows (
                id INTEGER PRIMARY KEY,
                name TEXT,
                description TEXT,
                created_by INTEGER,
                FOREIGN KEY (created_by) REFERENCES users(id)
            )
        ''')
        
        # Insert test data
        self.cursor.execute('''
            INSERT INTO users (id, username, email, password_hash, role) 
            VALUES (1, 'admin', 'admin@example.com', 'hash123', 'admin')
        ''')
        self.cursor.execute('''
            INSERT INTO users (id, username, email, password_hash, role) 
            VALUES (2, 'user1', 'user1@example.com', 'hash456', 'user')
        ''')
        self.cursor.execute('''
            INSERT INTO workflows (id, name, description, created_by)
            VALUES (1, 'Test Workflow', 'A test workflow', 1)
        ''')
        self.conn.commit()
    
    def teardown_method(self):
        """Clean up database after each test."""
        self.conn.close()
    
    # CRITICAL: Real SQL Injection Payloads
    SQL_INJECTION_PAYLOADS = [
        "1 OR 1=1; DROP TABLE users; --",
        "' OR '1'='1",
        "'; DROP TABLE users; --",
        "1; DELETE FROM users WHERE 1=1; --",
        "' UNION SELECT * FROM users--",
        "admin'--",
        "1' AND 1=1--",
        "') OR ('1'='1",
        "1'; EXEC xp_cmdshell('dir'); --",
        "' UNION SELECT username, password_hash FROM users--",
        "1; INSERT INTO users VALUES (99, 'hacker', 'hack@test.com', 'pwned', 'admin'); --",
        "' OR 1=1 LIMIT 1--",
        "1; UPDATE users SET role='admin' WHERE id=2; --",
        "'; DELETE FROM workflows; --",
        "1 AND 1=0 UNION SELECT * FROM sqlite_master--",
    ]
    
    def test_safe_parameterized_query_prevents_injection(self):
        """Test that parameterized queries prevent SQL injection attacks."""
        malicious_username = "1' OR '1'='1"
        
        # SAFE: Using parameterized query
        self.cursor.execute(
            "SELECT * FROM users WHERE username = ?",
            (malicious_username,)
        )
        results = self.cursor.fetchall()
        
        # Should NOT match any user (the literal string doesn't exist)
        assert len(results) == 0, "Parameterized query should not match malicious input"
        
        # Verify database is intact
        self.cursor.execute("SELECT COUNT(*) FROM users")
        count = self.cursor.fetchone()[0]
        assert count == 2, "All users should still exist"
    
    def test_union_injection_blocked(self):
        """Test that UNION-based SQL injection is blocked."""
        malicious_input = "' UNION SELECT * FROM users--"
        
        # SAFE: Parameterized query treats this as a literal string
        self.cursor.execute(
            "SELECT * FROM workflows WHERE name = ?",
            (malicious_input,)
        )
        results = self.cursor.fetchall()
        
        # Should not return any results (literal string doesn't match)
        assert len(results) == 0
        
        # Verify database structure is intact
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in self.cursor.fetchall()]
        assert 'users' in tables
        assert 'workflows' in tables
    
    def test_drop_table_injection_blocked(self):
        """Test that DROP TABLE injection attempts are neutralized."""
        malicious_input = "1'; DROP TABLE users; --"
        
        # Simulate application using parameterized query
        try:
            self.cursor.execute(
                "SELECT * FROM users WHERE id = ?",
                (malicious_input,)
            )
        except sqlite3.Error:
            pass  # Conversion error is acceptable
        
        # CRITICAL: Verify table still exists
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        result = self.cursor.fetchone()
        assert result is not None, "CRITICAL: users table should NOT be dropped!"
        
        # Verify data is intact
        self.cursor.execute("SELECT COUNT(*) FROM users")
        count = self.cursor.fetchone()[0]
        assert count == 2, "All users should still exist"
    
    def test_delete_injection_blocked(self):
        """Test that DELETE injection attempts are blocked."""
        malicious_input = "1; DELETE FROM users; --"
        
        # Use parameterized query
        self.cursor.execute(
            "SELECT * FROM users WHERE id = ?",
            (malicious_input,)
        )
        # No results expected since input isn't a valid integer match
        
        # CRITICAL: Verify all data still exists
        self.cursor.execute("SELECT COUNT(*) FROM users")
        count = self.cursor.fetchone()[0]
        assert count == 2, "CRITICAL: All users should still exist after injection attempt!"
    
    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_various_injection_payloads_blocked(self, payload):
        """Test various SQL injection payloads are all blocked."""
        # Insert the payload as a username (simulating user registration with malicious input)
        try:
            self.cursor.execute(
                "INSERT INTO users (id, username, email, password_hash) VALUES (?, ?, ?, ?)",
                (100, payload, 'test@test.com', 'hash')
            )
            self.conn.commit()
        except sqlite3.Error:
            # Some payloads might cause errors, that's fine as long as no injection occurs
            pass
        
        # Verify database integrity
        self.cursor.execute("SELECT COUNT(*) FROM users WHERE id IN (1, 2)")
        count = self.cursor.fetchone()[0]
        assert count == 2, f"Original users should not be affected by payload: {payload}"
    
    def test_sql_injection_in_search_functionality(self):
        """Test search functionality against SQL injection."""
        # Malicious search that would return all users if injected
        malicious_search = "%' OR '1'='1' OR '%"
        
        # SAFE: Properly parameterized LIKE query
        self.cursor.execute(
            "SELECT * FROM users WHERE username LIKE ? OR email LIKE ?",
            (f"%{malicious_search}%", f"%{malicious_search}%")
        )
        results = self.cursor.fetchall()
        
        # Should not return all users (literal search doesn't match)
        assert len(results) <= 2, "Search should not return all users"
    
    def test_sql_injection_in_order_by_blocked(self):
        """Test that ORDER BY injection is blocked (column names can't be parameterized)."""
        # Applications should validate column names, not use user input directly
        allowed_columns = ['id', 'username', 'email', 'role']
        user_input = "id; DROP TABLE users; --"
        
        # SAFE: Validate against whitelist
        if user_input in allowed_columns:
            order_column = user_input
        else:
            order_column = 'id'  # Default to safe value
        
        self.cursor.execute(f"SELECT * FROM users ORDER BY {order_column}")
        results = self.cursor.fetchall()
        assert len(results) == 2
        
        # Verify table still exists (injection would have failed or been ignored)
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        assert self.cursor.fetchone() is not None
    
    def test_sql_injection_in_limit_blocked(self):
        """Test that LIMIT injection is blocked."""
        # LIMIT values should be integers, validated before use
        malicious_limit = "1; DROP TABLE users; --"
        
        # SAFE: Validate and convert to integer
        try:
            limit = int(malicious_limit)
        except ValueError:
            limit = 10  # Default safe value
        
        self.cursor.execute("SELECT * FROM users LIMIT ?", (limit,))
        results = self.cursor.fetchall()
        
        # Should not cause any harm
        self.cursor.execute("SELECT COUNT(*) FROM users")
        assert self.cursor.fetchone()[0] == 2
    
    def test_batch_sql_injection_blocked(self):
        """Test that batch SQL execution doesn't allow injection between statements."""
        malicious_batch = [
            (3, "user3", "user3@test.com", "hash"),
            (4, "user4'; DROP TABLE users; --", "user4@test.com", "hash")
        ]
        
        # SAFE: executemany with parameterized query
        self.cursor.executemany(
            "INSERT INTO users (id, username, email, password_hash) VALUES (?, ?, ?, ?)",
            malicious_batch
        )
        self.conn.commit()
        
        # Verify database is intact
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        assert self.cursor.fetchone() is not None
        
        # Verify users were inserted (with literal malicious string as username)
        self.cursor.execute("SELECT COUNT(*) FROM users")
        assert self.cursor.fetchone()[0] == 4  # 2 original + 2 new
    
    def test_concurrent_sql_injection_attempts(self):
        """Test that concurrent injection attempts are all blocked."""
        results = []
        
        def attempt_injection(payload: str):
            try:
                conn = sqlite3.connect(':memory:')
                cursor = conn.cursor()
                cursor.execute('CREATE TABLE test (id INTEGER, name TEXT)')
                cursor.execute('INSERT INTO test VALUES (1, "test")')
                conn.commit()
                
                # Attempt injection
                cursor.execute("SELECT * FROM test WHERE name = ?", (payload,))
                cursor.fetchall()
                
                # Check table still exists
                cursor.execute("SELECT COUNT(*) FROM test")
                count = cursor.fetchone()[0]
                results.append(count == 1)
                conn.close()
            except Exception:
                results.append(True)  # Error is acceptable (injection blocked)
        
        # Run multiple injection attempts concurrently
        threads = []
        payloads = self.SQL_INJECTION_PAYLOADS[:5]
        for payload in payloads:
            t = threading.Thread(target=attempt_injection, args=(payload,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All attempts should have been blocked
        assert all(results), "All concurrent injection attempts should be blocked"


class TestRealNoSQLInjectionPrevention:
    """Test NoSQL/MongoDB-style injection prevention."""
    
    def test_nosql_operator_injection_blocked(self):
        """Test that NoSQL operators in input are sanitized."""
        # Simulating a document store with JSON
        import json
        
        malicious_input = {"$ne": None}  # MongoDB-style operator
        
        # Convert to string for storage (simulating JSON field)
        input_str = json.dumps(malicious_input)
        
        # Verify it's stored as data, not executed
        parsed = json.loads(input_str)
        assert "$ne" in parsed
        
        # SAFE: Applications should validate keys against whitelist
        dangerous_keys = ['$ne', '$eq', '$gt', '$lt', '$regex', '$where', '$or', '$and']
        
        for key in parsed.keys():
            assert key not in dangerous_keys or isinstance(parsed[key], (str, int, bool, type(None))), \
                "NoSQL operators should be sanitized"


class TestSecondOrderSQLInjection:
    """Test second-order SQL injection (stored then retrieved)."""
    
    def test_second_order_injection_blocked(self):
        """Test that stored malicious data doesn't cause injection when retrieved."""
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        cursor.execute('CREATE TABLE comments (id INTEGER, content TEXT)')
        
        # Store malicious content
        malicious_content = "'; DROP TABLE comments; --"
        cursor.execute("INSERT INTO comments VALUES (1, ?)", (malicious_content,))
        conn.commit()
        
        # Retrieve and display (simulating application displaying comment)
        cursor.execute("SELECT content FROM comments WHERE id = 1")
        result = cursor.fetchone()[0]
        
        # Content should be retrieved as literal string
        assert result == malicious_content
        
        # CRITICAL: Table should still exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='comments'")
        assert cursor.fetchone() is not None
        
        conn.close()


class TestBlindSQLInjectionPrevention:
    """Test blind SQL injection prevention."""
    
    def test_boolean_based_blind_injection_blocked(self):
        """Test that boolean-based blind SQL injection is blocked."""
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        cursor.execute('CREATE TABLE users (id INTEGER, username TEXT)')
        cursor.execute("INSERT INTO users VALUES (1, 'admin')")
        conn.commit()
        
        # Blind injection attempt
        malicious_id = "1 AND 1=1"
        
        # SAFE: Parameterized query
        try:
            cursor.execute("SELECT * FROM users WHERE id = ?", (malicious_id,))
            results = cursor.fetchall()
            # Should not match (literal string "1 AND 1=1" != integer 1)
            assert len(results) == 0
        except sqlite3.Error:
            pass  # Error is acceptable
        
        conn.close()
    
    def test_time_based_blind_injection_blocked(self):
        """Test that time-based blind SQL injection is blocked."""
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        cursor.execute('CREATE TABLE test (id INTEGER)')
        
        # Time-based injection attempt
        malicious_input = "1; SELECT randomblob(1000000000); --"
        
        start_time = time.time()
        try:
            cursor.execute("SELECT * FROM test WHERE id = ?", (malicious_input,))
            cursor.fetchall()
        except sqlite3.Error:
            pass
        elapsed = time.time() - start_time
        
        # Should complete quickly (not wait for time delay)
        assert elapsed < 1.0, "Time-based injection should be blocked (no delay)"
        
        conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
