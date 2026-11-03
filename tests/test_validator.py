"""
tests/test_validator.py
=======================
Unit tests for QueryMate SQL security validator.
Verifies that:
1. Destructive statements (DROP, DELETE, UPDATE, INSERT) are blocked.
2. Chained queries (semicolon injection) are rejected.
3. System tables (sqlite_master) are forbidden.
4. Comment-hidden attacks are safely caught and rejected.
5. Valid queries pass and receive automatic LIMIT injection.
"""

import sys
import os
import unittest

# Add parent directory to path so imports work smoothly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from validator import validate_sql, SQLValidationError


class TestSQLValidator(unittest.TestCase):

    def setUp(self):
        self.allowed_tables = {"students", "subjects", "marks", "attendance"}

    def test_block_drop_table(self):
        """DROP TABLE must be immediately rejected."""
        query = "DROP TABLE students;"
        with self.assertRaises(SQLValidationError) as ctx:
            validate_sql(query, self.allowed_tables)
        self.assertIn("DROP", str(ctx.exception).upper())

    def test_block_delete_from(self):
        """DELETE statements must be rejected."""
        query = "DELETE FROM marks WHERE score < 40;"
        with self.assertRaises(SQLValidationError) as ctx:
            validate_sql(query, self.allowed_tables)
        self.assertIn("DELETE", str(ctx.exception).upper())

    def test_block_chained_drop_query(self):
        """Chained query injection via semicolon must be rejected."""
        query = "SELECT 1; DROP TABLE marks;"
        with self.assertRaises(SQLValidationError) as ctx:
            validate_sql(query, self.allowed_tables)
        error_msg = str(ctx.exception).lower()
        self.assertTrue("multiple" in error_msg or "semicolon" in error_msg or "drop" in error_msg)

    def test_block_update_statement(self):
        """UPDATE statements must be rejected."""
        query = "UPDATE students SET name='x' WHERE student_id=1;"
        with self.assertRaises(SQLValidationError) as ctx:
            validate_sql(query, self.allowed_tables)
        self.assertIn("UPDATE", str(ctx.exception).upper())

    def test_block_sqlite_master(self):
        """Access to internal sqlite_master table must be blocked."""
        query = "SELECT * FROM sqlite_master;"
        with self.assertRaises(SQLValidationError) as ctx:
            validate_sql(query, self.allowed_tables)
        self.assertIn("sqlite_master", str(ctx.exception).lower())

    def test_block_comment_hidden_drop(self):
        """Comment-obfuscated destructive statements must be stripped and rejected."""
        query = "/* safe query prefix */ DROP TABLE marks;"
        with self.assertRaises(SQLValidationError) as ctx:
            validate_sql(query, self.allowed_tables)
        self.assertIn("DROP", str(ctx.exception).upper())

    def test_block_inline_comment_injection(self):
        """Inline comment hiding additional statement must be rejected."""
        query = "SELECT * FROM students; -- comment \n DROP TABLE marks;"
        with self.assertRaises(SQLValidationError) as ctx:
            validate_sql(query, self.allowed_tables)
        self.assertTrue(any(word in str(ctx.exception).lower() for word in ["multiple", "drop", "semicolon"]))

    def test_block_unauthorized_table(self):
        """Queries accessing tables not in schema must be rejected."""
        query = "SELECT * FROM secret_passwords;"
        with self.assertRaises(SQLValidationError) as ctx:
            validate_sql(query, self.allowed_tables)
        self.assertIn("secret_passwords", str(ctx.exception).lower())

    def test_valid_join_query_adds_limit(self):
        """Valid JOIN query should pass and have LIMIT 100 injected."""
        query = (
            "SELECT s.name, m.score "
            "FROM students s "
            "JOIN marks m ON s.student_id = m.student_id "
            "WHERE m.unit = 'Unit 2'"
        )
        validated = validate_sql(query, self.allowed_tables, limit=100)
        self.assertTrue(validated.startswith("SELECT s.name, m.score"))
        self.assertTrue(validated.endswith("LIMIT 100"))

    def test_query_with_existing_limit_preserved(self):
        """If query already has LIMIT, it should not double-add LIMIT."""
        query = "SELECT * FROM students LIMIT 10"
        validated = validate_sql(query, self.allowed_tables, limit=100)
        self.assertEqual(validated, "SELECT * FROM students LIMIT 10")
        self.assertNotIn("LIMIT 100", validated)

    def test_empty_query_rejected(self):
        """Empty query must raise SQLValidationError."""
        with self.assertRaises(SQLValidationError):
            validate_sql("   ", self.allowed_tables)


if __name__ == "__main__":
    unittest.main()

