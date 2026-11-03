"""
validator.py
============
Security and safety validation module for user/LLM-generated SQL queries.
Enforces multi-layer defense:
1. Strips all inline and multi-line comments.
2. Rejects multiple/chained SQL statements.
3. Enforces read-only initiation (must begin with SELECT or WITH).
4. Rejects blocked destructive or administrative SQL keywords.
5. Enforces a strict table whitelist (blocks system tables like sqlite_master).
6. Automatically appends a LIMIT clause if one is missing.
"""

import re
from typing import Iterable, Set
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Token
from sqlparse.tokens import Keyword, DML


BLOCKED_KEYWORDS: Set[str] = {
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "CREATE",
    "REPLACE",
    "TRUNCATE",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "VACUUM",
    "EXEC",
    "EXECUTE",
    "REINDEX",
    "GRANT",
    "REVOKE",
}

BLOCKED_SYSTEM_TABLES: Set[str] = {
    "sqlite_master",
    "sqlite_temp_master",
    "sqlite_schema",
    "sqlite_temp_schema",
    "sqlite_sequence",
    "sqlite_stat1",
    "sqlite_stat4",
}


class SQLValidationError(ValueError):
    """Custom exception raised when a query fails security validation."""
    pass


def strip_sql_comments(sql: str) -> str:
    """
    Strips single-line (--) and multi-line (/* */) SQL comments
    using sqlparse formatting and regex sanitization.
    """
    # sqlparse comment stripping
    formatted = sqlparse.format(sql, strip_comments=True).strip()

    # Secondary regex pass to guarantee removal of comment patterns
    # Remove multi-line /* ... */
    no_block_comments = re.sub(r"/\*.*?\*/", " ", formatted, flags=re.DOTALL)
    # Remove single-line -- ...
    no_line_comments = re.sub(r"--[^\r\n]*", " ", no_block_comments)

    return no_line_comments.strip()


def extract_tables_from_query(sql: str) -> Set[str]:
    """
    Extracts table names referenced in FROM and JOIN clauses.
    Also handles CTEs (Common Table Expressions) declared in WITH clauses.
    """
    parsed = sqlparse.parse(sql)
    if not parsed:
        return set()

    statement = parsed[0]
    tables = set()
    ctes = set()

    # Track CTE declarations: WITH cte_name AS (...)
    tokens = [t for t in statement.tokens if not t.is_whitespace]
    for i, token in enumerate(tokens):
        if token.ttype is Keyword.CTE or (token.is_keyword and token.value.upper() == "WITH"):
            # The next non-space token is often an Identifier or IdentifierList
            if i + 1 < len(tokens):
                cte_token = tokens[i + 1]
                if isinstance(cte_token, Identifier):
                    ctes.add(cte_token.get_real_name().lower())
                elif isinstance(cte_token, IdentifierList):
                    for iden in cte_token.get_identifiers():
                        ctes.add(iden.get_real_name().lower())

    def process_token(token, inside_from=False, inside_join=False):
        nonlocal inside_from_flag, inside_join_flag

        if token.is_whitespace:
            return

        if token.ttype is Keyword or token.ttype is DML:
            val = token.value.upper()
            if val == "FROM":
                inside_from_flag = True
                inside_join_flag = False
                return
            elif "JOIN" in val:
                inside_join_flag = True
                inside_from_flag = False
                return
            elif val in {"WHERE", "GROUP", "ORDER", "HAVING", "LIMIT", "UNION", "EXCEPT", "INTERSECT", "WINDOW"}:
                inside_from_flag = False
                inside_join_flag = False
                return

        if inside_from_flag or inside_join_flag:
            if isinstance(token, IdentifierList):
                for identifier in token.get_identifiers():
                    real_name = identifier.get_real_name()
                    if real_name:
                        tables.add(real_name.lower())
            elif isinstance(token, Identifier):
                real_name = token.get_real_name()
                if real_name:
                    tables.add(real_name.lower())
            elif token.ttype is None and hasattr(token, "tokens"):
                for subtoken in token.tokens:
                    process_token(subtoken)
            else:
                # Raw symbol token
                raw_name = token.value.strip("`\"'[] ;")
                if raw_name and raw_name.isidentifier():
                    tables.add(raw_name.lower())

            if inside_join_flag:
                # JOIN consumes only one table target
                inside_join_flag = False

    inside_from_flag = False
    inside_join_flag = False

    for token in statement.tokens:
        process_token(token)

    # Secondary regex extraction to catch patterns like "FROM table_name" or "JOIN table_name"
    regex_tables = re.findall(
        r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        sql,
        re.IGNORECASE,
    )
    for tbl in regex_tables:
        tables.add(tbl.lower())

    # Exclude declared CTEs from external table validation
    return {t for t in tables if t not in ctes}


def validate_sql(sql: str, allowed_tables: Iterable[str], limit: int = 100) -> str:
    """
    Validates the safety and read-only compliance of an SQL query.

    Checks enforced:
    1. Strips all inline and multi-line comments.
    2. Only ONE statement allowed (no chained statements with semicolons).
    3. Query must start with SELECT or WITH.
    4. Blocklist enforcement (DROP, DELETE, UPDATE, INSERT, ALTER, etc.).
    5. Table whitelist (must only use authorized database tables; blocks sqlite_master).
    6. Automatically injects LIMIT clause if missing.

    Parameters:
        sql: The SQL string to validate.
        allowed_tables: Iterable of allowed table names (e.g. {'students', 'marks'}).
        limit: Default row limit to inject if LIMIT is absent.

    Returns:
        Cleaned, validated, safe SQL query string.

    Raises:
        SQLValidationError: When the query fails any security rule.
    """
    if not sql or not sql.strip():
        raise SQLValidationError("Query is empty. Please provide a valid SQL statement.")

    # 1. Strip comments
    cleaned = strip_sql_comments(sql).strip()
    if not cleaned:
        raise SQLValidationError("Query is empty after comment removal.")

    # 2. Only ONE statement allowed
    parsed_statements = [s for s in sqlparse.parse(cleaned) if str(s).strip()]
    if len(parsed_statements) == 0:
        raise SQLValidationError("No valid SQL statement found.")
    if len(parsed_statements) > 1:
        raise SQLValidationError(
            "Multiple SQL statements detected (chained queries via semicolon). "
            "Only a single statement is permitted."
        )

    # Check for unescaped internal semicolons that could chain statements
    raw_cleaned_no_trailing = cleaned.rstrip("; \t\n")
    if ";" in raw_cleaned_no_trailing:
        raise SQLValidationError(
            "Semicolon detected inside statement. Chaining multiple commands is forbidden."
        )

    # 3. Must start with SELECT or WITH
    tokens = cleaned.split()
    first_word = tokens[0].upper()
    if first_word not in {"SELECT", "WITH"}:
        raise SQLValidationError(
            f"Unauthorized statement type '{first_word}'. "
            "Only read-only SELECT and WITH queries are permitted."
        )

    # 4. Keyword blocklist (whole-word regex match)
    for blocked_kw in BLOCKED_KEYWORDS:
        pattern = rf"\b{re.escape(blocked_kw)}\b"
        if re.search(pattern, cleaned, re.IGNORECASE):
            raise SQLValidationError(
                f"Blocked keyword detected: '{blocked_kw}'. "
                "Data modification and DDL operations are strictly prohibited."
            )

    # Check for blocked system tables anywhere in query
    for sys_tbl in BLOCKED_SYSTEM_TABLES:
        pattern = rf"\b{re.escape(sys_tbl)}\b"
        if re.search(pattern, cleaned, re.IGNORECASE):
            raise SQLValidationError(
                f"Access to internal system table '{sys_tbl}' is strictly forbidden."
            )

    # 5. Table whitelist validation
    allowed_set = {t.lower() for t in allowed_tables}
    extracted_tables = extract_tables_from_query(cleaned)

    # If tables were extracted, verify each one is in the allowed whitelist
    for table_name in extracted_tables:
        if table_name not in allowed_set:
            raise SQLValidationError(
                f"Unauthorized table '{table_name}'. "
                f"Allowed tables: {', '.join(sorted(allowed_set))}."
            )

    # 6. Add LIMIT clause if missing
    # Check if LIMIT exists in the outer query
    limit_match = re.search(r"\bLIMIT\s+\d+\b", cleaned, re.IGNORECASE)
    final_sql = cleaned.rstrip("; \t\n")
    if not limit_match:
        final_sql = f"{final_sql} LIMIT {limit}"

    return final_sql

