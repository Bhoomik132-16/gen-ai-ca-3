"""
db.py
=====
Database management module for QueryMate.
Handles:
1. SQLite database initialization and seeding with extensive data (50 students, 6 subjects, 900 marks).
2. Schema extraction with column metadata, relationships, and sample rows.
3. Read-only connection creation (mode=ro).
4. Safe query execution returning pandas DataFrames.
"""

import os
import sqlite3
from pathlib import Path
from typing import List, Set
import pandas as pd

from sample_data import (
    STUDENTS_DATA,
    SUBJECTS_DATA,
    generate_marks_data,
    generate_attendance_data,
)

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "querymate.db")
ALLOWED_TABLES: Set[str] = {"students", "subjects", "marks", "attendance"}


def get_db_path(custom_path: str = None) -> str:
    """Returns absolute path to SQLite database."""
    return os.path.abspath(custom_path or DEFAULT_DB_PATH)


def init_db(db_path: str = None, force_refresh: bool = False) -> str:
    """
    Initializes the SQLite database with rich mock data.
    If force_refresh is True or the student count is less than 50, recreates tables.
    """
    target_path = get_db_path(db_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    conn = sqlite3.connect(target_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA foreign_keys = ON;")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            class TEXT NOT NULL,
            division TEXT NOT NULL
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            subject_id INTEGER PRIMARY KEY,
            subject_name TEXT NOT NULL UNIQUE
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS marks (
            mark_id INTEGER PRIMARY KEY,
            student_id INTEGER NOT NULL,
            subject_id INTEGER NOT NULL,
            unit TEXT NOT NULL,
            score INTEGER NOT NULL,
            max_score INTEGER NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(student_id),
            FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            student_id INTEGER NOT NULL,
            subject_id INTEGER NOT NULL,
            classes_held INTEGER NOT NULL,
            classes_attended INTEGER NOT NULL,
            PRIMARY KEY (student_id, subject_id),
            FOREIGN KEY (student_id) REFERENCES students(student_id),
            FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
        );
    """)

    # Check student count
    cursor.execute("SELECT COUNT(*) FROM students;")
    count = cursor.fetchone()[0]

    if count < 50 or force_refresh:
        # Clear existing data to load the full 50-student dataset
        cursor.execute("DELETE FROM attendance;")
        cursor.execute("DELETE FROM marks;")
        cursor.execute("DELETE FROM subjects;")
        cursor.execute("DELETE FROM students;")

        # Populate 50 students
        cursor.executemany(
            "INSERT INTO students (student_id, name, class, division) VALUES (?, ?, ?, ?);",
            STUDENTS_DATA,
        )

        # Populate 6 subjects
        cursor.executemany(
            "INSERT INTO subjects (subject_id, subject_name) VALUES (?, ?);",
            SUBJECTS_DATA,
        )

        # Populate 900 marks
        marks_data = generate_marks_data()
        cursor.executemany(
            "INSERT INTO marks (mark_id, student_id, subject_id, unit, score, max_score) VALUES (?, ?, ?, ?, ?, ?);",
            marks_data,
        )

        # Populate 300 attendance records
        attendance_data = generate_attendance_data()
        cursor.executemany(
            "INSERT INTO attendance (student_id, subject_id, classes_held, classes_attended) VALUES (?, ?, ?, ?);",
            attendance_data,
        )

        conn.commit()

    conn.close()
    return target_path


def get_connection(db_path: str = None) -> sqlite3.Connection:
    """
    Opens an SQLite connection in strictly READ-ONLY mode using URI 'file:path?mode=ro'.
    """
    target_path = Path(get_db_path(db_path)).resolve()
    uri = f"{target_path.as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=10.0)


def get_allowed_tables() -> Set[str]:
    """Returns the set of valid table names allowed in user queries."""
    return ALLOWED_TABLES.copy()


def get_schema(db_path: str = None) -> str:
    """
    Extracts and formats the database schema including table names, columns,
    data types, keys, relationships, and 2-3 sample rows per table.
    """
    target_path = get_db_path(db_path)
    if not os.path.exists(target_path):
        init_db(target_path)

    conn = get_connection(target_path)
    cursor = conn.cursor()

    schema_parts: List[str] = []

    for table_name in sorted(ALLOWED_TABLES):
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        col_desc = [f"{col[1]} ({col[2]}){' [PK]' if col[5] else ''}" for col in columns]

        cursor.execute(f"PRAGMA foreign_key_list({table_name});")
        fks = cursor.fetchall()
        fk_desc = [f"FK {fk[3]} -> {fk[2]}({fk[4]})" for fk in fks]

        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
        sample_rows = cursor.fetchall()
        col_names = [col[1] for col in columns]

        table_info = [
            f"Table: {table_name}",
            f"  Columns: {', '.join(col_desc)}",
        ]
        if fk_desc:
            table_info.append(f"  Foreign Keys: {', '.join(fk_desc)}")

        table_info.append("  Sample Rows:")
        for row in sample_rows:
            row_dict = dict(zip(col_names, row))
            table_info.append(f"    {row_dict}")

        schema_parts.append("\n".join(table_info))

    conn.close()
    return "\n\n".join(schema_parts)


def run_query(sql: str, db_path: str = None, timeout: float = 10.0) -> pd.DataFrame:
    """
    Executes a SQL query against the read-only database and returns a pandas DataFrame.
    """
    target_path = get_db_path(db_path)
    conn = get_connection(target_path)
    try:
        df = pd.read_sql_query(sql, conn)
        return df
    finally:
        conn.close()
