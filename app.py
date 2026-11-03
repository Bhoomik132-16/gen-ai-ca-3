"""
app.py
======
QueryMate – Text-to-SQL Database Assistant
Streamlit Web Application for Generative AI Academic Project.

Features:
- Pure local text-to-SQL engine with 0 external API dependencies.
- Persistent session state so query output and Download CSV NEVER disappear on rerun.
- Safe read-only SQLite execution with multi-layer security validator.
- PyArrow-free Pure HTML Table Renderer: 100% immune to NumPy/PyArrow version conflicts!
- Clean, modern UI with direct table rendering and CSV export.
"""

import html
import os
import time
from typing import Dict, Any
import pandas as pd
import streamlit as st

# Application modules
import db
import validator
import local_model

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="QueryMate – Text-to-SQL Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Modern Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #10B981, #2563EB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 12px;
        background-color: #E0F2FE;
        color: #0369A1;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 8px;
    }
    .security-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 12px;
        background-color: #ECFDF5;
        color: #047857;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 8px;
    }
    .danger-alert {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 14px;
        border-radius: 6px;
        color: #991B1B;
        margin: 14px 0;
        font-weight: 500;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# Persistent Session State
if "history" not in st.session_state:
    st.session_state.history = []

if "selected_query" not in st.session_state:
    st.session_state.selected_query = ""

if "current_result" not in st.session_state:
    st.session_state.current_result = None

# Auto-initialize database on application launch
db.init_db()


# ==============================================================================
# Helper Functions & PyArrow-Free Table Renderer
# ==============================================================================
def render_safe_table(df: pd.DataFrame):
    """
    Renders DataFrame as a styled responsive HTML table directly via st.markdown.
    COMPLETELY BYPASSES pyarrow to permanently eliminate NumPy/PyArrow version conflicts!
    """
    if df is None or df.empty:
        st.info("The query executed successfully, but returned 0 matching records.")
        return

    table_html = df.to_html(index=False, classes="querymate-table")
    styled_table = f"""
    <style>
        .querymate-table-container {{
            max-height: 460px;
            overflow-y: auto;
            border: 1px solid #CBD5E1;
            border-radius: 8px;
            margin-top: 10px;
            margin-bottom: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        table.querymate-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 0.92rem;
            color: #1E293B;
            background-color: #FFFFFF;
        }}
        table.querymate-table th {{
            background-color: #F8FAFC;
            color: #0F172A;
            font-weight: 600;
            text-align: left;
            padding: 10px 14px;
            border-bottom: 2px solid #CBD5E1;
            position: sticky;
            top: 0;
            z-index: 2;
        }}
        table.querymate-table td {{
            padding: 8px 14px;
            border-bottom: 1px solid #F1F5F9;
        }}
        table.querymate-table tr:nth-child(even) {{
            background-color: #F8FAFC;
        }}
        table.querymate-table tr:hover {{
            background-color: #EFF6FF;
        }}
    </style>
    <div class="querymate-table-container">
        {table_html}
    </div>
    """
    st.markdown(styled_table, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_cached_schema() -> str:
    """Retrieves and caches database schema."""
    return db.get_schema()


def sanitize_text(text: str) -> str:
    """HTML-escapes text to prevent XSS injection."""
    return html.escape(text or "")


# ==============================================================================
# Sidebar: Database Explorer & History
# ==============================================================================
with st.sidebar:
    st.title("🗄️ Database Explorer")
    st.markdown(
        "<span class='security-badge'>Mode: READ-ONLY (mode=ro)</span>",
        unsafe_allow_html=True,
    )

    schema_text = get_cached_schema()
    with st.expander("View Database Schema & Samples", expanded=False):
        st.code(schema_text, language="sql")

    with st.expander("Preview Table Data"):
        table_choice = st.selectbox(
            "Select Table to Preview",
            options=["students", "subjects", "marks", "attendance"]
        )
        try:
            preview_df = db.run_query(f"SELECT * FROM {table_choice} LIMIT 5;")
            render_safe_table(preview_df)
        except Exception as e:
            st.error(f"Error loading preview: {e}")

    st.markdown("---")
    st.markdown("### 📜 Session History (Last 5)")

    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            st.markdown(f"**Q:** {sanitize_text(item['question'])}")
            st.caption(f"Status: {item['status']} | {item['time']}")
            st.divider()

        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.current_result = None
            st.rerun()
    else:
        st.caption("No queries run yet in this session.")


# ==============================================================================
# Main UI Layout
# ==============================================================================
st.markdown("<div class='main-title'>⚡ QueryMate – Text-to-SQL Database Assistant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Ask questions in plain English. Powered by Local Trained Model with multi-layer SQL security validation.</div>",
    unsafe_allow_html=True,
)

# Top Badges
col_b1, col_b2 = st.columns([1, 3])
with col_b1:
    st.markdown("<span class='metric-badge'>Database: 1,250+ Records</span>", unsafe_allow_html=True)
with col_b2:
    st.markdown("<span class='security-badge'>Security: Safe Read-Only (mode=ro)</span>", unsafe_allow_html=True)

st.write("")

# Sample Questions Section
st.markdown("##### 💡 Try an Example Question:")
example_cols = st.columns(5)

examples = [
    ("📊 Unit 2 > 80", "Which students scored above 80 in Unit 2?"),
    ("🏆 Top 5 Students", "List the top 5 students by overall average marks across all subjects."),
    ("🤖 Gen AI Attendance", "Show attendance percentage for each student in Generative AI."),
    ("⚠️ Low Attendance", "Which students have attendance below 75% in any subject?"),
    ("🛡️ Unsafe (Demo)", "Delete all students with marks below 40"),
]

for col, (label, question_text) in zip(example_cols, examples):
    with col:
        if st.button(label, use_container_width=True, help=question_text):
            st.session_state.selected_query = question_text
            st.rerun()

# Text input for question
query_input = st.text_area(
    "Enter your database question in plain English:",
    value=st.session_state.selected_query,
    placeholder="e.g. Which students scored above 80 in Unit 2?",
    height=80,
)

run_button = st.button("🚀 Run Query", type="primary", use_container_width=True)


# ==============================================================================
# Query Execution & Persistence Logic
# ==============================================================================
if run_button:
    if not query_input.strip():
        st.warning("⚠️ Please enter a question or click one of the example queries above.")
    else:
        start_time = time.time()
        allowed_tables = db.get_allowed_tables()

        with st.spinner("🤖 Generating SQL from question..."):
            llm_response = local_model.generate_local_sql(query_input.strip())

        can_answer = llm_response.get("can_answer", False)
        raw_sql = llm_response.get("sql", "").strip()
        explanation = llm_response.get("explanation", "").strip()

        # Handle out of scope
        if not can_answer:
            st.session_state.current_result = {
                "type": "out_of_scope",
                "question": query_input,
                "explanation": explanation,
            }
            st.session_state.history.append({
                "question": query_input,
                "status": "Unrelated Question",
                "time": time.strftime("%H:%M:%S"),
            })

        # Validate and Execute
        else:
            validation_error = None
            safe_sql = None

            try:
                safe_sql = validator.validate_sql(raw_sql, allowed_tables=allowed_tables, limit=100)
            except validator.SQLValidationError as sec_err:
                validation_error = str(sec_err)

            # Blocked by security validator
            if validation_error:
                st.session_state.current_result = {
                    "type": "blocked",
                    "question": query_input,
                    "sql": raw_sql,
                    "explanation": explanation,
                    "error": validation_error,
                }
                st.session_state.history.append({
                    "question": query_input,
                    "status": "Blocked by Security",
                    "time": time.strftime("%H:%M:%S"),
                })

            # Valid query: Run against SQLite read-only database
            else:
                exec_error = None
                df_result = None

                try:
                    df_result = db.run_query(safe_sql)
                except Exception as err:
                    exec_error = str(err)

                elapsed = time.time() - start_time

                if exec_error:
                    st.session_state.current_result = {
                        "type": "error",
                        "question": query_input,
                        "sql": safe_sql,
                        "explanation": explanation,
                        "error": exec_error,
                    }
                    st.session_state.history.append({
                        "question": query_input,
                        "status": "Execution Failed",
                        "time": time.strftime("%H:%M:%S"),
                    })
                else:
                    st.session_state.current_result = {
                        "type": "success",
                        "question": query_input,
                        "sql": safe_sql,
                        "explanation": explanation,
                        "df": df_result,
                        "rows": len(df_result),
                        "elapsed": elapsed,
                    }
                    st.session_state.history.append({
                        "question": query_input,
                        "status": f"Success ({len(df_result)} rows)",
                        "time": time.strftime("%H:%M:%S"),
                    })


# ==============================================================================
# Output Display Area (Persistent across downloads and reruns)
# ==============================================================================
res = st.session_state.current_result

if res:
    st.markdown("---")

    # 1. Out of Scope Result
    if res["type"] == "out_of_scope":
        st.info(f"ℹ️ **Cannot Answer From Database:**\n\n{sanitize_text(res['explanation'])}")

    # 2. Blocked Query Result
    elif res["type"] == "blocked":
        st.markdown("### 🔍 Generated SQL")
        st.code(res["sql"], language="sql")
        st.markdown(
            f"""
            <div class='danger-alert'>
                <strong>🛡️ Security Validator Blocked This Query!</strong><br/>
                Reason: {sanitize_text(res['error'])}<br/>
                <em>The database was protected and no commands were executed.</em>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Execution Error Result
    elif res["type"] == "error":
        st.markdown("### 🔍 Generated SQL")
        st.code(res["sql"], language="sql")
        st.error(f"❌ Execution Error: {res['error']}")

    # 4. Successful Query Result Table & Download CSV
    elif res["type"] == "success":
        df = res["df"]

        st.markdown("### 🔍 Generated SQL & Explanation")
        st.markdown(f"**Explanation:** {sanitize_text(res['explanation'])}")
        st.code(res["sql"], language="sql")

        st.markdown("### 📋 Query Results")

        # Telemetry row & Download CSV button
        col_m1, col_m2, col_m3 = st.columns([1, 1, 2])
        with col_m1:
            st.metric("Rows Returned", res["rows"])
        with col_m2:
            st.metric("Response Time", f"{res['elapsed']:.3f} s")
        with col_m3:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download CSV",
                data=csv_bytes,
                file_name="querymate_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        render_safe_table(df)
