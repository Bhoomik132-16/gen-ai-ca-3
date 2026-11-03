# ⚡ QueryMate – Text-to-SQL Database Assistant

A full-stack Generative AI application that converts natural language questions into safe, optimized SQL queries using an **Offline Trained Semantic Model** (0 API keys required) or optional cloud models (Groq / Gemini), validates them for security against injection, and executes them against a read-only SQLite database.

---

## 🎯 Key Highlights

1. **100% Offline & Free (Zero API Keys Needed)**: Trained local semantic NLP engine translates English queries into SQL in under 5 milliseconds with zero network requests.
2. **Rich Academic Database (1,250+ Records)**:
   - **50 Students** across Divisions A, B, and C.
   - **6 Core Subjects** (Generative AI, Data Structures, DBMS, Operating Systems, Computer Networks, Machine Learning).
   - **900 Marks Entries** across Unit 1, Unit 2, and Unit 3.
   - **300 Attendance Records** (with realistic high, average, and <75% attendance).
3. **Resilient UI**: Built-in compatibility layer avoids PyArrow / NumPy 2.0 version mismatch errors.
4. **Physical Read-Only Safety**: Database opened with SQLite `mode=ro`.
5. **Multi-Layer Validator**: Enforces keyword blocklists, table whitelisting, and automatic `LIMIT 100` injection.

---

## 🚀 How to Run (Single Command - No venv needed)

```powershell
# 1. Open terminal in the project directory
cd "d:\CODING\gen ai ca-2"

# 2. Run the application
streamlit run app.py
```

The app will open automatically at **`http://localhost:8501`**.

---

## 🧪 8 Test Questions to Try in the UI

1. `Which students scored above 80 in Unit 2?`
2. `List the top 5 students by overall average marks across all subjects.`
3. `Show attendance percentage for each student in Generative AI.`
4. `Which students have attendance below 75% in any subject?`
5. `What is the average score for each subject across all units?`
6. `Which students have attendance below 75% AND scored below 40 in any subject?`
7. `Delete all students with marks below 40` *(Demonstrates safety validator blocking unsafe SQL)*
8. `What is the capital of Australia?` *(Demonstrates out-of-scope question rejection)*
