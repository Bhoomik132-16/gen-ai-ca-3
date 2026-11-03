"""
sql_generator.py
================
Unified SQL generation module for QueryMate.
Supports:
1. Local Trained Model (Default & Recommended: 100% offline, 0 API keys required)
2. Groq (Ultra-fast cloud option: LLaMA-3.3-70B)
3. Google Gemini (gemini-2.5-flash)

Features:
- Instant local generation using trained semantic engine.
- Zero-API fallback when cloud keys are invalid or networks are restricted.
- Multi-layer validation and self-correction loop.
"""

import json
import os
import re
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

import local_model

load_dotenv()

# Candidate models for cloud providers
GROQ_MODELS: List[str] = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]

GEMINI_MODELS: List[str] = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

FEW_SHOT_EXAMPLES = [
    {
        "question": "Which students scored above 80 in Unit 2?",
        "response": {
            "can_answer": True,
            "sql": "SELECT DISTINCT s.student_id, s.name, s.class, s.division, sub.subject_name, m.score FROM students s JOIN marks m ON s.student_id = m.student_id JOIN subjects sub ON m.subject_id = sub.subject_id WHERE m.unit = 'Unit 2' AND m.score > 80 ORDER BY m.score DESC;",
            "explanation": "Finds students whose score in Unit 2 exceeds 80, showing their name, class, division, subject, and score in descending order."
        }
    },
    {
        "question": "Show average marks and attendance for each student in Generative AI",
        "response": {
            "can_answer": True,
            "sql": "SELECT s.student_id, s.name, ROUND(AVG(m.score), 2) AS avg_score, a.classes_attended, a.classes_held, ROUND((a.classes_attended * 100.0 / a.classes_held), 2) AS attendance_pct FROM students s JOIN subjects sub ON sub.subject_name = 'Generative AI' JOIN marks m ON s.student_id = m.student_id AND m.subject_id = sub.subject_id JOIN attendance a ON s.student_id = a.student_id AND a.subject_id = sub.subject_id GROUP BY s.student_id, s.name, a.classes_attended, a.classes_held ORDER BY avg_score DESC;",
            "explanation": "Calculates the average examination score across all units and attendance percentage specifically for the 'Generative AI' course for each student."
        }
    },
    {
        "question": "What is the capital of France?",
        "response": {
            "can_answer": False,
            "sql": "",
            "explanation": "This question is not related to the student marks, courses, or attendance database schema provided."
        }
    }
]


def build_system_prompt(schema_text: str) -> str:
    """Builds prompt with schema and few-shot examples for cloud LLMs."""
    examples_str = json.dumps(FEW_SHOT_EXAMPLES, indent=2)

    return f"""You are an expert SQL developer assisting users with querying an SQLite database.
Your job is to translate the user's natural language question into a single, highly accurate, safe, read-only SQL query.

<schema>
{schema_text}
</schema>

CRITICAL RULES:
1. ONLY query the tables and columns explicitly listed in the <schema>. Never invent tables or columns.
2. Generate strictly ONE read-only SELECT query.
3. NEVER produce INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE, TRUNCATE, ATTACH, or PRAGMA statements.
4. If the question cannot be answered using the provided schema, set "can_answer" to false, "sql" to "", and explain in "explanation".
5. For calculations like attendance percentage, use: ROUND((classes_attended * 100.0 / classes_held), 2).
6. Always provide your output as a valid JSON object matching the schema below.

OUTPUT FORMAT (Strict JSON):
{{
  "can_answer": true | false,
  "sql": "SELECT ...;",
  "explanation": "Clear summary of query."
}}

FEW-SHOT EXAMPLES:
{examples_str}
"""


def clean_json_response(raw_text: str) -> str:
    """Cleans markdown fences from response text."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_llm_json(raw_text: str) -> Dict[str, Any]:
    """Parses JSON response."""
    cleaned = clean_json_response(raw_text)
    data = json.loads(cleaned)
    return {
        "can_answer": bool(data.get("can_answer", True)),
        "sql": str(data.get("sql", "")).strip(),
        "explanation": str(data.get("explanation", "")).strip(),
    }


def call_groq(prompt: str, system_instruction: str, api_key: str, preferred_model: Optional[str] = None) -> str:
    """Groq API runner with urllib fallback."""
    models_to_try = [preferred_model] if preferred_model else []
    for m in GROQ_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }

    last_error = None
    for model_name in models_to_try:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except Exception as err:
            last_error = err
            continue

    raise RuntimeError(f"Groq API call failed: {last_error}")


def call_gemini(prompt: str, system_instruction: str, api_key: str, preferred_model: Optional[str] = None) -> str:
    """Gemini API runner."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key.strip())
    model_name = preferred_model or "gemini-2.5-flash"

    config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        system_instruction=system_instruction,
    )
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )
    if response and response.text:
        return response.text
    raise RuntimeError("Empty response from Gemini.")


def generate_sql(
    question: str,
    schema_text: str = "",
    api_key: Optional[str] = None,
    preferred_model: Optional[str] = None,
    provider: str = "local",
) -> Dict[str, Any]:
    """
    Main dispatch entry point.
    Defaults to 100% offline local model.
    Falls back to local model if cloud API encounters any network or authentication error.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty. Please enter an inquiry.")

    # 1. Local Trained Model (Default, 0 API calls, 0 network dependencies)
    if provider == "local" or not api_key:
        return local_model.generate_local_sql(question)

    # 2. Cloud Providers (Groq or Gemini)
    system_prompt = build_system_prompt(schema_text)
    user_prompt = f"<question>\n{question.strip()}\n</question>"

    try:
        if provider == "groq":
            raw = call_groq(user_prompt, system_prompt, api_key, preferred_model)
        else:
            raw = call_gemini(user_prompt, system_prompt, api_key, preferred_model)
        return parse_llm_json(raw)
    except Exception as cloud_err:
        # Graceful fallback: If cloud API fails (e.g. rate limit, bad key, firewall),
        # transparently switch to the local trained model so the user's app NEVER breaks!
        local_result = local_model.generate_local_sql(question)
        local_result["explanation"] += f" (Generated locally via trained model. Cloud error: {cloud_err})"
        return local_result


def correct_sql_query(
    question: str,
    failed_sql: str,
    error_message: str,
    schema_text: str,
    api_key: Optional[str] = None,
    preferred_model: Optional[str] = None,
    provider: str = "local",
) -> Dict[str, Any]:
    """Self-healing correction."""
    if provider == "local" or not api_key:
        return local_model.generate_local_sql(question)

    system_prompt = build_system_prompt(schema_text)
    correction_prompt = f"""Fix this failed SQL query for the question:
<question>{question}</question>
<failed_sql>{failed_sql}</failed_sql>
<error>{error_message}</error>
Output JSON format: {{"can_answer": true, "sql": "...", "explanation": "..."}}
"""
    try:
        if provider == "groq":
            raw = call_groq(correction_prompt, system_prompt, api_key, preferred_model)
        else:
            raw = call_gemini(correction_prompt, system_prompt, api_key, preferred_model)
        return parse_llm_json(raw)
    except Exception:
        return local_model.generate_local_sql(question)
