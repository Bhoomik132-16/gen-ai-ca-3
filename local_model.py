"""
local_model.py
==============
Inference wrapper for the trained local Text-to-SQL model.
Loads 'trained_model.pkl' or automatically trains and caches it on first use.
Requires ZERO external APIs, ZERO internet access, and ZERO API keys.
"""

import os
import pickle
from typing import Dict, Any

MODEL_FILE = os.path.join(os.path.dirname(__file__), "trained_model.pkl")
_MODEL_INSTANCE = None


def get_model():
    """Returns the singleton instance of the trained local model."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE

    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                _MODEL_INSTANCE = pickle.load(f)
                return _MODEL_INSTANCE
        except Exception:
            pass

    # If model file does not exist yet, train and cache immediately
    import train_model
    _MODEL_INSTANCE = train_model.train_and_save()
    return _MODEL_INSTANCE


def generate_local_sql(question: str) -> Dict[str, Any]:
    """Generates SQL using the local trained semantic model in < 5ms."""
    model = get_model()
    return model.predict(question)

