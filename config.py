"""
config.py — Centralized, safe environment configuration reader.

Design decisions:
- Uses python-dotenv to load .env with override=False so CI/CD environment
  variables are never silently overwritten by a local .env file.
- Validates ALL required keys at module import time → fail-fast before
  Streamlit even attempts to render a page.
- Exposes only typed constants; no raw os.getenv() calls are scattered
  elsewhere in the codebase.
- NEVER logs, prints, or exposes key values in error messages or stack traces.

Usage:
    from config import OPENAI_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY
"""

import os
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env file from the directory containing this file.
# override=False means environment variables already set in the process
# (e.g., from a CI secret or Docker env) take precedence over the .env file.
# ---------------------------------------------------------------------------
_env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=_env_path, override=False)


def _require(key: str) -> str:
    """
    Retrieve a required environment variable by name.

    Raises:
        EnvironmentError: If the variable is missing or empty.
                          The error message never includes the key value.
    """
    value = os.getenv(key, "").strip()
    if not value:
        raise EnvironmentError(
            f"[config] Required environment variable '{key}' is missing or empty. "
            f"Ensure it is set in your .env file or system environment."
        )
    return value


# ---------------------------------------------------------------------------
# Public constants — import these throughout the application.
# ---------------------------------------------------------------------------

OPENAI_API_KEY: str = _require("OPENAI_API_KEY")
"""OpenAI API key configuration variable slot (used for Groq API key string)."""

SUPABASE_URL: str = _require("SUPABASE_URL")
"""Supabase project REST API URL (e.g. https://xxxx.supabase.co)."""

SUPABASE_ANON_KEY: str = _require("SUPABASE_ANON_KEY")
"""Supabase anon/public key for row-level-security-gated DB operations."""

# ---------------------------------------------------------------------------
# AI model settings — Swapped to Free Groq Tier for Hackathon
# ---------------------------------------------------------------------------

OPENAI_BASE_URL: str = "https://api.groq.com/openai/v1"
"""Redirects standard OpenAI client calls to Groq's free cloud infrastructure endpoint."""

OPENAI_MODEL: str = "llama-3.3-70b-versatile"
"""Free tier model identifier. Blazing fast alternative that costs zero credits."""

OPENAI_TEMPERATURE: float = 0.0
"""
Deterministic temperature. Set to 0.0 per PRD safety requirement to minimize
creative drift into unguarded / hallucinated medical advice territory.
"""

OPENAI_MAX_TOKENS: int = 500
"""
Maximum output tokens per AI assistant turn. Kept low to prevent long-form
responses that might ramble into unsafe territory outside the system prompt scope.
"""

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------

APP_NAME: str = "AMR Awareness & Risk Checker"
APP_VERSION: str = "1.0.0"
APP_TAGLINE: str = "Understanding Antibiotic Resistance — Education, Not Diagnosis"