"""
database.py — Supabase PostgreSQL connection wrapper (Demo Optimized).

All database I/O for the AMR Awareness app flows through this module.
Design principles:
  - Every public function is wrapped in try/except — a DB failure NEVER
    crashes the Streamlit UI. Errors are printed to console only.
  - No function leaks raw Supabase exceptions or credentials to the UI layer.
  - The Supabase client is instantiated once as a module-level singleton.
  - Reference snippets are cached (via @st.cache_data) to avoid repeated
    network calls on every chat interaction.
"""

from __future__ import annotations

import re
import traceback
from datetime import datetime, timezone
from typing import Optional

import streamlit as st
from supabase import create_client, Client

import config

# ---------------------------------------------------------------------------
# Module-level Supabase client singleton
# ---------------------------------------------------------------------------

_supabase: Optional[Client] = None

def _get_client() -> Optional[Client]:
    """
    Return the module-level Supabase client, initialising it on first call.
    Uses a module-level singleton so we don't open a new connection per call.
    """
    global _supabase
    if _supabase is None:
        try:
            _supabase = create_client(config.SUPABASE_URL, config.SUPABASE_ANON_KEY)
        except Exception as exc:
            print(f"[database._get_client] Initialization failed: {exc}. Using fallback mode.")
            _supabase = None
    return _supabase

# ---------------------------------------------------------------------------
# Helper: silent error handler
# ---------------------------------------------------------------------------

def _log_db_error(fn_name: str, exc: Exception) -> None:
    """
    Print a DB error to console without propagating it to the UI.
    """
    print(f"[database.{fn_name}] DB error (non-fatal): {type(exc).__name__}: {exc}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# HIGH-FIDELITY FALLBACK DATASET (Keeps UI populated if DB tables are missing)
# ---------------------------------------------------------------------------

MOCK_SNIPPETS = [
    {
        "snippet_id": "mock-1",
        "source_org": "World Health Organization (WHO)",
        "title": "The Mechanics of Antimicrobial Resistance",
        "tags": ["resistance", "amr", "mutation", "bacteria", "antibiotic", "infection"],
        "snippet_text": "Antimicrobial resistance occurs naturally over time, usually through genetic changes. However, misuse and overuse of antimicrobials in humans and livestock are accelerating this process, leaving standard treatments ineffective.",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance",
        "is_myth_fact": False
    },
    {
        "snippet_id": "mock-2",
        "source_org": "Centers for Disease Control (CDC)",
        "title": "Antibiotics vs Common Analgesics",
        "tags": ["paracetamol", "fever", "analgesic", "virus", "cold", "flu", "cough"],
        "snippet_text": "Paracetamol is a symptom-management medication used to treat pain and reduce high fevers. It possesses absolutely zero antimicrobial properties. Using antibiotics to treat viral fevers provides zero benefit and actively drives community resistance patterns.",
        "source_url": "https://www.cdc.gov/antibiotic-use/index.html",
        "is_myth_fact": False
    },
    {
        "snippet_id": "mock-3",
        "source_org": "Global AMR Stewardship Alliance",
        "title": "Myth: Stopping Early is Harmless",
        "tags": ["resistance", "course", "duration", "myth", "stop"],
        "snippet_text": "MYTH: You can stop taking antibiotics as soon as your symptoms clear up. FACT: Halting therapy prematurely before eradicating the target colony allows partially resistant survivors to undergo selective mutation and propagate.",
        "source_url": "https://www.who.int",
        "is_myth_fact": True
    }
]

# ---------------------------------------------------------------------------
# SESSION MANAGEMENT
# ---------------------------------------------------------------------------

def create_session(session_id: str) -> bool:
    try:
        client = _get_client()
        if client is None:
            raise RuntimeError("Supabase client is offline.")
        client.table("sessions").insert({
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "gate_accepted_at": None,
        }).execute()
        return True
    except Exception as exc:
        _log_db_error("create_session", exc)
        print(f"📝 [LOCAL LOG] Session Created -> {session_id}")
        return False

def update_gate_accepted(session_id: str) -> bool:
    try:
        client = _get_client()
        if client is None:
            raise RuntimeError("Supabase client is offline.")
        client.table("sessions").update({
            "gate_accepted_at": datetime.now(timezone.utc).isoformat(),
        }).eq("session_id", session_id).execute()
        return True
    except Exception as exc:
        _log_db_error("update_gate_accepted", exc)
        print(f"📝 [LOCAL LOG] Disclaimer Accepted for Session -> {session_id}")
        return False

# ---------------------------------------------------------------------------
# QUIZ LOGGING
# ---------------------------------------------------------------------------

def log_quiz_result(session_id: str, answers: dict, risk_score: int, risk_band: str, viral_flag: bool) -> bool:
    try:
        client = _get_client()
        if client is None:
            raise RuntimeError("Supabase client is offline.")
        client.table("quiz_logs").insert({
            "session_id": session_id,
            "answers": answers,
            "risk_score": risk_score,
            "risk_band": risk_band,
            "viral_flag": viral_flag,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
        return True
    except Exception as exc:
        _log_db_error("log_quiz_result", exc)
        print(f"📝 [LOCAL LOG] Quiz Saved -> Score: {risk_score} | Band: {risk_band}")
        return False

# ---------------------------------------------------------------------------
# VIRAL INTERCEPT LOGGING
# ---------------------------------------------------------------------------

def log_viral_intercept(session_id: str, source: str, raw_trigger_text: str) -> bool:
    assert source in ("quiz", "chat"), f"Invalid source '{source}'; must be 'quiz' or 'chat'."
    try:
        client = _get_client()
        if client is None:
            raise RuntimeError("Supabase client is offline.")
        client.table("intercepted_viral_queries").insert({
            "session_id": session_id,
            "source": source,
            "raw_trigger_text": raw_trigger_text[:500],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
        return True
    except Exception as exc:
        _log_db_error("log_viral_intercept", exc)
        print(f"📝 [LOCAL LOG] Feature C Intercept -> Source: {source} | Trigger: {raw_trigger_text[:500]}")
        return False

# ---------------------------------------------------------------------------
# CHAT LOGGING
# ---------------------------------------------------------------------------

def log_chat_turn(session_id: str, user_message: str, assistant_response: str, was_blocked_by_guardrail: bool, matched_snippet_ids: Optional[list[str]] = None) -> bool:
    try:
        client = _get_client()
        if client is None:
            raise RuntimeError("Supabase client is offline.")
        client.table("chat_logs").insert({
            "session_id": session_id,
            "user_message": user_message[:2000],
            "assistant_response": assistant_response[:4000],
            "was_blocked_by_guardrail": was_blocked_by_guardrail,
            "matched_snippet_ids": matched_snippet_ids or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
        return True
    except Exception as exc:
        _log_db_error("log_chat_turn", exc)
        print(f"📝 [LOCAL LOG] Chat Log Intercepted -> Guardrail Triggered: {was_blocked_by_guardrail}")
        return False

# ---------------------------------------------------------------------------
# REFERENCE SNIPPETS  (Demo Presentation Optimized)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_reference_snippets() -> list[dict]:
    try:
        client = _get_client()
        if client is None:
            raise RuntimeError("Supabase client offline.")
        response = client.table("reference_snippets").select("*").execute()
        if response.data and len(response.data) > 0:
            return response.data
        return MOCK_SNIPPETS
    except Exception as exc:
        _log_db_error("get_reference_snippets", exc)
        return MOCK_SNIPPETS

def search_snippets_by_tags(query_text: str, snippets: list[dict], top_k: int = 3) -> list[dict]:
    """
    Keyword-overlap matcher with presentation fallback logic to prevent empty AI context windows.
    """
    if not snippets:
        return []
    
    if not query_text.strip():
        return snippets[:top_k]

    query_tokens = set(re.sub(r"[^\w\s]", "", query_text.lower()).split())

    scored: list[tuple[int, dict]] = []
    for snippet in snippets:
        score = 0
        tags = [t.lower() for t in (snippet.get("tags") or [])]
        score += sum(1 for token in query_tokens if token in tags) * 3

        title_tokens = set(re.sub(r"[^\w\s]", "", (snippet.get("title") or "").lower()).split())
        score += len(query_tokens & title_tokens) * 2

        text_tokens = set(re.sub(r"[^\w\s]", "", (snippet.get("snippet_text") or "").lower()).split())
        score += len(query_tokens & text_tokens)

        if score > 0:
            scored.append((score, snippet))

    scored.sort(key=lambda x: x[0], reverse=True)
    
    # PRESENTATION SAFEGUARD: If no keyword matched, return the first few snippets 
    # instead of an empty list so the AI always stays responsive on stage.
    if not scored:
        return snippets[:top_k]
        
    return [s for _, s in scored[:top_k]]

def get_myths_and_facts():
    """Fetches educational myths/facts from Supabase, with a high-fidelity local fallback."""
    try:
        supabase = _get_client()
        if supabase:
            response = supabase.table("myths_facts").select("*").execute()
            if response.data and len(response.data) > 0:
                return response.data
    except Exception as e:
        print(f"Database read error context managed: {e}")
    
    # Using your high-fidelity lowercase array directly as the backup
    return [
        {
            "myth": "Antibiotics cure colds and flu.",
            "fact": "Colds and flu are caused by viruses. Antibiotics only work on bacteria — they have zero effect on viral infections and can cause harm.",
            "source_org": "WHO",
            "source_url": "https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance",
        },
        {
            "myth": "Stopping antibiotics early is fine once you feel better.",
            "fact": "Feeling better doesn't mean all bacteria are dead. Stopping early lets the strongest bacteria survive and develop resistance. Always complete the full course.",
            "source_org": "ICMR",
            "source_url": "https://main.icmr.gov.in/content/antimicrobial-resistance",
        },
        {
            "myth": "Sharing antibiotics with a sick family member is helpful.",
            "fact": "A prescription is for a specific person, infection, and dose. Sharing means wrong drug, wrong dose, and an incomplete course — ideal conditions for creating resistant bacteria.",
            "source_org": "WHO",
            "source_url": "https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance",
        },
        {
            "myth": "Stronger antibiotics are always better for serious infections.",
            "fact": "Broad-spectrum and 'stronger' antibiotics accelerate resistance and kill beneficial gut bacteria. Targeted, appropriate-spectrum antibiotics chosen by a doctor are always preferable.",
            "source_org": "CDC",
            "source_url": "https://www.cdc.gov/antibiotic-use/index.html",
        },
        {
            "myth": "Keeping antibiotics at home is safe and convenient.",
            "fact": "Stockpiling creates easy access for self-medication. Stored antibiotics are often past expiry, incorrectly stored, and used without a diagnosis — all major resistance risks.",
            "source_org": "ICMR",
            "source_url": "https://main.icmr.gov.in/content/antimicrobial-resistance",
        },
        {
            "myth": "Antibiotic resistance only affects people who misuse antibiotics.",
            "fact": "Resistance spreads through communities, hospitals, food, and water. Even if you use antibiotics correctly, resistant bacteria created elsewhere can infect you.",
            "source_org": "WHO",
            "source_url": "https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance",
        }
    ]