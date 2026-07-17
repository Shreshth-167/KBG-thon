"""Quick import + syntax verification for all three app modules."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
errors = []

# 1. config.py
try:
    import config
    assert hasattr(config, "OPENAI_API_KEY"), "Missing OPENAI_API_KEY"
    assert hasattr(config, "SUPABASE_URL"), "Missing SUPABASE_URL"
    assert hasattr(config, "SUPABASE_ANON_KEY"), "Missing SUPABASE_ANON_KEY"
    assert config.OPENAI_TEMPERATURE == 0.0, "temperature must be 0.0"
    assert config.OPENAI_MODEL == "gpt-4o-mini", "Model must be gpt-4o-mini"
    print(f"[OK] config.py — model={config.OPENAI_MODEL}, temp={config.OPENAI_TEMPERATURE}")
except Exception as e:
    errors.append(f"[FAIL] config.py: {e}")

# 2. database.py — syntax check only (no live DB call)
try:
    import ast, pathlib
    src = pathlib.Path("database.py").read_text(encoding="utf-8")
    ast.parse(src)
    print("[OK] database.py — syntax valid")
    # Check all required functions exist
    required_fns = [
        "create_session", "update_gate_accepted", "log_quiz_result",
        "log_viral_intercept", "log_chat_turn",
        "get_reference_snippets", "search_snippets_by_tags", "get_myth_fact_snippets"
    ]
    for fn in required_fns:
        if f"def {fn}" not in src:
            errors.append(f"[FAIL] database.py: missing function '{fn}'")
        else:
            print(f"       ✓ {fn}")
except Exception as e:
    errors.append(f"[FAIL] database.py: {e}")

# 3. app.py — syntax check
try:
    src = pathlib.Path("app.py").read_text(encoding="utf-8")
    ast.parse(src)
    print("[OK] app.py — syntax valid")
    # Check key safety components
    checks = {
        "st.stop()": "Feature A safety gate",
        "temperature=config.OPENAI_TEMPERATURE": "Layer 3 temperature lock",
        "_EMERGENCY_RE": "Layer 1a emergency regex",
        "_VIRAL_RE": "Layer 1b viral regex",
        "_DOSAGE_RE": "Layer 5 output dosage filter",
        "_PRESCRIPTIVE_RE": "Layer 5 prescriptive phrase filter",
        "_DRUG_BRAND_RE": "Layer 5 drug brand filter",
        "_SAFE_FALLBACK_RESPONSE": "Guardrail fallback response",
        "log_viral_intercept": "Feature C DB logging",
        "guardrail_fired": "Guardrail block logging",
        "gate_ok": "Session state gate key",
        "user_session_id": "Anonymous session UUID",
    }
    for pattern, label in checks.items():
        if pattern in src:
            print(f"       ✓ {label}")
        else:
            errors.append(f"[FAIL] app.py: missing '{pattern}' ({label})")
except Exception as e:
    errors.append(f"[FAIL] app.py: {e}")

# 4. Check packages importable
for pkg, name in [("streamlit", "streamlit"), ("openai", "openai"), ("supabase", "supabase"), ("dotenv", "python-dotenv")]:
    try:
        __import__(pkg)
        print(f"[OK] Package '{name}' importable")
    except ImportError as e:
        errors.append(f"[FAIL] Package '{name}' not importable: {e}")

print()
if errors:
    print("=== FAILURES ===")
    for err in errors:
        print(err)
    sys.exit(1)
else:
    print("=== ALL CHECKS PASSED ✅ ===")
    print("Run with: streamlit run app.py")
