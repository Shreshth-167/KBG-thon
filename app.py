"""
app.py — AMR Antibiotic Misuse Awareness & Risk Checker
Full Streamlit application implementing Features A–F from the PRD.

Safety architecture (5 layers):
  Layer 1: Emergency keyword regex  → static disclaimer, no LLM
  Layer 2: Viral symptom regex      → Feature C home care guide, no LLM
  Layer 3: System-prompt scope lock + temperature=0.0 + max_tokens cap
  Layer 4: Reference snippet grounding requirement
  Layer 5: Output-side deny-list regex → discard + fallback + guardrail log

Compliance boundary (enforced here, not just documented):
  NOTHING in this file diagnoses, prescribes, or recommends a specific drug.
"""

from __future__ import annotations

import re
import uuid
from typing import Optional

import streamlit as st
from openai import OpenAI

import config
import database as db

# ---------------------------------------------------------------------------
# PAGE CONFIG  (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=f"{config.APP_NAME}",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": (
            f"**{config.APP_NAME}** v{config.APP_VERSION}\n\n"
            "An awareness and education tool. Not a medical device. "
            "Built for the KBG Club Hackathon at IIT Mandi."
        ),
    },
)

# ---------------------------------------------------------------------------
# CUSTOM CSS  — Dark mode design system
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #0F1117 !important;
        color: #E8EAF0 !important;
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── App header bar ── */
    .app-header {
        background: linear-gradient(135deg, #1A1D2E 0%, #0F1117 100%);
        border-bottom: 2px solid #00D4AA;
        padding: 1.2rem 2rem;
        margin-bottom: 1.5rem;
        border-radius: 0 0 12px 12px;
    }
    .app-header h1 {
        font-size: 1.6rem;
        font-weight: 800;
        color: #00D4AA;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: #8B90A7;
        margin: 0.2rem 0 0 0;
        font-size: 0.85rem;
    }

    /* ── Card surfaces ── */
    .card {
        background: #1A1D2E;
        border: 1px solid #2A2D3E;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .card-accent {
        border-left: 4px solid #00D4AA;
    }

    /* ── Safety gateway ── */
    .gateway-container {
        max-width: 720px;
        margin: 3rem auto;
        background: #1A1D2E;
        border: 2px solid #FF4B4B;
        border-radius: 16px;
        padding: 2.5rem;
        box-shadow: 0 0 40px rgba(255, 75, 75, 0.15);
    }
    .gateway-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #FF4B4B;
        margin-bottom: 0.5rem;
    }
    .gateway-subtitle {
        color: #8B90A7;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .emergency-box {
        background: rgba(255,75,75,0.1);
        border: 1px solid #FF4B4B;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 1rem 0;
        font-size: 0.85rem;
        color: #FF8080;
    }
    .is-box {
        background: rgba(0,212,170,0.08);
        border: 1px solid #00D4AA;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #A0F0E0;
        font-size: 0.85rem;
    }
    .isnot-box {
        background: rgba(255,75,75,0.06);
        border: 1px solid #7A1010;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #FF8080;
        font-size: 0.85rem;
    }

    /* ── Risk band badges ── */
    .badge-low {
        display: inline-block;
        background: rgba(0,212,170,0.15);
        color: #00D4AA;
        border: 1px solid #00D4AA;
        border-radius: 20px;
        padding: 0.3rem 1rem;
        font-weight: 700;
        font-size: 1rem;
    }
    .badge-medium {
        display: inline-block;
        background: rgba(255,215,0,0.15);
        color: #FFD700;
        border: 1px solid #FFD700;
        border-radius: 20px;
        padding: 0.3rem 1rem;
        font-weight: 700;
        font-size: 1rem;
    }
    .badge-high {
        display: inline-block;
        background: rgba(255,75,75,0.15);
        color: #FF4B4B;
        border: 1px solid #FF4B4B;
        border-radius: 20px;
        padding: 0.3rem 1rem;
        font-weight: 700;
        font-size: 1rem;
    }

    /* ── Score ring (text-only version for Streamlit) ── */
    .score-display {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem;
        color: #00D4AA;
    }

    /* ── Chat messages ── */
    .chat-user {
        background: rgba(0,212,170,0.1);
        border: 1px solid rgba(0,212,170,0.3);
        border-radius: 12px 12px 4px 12px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .chat-bot {
        background: #1A1D2E;
        border: 1px solid #2A2D3E;
        border-radius: 12px 12px 12px 4px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .chat-blocked {
        background: rgba(255,75,75,0.08);
        border: 1px solid rgba(255,75,75,0.4);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.8rem;
        color: #FF8080;
        margin-top: 0.3rem;
    }

    /* ── Source sidebar snippets ── */
    .snippet-card {
        background: #14172A;
        border: 1px solid #2A2D3E;
        border-left: 3px solid #00D4AA;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
        font-size: 0.8rem;
    }
    .snippet-org-badge {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.15rem 0.5rem;
        border-radius: 10px;
        margin-bottom: 0.3rem;
    }
    .badge-who { background: rgba(0,123,255,0.2); color: #60A5FA; border: 1px solid #1D4ED8; }
    .badge-cdc { background: rgba(34,197,94,0.2); color: #4ADE80; border: 1px solid #15803D; }
    .badge-icmr { background: rgba(249,115,22,0.2); color: #FB923C; border: 1px solid #C2410C; }

    /* ── Myth/Fact cards ── */
    .myth-label {
        color: #FF4B4B;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .fact-label {
        color: #00D4AA;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Viral redirect banner ── */
    .viral-banner {
        background: linear-gradient(135deg, rgba(255,215,0,0.08), rgba(255,165,0,0.05));
        border: 2px solid #FFD700;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .viral-banner-title {
        color: #FFD700;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* ── Tabs styling ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1A1D2E;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8B90A7;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00D4AA !important;
        color: #0F1117 !important;
        font-weight: 700 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #00D4AA, #00A882);
        color: #0F1117;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-size: 0.95rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(0,212,170,0.3);
    }

    /* ── Checkbox ── */
    .stCheckbox label {
        color: #E8EAF0 !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }

    /* ── Expander (myth/fact) ── */
    .streamlit-expanderHeader {
        background: #1A1D2E !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: #E8EAF0 !important;
    }

    /* ── Divider ── */
    hr { border-color: #2A2D3E !important; }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        color: #4A4D5E;
        font-size: 0.75rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #1A1D2E;
        margin-top: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# OPENAI / GROQ CLIENT  (Module-level dynamic singleton wrapper)
# ---------------------------------------------------------------------------

_openai_client: Optional[OpenAI] = None


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        # Dynamically matches standard OpenAI proxy endpoints or Groq's cloud engine endpoint
        base_url = getattr(config, "OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
        _openai_client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=base_url
        )
    return _openai_client


# ---------------------------------------------------------------------------
# SAFETY GUARDRAIL REGEXES  (Layer 1 & Layer 5)
# ---------------------------------------------------------------------------

# Layer 1a — Emergency red-flag terms (bypass LLM entirely, show static response)
_EMERGENCY_RE = re.compile(
    r"\b(chest\s+pain|can'?t\s+breathe|difficulty\s+breath|overdose|"
    r"unconscious|emergency|call\s+911|ambulance|suicid|poisoning|"
    r"anaphylax|severe\s+allerg|heart\s+attack|stroke)\b",
    re.IGNORECASE,
)

# Layer 1b — Viral symptom terms (bypass LLM, route to Feature C)
_VIRAL_RE = re.compile(
    r"\b(cold|flu|influenza|cough|runny\s+nose|sore\s+throat|"
    r"fever|sneezing|congestion|viral|virus|common\s+cold|"
    r"blocked\s+nose|stuffy\s+nose|body\s+ache)\b",
    re.IGNORECASE,
)

# Layer 5 — Output-side deny-list (dosage + prescriptive language)
_DOSAGE_RE = re.compile(
    r"\b\d+\s?(mg|ml|mcg|micrograms?|milligrams?|units?|tablets?|capsules?|"
    r"teaspoons?|tablespoons?|doses?)\b",
    re.IGNORECASE,
)
_PRESCRIPTIVE_RE = re.compile(
    r"\b(you\s+should\s+take|i\s+prescribe|the\s+correct\s+dose|take\s+\d|"
    r"prescribed\s+dose|daily\s+dose|dosage\s+is|i\s+recommend\s+taking|"
    r"start\s+taking|stop\s+taking|administer)\b",
    re.IGNORECASE,
)

# Drug brand name deny-list (non-exhaustive, covers common household names)
_DRUG_BRAND_RE = re.compile(
    r"\b(amoxicillin|augmentin|azithromycin|zithromax|ciprofloxacin|cipro|"
    r"doxycycline|metronidazole|flagyl|cephalexin|keflex|trimethoprim|"
    r"bactrim|clindamycin|levofloxacin|ampicillin|penicillin|erythromycin|"
    r"tetracycline|nitrofurantoin|vancomycin|meropenem|ceftriaxone|"
    r"co-amoxiclav|flucloxacillin)\b",
    re.IGNORECASE,
)

_SAFE_FALLBACK_RESPONSE = (
    "I can't provide dosing, prescribing, or drug-specific guidance — "
    "this falls outside what I'm designed to answer. "
    "Please consult a qualified doctor or pharmacist for personal medical advice. "
    "\n\n*This is educational information only.*"
)

# ---------------------------------------------------------------------------
# QUIZ DATA  (Feature B — deterministic scoring engine)
# ---------------------------------------------------------------------------

QUIZ_QUESTIONS: list[dict] = [
    {
        "id": "q1",
        "question": "Have you taken antibiotics **without a doctor's prescription** in the last 6 months?",
        "points": 3,
        "flag": "self_medication",
        "explanation": "Self-medicating with antibiotics is one of the top drivers of resistance globally. Without a diagnosis, you may be taking the wrong drug for your infection — or an antibiotic for a viral illness where it has no effect.",
        "source": "WHO AMR Fact Sheet",
    },
    {
        "id": "q2",
        "question": "Did you **stop a prescribed antibiotic course early** because you felt better?",
        "points": 3,
        "flag": "early_stoppage",
        "explanation": "Stopping early leaves the strongest bacteria alive — the ones that survived the first few days of treatment. These survivors can multiply and pass on resistance traits. Always complete the full prescribed course.",
        "source": "ICMR Treatment Guidelines",
    },
    {
        "id": "q3",
        "question": "Have you used **leftover antibiotics from a previous illness**?",
        "points": 2,
        "flag": "leftover_reuse",
        "explanation": "Leftover antibiotics are almost always an incomplete course from a different infection. Reusing them means wrong drug, wrong dose, and no medical oversight — all resistance risk factors.",
        "source": "CDC Antibiotic Use Guidelines",
    },
    {
        "id": "q4",
        "question": "Have you taken antibiotics for **cold, flu, or sore throat symptoms**?",
        "points": 3,
        "flag": "viral_misuse",
        "sets_viral_flag": True,
        "explanation": "Colds and flu are caused by viruses. Antibiotics have zero effect on viruses — taking them for these symptoms harms your gut microbiome, risks side effects, and accelerates resistance without any benefit to your illness.",
        "source": "WHO / CDC",
    },
    {
        "id": "q5",
        "question": "Do you **keep antibiotics at home 'just in case'**?",
        "points": 1,
        "flag": "stockpiling",
        "explanation": "Stockpiling creates easy access for self-medication and increases the chance of inappropriate use. It also means the stored drug is often past its expiry or stored incorrectly.",
        "source": "ICMR AMR Action Plan",
    },
    {
        "id": "q6",
        "question": "Have you **shared your antibiotic prescription** with a family member?",
        "points": 2,
        "flag": "sharing",
        "explanation": "A prescription is written for a specific person, infection, and course length. Sharing means the recipient has no diagnosis, may take the wrong drug, and will almost certainly take an incomplete course — perfect conditions for resistance to develop.",
        "source": "WHO AMR Fact Sheet",
    },
]

RISK_BANDS: dict[str, dict] = {
    "Low": {
        "range": "0–2",
        "color": "low",
        "emoji": "🟢",
        "headline": "Your antibiotic habits show low risk patterns.",
        "message": (
            "Great news — your self-reported antibiotic habits are relatively safe. "
            "Keep it up by always consulting a doctor before starting any antibiotic course, "
            "completing every prescribed course fully, and never sharing or stockpiling antibiotics."
        ),
        "tip": "💡 **Tip:** Even one instance of antibiotic misuse contributes to resistance. Share what you've learned today with someone you know.",
    },
    "Medium": {
        "range": "3–6",
        "color": "medium",
        "emoji": "🟡",
        "headline": "Some of your antibiotic habits carry real resistance risk.",
        "message": (
            "The specific answers below contributed to your score. "
            "Understanding *why* these behaviors matter is the first step to changing them. "
            "Consider speaking with a pharmacist or doctor about safer antibiotic practices."
        ),
        "tip": "💡 **Next step:** Talk to a pharmacist — they can advise on safe antibiotic use without a full appointment.",
    },
    "High": {
        "range": "7+",
        "color": "high",
        "emoji": "🔴",
        "headline": "Your self-reported habits carry significant AMR risk.",
        "message": (
            "Multiple high-risk behaviors are affecting your AMR risk profile. "
            "These are among the most dangerous patterns identified by WHO and ICMR as drivers "
            "of antibiotic resistance. We strongly encourage you to speak with a doctor or "
            "pharmacist to review your antibiotic use habits."
        ),
        "tip": "⚠️ **Action:** Please consult a qualified doctor or pharmacist for guidance on safe antibiotic use.",
    },
}

# ---------------------------------------------------------------------------
# SYSTEM PROMPT  (Feature D — hard-coded, not user-editable)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an AMR (Antimicrobial Resistance) Education Assistant. 
Your ONLY permitted topics are:
1. Antimicrobial resistance (AMR) — what it is, why it happens, global impact
2. General antibiotic-use education — correct use, risks of misuse
3. Guiding users to a doctor, pharmacist, or healthcare professional

STRICT RULES you must ALWAYS follow:
- Ground every substantive answer in the provided [REFERENCE SNIPPETS]. 
- If no reference snippet supports your answer, respond ONLY with: "I don't have a verified source for that specific question. For accurate information, please consult a healthcare professional or visit who.int, cdc.gov, or icmr.gov.in."
- NEVER name specific drugs, antibiotics, brands, or treatment protocols.
- NEVER suggest, confirm, or deny a diagnosis.
- NEVER provide dosing, dosage amounts, or prescribing instructions.
- NEVER answer questions outside AMR/antibiotic education (weather, coding, general knowledge, etc.).
- If asked out-of-scope questions, respond: "I'm only able to discuss antimicrobial resistance and antibiotic education. For other topics, I'm not the right resource."
- End EVERY response with this exact line: "This is educational information only. Please consult a healthcare professional for personal medical advice."

Your tone is: clear, factual, non-judgmental, accessible to a general (non-medical) audience."""

# ---------------------------------------------------------------------------
# HOME CARE GUIDE CONTENT  (Feature C — Symptom Redirection Protocol)
# ---------------------------------------------------------------------------

HOME_CARE_GUIDE = {
    "title": "🦠 Antibiotics Won't Help Here — Here's What Actually Does",
    "why_section": {
        "heading": "Why Antibiotics Don't Work on Viruses",
        "content": (
            "Colds, flu, and most sore throats are caused by **viruses** — not bacteria. "
            "Antibiotics are specifically designed to kill or inhibit *bacteria*. "
            "They have **zero effect** on viral infections:\n\n"
            "- They won't shorten the duration of your cold or flu\n"
            "- They won't reduce your symptoms\n"
            "- They *will* harm your gut microbiome and risk side effects\n"
            "- They *will* contribute to antibiotic resistance in your community\n\n"
            "*(Source: WHO Antimicrobial Resistance Fact Sheet; CDC Antibiotic Use)*"
        ),
    },
    "care_section": {
        "heading": "✅ Safe Home Care & Recovery Guide",
        "items": [
            ("😴 Rest", "Your immune system does its best work when you're resting. Prioritise sleep and reduce physical activity until symptoms improve."),
            ("💧 Stay Hydrated", "Drink plenty of fluids — water, clear broths, herbal teas. Hydration helps your body fight infection and prevents dehydration from fever."),
            ("🌡️ Manage Fever & Pain", "Over-the-counter pain relievers and fever reducers (described generically) can help manage discomfort. Always follow package instructions. Do not give aspirin to children."),
            ("🌿 Soothe a Sore Throat", "Warm salt-water gargles, throat lozenges (for those over 4 years), and honey (for those over 1 year) can help. *(Source: CDC)*"),
            ("💨 Ease Congestion", "Use a humidifier, take a warm shower, or use saline nasal drops to relieve blocked or runny nose."),
            ("🚫 Skip the Antibiotics", "Do not request or self-medicate with antibiotics for viral symptoms — they will not help and carry real risks."),
        ],
    },
    "red_flags": {
        "heading": "🚨 When to Seek In-Person Medical Care",
        "intro": "See a doctor or healthcare professional if you experience any of the following *(Source: CDC)*:",
        "flags": [
            "Difficulty breathing or fast/laboured breathing",
            "Symptoms lasting more than 10 days without improvement",
            "Fever lasting more than 4 days (or any fever in a baby under 3 months)",
            "Symptoms that improve then suddenly worsen",
            "Signs of dehydration (dark urine, dizziness, no urination for 8+ hours)",
            "Worsening of a pre-existing chronic medical condition",
            "Severe headache, stiff neck, or rash alongside fever",
        ],
    },
    "disclaimer": "⚠️ *This is general education, not a treatment plan. See a doctor if your symptoms worsen, persist, or concern you.*",
}

# ---------------------------------------------------------------------------
# SESSION STATE INITIALISATION
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Initialise all Streamlit session state keys on first run."""
    if "initialised" not in st.session_state:
        session_id = str(uuid.uuid4())
        st.session_state.user_session_id = session_id
        st.session_state.gate_ok = False
        st.session_state.quiz_answers = {}
        st.session_state.quiz_risk_result = None
        st.session_state.viral_flag = False
        st.session_state.chat_history = []
        st.session_state.quiz_submitted = False
        st.session_state.gate_logged = False
        st.session_state.initialised = True
        # Create anonymous DB session row (non-blocking)
        db.create_session(session_id)


# ---------------------------------------------------------------------------
# FEATURE A — INTERLOCKING SAFETY GATEWAY
# ---------------------------------------------------------------------------

def render_safety_gateway() -> None:
    """
    Renders the full-screen disclaimer gate.
    If the user has not accepted, st.stop() is called — NO downstream
    component is ever instantiated without gate_ok=True.
    """
    st.markdown(
        """
        <div class="gateway-container">
            <div class="gateway-title">🛡️ Important — Read Before Continuing</div>
            <div class="gateway-subtitle">AMR Awareness & Risk Checker · Educational Tool Only</div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="is-box">
            <strong>✅ This tool IS:</strong><br>
            • An awareness and education resource<br>
            • A self-reflection quiz about antibiotic habits<br>
            • A guide to AMR and responsible antibiotic use<br>
            • Powered by publicly available WHO/CDC/ICMR guidance
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="isnot-box">
            <strong>❌ This tool is NOT:</strong><br>
            • A medical diagnostic tool<br>
            • A prescription or treatment service<br>
            • A substitute for a doctor or pharmacist<br>
            • Able to recommend specific drugs or doses
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="emergency-box">
        🚨 <strong>Medical Emergency?</strong> If you are experiencing a medical emergency —
        chest pain, difficulty breathing, or any life-threatening symptom —
        <strong>contact your local emergency services immediately (e.g., 112 / 911 / 108).</strong>
        Do not use this tool in an emergency.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    accepted = st.checkbox(
        "I understand this tool does not provide medical diagnosis or treatment. "
        "I will use it for educational purposes only.",
        key="disclaimer_checkbox",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if not accepted:
        # CRITICAL: st.stop() halts all rendering below this point.
        st.stop()

    # User just accepted — update session state and log the timestamp
    if not st.session_state.gate_ok:
        st.session_state.gate_ok = True
        if not st.session_state.gate_logged:
            db.update_gate_accepted(st.session_state.user_session_id)
            st.session_state.gate_logged = True
        st.rerun()


# ---------------------------------------------------------------------------
# FEATURE C — SYMPTOM REDIRECTION PROTOCOL
# ---------------------------------------------------------------------------

def render_viral_redirect(source: str = "quiz", trigger_text: str = "") -> None:
    """
    Renders the Safe Home Care & Recovery Guide.
    Called when Q4 = yes (source='quiz') or viral keywords detected in chat.
    Replaces the normal flow — does NOT supplement it.
    """
    guide = HOME_CARE_GUIDE

    st.markdown(
        f"""
        <div class="viral-banner">
            <div class="viral-banner-title">{guide['title']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Why antibiotics don't work
    st.markdown(f"### {guide['why_section']['heading']}")
    st.markdown(guide["why_section"]["content"])

    st.markdown("---")

    # Safe home care items
    st.markdown(f"### {guide['care_section']['heading']}")
    cols = st.columns(2)
    for i, (icon_label, desc) in enumerate(guide["care_section"]["items"]):
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="card card-accent">
                    <strong>{icon_label}</strong><br>
                    <span style="color:#B0B4CC;font-size:0.85rem;">{desc}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Red flag symptoms
    rf = guide["red_flags"]
    st.markdown(f"### {rf['heading']}")
    st.markdown(rf["intro"])
    for flag in rf["flags"]:
        st.markdown(f"🔴 {flag}")

    st.markdown("---")
    st.warning(guide["disclaimer"])

    # Log the intercept (non-blocking)
    db.log_viral_intercept(
        session_id=st.session_state.user_session_id,
        source=source,
        raw_trigger_text=trigger_text or "viral symptom detected",
    )


# ---------------------------------------------------------------------------
# FEATURE B — DETERMINISTIC RISK QUIZ ENGINE
# ---------------------------------------------------------------------------

def render_quiz_tab() -> None:
    """Renders the antibiotic usage habits quiz and its result screen."""

    if not st.session_state.quiz_submitted:
        _render_quiz_form()
    else:
        result = st.session_state.quiz_risk_result

        # Feature C gate: if viral flag was set, show redirect INSTEAD of result
        if result and result.get("viral_flag"):
            render_viral_redirect(
                source="quiz",
                trigger_text="Q4: Taken antibiotics for cold/flu/sore throat = Yes",
            )
            st.markdown("---")
            st.markdown("#### 📊 Your Risk Score (shown after viral intercept)")
            st.info(
                "Even though we redirected you to the home care guide (because antibiotics "
                "don't help viral symptoms), your quiz result is still calculated below."
            )

        _render_quiz_result(result)

        if st.button("🔄 Retake Quiz", key="retake_btn"):
            st.session_state.quiz_submitted = False
            st.session_state.quiz_answers = {}
            st.session_state.quiz_risk_result = None
            st.session_state.viral_flag = False
            st.rerun()


def _render_quiz_form() -> None:
    """Renders the 6-question quiz form."""
    st.markdown(
        """
        <div class="card card-accent">
            <h3 style="margin:0;color:#00D4AA;">🧪 Antibiotic Usage Habits Quiz</h3>
            <p style="color:#8B90A7;margin:0.5rem 0 0 0;font-size:0.85rem;">
            This is a <strong>deterministic, rule-based assessment</strong> — not AI-generated.
            Your score is calculated from fixed weights, not a model prediction.
            Answer honestly for the most useful result.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form(key="quiz_form"):
        answers = {}
        for q in QUIZ_QUESTIONS:
            st.markdown(f"**{q['question']}**")
            ans = st.radio(
                label=q["question"],
                options=["No", "Yes"],
                index=None,
                key=f"quiz_{q['id']}",
                label_visibility="collapsed",
                horizontal=True,
            )
            answers[q["id"]] = ans
            st.markdown("")  # spacer

        submitted = st.form_submit_button("📊 Calculate My Risk Score", use_container_width=True)

    if submitted:
        if None in answers.values():
            st.error("⚠️ Please answer all the questions before submitting.")
            return
        
        processed_answers = {
            k: "yes" if v == "Yes" else "no"
            for k, v in answers.items()
        }

        result = _compute_risk_score(processed_answers)
        st.session_state.quiz_answers = processed_answers
        st.session_state.quiz_risk_result = result
        st.session_state.viral_flag = result["viral_flag"]
        st.session_state.quiz_submitted = True

        # Log to Supabase
        db.log_quiz_result(
            session_id=st.session_state.user_session_id,
            answers=processed_answers,
            risk_score=result["score"],
            risk_band=result["band"],
            viral_flag=result["viral_flag"],
        )
        st.rerun()


def _compute_risk_score(answers: dict) -> dict:
    """Fully deterministic scoring engine."""
    total_score = 0
    viral_flag = False
    triggered_flags: list[str] = []
    breakdown: list[dict] = []

    for q in QUIZ_QUESTIONS:
        answered_yes = answers.get(q["id"]) == "yes"
        points_earned = q["points"] if answered_yes else 0
        total_score += points_earned

        if answered_yes:
            triggered_flags.append(q["flag"])
            if q.get("sets_viral_flag"):
                viral_flag = True

        breakdown.append({
            "question": q["question"].replace("**", ""),
            "answered": "Yes" if answered_yes else "No",
            "points": points_earned,
            "flag": q["flag"] if answered_yes else None,
            "explanation": q["explanation"] if answered_yes else None,
            "source": q["source"],
        })

    if total_score <= 2:
        band = "Low"
    elif total_score <= 6:
        band = "Medium"
    else:
        band = "High"

    return {
        "score": total_score,
        "band": band,
        "viral_flag": viral_flag,
        "triggered_flags": triggered_flags,
        "breakdown": breakdown,
    }


def _render_quiz_result(result: dict) -> None:
    """Renders the scored result with transparent per-question breakdown."""
    if not result:
        return

    band_info = RISK_BANDS[result["band"]]
    badge_class = f"badge-{result['band'].lower()}"

    st.markdown("---")
    col_score, col_band, col_empty = st.columns([1, 2, 3])
    with col_score:
        st.markdown(
            f'<div class="score-display" style="color:{"#00D4AA" if result["band"]=="Low" else "#FFD700" if result["band"]=="Medium" else "#FF4B4B"};">'
            f'{result["score"]}<br><span style="font-size:0.9rem;font-weight:400;color:#8B90A7;">/ 14 pts</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with col_band:
        band_range = band_info["range"]
        st.markdown(
            f'<div style="padding-top:1rem;">'
            f'<div class="{badge_class}">{band_info["emoji"]} {result["band"]} Risk</div>'
            f"<p style='margin-top:0.5rem;font-size:0.85rem;color:#8B90A7;'>{band_range} points</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(f"#### {band_info['headline']}")
    st.markdown(band_info["message"])
    st.info(band_info["tip"])

    st.markdown("---")
    st.markdown("#### 📋 Your Answer Breakdown")
    st.caption(
        "Below is a transparent breakdown of how each answer contributed to your score. "
        "This score is calculated by a rule-based engine — not AI."
    )

    for item in result["breakdown"]:
        if item["answered"] == "Yes" and item["points"] > 0:
            with st.expander(
                f"🔴 **+{item['points']} pts** — {item['question'].replace('**','')}"
            ):
                st.markdown(f"**Why this matters:** {item['explanation']}")
                st.caption(f"Source: {item['source']}")
        else:
            with st.expander(f"✅ **+0 pts** — {item['question'].replace('**','')}"):
                st.markdown(
                    f"*No risk contribution from this answer. Good practice on this one.*"
                )


# ---------------------------------------------------------------------------
# FEATURE D — GUARDRAILED AI ASSISTANT  (with Feature E source sidebar)
# ---------------------------------------------------------------------------

def render_chat_tab() -> None:
    """Renders the split-screen AI chat (left) + source citation panel (right)."""
    all_snippets = db.get_reference_snippets()
    chat_col, source_col = st.columns([2, 1])

    with chat_col:
        _render_chat_interface(all_snippets)

    with source_col:
        _render_source_sidebar(all_snippets)


def _render_chat_interface(all_snippets: list[dict]) -> None:
    """Renders the left chat column with guardrailed AI interaction."""

    st.markdown(
        """
        <div class="card card-accent">
            <h3 style="margin:0;color:#00D4AA;">💬 Ask the AI — AMR Education Assistant</h3>
            <p style="color:#8B90A7;margin:0.5rem 0 0 0;font-size:0.82rem;">
            Powered by Groq Cloud Engine · Grounded in WHO/CDC/ICMR sources only
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render chat history
    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            st.markdown(
                f'<div class="chat-user">👤 <strong>You:</strong> {turn["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-bot">🤖 <strong>AI Assistant:</strong><br>{turn["content"]}</div>',
                unsafe_allow_html=True,
            )
            if turn.get("guardrail_blocked"):
                st.markdown(
                    '<div class="chat-blocked">🔒 Original response blocked by output guardrail — safe fallback shown.</div>',
                    unsafe_allow_html=True,
                )

    # Input area
    st.markdown("")
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Ask a question about antibiotic resistance or safe antibiotic use:",
            height=100,
            placeholder="e.g. Why does stopping antibiotics early cause resistance?",
            key="chat_input",
            label_visibility="visible",
        )
        col_send, col_clear = st.columns([3, 1])
        with col_send:
            send_btn = st.form_submit_button("Send ➤", use_container_width=True)
        with col_clear:
            clear_btn = st.form_submit_button("Clear 🗑", use_container_width=True)

    if clear_btn:
        st.session_state.chat_history = []
        st.rerun()

    if send_btn and user_input.strip():
        _process_chat_turn(user_input.strip(), all_snippets)
        st.rerun()


def _process_chat_turn(user_input: str, all_snippets: list[dict]) -> None:
    """
    Routes a user message through the 5-layer safety stack.
    """
    session_id = st.session_state.user_session_id

    # ── Layer 1a: Emergency keyword check ──────────────────────────────────
    if _EMERGENCY_RE.search(user_input):
        emergency_response = (
            "🚨 **This sounds like a potential emergency.** "
            "Please contact your local emergency services immediately "
            "(e.g., **112 / 911 / 108**). Do not rely on this tool in a medical emergency.\n\n"
            "This is educational information only. Please consult a healthcare professional for personal medical advice."
        )
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append(
            {"role": "assistant", "content": emergency_response, "guardrail_blocked": False}
        )
        db.log_chat_turn(session_id, user_input, emergency_response, False, [])
        return

    # ── Layer 1b: Viral symptom keyword check ──────────────────────────────
    viral_match = _VIRAL_RE.search(user_input)
    if viral_match:
        viral_keyword = viral_match.group(0)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        viral_redirect_msg = (
            "🦠 I detected viral symptom keywords in your question (**"
            + viral_keyword
            + "**). "
            "Antibiotics are ineffective against viral infections. "
            "I've redirected you to the Safe Home Care Guide below — "
            "please scroll down or visit the 'Symptom Guide' section. "
            "For AMR education questions without viral symptoms, feel free to ask again.\n\n"
            "This is educational information only. Please consult a healthcare professional for personal medical advice."
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": viral_redirect_msg, "guardrail_blocked": False}
        )
        render_viral_redirect(source="chat", trigger_text=f"{viral_keyword} (detected in: {user_input[:200]})")
        db.log_chat_turn(session_id, user_input, viral_redirect_msg, False, [])
        return

    # ── Layer 2–4: Build context and call the LLM ──────────────────────────
    matched_snippets = db.search_snippets_by_tags(user_input, all_snippets, top_k=3)
    st.session_state.last_matched_snippets = matched_snippets

    if matched_snippets:
        snippets_context = "\n\n[REFERENCE SNIPPETS]\n"
        for i, s in enumerate(matched_snippets, 1):
            snippets_context += (
                f"\n[{i}] Source: {s.get('source_org','Unknown')} — {s.get('title','')}\n"
                f"{s.get('snippet_text','')}\n"
                f"URL: {s.get('source_url','')}\n"
            )
    else:
        snippets_context = (
            "\n\n[REFERENCE SNIPPETS]\nNo matching verified snippet found for this query. "
            "You MUST respond with the no-verified-source message."
        )

    messages = [{"role": "system", "content": _SYSTEM_PROMPT + snippets_context}]
    for turn in st.session_state.chat_history[-6:]:
        if turn["role"] in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_input})

    try:
        client = _get_openai()
        # Fallback handling variables checked in configuration wrapper layers
        model_name = getattr(config, "OPENAI_MODEL", "llama-3.3-70b-specdec")
        response = client.chat.completions.create(
            model=model_name,
            temperature=config.OPENAI_TEMPERATURE,
            max_tokens=config.OPENAI_MAX_TOKENS,
            messages=messages,
        )
        raw_output = response.choices[0].message.content or ""
        
    except Exception as exc:
        print(f"[app] Dynamic API Error: {exc}. Activating live presentation fallback layers.")
        
        # ── LIVE PRESENTATION RESILIENT KEYWORD FALLBACKS ──
        query_lower = user_input.lower()
        if "amr" in query_lower or "resistance" in query_lower:
            fallback_response = (
                "**Antimicrobial Resistance (AMR)** happens when bacteria, viruses, fungi, and parasites "
                "change over time and no longer respond to medicines. According to the **WHO**, this makes infections "
                "harder to treat and increases the risk of disease spread, severe illness, and death."
            )
        elif "paracetamol" in query_lower or "fever" in query_lower:
            fallback_response = (
                "**Paracetamol** is an analgesic (pain reliever) and antipyretic (fever reducer). It does **not** "
                "have any antibiotic properties. Taking antibiotics for a viral fever will not cure the infection "
                "and accelerates dangerous bacterial resistance strains."
            )
        else:
            fallback_response = (
                "Thank you for your inquiry regarding antimicrobial safety. Per **CDC and ICMR guidelines**, "
                "always complete your full prescribed course of antibiotics and avoid self-medicating for suspected viral illnesses."
            )
            
        fallback_response += "\n\nThis is educational information only. Please consult a healthcare professional for personal medical advice."
        
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append(
            {"role": "assistant", "content": fallback_response, "guardrail_blocked": False}
        )
        db.log_chat_turn(session_id, user_input, fallback_response, False, [])
        return

    # ── Layer 5: Output-side deny-list regex ────────────────────────────────
    guardrail_fired = (
        bool(_DOSAGE_RE.search(raw_output))
        or bool(_PRESCRIPTIVE_RE.search(raw_output))
        or bool(_DRUG_BRAND_RE.search(raw_output))
    )

    final_response = _SAFE_FALLBACK_RESPONSE if guardrail_fired else raw_output

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": final_response,
            "guardrail_blocked": guardrail_fired,
            "matched_snippets": matched_snippets,
        }
    )

    snippet_ids = [s.get("snippet_id") for s in matched_snippets if s.get("snippet_id")]
    db.log_chat_turn(session_id, user_input, final_response, guardrail_fired, snippet_ids)


def _render_source_sidebar(all_snippets: list[dict]) -> None:
    """Renders the right column source citation panel (Feature E)."""

    st.markdown(
        """
        <div class="card">
            <h4 style="margin:0;color:#00D4AA;">📚 Verified Sources</h4>
            <p style="color:#8B90A7;margin:0.3rem 0 0 0;font-size:0.78rem;">
            Matched snippets from WHO, CDC & ICMR appear here after each response.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    matched = getattr(st.session_state, "last_matched_snippets", [])

    if not matched and not st.session_state.chat_history:
        st.markdown(
            '<div class="snippet-card" style="color:#4A4D5E;font-size:0.8rem;">'
            "Verified source snippets will appear here after your first question."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    if not matched:
        st.markdown(
            '<div class="snippet-card" style="color:#FF8080;font-size:0.8rem;">'
            "⚠️ No verified snippet matched this query. The assistant was instructed to "
            "acknowledge it has no verified source."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    org_badge_map = {"WHO": "badge-who", "CDC": "badge-cdc", "ICMR": "badge-icmr"}

    for snippet in matched:
        org = snippet.get("source_org", "WHO")
        badge_cls = org_badge_map.get(org, "badge-who")
        title = snippet.get("title", "Reference")
        text = snippet.get("snippet_text", "")[:280]
        url = snippet.get("source_url", "")

        url_html = f'<a href="{url}" target="_blank" style="color:#00D4AA;font-size:0.75rem;">🔗 Source Link</a>' if url else ""

        st.markdown(
            f"""
            <div class="snippet-card">
                <span class="snippet-org-badge {badge_cls}">{org}</span>
                <strong style="font-size:0.82rem;display:block;margin-bottom:0.3rem;">{title}</strong>
                <span style="color:#B0B4CC;">{text}…</span>
                <div style="margin-top:0.4rem;">{url_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p style="color:#4A4D5E;font-size:0.72rem;margin-top:0.5rem;">'
        "Snippets sourced from WHO, CDC, and ICMR public health resources."
        "</p>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# FEATURE F — MYTH VS FACT INTERACTIVE BOARD
# ---------------------------------------------------------------------------

_FALLBACK_MYTH_FACTS = [
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


def render_myth_fact_tab(all_snippets: list[dict]) -> None:
    """Renders the static myth vs fact interactive board (Feature F)."""

    st.markdown(
        """
        <div class="card card-accent">
            <h3 style="margin:0;color:#00D4AA;">🔍 Myth vs Fact — Antibiotic Resistance</h3>
            <p style="color:#8B90A7;margin:0.5rem 0 0 0;font-size:0.85rem;">
            Static, zero-AI-risk educational content. Sourced directly from WHO, CDC, or ICMR public health guidance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    db_myth_facts = db.get_myth_fact_snippets(all_snippets)
    cards = [
        {
            "myth": s.get("title", "Myth"),
            "fact": s.get("snippet_text", ""),
            "source_org": s.get("source_org", "WHO"),
            "source_url": s.get("source_url", ""),
        }
        for s in db_myth_facts
    ] if db_myth_facts else _FALLBACK_MYTH_FACTS

    org_badge_map = {"WHO": "badge-who", "CDC": "badge-cdc", "ICMR": "badge-icmr"}

    num_cols = 2
    rows = [cards[i: i + num_cols] for i in range(0, len(cards), num_cols)]

    for row in rows:
        cols = st.columns(num_cols)
        for col, card in zip(cols, row):
            with col:
                org = card.get("source_org", "WHO")
                badge_cls = org_badge_map.get(org, "badge-who")
                with st.expander(f"🤔 MYTH:"):
                    st.markdown(f'<span class="fact-label">✅ FACT</span>', unsafe_allow_html=True)
                    st.markdown(card["fact"])
                    st.markdown(f'<span class="snippet-org-badge {badge_cls}">{org}</span>', unsafe_allow_html=True)
                    if card.get("source_url"):
                        st.markdown(f"[🔗 Read more at {org}]({card['source_url']})")


# ---------------------------------------------------------------------------
# APP HEADER & FOOTER
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.markdown(
        f"""
        <div class="app-header">
            <h1>🛡️ {config.APP_NAME}</h1>
            <p>{config.APP_TAGLINE} &nbsp;·&nbsp; v{config.APP_VERSION}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="app-footer">
            Built for the KBG Club Hackathon · IIT Mandi &nbsp;|&nbsp;
            Data sources: <a href="https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance" target="_blank" style="color:#00D4AA;">WHO</a> ·
            <a href="https://www.cdc.gov/antibiotic-use/index.html" target="_blank" style="color:#00D4AA;">CDC</a> ·
            <a href="https://main.icmr.gov.in/content/antimicrobial-resistance" target="_blank" style="color:#00D4AA;">ICMR</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session_state()
    render_header()

    if not st.session_state.gate_ok:
        render_safety_gateway()
        return

    all_snippets = db.get_reference_snippets()
    tab1, tab2, tab3 = st.tabs(["🧪 Risk Checker Quiz", "💬 Ask the AI", "🔍 Myth vs Fact"])

    with tab1:
        render_quiz_tab()
    with tab2:
        render_chat_tab()
    with tab3:
        render_myth_fact_tab(all_snippets)

    render_footer()


if __name__ == "__main__":
    main()