# Product Requirement Document
## AI-Based Antibiotic Misuse Awareness & Risk Checker

**Track:** TRK-PHA | **Difficulty:** Easy-Medium | **Team size:** 2 | **Build window:** 48 hours
**Stack:** Python (Streamlit) · OpenAI API (gpt-4o-mini) · PostgreSQL via Supabase

---

## 1. Executive Summary & Objective

### 1.1 Problem Statement
Antimicrobial resistance (AMR) is accelerated by everyday misuse: self-medication, incomplete courses, and antibiotic use for viral infections (cold/flu) where they have no effect. Most people don't have an accessible, judgment-free way to check whether their own antibiotic habits are risky, or to understand *why*.

### 1.2 Objective
Build a **web-based awareness and risk-literacy tool** that:
- Helps a user reflect on their antibiotic usage habits via a deterministic quiz (not an AI diagnosis).
- Flags unsafe patterns (self-medication, early stoppage, viral-infection misuse) and explains *why* they're risky.
- Answers AMR-related questions through a guardrailed AI assistant, always citing WHO/CDC/ICMR source material.
- Redirects viral-symptom queries away from antibiotic-seeking behavior toward safe supportive care guidance.

### 1.3 Non-Negotiable Compliance Boundary
> **Nothing built here diagnoses anyone, prescribes anything, or recommends a specific drug, dose, or brand.**

This boundary is enforced at three independent layers so a single point of failure can't break it:
1. **UI layer** — a blocking disclaimer gate (Feature A) the user must acknowledge before any interaction is possible.
2. **Logic layer** — the risk quiz (Feature B) is rule-based/deterministic, not LLM-based, so risk output is never model-generated.
3. **Model layer** — the AI assistant (Feature D) runs a hard system-prompt guardrail plus output-side keyword/regex screening (Section 5) that blocks prescriptive language before it reaches the user.

### 1.4 Success Criteria for Judging
- Awareness content is clear and jargon-free for a general (non-medical) audience.
- Risk logic is transparent and honest — no black-box "AI says you're high risk."
- The safety disclaimer and guardrails are functionally real (testable), not a static paragraph.

---

## 2. User Persona & App Flow

### 2.1 Primary Persona
**"Reactive Rohan"** — 20s–40s, self-medicates when he feels ill, has leftover antibiotics at home, trusts the pharmacist over a doctor for minor illnesses, has never heard the term "antimicrobial resistance" explicitly but has vaguely heard "antibiotics stop working."

Secondary persona: **"Concerned Caregiver"** — parent/guardian managing medication for a child or elderly relative, wants a fast risk check, not a chatbot essay.

### 2.2 Text-Based Architecture / App Flow

```
┌─────────────────────────────────────────────────────────────┐
│  ENTRY POINT: Streamlit app load                             │
└───────────────────────────┬───────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ FEATURE A: Interlocking Safety Gateway                       │
│  - Full-screen modal/blocking layout                         │
│  - Checkbox: "I understand this is not medical advice"       │
│  - st.stop() halts render of ANY other component until       │
│    checkbox == True                                           │
└───────────────────────────┬───────────────────────────────────┘
                             ▼ (gate passed → session_state.gate_ok=True)
┌─────────────────────────────────────────────────────────────┐
│ HOME: Tab-based navigation (st.tabs)                          │
│  [1] Risk Checker Quiz   [2] Ask the AI   [3] Myth vs Fact     │
└───────┬─────────────────────────┬──────────────────┬──────────┘
        ▼                         ▼                  ▼
┌────────────────────┐  ┌──────────────────────┐  ┌────────────────┐
│ FEATURE B: Quiz     │  │ FEATURE D+E: Chat +   │  │ FEATURE F:      │
│ Rule-based Q&A      │  │ Source Sidebar         │  │ Myth/Fact board │
│                     │  │                        │  │ (static, no AI) │
│ IF viral symptom    │  │ Split screen:          │  └─────────────────┘
│ flag == True        │  │  Left = chat            │
│    │                │  │  Right = matched WHO/   │
│    ▼                │  │  CDC/ICMR snippet(s)    │
│ FEATURE C:          │  │                        │
│ Symptom Redirection │  │ System prompt +         │
│ Protocol → Safe     │  │ output guardrail        │
│ Home Care Guide     │  │ (Section 5)             │
│ (intercepts, skips  │  └───────────┬────────────┘
│  antibiotic loop)   │              ▼
│    │                │      Supabase: chat_logs
│    ▼ (no viral flag)│
│ Deterministic risk  │
│ scoring engine       │
│ → Low / Medium /    │
│   High + explanation │
│    │                │
│    ▼                │
│ Supabase: quiz_logs │
└─────────────────────┘
```

### 2.3 Session State Keys (Streamlit)
`gate_ok`, `quiz_answers`, `quiz_risk_result`, `viral_flag`, `chat_history`, `user_session_id` (anonymous UUID, no PII collected).

---

## 3. Core Feature Specifications

### Feature A — The Interlocking Safety Gateway
**Goal:** Make the disclaimer a functional gate, not decoration.

- Renders as the *only* visible component on first load (`st.session_state.gate_ok` defaults to `False`).
- Content: what the tool is (awareness/education), what it is NOT (diagnosis, prescription, emergency service), and an emergency-care disclaimer ("If you are experiencing a medical emergency, contact local emergency services immediately").
- A single checkbox: *"I understand this tool does not provide medical diagnosis or treatment."*
- Implementation: if checkbox is unchecked, call `st.stop()` immediately after rendering the gate — this halts execution before any tab, quiz, or chat component is instantiated. No component downstream is reachable via URL, tab click, or session replay without `gate_ok=True`.
- Re-shown if the user starts a brand-new session (new `user_session_id`); persists across tab switches within a session via `st.session_state`.

**Acceptance test:** Disabling JS/refreshing without checking the box must never expose the quiz or chat inputs.

---

### Feature B — Antibiotic Usage Habits Quiz (Deterministic Risk Engine)
**Goal:** Transparent, non-AI risk scoring — this is a scored survey, not a model prediction.

**Sample question set (5–7 questions, single/multi-select):**
1. Have you taken antibiotics without a doctor's prescription in the last 6 months? (Y/N)
2. Did you stop a prescribed antibiotic course early because you felt better? (Y/N)
3. Have you used leftover antibiotics from a previous illness? (Y/N)
4. Have you taken antibiotics for cold, flu, or sore throat symptoms? (Y/N) → also sets `viral_flag`
5. Do you keep antibiotics at home "just in case"? (Y/N)
6. Have you shared your antibiotic prescription with a family member? (Y/N)

**Scoring logic (fully deterministic, weights hard-coded, not LLM-derived):**

| Behavior flagged | Points |
|---|---|
| Self-medication (no prescription) | 3 |
| Early stoppage | 3 |
| Reused leftover antibiotics | 2 |
| Used for viral symptoms | 3 (+ triggers Feature C) |
| Stockpiling "just in case" | 1 |
| Shared prescription | 2 |

**Risk bands:**
- 0–2 → **Low risk** — reinforcing message, still show 1 educational tip.
- 3–6 → **Medium risk** — explain which specific answers drove the score, in plain language.
- 7+ → **High risk** — explain which behaviors are most dangerous for AMR, strongly worded "consult a doctor/pharmacist for guidance" CTA.

Every result screen shows: total score, band, **and** a per-question breakdown of *why* ("You scored higher because you reported stopping a course early — this is one of the top contributors to resistant bacteria surviving treatment"). This transparency is what satisfies "honesty of the risk logic" in the judging rubric.

---

### Feature C — Symptom Redirection Protocol
**Goal:** Intercept viral-symptom-driven antibiotic-seeking before it reinforces misuse.

- Triggered when quiz Q4 (or equivalent) is answered "Yes" (self-reported antibiotic use for cold/flu/sore throat), **or** if the AI chat detects viral-symptom language via a lightweight keyword check (`cold`, `flu`, `sore throat`, `runny nose`, `viral`, etc.) before it ever reaches the LLM.
- On trigger: the normal quiz-scoring flow (for that path) or chat response is **replaced**, not supplemented, with a structured **"Safe Home Care & Recovery Guide"** screen:
  - What viral infections are and why antibiotics don't work on them (plain-language AMR explainer).
  - CDC-sourced supportive care guidance: rest, fluids, OTC symptom relief categories (described generically — no dosing), when to seek in-person care (red-flag symptoms list from CDC).
  - Explicit line: *"This is general education, not a treatment plan. See a doctor if symptoms worsen or persist."*
- Logged to `intercepted_viral_queries` table (Section 4) for the metrics/judging dashboard — this is your strongest "the safety logic is real" evidence.

---

### Feature D — Guardrailed AI Assistant
**Goal:** An LLM that can explain AMR concepts but structurally cannot prescribe.

- Model: `gpt-4o-mini`, **temperature = 0.0** (deterministic, minimizes creative drift into medical advice).
- **System prompt (hard-coded, not user-editable):**
  - Scope: may only discuss antimicrobial resistance, general antibiotic-use education, and how to navigate to a doctor/pharmacist.
  - Explicit refusal instruction: must refuse to name specific drugs, doses, brands, or treatment plans, and must refuse to confirm/deny a diagnosis, and must respond with a fixed refusal + "see a healthcare professional" message if asked to do so.
  - Must ground every substantive claim in the WHO/CDC/ICMR reference corpus (Feature E) — no answer without a matched source snippet.
- **Pre-model filter:** user input is scanned for viral-symptom keywords first (routes to Feature C) and for red-flag emergency language (routes to an emergency-disclaimer message, bypassing the LLM entirely).
- **Post-model filter:** output is scanned against a deny-list of prescriptive patterns (drug-name regex list, dosage-pattern regex like `\d+\s?mg`, phrases like "you should take/prescribe") before being shown to the user; a match triggers a safe fallback response instead of the raw model output (Section 5 has full detail).

---

### Feature E — Verified Source Citation Sidebar
**Goal:** Every AI answer is visibly traceable to a real public-health source — this directly kills "hallucination" risk in judging.

- Split-screen layout: `st.columns([2,1])` — chat on the left, source panel on the right.
- Backend: a small curated **reference snippet table** (Section 4, `reference_snippets`) pre-loaded from WHO AMR fact sheets, CDC antibiotic-use resources, and ICMR treatment guidelines (chunked into short, tagged passages).
- Retrieval approach (kept lean for 48 hrs): simple keyword/embedding-lite match between the user's question and snippet tags/titles — **not** a full RAG pipeline unless time allows (see MVP matrix). The top 1–3 matching snippets render in the right panel alongside the chat turn, each tagged with its source (WHO / CDC / ICMR) and a direct citation label.
- If no snippet matches confidently, the assistant says so explicitly rather than answering ungrounded.

---

### Feature F — Myth-vs-Fact Interactive Board
**Goal:** Static, zero-AI-risk educational content — cheap to build, high judging value for "clarity of awareness content."

- Static card grid (no LLM call): e.g. "Myth: Antibiotics cure colds." / "Fact: Colds are viral; antibiotics only work on bacteria." Flip/reveal interaction via `st.expander` or a simple click-toggle.
- Content sourced directly from the same `reference_snippets` table used by Feature E, so myths/facts and chat citations stay consistent.
- 8–10 cards is sufficient; don't over-invest here (see MVP matrix).

---

## 4. Database Schema Design (Supabase / PostgreSQL)

```sql
-- Anonymous session tracking (no PII)
CREATE TABLE sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    gate_accepted_at TIMESTAMPTZ
);

-- Quiz responses + deterministic risk output
CREATE TABLE quiz_logs (
    quiz_log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(session_id),
    answers JSONB NOT NULL,          -- {"q1": "yes", "q2": "no", ...}
    risk_score INT NOT NULL,
    risk_band TEXT NOT NULL CHECK (risk_band IN ('Low','Medium','High')),
    viral_flag BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Every time Feature C intercepts a viral-symptom flow
CREATE TABLE intercepted_viral_queries (
    intercept_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(session_id),
    source TEXT NOT NULL CHECK (source IN ('quiz','chat')),
    raw_trigger_text TEXT,           -- keyword or answer that triggered redirect
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Chat turn history for the guardrailed assistant
CREATE TABLE chat_logs (
    chat_log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(session_id),
    user_message TEXT NOT NULL,
    assistant_response TEXT NOT NULL,
    was_blocked_by_guardrail BOOLEAN NOT NULL DEFAULT FALSE,
    matched_snippet_ids UUID[],      -- FK-like array into reference_snippets
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Curated WHO / CDC / ICMR knowledge base used by Feature E and F
CREATE TABLE reference_snippets (
    snippet_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_org TEXT NOT NULL CHECK (source_org IN ('WHO','CDC','ICMR')),
    title TEXT NOT NULL,
    tags TEXT[] NOT NULL,            -- for keyword matching, e.g. {'cold','viral','myth'}
    snippet_text TEXT NOT NULL,      -- short paraphrased passage, cite-able
    source_url TEXT,
    is_myth_fact BOOLEAN NOT NULL DEFAULT FALSE
);

-- Indexes for the two hot lookup paths
CREATE INDEX idx_quiz_logs_session ON quiz_logs(session_id);
CREATE INDEX idx_chat_logs_session ON chat_logs(session_id);
CREATE INDEX idx_reference_tags ON reference_snippets USING GIN (tags);
```

**Metrics this schema gives you for free (useful for your pitch deck):** total sessions, % completing gate → quiz, average risk score distribution, count of viral-intercepts (proof the safety logic actually fires), and guardrail block rate on chat.

---

## 5. Verification & AI Safety Guardrails

Layered defense — no single layer is trusted alone.

**Layer 1 — Input classification (pre-model, deterministic, cheap):**
- Regex/keyword scan for viral-symptom terms → route to Feature C, skip the LLM call entirely.
- Regex/keyword scan for emergency/red-flag terms (e.g. "can't breathe", "chest pain", "overdose") → immediate static emergency-disclaimer response, LLM never called.

**Layer 2 — System-prompt scope lock:**
- Model is instructed it may only answer using the retrieved `reference_snippets` context (Feature E retrieval), and must decline (with a fixed message) if asked for dosing, prescriptions, drug names, or diagnosis.
- Temperature 0.0 and a low `max_tokens` cap reduce drift and rambling into unguarded territory.

**Layer 3 — Grounding requirement:**
- If no `reference_snippets` match the query with reasonable confidence, the assistant is instructed to say it doesn't have a verified source for that, rather than answer from general model knowledge. This is the single biggest hallucination-killer and directly matches the judging criterion.

**Layer 4 — Output-side regex/deny-list filter (post-model, before render):**
- Dosage pattern matches (e.g. `\d+\s?(mg|ml|mcg)`), drug-brand-name list, and prescriptive phrase patterns ("you should take", "I prescribe", "the correct dose is") are checked against the raw model output.
- Any match → discard model output, show a fixed safe-fallback message ("I can't provide dosing or prescribing guidance — please consult a doctor or pharmacist"), and log `was_blocked_by_guardrail = TRUE`.

**Layer 5 — Manual content verification (pre-hackathon, one-time):**
- All `reference_snippets` rows are populated by a human from the actual WHO/CDC/ICMR source pages (not AI-generated), satisfying the deliverable requirement "AI-generated but verified content" — content the AI *cites* is human-verified even if UI copy elsewhere is AI-drafted-then-reviewed.

**Testing checklist before demo:**
- [ ] Try to jailbreak the assistant into naming a drug/dose — confirm Layer 4 catches it.
- [ ] Ask a cold/flu question in chat — confirm it redirects to Feature C, not the LLM.
- [ ] Ask an out-of-scope question (e.g. "what's the weather") — confirm scoped refusal.
- [ ] Skip the disclaimer checkbox and try to reach the quiz via tab click — confirm `st.stop()` blocks it.

---

## 6. 48-Hour MVP Scope Matrix

| Feature | Hour 0–12 (Day 1 AM) | Hour 12–24 (Day 1 PM) | Hour 24–36 (Day 2 AM) | Hour 36–48 (Day 2 PM) | KEEP / CUT |
|---|---|---|---|---|---|
| A. Safety Gateway | Build + `st.stop()` gate | — | — | Final polish | **KEEP — non-negotiable, build first** |
| B. Risk Quiz (deterministic) | Question set + scoring logic | Result screen w/ breakdown | Supabase logging | — | **KEEP — core deliverable** |
| C. Symptom Redirection | — | Keyword trigger + static guide screen | Wire into both quiz & chat | Logging to `intercepted_viral_queries` | **KEEP — judged safety criterion** |
| D. Guardrailed Chat | — | System prompt + temp=0.0 + input filter | Output regex filter | Testing/jailbreak hardening | **KEEP — but scope prompt tightly, no fancy memory** |
| E. Source Sidebar | — | Populate `reference_snippets` (manual, ~15–20 rows) | Keyword-match retrieval + split UI | — | **KEEP — simple keyword match only, NOT full RAG/embeddings unless ahead of schedule** |
| F. Myth-vs-Fact Board | — | — | Build from same snippet table | Styling pass | **KEEP but LOW priority — build only after B, C, D, E work** |
| Database (Supabase) | Schema migration (Section 4) | — | Wire logging calls into B, C, D | — | **KEEP — required for judging metrics** |
| Auth / user accounts | — | — | — | — | **CUT — anonymous UUID sessions only** |
| Full RAG / vector embeddings | — | — | — | — | **CUT unless time allows — keyword/tag match is sufficient for 48h** |
| Multi-language support | — | — | — | — | **CUT — stretch goal only** |
| Mobile-native app | — | — | — | — | **CUT — Streamlit web only** |
| Analytics dashboard UI | — | — | — | If time remains | **CUT-to-stretch — raw Supabase table view is enough for pitch** |

**Sequencing rule for the 2-person team:** Person 1 owns Streamlit UI/UX (A, B result screens, F); Person 2 owns backend/AI logic (Supabase schema, C's trigger logic, D's guardrails, E's retrieval). Both features B and D depend on Supabase existing first — **migrate the schema in the first 2 hours**, before any feature work, so nothing gets blocked waiting on it.

**Definition of Done for demo:** A judge can (1) hit the gate and be blocked without checking the box, (2) take the quiz and get a scored, explained result, (3) trigger Feature C by mentioning "flu" in either the quiz or chat, and (4) try to jailbreak the chatbot into giving a dose and see it refuse — all four, live, in under 5 minutes.
