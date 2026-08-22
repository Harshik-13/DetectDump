# HACKATHON OPERATING CONSTITUTION

## PRIMARY OBJECTIVE

Build the smallest possible product that delivers a powerful, obvious, judge-visible solution and works reliably during demonstration.

---

## DECISION FILTER

Before every action, answer: **"Does this directly help us produce a more convincing, reliable hackathon prototype?"**

- YES → Proceed
- NO → Don't do it
- Uncertain → Stop, inspect, reason, don't guess

---

## PRIORITY ORDER (Highest → Lowest)

1. Broken primary demo
2. Missing primary functionality
3. Critical bug
4. Major UX problem
5. Important security issue
6. Demo reliability
7. Differentiating feature
8. UI polish
9. Secondary functionality
10. Everything else

---

## WHAT MATTERS

| Hackathon | NOT Hackathon |
|-----------|---------------|
| Demonstrable value | Massive scale |
| Reliability | Enterprise architecture |
| Speed | Perfect abstraction |
| Simplicity | Excessive modularity |
| Polish | Future hypotheticals |
| Differentiation | Complex infrastructure |

---

## RULES

- **SCOPE IS SACRED** — Never independently add features, pages, workflows, integrations, or services
- **NEVER GUESS** — Inspect code before modifying; state uncertainty; stop and ask if blocked
- **100% CONFIDENCE RULE** — No consequential changes without complete understanding
- **MINIMUM VIABLE FIRST** — Core → Working → Verify → Polish
- **PREFER SIMPLE TECH** — Don't add dependencies unless existing stack can't solve it
- **NO PREMATURE ARCHITECTURE** — No microservices, message queues, complex caching, etc.
- **VALIDATE AT SERVER** — Never trust client-provided data for important rules
- **NEVER BREAK WORKING CODE** — Inspect → Understand → Smallest change → Test
- **TEST P0 FIRST** — Primary demo workflow must work before anything else

---

## STOP CONDITION

Stop building when:
- Core workflow works
- Demo is reliable
- UI is polished
- Important security addressed
- Deployment works
- Critical tests pass
- Demo has been rehearsed

Then switch to: Testing → Demo rehearsal → Bug fixing → Presentation prep

---

## DEMO REQUIREMENTS

Primary demo must be:
- Predictable
- Fast
- Repeatable
- Easy to understand
- Resistant to common failures

---

## SELF-REVIEW CHECKLIST

Before declaring any task complete:

- [ ] Requirement actually satisfied
- [ ] No unnecessary features added
- [ ] No unnecessary dependencies added
- [ ] Existing functionality preserved
- [ ] Primary workflow still works
- [ ] Important failure cases considered
- [ ] Security boundary considered
- [ ] Relevant tests executed
- [ ] UI behavior checked if applicable
- [ ] No obvious regression introduced

---

## BUG CLASSIFICATION

| Priority | Definition | Action |
|----------|------------|--------|
| P0 | Breaks demo/core | Fix immediately |
| P1 | Damages UX/functionality | Fix before polish |
| P2 | Visible but non-critical | Fix if time |
| P3 | Cosmetic/minor | Fix last |

---

## WHEN DISCOVERING A BETTER IDEA

Report format:
```
CURRENT APPROACH: ...
ALTERNATIVE: ...
BENEFIT: ...
COST: ...
DEMO IMPACT: ...
RECOMMENDATION: ...
```
Then wait for approval if significant.

---

## JUDGE PERSPECTIVE

Continuously evaluate:
- What problem is being solved?
- Is it immediately understandable?
- Can value be demonstrated quickly?
- What makes this different?
- What is the "wow" moment?
- Does it feel intentional and complete?

---

## FINAL PRINCIPLE

> **MAXIMIZE THE PROBABILITY THAT A JUDGE EXPERIENCES A SMALL, POLISHED, RELIABLE PRODUCT AND IMMEDIATELY UNDERSTANDS WHY IT IS GOOD.**

```
Small > Large
Working > Complete
Reliable > Sophisticated
Clear > Complex
Polished > Feature-heavy
Demonstrable > Theoretical
Necessary > "Nice to have"
Verified > Assumed
```

---

## CURRENT PROJECT STATUS

**Repository**: https://github.com/Harshik-13/DetectDump.git
**Latest tag**: `v2.0-fastapi-ui`
**Branch**: `main`

### Architecture

```
Frontend (ui/detectdump.html)
    ↓ fetch API
Backend (app_server.py — FastAPI)
    ↓ threading
CV Pipeline (YOLOv8n + ByteTrack + Temporal Engine + VLM)
    ↓
Results → JSON → Frontend renders
```

### Entry Points

| Command | Purpose |
|---------|---------|
| `python app_server.py` | Start full application (FastAPI + UI) → http://127.0.0.1:8080 |
| `python dumping_detector.py <video>` | CLI pipeline (standalone) |
| `streamlit run app.py` | Legacy Streamlit UI |

### Key Files

| File | Role |
|------|------|
| `app_server.py` | FastAPI backend — serves UI + REST API for analysis |
| `ui/detectdump.html` | Production UI — 4-stage flow connected to backend |
| `dumping_detector.py` | Standalone CLI pipeline |
| `temporal_engine.py` | Core state machine (IDLE→OBSERVING→SUSPICIOUS→ACTOR_LEFT→DUMPING_CANDIDATE) |
| `vlm_verify.py` | VLM verification via OpenRouter (GPT-4o-mini) |
| `app.py` | Legacy Streamlit UI (preserved, not primary) |

### Test Infrastructure

- `test_temporal_engine.py` — 9 unit/integration tests (all passing)
- `test_videos/` — validation videos (normal, sports ball, different environment)
- CPU-only: Intel Core Ultra 5 125H, PyTorch 2.13.0+cpu, ~7-9 FPS

### Git Tags

| Tag | Description |
|-----|-------------|
| `v0.2-phase4-vlm` | VLM verification added |
| `v0.3-phase5-ui` | Streamlit demo UI |
| `v1.0-code-freeze` | Critical fix pass |
| `v1.1-generalized` | Generalized detection (behavioral, not waste-class-specific) |
| `v2.0-fastapi-ui` | FastAPI backend + reference UI frontend |
