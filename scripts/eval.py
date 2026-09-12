"""
Evaluation harness for MyAI Memory.

Answers: "Does injecting this memory make the LLM answer better than
without it?" with a reproducible, layered methodology.

Three checks, cheapest first (named by role to avoid colliding with the memory
hierarchy L0-L3 in store.py):
  check_retrieval : deterministic hit-rate of recall() against gold facts. No API calls.
  check_pipeline  : extraction precision/recall + persona sanity. No API calls.
  eval_ab         : paired with-memory vs without-memory answers, graded by an
                    LLM-as-judge. Costs API calls; skippable with --no-judge.

Usage:
  python scripts/eval.py                 # full run (fresh eval DB, all checks)
  python scripts/eval.py --no-judge      # free checks only (no API calls)
  python scripts/eval.py --keep-db       # don't wipe before seeding
  python scripts/eval.py --db foo.db     # use a specific DB file
  python scripts/eval.py --limit 3       # only first 3 questions

IMPORTANT: by default this uses a SEPARATE database (`eval_memory.db`) so it
never touches your live `my_memory.db`. The default `--fresh` wipes that eval
DB only. Pass `--db my_memory.db --keep-db` to point at real data non-destructively.

Env vars (same as main.py / test_pipeline.py):
  LLM_API_KEY   : API key for the model under test
  LLM_BASE_URL  : OpenAI-compatible endpoint
  LLM_MODEL     : model under test (default: deepseek-chat)
  JUDGE_MODEL   : model used to grade answers (default: LLM_MODEL)

`.env.proxy.local` (then `.env.proxy`) is auto-loaded if present, matching
proxy.py. Real environment variables always take precedence over the file, so
`LLM_API_KEY=... python scripts/eval.py` still wins.

Note: the judging model SHOULD differ from the model under test to reduce
self-preference bias. Set JUDGE_MODEL explicitly for a cleaner signal.
"""

import os
import sys
import re
import json
import argparse

from openai import OpenAI
from dotenv import load_dotenv

# Make the repo root importable (store/recall/extract/aggregate) and the
# scripts dir importable (test_pipeline for the shared seed corpus).
# This MUST run before the local imports below — otherwise `import store`
# only resolves when the CWD happens to be the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

from store import MemoryStore  # noqa: E402
from recall import recall, format_for_system_prompt, MAX_CONTEXT_CHARS  # noqa: E402
from extract import extract_atoms, deduplicate_atoms  # noqa: E402
from aggregate import build_scenarios, build_persona  # noqa: E402
from test_pipeline import SEED_CONVERSATIONS  # noqa: E402  (shared frozen corpus)

load_dotenv(os.path.join(_REPO_ROOT, ".env.proxy.local"))
load_dotenv(os.path.join(_REPO_ROOT, ".env.proxy"))


# ── Config ──

LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA — hand-written, frozen. Edit here only; logic reads from these.
# ═══════════════════════════════════════════════════════════════════════
#
# A "fact" is matched by keyword GROUPS. A fact is PRESENT in some text iff
# EVERY group has at least one of its tokens/alternatives present. This is
# robust to the LLM paraphrasing (we match meaning, not exact strings).
#
#   groups=[["backend","engineer"], ["fintech"]]  -> needs a backend/engineer
#                                                    term AND a fintech term.

KNOWN_FACTS = [
    {"id": "name",           "desc": "User's name is Alex",                              "groups": [["alex"]]},
    {"id": "role",           "desc": "User is a backend engineer",                       "groups": [["backend", "engineer"]]},
    {"id": "location",       "desc": "User lives in Singapore",                          "groups": [["singapore"]]},
    {"id": "stack",          "desc": "User writes Go microservices on Kubernetes",       "groups": [["go", "golang"], ["kubernetes", "k8s"]]},
    {"id": "team",           "desc": "User leads the payments squad of six",             "groups": [["payments", "payment"], ["six", "6"]]},
    {"id": "lang_pref",      "desc": "User prefers Python over Go for quick scripts",    "groups": [["python"]]},
    {"id": "coffee",         "desc": "User drinks black coffee, no sugar",               "groups": [["coffee"], ["black", "no sugar"]]},
    {"id": "running",        "desc": "User runs 5km every morning",                      "groups": [["5km", "5 km", "5k"], ["morning", "run"]]},
    {"id": "migration_dec",  "desc": "Decided to migrate monolith to microservices",     "groups": [["monolith", "microservices", "microservice"], ["migrat"]]},
    {"id": "migration_why",  "desc": "Migration driven by scalability / Black Friday",   "groups": [["scalab", "black friday", "traffic"]]},
    {"id": "migration_time", "desc": "Migration to finish by end of Q3",                 "groups": [["q3", "third quarter"]]},
    {"id": "talk",           "desc": "Gave a Go meetup talk about gRPC",                 "groups": [["grpc", "meetup", "talk"]]},
    {"id": "promotion",      "desc": "Promoted to senior engineer in June",              "groups": [["promot"], ["senior"], ["june"]]},
    {"id": "eating",         "desc": "Eating healthier; cutting sugary drinks",          "groups": [["vegetable", "fried", "health", "sugary", "sugar"]]},
    {"id": "daughter",       "desc": "Has a daughter named Mia",                         "groups": [["mia"]]},
    {"id": "daughter_age",   "desc": "Daughter is five years old",                       "groups": [["five", "5"]]},
    {"id": "editor",         "desc": "Uses Neovim as main editor",                       "groups": [["neovim"]]},
    {"id": "themes",         "desc": "Prefers dark themes in tools",                     "groups": [["dark theme", "dark themes"]]},
    {"id": "learning",       "desc": "Learning Rust to build a CLI tool",                "groups": [["rust"]]},
    {"id": "cert",           "desc": "Wants AWS Solutions Architect cert by December",   "groups": [["aws"], ["december", "dec"]]},
    {"id": "remote",         "desc": "Works remotely Tuesdays and Thursdays",            "groups": [["remot"], ["tuesday", "thursday", "tue", "thu"]]},
    {"id": "meetings",       "desc": "Avoids meetings before 10am",                     "groups": [["meeting"], ["10am", "10 am"]]},
    {"id": "mentoring",      "desc": "Mentors two junior engineers",                     "groups": [["mentor"], ["junior"]]},
    {"id": "travel",         "desc": "Plans a Japan trip in October with wife",         "groups": [["japan"], ["october", "oct"]]},
    {"id": "funding",        "desc": "Startup closed a $10M Series A",                   "groups": [["series a"], ["10m", "10 million"]]},
    {"id": "dislike_food",   "desc": "Dislikes pineapple on pizza",                      "groups": [["pineapple"], ["pizza"]]},
]

_FACT_BY_ID = {f["id"]: f for f in KNOWN_FACTS}

# Each question is memory-dependent: without memory the model cannot know it.
QUESTIONS = [
    {"q": "What's my name and what do I do for work?",           "facts": ["name", "role"]},
    {"q": "Where do I live and what's my tech stack?",           "facts": ["location", "stack"]},
    {"q": "What's my morning routine?",                          "facts": ["coffee", "running"]},
    {"q": "What did we decide about the monolith migration, and why?",
                                                                 "facts": ["migration_dec", "migration_why"]},
    {"q": "When do we plan to finish the migration?",            "facts": ["migration_time"]},
    {"q": "Tell me about my family.",                            "facts": ["daughter", "daughter_age"]},
    {"q": "What did I do at the Go meetup recently?",            "facts": ["talk"]},
    {"q": "Any recent career milestones?",                       "facts": ["promotion"]},
    {"q": "What are my dietary goals?",                          "facts": ["eating"]},
    {"q": "Do I prefer Python or Go for quick scripts?",         "facts": ["lang_pref"]},
    {"q": "What editor do I use?",                              "facts": ["editor", "themes"]},
    {"q": "Any new language am I learning?",                    "facts": ["learning"]},
    {"q": "Do I have any certification goals?",                 "facts": ["cert"]},
    {"q": "What's my remote work schedule?",                    "facts": ["remote"]},
    {"q": "When do I avoid meetings?",                          "facts": ["meetings"]},
    {"q": "Do I mentor anyone at work?",                        "facts": ["mentoring"]},
    {"q": "Any travel plans coming up?",                        "facts": ["travel"]},
    {"q": "What recent news does my company have?",             "facts": ["funding"]},
    {"q": "Is there a food I dislike?",                         "facts": ["dislike_food"]},
]

# ═══════════════════════════════════════════════════════════════════════
# 2. Matching helpers
# ═══════════════════════════════════════════════════════════════════════


def _norm(s: str) -> str:
    """Lowercase and collapse non-alphanumerics to single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def fact_present(text: str, groups: list[list[str]]) -> bool:
    """True iff every group has >=1 alternative appearing in `text`."""
    t = _norm(text)
    for group in groups:
        if not any(_norm(g) in t for g in group):
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════
# 3. Setup — seed L0 and run the real pipeline (L0 -> L1 -> L2 -> L3)
# ═══════════════════════════════════════════════════════════════════════


def setup_pipeline(store: MemoryStore, client: OpenAI, session_id: str, fresh: bool):
    """Wipe (optionally), seed L0, then run extraction + aggregation."""
    if fresh:
        # L2/L3 are global (aggregation pulls every atom), so a partial wipe
        # would leave stale atoms/scenarios/persona behind — wipe all layers.
        store.db.execute("DELETE FROM conversations WHERE session_id=?", (session_id,))
        store.db.execute("DELETE FROM pipeline_state WHERE session_id=?", (session_id,))
        store.db.execute("DELETE FROM atoms")
        store.db.execute("DELETE FROM scenarios")
        store.db.execute("UPDATE persona SET content='', updated_at=0 WHERE id=1")
        store.db.commit()
        print(f"[setup] wiped session '{session_id}' + all atoms/scenarios/persona")

    # L0
    store.add_conversation(session_id, SEED_CONVERSATIONS)
    print(f"[setup] seeded {len(SEED_CONVERSATIONS)} L0 messages")

    # L1 — extract + dedup
    msgs = store.get_unprocessed_conversations(session_id, limit=200)
    raw = [{"id": m["id"], "role": m["role"], "content": m["content"]} for m in msgs]
    new_atoms = extract_atoms(client, LLM_MODEL, raw)
    if new_atoms:
        existing = store.search_similar_atoms(new_atoms[0]["content"], limit=10)
        existing_dicts = [
            {"atom_type": e["atom_type"], "content": e["content"]} for e in existing
        ]
        deduped = deduplicate_atoms(client, LLM_MODEL, new_atoms, existing_dicts)
    else:
        deduped = []
    store.add_atoms(deduped)
    store.mark_extraction_done(session_id)
    print(f"[setup] L1: {len(deduped)} atoms (from {len(new_atoms)} raw)")

    # L2 — scenarios
    all_atoms = store.get_all_atoms(limit=200)
    atom_dicts = [
        {"id": a["id"], "type": a["atom_type"], "content": a["content"]} for a in all_atoms
    ]
    scenarios = build_scenarios(client, LLM_MODEL, atom_dicts) if atom_dicts else []
    for s in scenarios:
        store.add_scenario(s)
    print(f"[setup] L2: {len(scenarios)} scenarios")

    # L3 — persona
    if scenarios:
        persona = build_persona(client, LLM_MODEL, scenarios, store.get_persona())
        store.set_persona(persona)
        store.mark_aggregation_done(session_id)
        print(f"[setup] L3: persona {len(persona)} chars")
    else:
        print("[setup] L3: skipped (no scenarios)")


# ═══════════════════════════════════════════════════════════════════════
# 4. Check 1 — deterministic retrieval (no API calls)
# ═══════════════════════════════════════════════════════════════════════


def check_retrieval(store: MemoryStore, questions: list[dict]) -> list[dict]:
    """For each question, measure how many gold facts recall() surfaces."""
    rows = []
    for item in questions:
        context = recall(store, item["q"])
        facts = item["facts"]
        hits = [fid for fid in facts if fact_present(context, _FACT_BY_ID[fid]["groups"])]
        rows.append({
            "q": item["q"],
            "hit": len(hits) / len(facts) if facts else 1.0,
            "budget_ok": len(context) <= MAX_CONTEXT_CHARS,
            "chars": len(context),
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════
# 5. Check 2 — pipeline quality gates (no API calls)
# ═══════════════════════════════════════════════════════════════════════


def check_pipeline(store: MemoryStore) -> dict:
    """Extraction precision/recall + persona coverage, against gold facts."""
    atoms = store.get_all_atoms(limit=500)
    atom_text = "\n".join(a["content"] for a in atoms)

    found = [f["id"] for f in KNOWN_FACTS if fact_present(atom_text, f["groups"])]
    ext_recall = len(found) / len(KNOWN_FACTS) if KNOWN_FACTS else 0.0

    matched_atoms = sum(
        1 for a in atoms
        if any(fact_present(a["content"], f["groups"]) for f in KNOWN_FACTS)
    )
    ext_precision = matched_atoms / len(atoms) if atoms else 0.0

    persona = store.get_persona()
    persona_facts = [f["id"] for f in KNOWN_FACTS if fact_present(persona, f["groups"])]

    return {
        "n_atoms": len(atoms),
        "extraction_recall": ext_recall,
        "extraction_precision": ext_precision,
        "persona_chars": len(persona),
        "persona_facts": len(persona_facts),
        "persona_ok": len(persona) < 1500,
    }


# ═══════════════════════════════════════════════════════════════════════
# 6. Check 3 — paired A/B with LLM-as-judge (the headline)
# ═══════════════════════════════════════════════════════════════════════

ANSWER_SYSTEM_PROMPT = (
    "You are a helpful personal assistant. Answer the user's question concisely. "
    "If memory context is provided and relevant, use it; otherwise answer normally."
)

JUDGE_SYSTEM_PROMPT = """You are a strict, impartial evaluator of an AI assistant's answer.

You are given REFERENCE FACTS (ground truth) and a QUESTION + ANSWER. Grade the
answer against the reference facts ONLY.

Return a JSON object with three integer fields, each 0-5:
  - "correctness": how accurately the answer states the reference facts (5 = all
    stated accurately, 0 = none accurate).
  - "completeness": how many reference facts the answer covers (5 = all covered,
    0 = none).
  - "grounding": how well the answer avoids inventing facts NOT in the reference
    facts (5 = nothing invented / fully grounded, 0 = mostly fabricated).

Output ONLY valid JSON, no commentary."""


def _call_llm(client: OpenAI, model: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def _judge(client: OpenAI, question: str, answer: str, ref_text: str) -> dict:
    user = (
        f"REFERENCE FACTS:\n{ref_text}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:\n{answer}"
    )
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:  # noqa: BLE001 — eval must not crash on a judge hiccup
        print(f"  [judge warn] {e}")
        data = {}

    def _num(key):
        try:
            return max(0, min(5, int(data.get(key, 0))))
        except (TypeError, ValueError):
            return 0

    return {
        "correctness": _num("correctness"),
        "completeness": _num("completeness"),
        "grounding": _num("grounding"),
    }


def eval_ab(store: MemoryStore, client: OpenAI, questions: list[dict]) -> list[dict]:
    """Paired with-memory vs without-memory answers, graded by the judge."""
    rows = []
    for item in questions:
        q = item["q"]
        ref_text = "\n".join(f"- {_FACT_BY_ID[fid]['desc']}" for fid in item["facts"])

        # Arm A — with memory
        context = format_for_system_prompt(recall(store, q))
        ans_with = _call_llm(client, LLM_MODEL, ANSWER_SYSTEM_PROMPT,
                             f"{context}\n\nUser question: {q}")
        # Arm B — without memory
        ans_without = _call_llm(client, LLM_MODEL, ANSWER_SYSTEM_PROMPT, q)

        s_with = _judge(client, q, ans_with, ref_text)
        s_without = _judge(client, q, ans_without, ref_text)

        rows.append({
            "q": q,
            "with": s_with,
            "without": s_without,
            "delta": s_with["correctness"] - s_without["correctness"],
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════
# 7. Reporting
# ═══════════════════════════════════════════════════════════════════════


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "\u2026"


def print_report(ret, pipe, ab) -> None:
    print("\n" + "=" * 78)
    print("MEMORY EFFECTIVENESS REPORT")
    print("=" * 78)

    # Check 1 — retrieval
    print("\n[Check 1] Retrieval — does recall() surface the gold facts?")
    print(f"  {'Question':<52} {'Hit':>5}  Budget")
    for r in ret:
        print(f"  {_trunc(r['q'], 52):<52} {r['hit']:>5.2f}  "
              f"{'ok' if r['budget_ok'] else 'OVER'} ({r['chars']}c)")
    mean_hit = sum(r["hit"] for r in ret) / len(ret) if ret else 0.0
    print(f"  {'MEAN RETRIEVAL HIT':<52} {mean_hit:>5.2f}")

    # Check 2 — pipeline gates
    print("\n[Check 2] Pipeline quality gates")
    print(f"  atoms={pipe['n_atoms']}  "
          f"extraction recall={pipe['extraction_recall']:.2f}  "
          f"precision={pipe['extraction_precision']:.2f}")
    print(f"  persona: {pipe['persona_facts']}/{len(KNOWN_FACTS)} facts, "
          f"{pipe['persona_chars']} chars "
          f"({'ok <1500' if pipe['persona_ok'] else 'TOO LONG'})")

    # Check 3 — A/B (headline)
    if ab:
        print("\n[Check 3] A/B — answers with vs without memory (judge 0-5)")
        print(f"  {'Question':<46} {'Acc_noMem':>9} {'Acc_mem':>8} {'Delta':>6}")
        print("  " + "-" * 72)
        for r in ab:
            print(f"  {_trunc(r['q'], 46):<46} "
                  f"{r['without']['correctness']:>9} "
                  f"{r['with']['correctness']:>8} "
                  f"{r['delta']:>+6.1f}")
        n = len(ab)
        mean_no = sum(r["without"]["correctness"] for r in ab) / n
        mean_with = sum(r["with"]["correctness"] for r in ab) / n
        mean_delta = sum(r["delta"] for r in ab) / n
        print("  " + "-" * 72)
        print(f"  {'MEAN CORRECTNESS':<46} {mean_no:>9.2f} {mean_with:>8.2f} "
              f"{mean_delta:>+6.2f}")
        print("\n  => The DELTA is the headline: how much memory improves answers.")
    else:
        print("\n[Check 3] skipped (--no-judge or no API key)")


# ═══════════════════════════════════════════════════════════════════════
# 8. Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="eval-test")
    parser.add_argument("--db", default="eval_memory.db",
                        help="DB file to use (default: eval_memory.db, NOT your live DB)")
    parser.add_argument("--fresh", action="store_true", default=None,
                        help="wipe before seeding (default: on)")
    parser.add_argument("--keep-db", action="store_true",
                        help="do NOT wipe before seeding")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip the A/B check (no API calls for it)")
    parser.add_argument("--limit", type=int, default=0,
                        help="only run the first N questions (0 = all)")
    args = parser.parse_args()

    fresh = not args.keep_db
    questions = QUESTIONS[: args.limit] if args.limit else QUESTIONS

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    store = MemoryStore(args.db)
    session_id = args.session

    try:
        # ── Setup (needs the LLM) ──
        setup_pipeline(store, client, session_id, fresh)

        # ── Checks 1 & 2 (free, deterministic) ──
        ret = check_retrieval(store, questions)
        pipe = check_pipeline(store)

        # ── Check 3 — A/B (API calls) ──
        if args.no_judge or not LLM_API_KEY:
            if not LLM_API_KEY and not args.no_judge:
                print("\n[eval] LLM_API_KEY not set — skipping the A/B check.")
            ab = []
        else:
            print(f"\n[eval] running A/B with model={LLM_MODEL}, judge={JUDGE_MODEL}")
            ab = eval_ab(store, client, questions)

        print_report(ret, pipe, ab)
    finally:
        store.close()


if __name__ == "__main__":
    main()
