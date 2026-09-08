"""
End-to-end pipeline test: seed many L0 conversations, then run L1 → L2 → L3.

Usage:
  python scripts/test_pipeline.py [--session SESSION_ID] [--fresh]

  --fresh   wipe the test session's data first (clean slate)
  --session  session id to use (default: "pipeline-test")

Flow:
  1. Seed N L0 conversations (realistic user/assistant turns across topics)
  2. L1: extract atoms from L0, dedup, persist
  3. L2: group atoms into scenarios, persist
  4. L3: distill scenarios into persona, persist
  5. Print a summary of each layer

Env vars (same as main.py):
  LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
"""

import os
import sys
import json
import time
import argparse

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from store import MemoryStore
from extract import extract_atoms, deduplicate_atoms
from aggregate import build_scenarios, build_persona

# ── Config ──
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# ── Seed conversations: many L0 turns across several topics ──
# Each entry is a {"role", "content"} dict. Realistic enough to yield durable atoms.
SEED_CONVERSATIONS = [
    # Topic 1: identity / work
    {"role": "user", "content": "Hi! I'm Alex, a backend engineer at a fintech startup in Singapore."},
    {"role": "assistant", "content": "Nice to meet you, Alex! What kind of backend work do you do?"},
    {"role": "user", "content": "I mostly write Go microservices and deploy them on Kubernetes."},
    {"role": "assistant", "content": "Got it — Go + K8s. What's your team like?"},
    {"role": "user", "content": "We're a team of six, and I lead the payments squad."},

    # Topic 2: preferences / habits
    {"role": "user", "content": "By the way, I really prefer Python over Go for quick scripts."},
    {"role": "assistant", "content": "Noted. Python for scripts, Go for services."},
    {"role": "user", "content": "I always start my day with a black coffee, no sugar."},
    {"role": "assistant", "content": "A classic. Do you have any other routines?"},
    {"role": "user", "content": "I run 5km every morning before work, rain or shine."},

    # Topic 3: decisions / plans
    {"role": "user", "content": "We decided to migrate our monolith to microservices next quarter."},
    {"role": "assistant", "content": "That's a big move. What's driving it?"},
    {"role": "user", "content": "Scalability — the monolith can't handle our Black Friday traffic."},
    {"role": "assistant", "content": "Makes sense. Any timeline?"},
    {"role": "user", "content": "We plan to finish the migration by end of Q3."},

    # Topic 4: events / milestones
    {"role": "user", "content": "Last month I gave a talk at a Go meetup about gRPC best practices."},
    {"role": "assistant", "content": "That's great! How did it go?"},
    {"role": "user", "content": "It went well — about 80 people attended."},
    {"role": "assistant", "content": "Congrats! Any other recent events?"},
    {"role": "user", "content": "I also got promoted to senior engineer in June."},

    # Topic 5: health / family
    {"role": "user", "content": "I'm trying to eat healthier — more vegetables, less fried food."},
    {"role": "assistant", "content": "Good goal. Anything specific you're avoiding?"},
    {"role": "user", "content": "I'm cutting out sugary drinks entirely."},
    {"role": "assistant", "content": "Solid plan. How's the family?"},
    {"role": "user", "content": "I have a daughter named Mia who's five years old."},
]


def seed_l0(store: MemoryStore, session_id: str):
    """Insert many L0 conversations."""
    store.add_conversation(session_id, SEED_CONVERSATIONS)
    print(f"[L0] seeded {len(SEED_CONVERSATIONS)} messages into session '{session_id}'")


def run_l1(store: MemoryStore, client: OpenAI, session_id: str):
    """L1: extract atoms from unprocessed L0 conversations."""
    msgs = store.get_unprocessed_conversations(session_id, limit=200)
    if not msgs:
        print("[L1] no unprocessed messages")
        return []

    raw = [{"id": m["id"], "role": m["role"], "content": m["content"]} for m in msgs]
    print(f"[L1] extracting from {len(raw)} messages...")
    new_atoms = extract_atoms(client, LLM_MODEL, raw)
    print(f"[L1] LLM returned {len(new_atoms)} raw atoms")

    if not new_atoms:
        return []

    # Dedup against existing atoms
    existing = store.search_similar_atoms(new_atoms[0]["content"], limit=10)
    existing_dicts = [
        {"atom_type": e["atom_type"], "content": e["content"]} for e in existing
    ]
    deduped = deduplicate_atoms(client, LLM_MODEL, new_atoms, existing_dicts)
    print(f"[L1] kept {len(deduped)} atoms after dedup (from {len(new_atoms)})")

    store.add_atoms(deduped)
    store.mark_extraction_done(session_id)
    return deduped


def run_l2(store: MemoryStore, client: OpenAI):
    """L2: group atoms into scenarios."""
    all_atoms = store.get_all_atoms(limit=200)
    if not all_atoms:
        print("[L2] no atoms to aggregate")
        return []

    atom_dicts = [
        {"id": a["id"], "type": a["atom_type"], "content": a["content"]}
        for a in all_atoms
    ]
    print(f"[L2] grouping {len(atom_dicts)} atoms into scenarios...")
    scenarios = build_scenarios(client, LLM_MODEL, atom_dicts)
    for s in scenarios:
        store.add_scenario(s)
    print(f"[L2] built {len(scenarios)} scenarios")
    return scenarios


def run_l3(store: MemoryStore, client: OpenAI, scenarios: list[dict]):
    """L3: distill scenarios into persona."""
    if not scenarios:
        print("[L3] no scenarios to distill")
        return ""
    existing_persona = store.get_persona()
    print("[L3] distilling persona...")
    persona = build_persona(client, LLM_MODEL, scenarios, existing_persona)
    store.set_persona(persona)
    store.mark_aggregation_done("pipeline-test")
    print(f"[L3] persona updated ({len(persona)} chars)")
    return persona


def summarize(store: MemoryStore, session_id: str):
    """Print a summary of all layers."""
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    state = store.get_pipeline_state(session_id)
    print(f"\n[state] conversations={state['conversation_count']} "
          f"extractions={state['extraction_count']}")

    atoms = store.get_all_atoms(limit=200)
    print(f"\n[L1] {len(atoms)} atoms:")
    for a in atoms:
        print(f"  - [{a['atom_type']}] {a['content']}")

    scenarios = store.list_scenarios()
    print(f"\n[L2] {len(scenarios)} scenarios:")
    for s in scenarios:
        detail = store.get_scenario(s["id"])
        atom_ids = json.loads(detail["source_atom_ids"])
        print(f"  - {detail['title']}  (atoms: {len(atom_ids)})")

    persona = store.get_persona()
    print(f"\n[L3] persona ({len(persona)} chars):")
    print(persona or "  (empty)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="pipeline-test")
    parser.add_argument("--fresh", action="store_true",
                        help="wipe the test session's data first")
    args = parser.parse_args()

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    store = MemoryStore("my_memory.db")
    session_id = args.session

    if args.fresh:
        # Full reset: wipe ALL layers so the test starts from a clean slate.
        # (L2/L3 aggregation is global — it pulls every atom across sessions —
        # so a partial wipe would leave stale atoms/scenarios/persona behind.)
        store.db.execute("DELETE FROM conversations WHERE session_id=?", (session_id,))
        store.db.execute("DELETE FROM pipeline_state WHERE session_id=?", (session_id,))
        store.db.execute("DELETE FROM atoms")
        store.db.execute("DELETE FROM scenarios")
        store.db.execute("UPDATE persona SET content='', updated_at=0 WHERE id=1")
        store.db.commit()
        print(f"[fresh] wiped session '{session_id}' + all atoms/scenarios/persona")

    # 1. Seed L0
    seed_l0(store, session_id)

    # 2. L1 extraction
    run_l1(store, client, session_id)

    # 3. L2 aggregation
    scenarios = run_l2(store, client)

    # 4. L3 persona
    run_l3(store, client, scenarios)

    # 5. Summary
    summarize(store, session_id)
    store.close()


if __name__ == "__main__":
    main()
