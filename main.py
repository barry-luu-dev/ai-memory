"""
MyMemory — Personal AI Memory System

Orchestration loop that ties together:
  - store.py    : SQLite + FTS5 storage (L0/L1/L2/L3)
  - extract.py  : L0 → L1 LLM extraction + dedup
  - aggregate.py: L1 → L2 → L3 LLM aggregation
  - recall.py   : BM25 search + context assembly
  - claude_adapter.py: Claude Code request parsing (optional)

Usage:
  python main.py                          # interactive mode
  python main.py --import chat.json       # import a chat export

Config via environment variables:
  LLM_API_KEY      : your LLM API key
  LLM_MODEL        : model name (default: gpt-4o-mini)
  LLM_BASE_URL     : custom base URL (default: OpenAI)
  EXTRACT_EVERY_N  : conversations per extraction (default: 5)
  AGGREGATE_EVERY_N: extractions per aggregation (default: 3)
"""

import os
import sys
import json
from openai import OpenAI

from store import MemoryStore
from extract import extract_atoms, deduplicate_atoms
from aggregate import build_scenarios, build_persona
from recall import recall

# ── Config ──

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
EXTRACT_EVERY_N = int(os.getenv("EXTRACT_EVERY_N", "5"))
AGGREGATE_EVERY_N = int(os.getenv("AGGREGATE_EVERY_N", "3"))


def run_extraction(store: MemoryStore, client: OpenAI, session_id: str):
    """Run L1 extraction on unprocessed conversations."""
    msgs = store.get_unprocessed_conversations(session_id, limit=20)
    if not msgs:
        print("  [no unprocessed messages]")
        return

    raw = [{"id": m["id"], "role": m["role"], "content": m["content"]} for m in msgs]
    new_atoms = extract_atoms(client, LLM_MODEL, raw)

    if not new_atoms:
        print("  [no atoms extracted]")
        store.mark_extraction_done(session_id)
        return

    # Dedup against existing similar atoms
    existing = store.search_similar_atoms(new_atoms[0]["content"], limit=10)
    existing_dicts = [
        {"atom_type": e["atom_type"], "content": e["content"]} for e in existing
    ]
    deduped = deduplicate_atoms(client, LLM_MODEL, new_atoms, existing_dicts)

    store.add_atoms(deduped)
    store.mark_extraction_done(session_id)
    print(f"  [extracted {len(deduped)} new atoms (filtered from {len(new_atoms)})]")


def run_aggregation(store: MemoryStore, client: OpenAI, session_id: str):
    """Run L2 scenario building + L3 persona generation."""
    all_atoms = store.get_all_atoms(limit=200)
    if not all_atoms:
        print("  [no atoms to aggregate]")
        return

    atom_dicts = [
        {"id": a["id"], "type": a["atom_type"], "content": a["content"]}
        for a in all_atoms
    ]

    # L2: build scenarios
    scenarios = build_scenarios(client, LLM_MODEL, atom_dicts)
    for s in scenarios:
        store.add_scenario(s)
    print(f"  [built {len(scenarios)} scenarios]")

    # L3: build persona (incremental — merge with existing)
    existing_persona = store.get_persona()
    persona = build_persona(client, LLM_MODEL, scenarios, existing_persona)
    store.set_persona(persona)
    store.mark_aggregation_done(session_id)
    print(f"  [updated persona ({len(persona)} chars)]")


def interactive_loop():
    """Simple interactive chat loop with memory capture + recall."""
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY) if LLM_BASE_URL else OpenAI(api_key=LLM_API_KEY)
    store = MemoryStore("my_memory.db")
    session_id = "default"

    state = store.get_pipeline_state(session_id)
    conv_count = state["conversation_count"]
    extr_count = state["extraction_count"]

    print("=" * 60)
    print("  MyMemory — Personal AI Memory System")
    print("  Type 'quit' or 'exit' to stop")
    print("  Type '!persona' to see your profile")
    print("  Type '!atoms' to search your memories")
    print("  Type '!force' to force extraction + aggregation")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit"):
            break

        # ── Special commands ──
        if user_input.startswith("!persona"):
            persona = store.get_persona()
            if persona:
                print(f"\n[Persona]\n{persona}")
            else:
                print("\n[No persona yet — keep chatting!]")
            continue

        if user_input.startswith("!atoms"):
            query = user_input[6:].strip() or ""
            atoms = store.search_atoms(query, limit=10)
            if atoms:
                print(f"\n[Memories ({len(atoms)} found)]")
                for a in atoms:
                    print(f"  [{a['atom_type']}] {a['content']}")
            else:
                print("\n[No memories found]")
            continue

        if user_input == "!force":
            print("  [forcing extraction...]")
            run_extraction(store, client, session_id)
            print("  [forcing aggregation...]")
            run_aggregation(store, client, session_id)
            continue

        # ── Normal turn ──

        # Recall: get memory context
        context = recall(store, user_input)

        # Simulate assistant response (in production, you'd call the LLM here
        # with the context injected into the system prompt)
        assistant_response = f"I've noted: {user_input}"

        # L0: store the conversation
        store.add_conversation(session_id, [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_response},
        ])
        conv_count += 1

        # Show memory context
        if context:
            preview = context[:200] + "..." if len(context) > 200 else context
            print(f"\n  [Memory: {len(context)} chars] {preview}")

        print(f"\nAssistant: {assistant_response}")

        # ── L1: extract every N conversations ──
        if conv_count % EXTRACT_EVERY_N == 0:
            print("  [triggering extraction...]")
            run_extraction(store, client, session_id)
            extr_count += 1

        # ── L2/L3: aggregate every M extractions ──
        if extr_count > 0 and extr_count % AGGREGATE_EVERY_N == 0:
            print("  [triggering aggregation...]")
            run_aggregation(store, client, session_id)

    store.close()


def import_chat(filepath: str):
    """Import a chat export (JSON array of {role, content} messages)."""
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY) if LLM_BASE_URL else OpenAI(api_key=LLM_API_KEY)
    store = MemoryStore("my_memory.db")

    with open(filepath, "r", encoding="utf-8") as f:
        messages = json.load(f)

    if not isinstance(messages, list):
        print("Error: expected a JSON array of messages")
        return

    session_id = os.path.splitext(os.path.basename(filepath))[0]
    store.add_conversation(session_id, messages)
    print(f"Imported {len(messages)} messages into session '{session_id}'")

    # Run extraction immediately
    print("Running extraction...")
    run_extraction(store, client, session_id)

    # Run aggregation
    print("Running aggregation...")
    run_aggregation(store, client, session_id)

    store.close()
    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--import":
        if len(sys.argv) < 3:
            print("Usage: python main.py --import <chat.json>")
            sys.exit(1)
        import_chat(sys.argv[2])
    else:
        interactive_loop()
