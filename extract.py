"""
L0 → L1 extraction: LLM extracts structured atoms from raw conversations.

Design extracted from MemoryCore/src/core/record/l1-extractor.ts:
  - Send recent L0 messages to LLM with a system prompt
  - LLM returns JSON array of atoms (fact/preference/decision/event)
  - Dedup against existing atoms before writing
  - Stable IDs based on content hash (same fact = same ID)
"""

import json
import hashlib
from openai import OpenAI

# ── System prompts (adapted from MemoryCore's l1-extraction prompts) ──

EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction engine. Your job is to read
conversation messages and extract durable facts, preferences, decisions, and events
that are worth remembering for future conversations.

Return a JSON object with an "atoms" array. Each atom must have:
  - "content": the fact itself (one clear, self-contained sentence)
  - "type": one of "fact", "preference", "decision", "event"
  - "source_msg_ids": list of message IDs this was extracted from

Rules:
  - Only extract information likely to be useful in future conversations.
  - Skip greetings, small talk, transient remarks, and chitchat.
  - A "preference" is something the user likes/dislikes/wants.
  - A "decision" is a conclusion or choice that was made.
  - An "event" is something that happened (past action, milestone).
  - A "fact" is any other durable piece of information.
  - If nothing is worth remembering, return {"atoms": []}.
  - Do NOT invent facts not present in the conversation.
  - Each atom should be a single, atomic piece of information.
"""

DEDUP_SYSTEM_PROMPT = """You detect duplicate facts. Given a list of EXISTING facts
and a list of NEW candidate facts, return the indices (0-based) of NEW candidates
that are NOT duplicates of any existing fact.

A duplicate means the same information expressed differently. For example:
  "User prefers Python" and "User likes Python" are duplicates.
  "User prefers Python" and "User uses JavaScript at work" are NOT duplicates.

Return: {"keep": [0, 2, 5]} — only the indices of non-duplicate new candidates."""


def _stable_id(content: str) -> str:
    """Generate a stable atom ID from content hash (same fact = same ID)."""
    return "atm_" + hashlib.md5(content.encode()).hexdigest()[:12]


def extract_atoms(
    client: OpenAI,
    model: str,
    messages: list[dict],
) -> list[dict]:
    """
    Call LLM to extract L1 atoms from L0 messages.

    Args:
        client: OpenAI client
        model: LLM model name (e.g. "gpt-4o-mini")
        messages: list of {"id": int, "role": str, "content": str}

    Returns:
        list of atom dicts with "id", "content", "type", "source_msg_ids"
    """
    # Format messages for the LLM
    formatted = "\n".join(
        f"[msg_{m['id']}] {m['role']}: {m['content']}" for m in messages
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract atoms from:\n\n{formatted}"},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    atoms = result if isinstance(result, list) else result.get("atoms", [])

    # Assign stable IDs
    for a in atoms:
        a["id"] = _stable_id(a["content"])

    return atoms


def deduplicate_atoms(
    client: OpenAI,
    model: str,
    new_atoms: list[dict],
    existing_atoms: list[dict],
) -> list[dict]:
    """
    Filter out atoms that duplicate existing ones.

    Args:
        client: OpenAI client
        model: LLM model name
        new_atoms: freshly extracted atoms
        existing_atoms: existing atoms from BM25 search (dedup candidates)

    Returns:
        only the non-duplicate atoms from new_atoms
    """
    if not existing_atoms:
        return new_atoms

    existing_text = "\n".join(
        f"- [{e['atom_type']}] {e['content']}" for e in existing_atoms
    )
    new_text = "\n".join(
        f"[{i}] [{a['type']}] {a['content']}" for i, a in enumerate(new_atoms)
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DEDUP_SYSTEM_PROMPT},
            {"role": "user", "content": f"EXISTING:\n{existing_text}\n\nNEW:\n{new_text}"},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    keep_indices = json.loads(response.choices[0].message.content).get("keep", [])
    return [new_atoms[i] for i in keep_indices if 0 <= i < len(new_atoms)]
