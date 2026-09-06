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

EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction engine for a personal
assistant. Your job is to read conversation messages and extract durable, useful
facts that are worth remembering across future conversations.

Return ONLY a JSON object with an "atoms" array. Each atom must be an object with:
  - "content": the fact itself, as one clear, self-contained sentence
  - "type": one of "fact", "preference", "decision", "event"
  - "source_msg_ids": list of message IDs this was extracted from

Write every "content" field in the SAME language as the conversation messages.
Keep JSON keys and the "type" values in English.

## What is worth remembering
Extract information that remains true and useful AFTER the conversation ends:
stable attributes, likes/dislikes, conclusions, plans, and things that happened.

## Rules (follow strictly)
1. ONLY extract from the provided messages. Do NOT invent, guess, or "fill in"
   facts that are not present or clearly implied by what was said.
2. Make each atom self-contained: it must make sense with no surrounding context.
   Do NOT use deictic words like "this", "that", "it", "here", "now" unless
   unambiguous. Use "The user..." as the subject when the message is about the user.
3. Merge strongly related or causally connected messages into ONE complete atom.
   Do not split one fact into fragments, and do not restate the same idea twice.
4. Skip: greetings, small talk, chitchat, transient remarks, one-off operational
   requests ("this time...", "for now..."), repeated content, and anything that
   is only about the assistant's own behavior or outputs.
5. Skip pure subjective emotion or venting that carries no durable, factual content.

## Type guide
- "preference": something the user likes/dislikes/wants/values, or a habit.
  Triggers: "I like", "I prefer", "I always", "please use X from now on".
- "decision": a conclusion or choice that was made (a chosen option, a plan, a
  settled outcome).
- "event": something that objectively happened (a past action, milestone, or
  completed activity), ideally with its time.
- "fact": any other durable piece of information (identity, occupation, tech
  stack, constraints, location, etc.).

If nothing is worth remembering, return {"atoms": []}.
Output ONLY valid JSON — no markdown code fences, no extra commentary.
"""

DEDUP_SYSTEM_PROMPT = """You detect duplicate or redundant facts.

You are given a list of EXISTING facts and a list of NEW candidate facts. Decide
which NEW candidates are worth keeping because they are NOT already covered by any
existing fact.

A candidate is a DUPLICATE (do NOT keep) when it expresses the SAME information as
an existing fact, even if worded differently:
  EXISTING: "User prefers Python"
  NEW:      "User likes Python"            -> duplicate, drop

A candidate is NOT a duplicate (KEEP it) when it adds genuinely NEW information
that no existing fact already states — even if the topic overlaps:
  EXISTING: "User prefers Python"
  NEW:      "User uses JavaScript at work"              -> new info, keep
  NEW:      "User prefers Python for data science"      -> adds a "data science"
             qualifier not present above                -> new info, keep

Decide between the existing fact and the new one: if the new candidate is more
specific, more recent, or corrects/adds a detail absent from every existing fact,
KEEP it. Only DROP it when it is fully implied by an existing fact and adds
nothing new. When two facts are complementary (both true, each adds something),
keep both rather than collapsing them.

Return a JSON object: {"keep": [0, 2, 5]} — the 0-based indices of the NEW
candidates you decided to KEEP. Output ONLY this JSON, no commentary.
"""


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
