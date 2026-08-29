"""
L1 → L2 → L3 aggregation: group atoms into scenarios, distill scenarios into persona.

Design extracted from MemoryCore/src/core/persona/persona-generator.ts
and MemoryCore/src/core/prompts/scene-extraction.ts:
  - L2: LLM groups related atoms into named scenarios (markdown summaries)
  - L3: LLM distills all scenarios into a single persona document
  - Stable scenario IDs from title hash
"""

import json
import hashlib
from openai import OpenAI

SCENARIO_SYSTEM_PROMPT = """You are a knowledge organizer. Given a set of memory atoms
(facts, preferences, decisions, events), group them into logical scenarios.

A scenario is a coherent topic, project, or domain that multiple atoms relate to.
For example: "Work Projects", "Health & Fitness", "Travel Plans", "Technical Preferences".

Return a JSON object with a "scenarios" array. Each scenario has:
  - "title": short descriptive name (2-5 words)
  - "content": markdown summary synthesizing the relevant atoms into a coherent narrative
  - "atom_ids": list of atom IDs that belong to this scenario

Rules:
  - Merge related atoms; don't create a scenario for every single atom.
  - Aim for 3-8 scenarios. If there are very few atoms, fewer is fine.
  - Each atom should belong to exactly one scenario.
  - Write the content as if explaining the topic to someone new.
"""

PERSONA_SYSTEM_PROMPT = """You are building a long-term user profile. Given a set of
scenarios (each summarizing a topic area of the user's life/work), synthesize a
concise persona document.

The persona should cover:
  - Core preferences and values
  - Recurring patterns and habits
  - Key facts and background
  - Important decisions and their rationale
  - Skills and expertise areas

Write in markdown with ## headings. Be concise but comprehensive. This document
will be injected into future conversations to help an AI understand the user quickly.
"""


def _stable_id(title: str) -> str:
    return "scn_" + hashlib.md5(title.encode()).hexdigest()[:10]


def build_scenarios(
    client: OpenAI,
    model: str,
    atoms: list[dict],
) -> list[dict]:
    """
    Group L1 atoms into L2 scenarios.

    Args:
        client: OpenAI client
        model: LLM model name
        atoms: list of {"id": str, "type": str, "content": str}

    Returns:
        list of scenario dicts with "id", "title", "content", "source_atom_ids"
    """
    atom_text = "\n".join(
        f"[{a['id']}] ({a['type']}) {a['content']}" for a in atoms
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SCENARIO_SYSTEM_PROMPT},
            {"role": "user", "content": f"Organize these atoms:\n\n{atom_text}"},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    scenarios = result if isinstance(result, list) else result.get("scenarios", [])

    for s in scenarios:
        s["id"] = _stable_id(s["title"])
        # Normalize field name
        if "atom_ids" in s:
            s["source_atom_ids"] = s.pop("atom_ids")

    return scenarios


def build_persona(
    client: OpenAI,
    model: str,
    scenarios: list[dict],
    existing_persona: str = "",
) -> str:
    """
    Distill L2 scenarios into L3 persona.

    Args:
        client: OpenAI client
        model: LLM model name
        scenarios: list of scenario dicts
        existing_persona: current persona content (for incremental update)

    Returns:
        markdown persona string
    """
    scenario_text = "\n\n---\n\n".join(
        f"## {s['title']}\n{s['content']}" for s in scenarios
    )

    user_message = f"Build persona from:\n\n{scenario_text}"
    if existing_persona:
        user_message = (
            f"Existing persona (update this, don't lose old info):\n\n"
            f"{existing_persona}\n\n"
            f"---\n\n"
            f"New scenarios to incorporate:\n\n{scenario_text}"
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PERSONA_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content
