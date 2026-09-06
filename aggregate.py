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

SCENARIO_SYSTEM_PROMPT = """You are a knowledge organizer for a long-term memory
system. Given a set of memory atoms (facts, preferences, decisions, events), you
consolidate them into a small number of coherent "scenario" documents — like the
topics of a personal knowledge base.

A scenario is a coherent topic, project, or domain that MULTIPLE atoms relate to,
for example: "Work & Career", "Health & Fitness", "Travel Plans", "Technical
Preferences", "Family". Each scenario summarizes what is known about that topic.

Return a JSON object with a "scenarios" array. Each scenario has:
  - "title": short, descriptive name (2-5 words)
  - "content": a coherent markdown narrative that SYNTHESIZES the relevant atoms.
    Weave the atoms into flowing prose that makes sense to someone new to the
    topic. Do NOT just paste the atoms as a bullet list.
  - "atom_ids": the list of atom IDs that belong to this scenario

Rules:
  - Merge related atoms into shared scenarios. Do NOT create a scenario for every
    single atom, and do NOT fragment one topic across several scenarios.
  - Aim for 3-8 scenarios; if there are very few atoms, fewer is fine.
  - Assign each atom to EXACTLY ONE scenario.
  - Do not invent facts. Only combine what the atoms actually state; if atoms
    conflict, reflect the nuance rather than silently hiding it.
  - Write "content" in the same language as the atoms.
Output ONLY valid JSON — no markdown code fences, no extra commentary.
"""

PERSONA_SYSTEM_PROMPT = """You are building a long-term user profile ("persona")
that will be injected into future conversations to help an AI understand the user
quickly and answer well.

You are given a set of scenarios, each summarizing a topic area of the user's life
or work. Synthesize a concise persona document.

Cover what the evidence supports:
  - Core preferences and values
  - Recurring patterns and habits
  - Key background facts
  - Important decisions and their rationale
  - Skills and expertise areas

Rules:
  - Base EVERYTHING on the provided scenarios. Do NOT invent or over-infer traits
    with no support. If little is known yet, keep it brief rather than guessing.
  - Weave facts into coherent prose where possible; use short headings or bullets
    only when a list genuinely aids readability.
  - Write in markdown with "##" headings.
  - Be concise: the persona is injected into the context window, so keep it short
    and dense (aim well under ~1500 characters).
  - Write in the same language as the scenario content.
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
