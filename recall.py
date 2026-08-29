"""
Memory recall: assemble relevant context for a new query.

Design extracted from MemoryCore/src/core/hooks/auto-recall.ts:
  - Top-down: L3 (persona) → L2 (scenarios) → L1 (atoms)
  - BM25 search on atoms via FTS5
  - Budget cap: never exceed MAX_CONTEXT_CHARS
"""

from store import MemoryStore

# Budget cap — prevents memory from eating the context window.
# From MemoryCore: recall results are capped by item count + char budget + timeout.
MAX_CONTEXT_CHARS = 3000
MAX_SCENARIOS = 3
MAX_ATOMS = 5


def recall(store: MemoryStore, query: str) -> str:
    """
    Assemble memory context for a new user query.

    Top-down approach (matching MemoryCore's recall strategy):
      1. L3 persona — always included (small, high-value)
      2. L2 scenarios — most relevant ones
      3. L1 atoms — BM25 search results

    Returns a formatted string ready to inject into the system prompt.
    """
    parts: list[str] = []

    # ── L3: Persona (always include) ──
    persona = store.get_persona()
    if persona:
        parts.append(f"## User Profile\n{persona}")

    # ── L2: Scenarios ──
    scenarios = store.list_scenarios()
    if scenarios:
        parts.append("## Relevant Context")
        for s in scenarios[:MAX_SCENARIOS]:
            detail = store.get_scenario(s["id"])
            if detail:
                # Truncate each scenario to ~500 chars in recall context
                content = detail["content"]
                if len(content) > 500:
                    content = content[:500] + "..."
                parts.append(f"### {detail['title']}\n{content}")

    # ── L1: Atoms (BM25 search) ──
    atoms = store.search_atoms(query, limit=MAX_ATOMS)
    if atoms:
        parts.append("## Related Facts")
        for a in atoms:
            parts.append(f"- [{a['atom_type']}] {a['content']}")

    # ── Budget cap ──
    context = "\n\n".join(parts)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n...(truncated)"

    return context


def format_for_system_prompt(context: str) -> str:
    """Wrap recall context for injection into a system prompt."""
    if not context:
        return ""

    return f"""<memory_context>
{context}
</memory_context>

When answering, use the above memory context if relevant. If the context
doesn't cover what the user is asking about, just answer normally."""
