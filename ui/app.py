"""
MyAI Memory — Streamlit explorer for the L0 → L3 memory store.

Read-only viewer over the SQLite database, reusing MemoryStore from store.py.

Run:
    streamlit run ui/app.py

Views (top-down, matching the recall flow):
    Persona (L3) → Scenarios (L2) → Atoms (L1) → Conversations (L0)
Plus a pipeline-state status bar and BM25/LIKE search.
"""

import json
import os
import sys
from datetime import datetime

import streamlit as st

# Make `store.py` importable when running from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from store import MemoryStore  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────────

def fmt_ts(epoch: float | None) -> str:
    """Format an epoch-seconds timestamp as a human-readable local time."""
    if not epoch:
        return "—"
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")


def parse_json_ids(raw: str) -> list:
    """Safely parse a JSON-array column (source_msg_ids / source_atom_ids)."""
    try:
        val = json.loads(raw) if raw else []
        return val if isinstance(val, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


ATOM_TYPE_COLORS = {
    "fact": "#3b82f6",       # blue
    "preference": "#22c55e",  # green
    "decision": "#f59e0b",    # amber
    "event": "#a855f7",       # purple
}


def atom_badge(atom_type: str) -> str:
    """Return a colored HTML badge for an atom type."""
    color = ATOM_TYPE_COLORS.get(atom_type, "#6b7280")
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:10px;font-size:0.75rem;font-weight:600;">'
        f'{atom_type}</span>'
    )


@st.cache_resource
def get_store(db_path: str) -> MemoryStore:
    """Open (and cache) a single MemoryStore connection."""
    return MemoryStore(db_path)


# ── Page config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MyAI Memory",
    page_icon="💾",
    layout="wide",
)

st.title("💾 MyAI Memory")
st.caption("Layered memory explorer — L0 conversations → L1 atoms → L2 scenarios → L3 persona")

# DB path (defaults to repo-root my_memory.db)
db_path = st.sidebar.text_input(
    "Database path",
    value=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "my_memory.db"),
)
store = get_store(db_path)

# ── Sidebar: pipeline state ─────────────────────────────────────────────

st.sidebar.header("Pipeline state")
sessions = store.list_sessions()
if sessions:
    sel_session = st.sidebar.selectbox("Session", sessions)
    state = store.get_pipeline_state(sel_session)
    st.sidebar.metric("Conversations", state["conversation_count"])
    st.sidebar.metric("Extractions", state["extraction_count"])
    st.sidebar.caption(f"Last extraction: {fmt_ts(state['last_extraction_at'])}")
    st.sidebar.caption(f"Last aggregation: {fmt_ts(state['last_aggregation_at'])}")
else:
    st.sidebar.info("No sessions yet — feed some conversations first.")

# ── Tabs ─────────────────────────────────────────────────────────────────

tab_conversations, tab_atoms, tab_scenarios, tab_persona = st.tabs(
    ["💬 Conversations (L0)", "🔬 Atoms (L1)", "📚 Scenarios (L2)", "👤 Persona (L3)"]
)

# ── L3: Persona ──────────────────────────────────────────────────────────

with tab_persona:
    persona = store.get_persona()
    if persona:
        st.markdown("### Long-term profile")
        st.markdown(persona)
    else:
        st.info("No persona yet — run aggregation to distill one from your scenarios.")

# ── L2: Scenarios ────────────────────────────────────────────────────────

with tab_scenarios:
    scenarios = store.list_scenarios()
    if not scenarios:
        st.info("No scenarios yet — run aggregation to group atoms into scenarios.")
    else:
        st.markdown(f"**{len(scenarios)} scenarios**")
        for s in scenarios:
            detail = store.get_scenario(s["id"])
            if not detail:
                continue
            with st.expander(f"📚 {detail['title']}", expanded=False):
                st.markdown(detail["content"])
                st.caption(f"Created {fmt_ts(detail['created_at'])}")
                atom_ids = parse_json_ids(detail.get("source_atom_ids", "[]"))
                if atom_ids:
                    st.markdown("**Source atoms:**")
                    for a in store.get_atoms_by_ids(atom_ids):
                        st.markdown(
                            f"- {atom_badge(a['atom_type'])} {a['content']}",
                            unsafe_allow_html=True,
                        )

# ── L1: Atoms ────────────────────────────────────────────────────────────

with tab_atoms:
    col_search, col_type = st.columns([3, 1])
    atom_query = col_search.text_input("Search atoms (BM25)", key="atom_search")
    atom_type_filter = col_type.selectbox(
        "Type", ["all", "fact", "preference", "decision", "event"], key="atom_type"
    )

    if atom_query.strip():
        atoms = store.search_atoms(atom_query, limit=50)
    else:
        atoms = store.get_all_atoms(limit=200)

    if atom_type_filter != "all":
        atoms = [a for a in atoms if a["atom_type"] == atom_type_filter]

    if not atoms:
        st.info("No atoms found.")
    else:
        st.markdown(f"**{len(atoms)} atoms**")
        for a in atoms:
            with st.expander(
                f"{a['atom_type']} · {a['content'][:80]}{'…' if len(a['content']) > 80 else ''}",
                expanded=False,
            ):
                st.markdown(atom_badge(a["atom_type"]), unsafe_allow_html=True)
                st.markdown(a["content"])
                st.caption(f"Created {fmt_ts(a['created_at'])}")
                # Traceability: which conversations produced this atom
                msg_ids = parse_json_ids(a.get("source_msg_ids", "[]"))
                if msg_ids:
                    st.markdown("**Source conversations:**")
                    for c in store.get_conversations_by_ids(msg_ids):
                        st.markdown(f"- **{c['role']}**: {c['content'][:200]}")

# ── L0: Conversations ────────────────────────────────────────────────────

with tab_conversations:
    col_search, col_session = st.columns([3, 1])
    conv_query = col_search.text_input("Search conversations (LIKE)", key="conv_search")
    conv_session = col_session.selectbox(
        "Session", ["all"] + sessions, key="conv_session"
    )

    if conv_query.strip():
        convs = store.search_conversations(
            conv_query,
            session_id=None if conv_session == "all" else conv_session,
            limit=50,
        )
    else:
        convs = store.get_conversations(
            session_id=None if conv_session == "all" else conv_session,
            limit=200,
        )

    if not convs:
        st.info("No conversations found.")
    else:
        st.markdown(f"**{len(convs)} messages**")
        for c in convs:
            role_icon = "🧑" if c["role"] == "user" else "🤖"
            with st.expander(
                f"{role_icon} [{c['role']}] {c['content'][:80]}{'…' if len(c['content']) > 80 else ''}",
                expanded=False,
            ):
                st.markdown(c["content"])
                st.caption(
                    f"#{c['id']} · {c['session_id']} · {fmt_ts(c['created_at'])}"
                )
