"""
MemoryStore — SQLite-backed L0/L1/L2/L3 storage with FTS5 BM25 search.

Design extracted from MemoryCore/src/core/store/sqlite.ts:
  - L0: raw conversations
  - L1: extracted atoms with FTS5 virtual table for BM25
  - L2: scenarios (grouped knowledge blocks)
  - L3: persona (single-row long-term profile)

Why SQLite (from the original codebase):
  1. Zero external dependencies — no server process needed
  2. Ships with Python stdlib (sqlite3 module)
  3. FTS5 = free BM25 full-text search, no vector DB required
  4. Single-file portability — backup = copy the file
  5. WAL mode for concurrent reads + writes
"""

import sqlite3
import json
import time


class MemoryStore:
    def __init__(self, db_path: str = "memory.db"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")  # WAL for concurrency
        self.db.execute("PRAGMA busy_timeout=5000")  # 5s busy timeout
        self._init_tables()

    def _init_tables(self):
        """Create all tables. Idempotent (IF NOT EXISTS)."""
        self.db.executescript("""
            -- L0: raw conversations
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL DEFAULT 'default',
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_conv_session
                ON conversations(session_id, id);

            -- L1: extracted atoms (facts, preferences, decisions, events)
            CREATE TABLE IF NOT EXISTS atoms (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                atom_type TEXT NOT NULL,
                source_msg_ids TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL
            );
            -- FTS5 for BM25 search (content-sync table)
            CREATE VIRTUAL TABLE IF NOT EXISTS atoms_fts
                USING fts5(content, content=atoms, content_rowid=rowid);

            -- L2: scenarios (grouped knowledge blocks)
            CREATE TABLE IF NOT EXISTS scenarios (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source_atom_ids TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL
            );

            -- L3: persona (single-row long-term profile)
            CREATE TABLE IF NOT EXISTS persona (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                content TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL DEFAULT 0
            );
            INSERT OR IGNORE INTO persona (id, content, updated_at)
                VALUES (1, '', 0);

            -- Pipeline state: track extraction/aggregation counts
            CREATE TABLE IF NOT EXISTS pipeline_state (
                session_id TEXT PRIMARY KEY,
                conversation_count INTEGER NOT NULL DEFAULT 0,
                extraction_count INTEGER NOT NULL DEFAULT 0,
                last_extraction_at REAL,
                last_aggregation_at REAL
            );
        """)
        self.db.commit()

    # ═══════════════════════════════════════════════════════════════
    # L0: Conversation writes & reads
    # ═══════════════════════════════════════════════════════════════

    def add_conversation(self, session_id: str, messages: list[dict]) -> list[int]:
        """Record raw messages. Returns list of inserted IDs."""
        now = time.time()
        ids = []
        for msg in messages:
            cur = self.db.execute(
                "INSERT INTO conversations (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, msg["role"], msg["content"], now),
            )
            ids.append(cur.lastrowid)

        # Increment conversation count
        self.db.execute(
            "INSERT INTO pipeline_state (session_id, conversation_count) "
            "VALUES (?, 1) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "conversation_count = conversation_count + 1",
            (session_id,),
        )
        self.db.commit()
        return ids

    def get_unprocessed_conversations(
        self, session_id: str, limit: int = 20
    ) -> list[dict]:
        """Get recent conversations not yet linked to any atom (for extraction)."""
        rows = self.db.execute(
            """
            SELECT id, role, content FROM conversations
            WHERE session_id = ?
              AND id NOT IN (
                  SELECT DISTINCT CAST(json_each.value AS INTEGER)
                  FROM atoms, json_each(atoms.source_msg_ids)
              )
            ORDER BY id ASC LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_conversations(
        self, query: str, session_id: str | None = None, limit: int = 5
    ) -> list[dict]:
        """Simple LIKE search on L0 (raw text)."""
        params = [f"%{query}%", limit]
        where = ""
        if session_id:
            where = "AND session_id = ?"
            params.insert(1, session_id)
        rows = self.db.execute(
            f"SELECT id, session_id, role, content, created_at "
            f"FROM conversations WHERE content LIKE ? {where} "
            f"ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    # ═══════════════════════════════════════════════════════════════
    # L1: Atom writes & reads
    # ═══════════════════════════════════════════════════════════════

    def add_atoms(self, atoms: list[dict]):
        """Insert or replace atoms. Rebuilds FTS index after."""
        now = time.time()
        for a in atoms:
            self.db.execute(
                "INSERT OR REPLACE INTO atoms (id, content, atom_type, "
                "source_msg_ids, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    a["id"],
                    a["content"],
                    a["type"],
                    json.dumps(a.get("source_msg_ids", [])),
                    now,
                ),
            )
        self.db.commit()
        # Rebuild FTS index to pick up new content
        self.db.execute("INSERT INTO atoms_fts(atoms_fts) VALUES('rebuild')")
        self.db.commit()

    def search_atoms(self, query: str, limit: int = 5) -> list[dict]:
        """BM25 search on L1 atoms via FTS5."""
        if not query.strip():
            # Return all atoms ordered by recency
            rows = self.db.execute(
                "SELECT id, content, atom_type, created_at "
                "FROM atoms ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

        rows = self.db.execute(
            """
            SELECT a.id, a.content, a.atom_type, a.created_at
            FROM atoms_fts f
            JOIN atoms a ON f.rowid = a.rowid
            WHERE atoms_fts MATCH ?
            ORDER BY rank LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_similar_atoms(self, query: str, limit: int = 5) -> list[dict]:
        """BM25 search for dedup candidates (same as search_atoms)."""
        return self.search_atoms(query, limit)

    def get_all_atoms(self, limit: int = 200) -> list[dict]:
        """Get all atoms (for aggregation)."""
        rows = self.db.execute(
            "SELECT id, content, atom_type, created_at "
            "FROM atoms ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ═══════════════════════════════════════════════════════════════
    # L2: Scenario writes & reads
    # ═══════════════════════════════════════════════════════════════

    def add_scenario(self, scenario: dict):
        """Insert or replace a scenario."""
        self.db.execute(
            "INSERT OR REPLACE INTO scenarios (id, title, content, "
            "source_atom_ids, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                scenario["id"],
                scenario["title"],
                scenario["content"],
                json.dumps(scenario.get("source_atom_ids", [])),
                time.time(),
            ),
        )
        self.db.commit()

    def list_scenarios(self) -> list[dict]:
        """List all scenario titles + IDs."""
        rows = self.db.execute(
            "SELECT id, title FROM scenarios ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_scenario(self, scenario_id: str) -> dict | None:
        """Get full scenario by ID."""
        row = self.db.execute(
            "SELECT * FROM scenarios WHERE id = ?", (scenario_id,)
        ).fetchone()
        return dict(row) if row else None

    # ═══════════════════════════════════════════════════════════════
    # L3: Persona
    # ═══════════════════════════════════════════════════════════════

    def get_persona(self) -> str:
        """Get the current persona content."""
        row = self.db.execute(
            "SELECT content FROM persona WHERE id = 1"
        ).fetchone()
        return row["content"] if row else ""

    def set_persona(self, content: str):
        """Update the persona."""
        self.db.execute(
            "UPDATE persona SET content = ?, updated_at = ? WHERE id = 1",
            (content, time.time()),
        )
        self.db.commit()

    # ═══════════════════════════════════════════════════════════════
    # Pipeline state
    # ═══════════════════════════════════════════════════════════════

    def get_pipeline_state(self, session_id: str) -> dict:
        """Get conversation/extraction counts for a session."""
        row = self.db.execute(
            "SELECT * FROM pipeline_state WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row:
            return dict(row)
        return {
            "session_id": session_id,
            "conversation_count": 0,
            "extraction_count": 0,
            "last_extraction_at": None,
            "last_aggregation_at": None,
        }

    def mark_extraction_done(self, session_id: str):
        """Record that extraction ran for this session."""
        self.db.execute(
            "UPDATE pipeline_state SET "
            "extraction_count = extraction_count + 1, "
            "last_extraction_at = ? "
            "WHERE session_id = ?",
            (time.time(), session_id),
        )
        self.db.commit()

    def mark_aggregation_done(self, session_id: str):
        """Record that aggregation ran for this session."""
        self.db.execute(
            "UPDATE pipeline_state SET last_aggregation_at = ? "
            "WHERE session_id = ?",
            (time.time(), session_id),
        )
        self.db.commit()

    def close(self):
        self.db.close()
