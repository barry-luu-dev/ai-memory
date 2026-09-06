#!/usr/bin/env bash
#
# Smoke test for proxy.py — no real API keys required.
#
# Starts a mock upstream (scripts/mock_upstream.py) that emits scripted SSE,
# points the proxy at it, then drives the flow and asserts that:
#   1. a fresh conversation triggers the AskUserQuestion session-init form
#   2. answering "Yes" initializes memory and forwards to upstream
#   3. the thinking-fix patches missing/null "thinking" -> ""
#   4. tool_use turns are reconstructed and captured (visible + stored in L0)
#
# Usage:
#   bash scripts/smoke_test.sh
#
# Exit code 0 = all checks passed.

set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO_DIR/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

PROXY_PORT=8097
MOCK_PORT=8098
SESSION="smoke-$$"
DB="$REPO_DIR/my_memory.db"

TMPDIR_="$(mktemp -d)"
trap 'cleanup' EXIT

cleanup() {
  [ -n "${PROXY_PID:-}" ] && kill "$PROXY_PID" 2>/dev/null
  [ -n "${MOCK_PID:-}" ] && kill "$MOCK_PID" 2>/dev/null
  rm -rf "$TMPDIR_"
  # remove this run's test rows from the real memory DB
  "$PY" - "$DB" "$SESSION" <<'PYEOF'
import sqlite3, sys
db, sess = sys.argv[1], sys.argv[2]
try:
    con = sqlite3.connect(db)
    con.execute("DELETE FROM conversations WHERE session_id=?", (sess,))
    con.execute("DELETE FROM pipeline_state WHERE session_id=?", (sess,))
    con.commit(); con.close()
except Exception as e:
    print("cleanup warn:", e)
PYEOF
}

fail() { echo "FAIL: $1"; exit 1; }
pass() { echo "PASS: $1"; }

echo "== starting mock upstream on :$MOCK_PORT =="
"$PY" "$REPO_DIR/scripts/mock_upstream.py" >"$TMPDIR_/mock.log" 2>&1 &
MOCK_PID=$!
sleep 1

echo "== starting proxy on :$PROXY_PORT (upstream -> mock) =="
PROXY_PORT=$PROXY_PORT UPSTREAM_BASE_URL="http://127.0.0.1:$MOCK_PORT" \
  UPSTREAM_API_KEY=dummy "$PY" "$REPO_DIR/proxy.py" >"$TMPDIR_/proxy.log" 2>&1 &
PROXY_PID=$!
sleep 2

# --- health ---
curl -s "http://127.0.0.1:$PROXY_PORT/health" | grep -q ok \
  && pass "health" || fail "health"

# --- REQ 1: fresh conversation -> expect session-init form ---
cat >"$TMPDIR_/req1.json" <<EOF
{"model":"deepseek-chat","stream":true,"messages":[{"role":"user","content":"hello"}]}
EOF
curl -s -N -X POST "http://127.0.0.1:$PROXY_PORT/v1/messages" \
  -H 'content-type: application/json' -H "x-conversation-id: $SESSION" \
  -d @"$TMPDIR_/req1.json" -o "$TMPDIR_/out1.txt"
grep -q 'AskUserQuestion' "$TMPDIR_/out1.txt" \
  && pass "fresh conversation returns AskUserQuestion form" \
  || fail "session-init form not returned"

# --- REQ 2: answer Yes -> initialize + forward to mock ---
cat >"$TMPDIR_/req2.json" <<EOF
{"model":"deepseek-chat","stream":true,"messages":[{"role":"user","content":"hello"},{"role":"tool","content":[{"type":"tool_result","tool_use_id":"toolu_x","content":[{"type":"text","text":"{\\"answers\\":{\\"q\\":\\"Yes, use my memory\\"}}"}]}]}]}
EOF
curl -s -N -X POST "http://127.0.0.1:$PROXY_PORT/v1/messages" \
  -H 'content-type: application/json' -H "x-conversation-id: $SESSION" \
  -d @"$TMPDIR_/req2.json" -o "$TMPDIR_/out2.txt"

grep -q '"type": "thinking", "thinking": ""' "$TMPDIR_/out2.txt" \
  && pass "thinking-fix patched content_block_start to \"\"" \
  || fail "content_block_start thinking not patched"

grep -q '"type": "thinking_delta", "thinking": ""' "$TMPDIR_/out2.txt" \
  && pass "thinking-fix patched thinking_delta to \"\"" \
  || fail "thinking_delta not patched"

# --- REQ 3: plain main turn to force a fresh capture (also exercises tool_use) ---
cat >"$TMPDIR_/req3.json" <<EOF
{"model":"deepseek-chat","stream":true,"messages":[{"role":"user","content":"follow up"}]}
EOF
curl -s -N -X POST "http://127.0.0.1:$PROXY_PORT/v1/messages" \
  -H 'content-type: application/json' -H "x-conversation-id: $SESSION" \
  -d @"$TMPDIR_/req3.json" -o /dev/null
sleep 1

# tool_use reconstruction visible in proxy capture log
grep -q '\[tool_use: Read\]' "$TMPDIR_/proxy.log" \
  && pass "tool_use reconstructed in capture log" \
  || fail "tool_use not captured in proxy log"

# stored into L0 for this session
"$PY" - "$DB" "$SESSION" <<'PYEOF' >"$TMPDIR_/dbcheck.txt" 2>&1
import sqlite3, sys
con = sqlite3.connect(sys.argv[1])
rows = con.execute(
  "SELECT role, content FROM conversations WHERE session_id=?",
  (sys.argv[2],)).fetchall()
for r in rows: print(r[0], "|", r[1])
PYEOF
grep -q '\[tool_use: Read\]' "$TMPDIR_/dbcheck.txt" \
  && pass "tool_use captured into L0 in my_memory.db" \
  || fail "tool_use not found in DB (dbcheck below)"
cat "$TMPDIR_/dbcheck.txt"

echo "== smoke test passed =="
