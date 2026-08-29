"""
MyMemory Proxy — sits between Claude Code and the Anthropic API.

Adds two features ported from the real MemoryProxy:
  1. Session init (AskUserQuestion form) — on a fresh conversation, ask
     whether to enable memory for this session. Adapted for solo use
     (no teams/agents — just "connect to my memory?").
  2. Header filtering — SKIP_REQUEST_HEADERS / SKIP_RESPONSE_HEADERS,
     faithful port from anthropicHandler.ts.

Flow per request:
  1. Resolve session key (x-conversation-id / x-session-id / ...)
  2. If fresh conversation → inject AskUserQuestion form (session init)
  3. If pending → parse the tool_result answer
  4. Classify request (skip fork/sidequery)
  5. Extract real user text, recall memory, inject into system prompt
  6. Forward to Anthropic (with filtered headers), stream response back
  7. Capture the turn into L0, trigger extraction/aggregation

Launch Claude Code with:
  export ANTHROPIC_BASE_URL=http://127.0.0.1:8096
  export ANTHROPIC_AUTH_TOKEN=anything
  claude
"""

import os
import json
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from store import MemoryStore
from extract import extract_atoms, deduplicate_atoms
from aggregate import build_scenarios, build_persona
from recall import recall, format_for_system_prompt
from claude_adapter import classify_cc_request, extract_clean_user_text

# ── Load .env.proxy.local (or .env.proxy) if present ──
load_dotenv(".env.proxy.local")
load_dotenv(".env.proxy")

# ── Config ──

UPSTREAM_KEY = os.getenv("UPSTREAM_API_KEY", "")
UPSTREAM_BASE = os.getenv("UPSTREAM_BASE_URL", "https://api.anthropic.com").rstrip("/")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8096"))
EXTRACT_EVERY_N = int(os.getenv("EXTRACT_EVERY_N", "5"))
AGGREGATE_EVERY_N = int(os.getenv("AGGREGATE_EVERY_N", "3"))

# LLM for extraction/aggregation (OpenAI-compatible). Defaults to DeepSeek.
# Override with LLM_BASE_URL / LLM_MODEL / LLM_API_KEY as needed.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

app = FastAPI()
store = MemoryStore("my_memory.db")

# Per-session turn counters (keyed by session key)
_turn_counters: dict[str, int] = {}
_extraction_counters: dict[str, int] = {}

# ── Header filtering (faithful port from anthropicHandler.ts) ──

SKIP_REQUEST_HEADERS = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    # internal identity headers — never forwarded upstream
    "x-tdai-user-key",
    "x-conversation-id",
    "x-session-id",
    "x-claude-code-session-id",
    "x-chat-id",
    "x-thread-id",
}

SKIP_RESPONSE_HEADERS = {
    "content-encoding",
    "transfer-encoding",
    "content-length",
    "connection",
}


def filter_request_headers(headers: dict) -> dict:
    """Copy request headers, dropping hop-by-hop + internal identity headers."""
    out = {}
    for k, v in headers.items():
        if k.lower() not in SKIP_REQUEST_HEADERS:
            out[k] = v
    out["content-type"] = "application/json"
    return out


def filter_response_headers(headers: dict) -> dict:
    """Copy response headers, dropping hop-by-hop headers."""
    return {k: v for k, v in headers.items() if k.lower() not in SKIP_RESPONSE_HEADERS}


# ── Session init (adapted for solo memory use) ──

# Session state machine: "uninitialized" → "pending_confirm" → "initialized"/"skipped"
# Keyed by session key. "initialized" means memory is enabled for this session.
_session_state: dict[str, str] = {}

# AskUserQuestion form constants (from claude-code/form.ts)
TOOL_NAME = "AskUserQuestion"
TOOLCALL_PREFIX = "toolu_cc_session_init_"
MEMORY_CONFIRM_TITLE = "Session Init — Connect to MyMemory?"
MEMORY_YES = "Yes, use my memory"
MEMORY_NO = "No, skip this session"
SKIP_HINT = '(Select "Skip" to bypass memory — nothing will be injected or captured)'


def is_fresh_conversation(messages: list) -> bool:
    """At most 1 user message, no assistant/tool → fresh conversation."""
    user_count = 0
    for m in messages:
        role = m.get("role", "")
        if role in ("assistant", "tool"):
            return False
        if role == "user":
            user_count += 1
            if user_count > 1:
                return False
    return user_count <= 1


def build_memory_confirm_form() -> dict:
    """Build the AskUserQuestion tool_use for the memory-confirm step."""
    return {
        "type": "tool_use",
        "id": TOOLCALL_PREFIX + "memory_confirm",
        "name": TOOL_NAME,
        "input": {
            "question": "Connect this session to your memory?" + SKIP_HINT,
            "header": "MyMemory",
            "options": [
                {"label": MEMORY_YES, "description": "Inject + capture memory for this session"},
                {"label": MEMORY_NO, "description": "Pass through, no memory"},
            ],
            "multiSelect": False,
        },
    }


def extract_memory_confirm_answer(content) -> str | None:
    """
    Parse the user's answer from the AskUserQuestion tool_result.
    Claude Code returns it as a JSON tool_result in a role:"tool" message.
    """
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        # Find the tool_result block
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                inner = block.get("content")
                if isinstance(inner, str):
                    text = inner
                elif isinstance(inner, list):
                    text = "".join(
                        b.get("text", "") for b in inner
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                else:
                    text = ""
                break
        else:
            return None
    else:
        return None

    # Try to parse JSON envelope { answers: { "q": "label" } }
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("answers"), dict):
            for val in parsed["answers"].values():
                if isinstance(val, str) and val.strip():
                    return val.strip()
    except json.JSONDecodeError:
        pass

    # Fall back to raw text
    return text.strip() or None


def resolve_session_key(headers: dict) -> str:
    """Extract conversation ID from headers (from session-key.ts)."""
    for h in ("x-conversation-id", "x-session-id", "x-claude-code-session-id",
              "x-deepseek-harness-session-id", "x-chat-id", "x-thread-id"):
        v = headers.get(h)
        if v and len(v) > 0:
            return v
    return "default"


# ── Memory pipeline helpers ──

def _client():
    from openai import OpenAI
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def _run_extraction(session_id: str):
    msgs = store.get_unprocessed_conversations(session_id, limit=20)
    if not msgs:
        return
    raw = [{"id": m["id"], "role": m["role"], "content": m["content"]} for m in msgs]
    new_atoms = extract_atoms(_client(), LLM_MODEL, raw)
    if not new_atoms:
        store.mark_extraction_done(session_id)
        return
    existing = store.search_similar_atoms(new_atoms[0]["content"], limit=10)
    existing_dicts = [
        {"atom_type": e["atom_type"], "content": e["content"]} for e in existing
    ]
    deduped = deduplicate_atoms(_client(), LLM_MODEL, new_atoms, existing_dicts)
    store.add_atoms(deduped)
    store.mark_extraction_done(session_id)


def _run_aggregation(session_id: str):
    all_atoms = store.get_all_atoms(limit=200)
    if not all_atoms:
        return
    atom_dicts = [
        {"id": a["id"], "type": a["atom_type"], "content": a["content"]}
        for a in all_atoms
    ]
    scenarios = build_scenarios(_client(), LLM_MODEL, atom_dicts)
    for s in scenarios:
        store.add_scenario(s)
    existing_persona = store.get_persona()
    persona = build_persona(_client(), LLM_MODEL, scenarios, existing_persona)
    store.set_persona(persona)
    store.mark_aggregation_done(session_id)


# ── System prompt injection ──

def _inject_memory(body: dict, context: str) -> dict:
    if not context:
        return body
    memory_block = format_for_system_prompt(context)
    system = body.get("system")
    if isinstance(system, str):
        body["system"] = system + "\n\n" + memory_block
    elif isinstance(system, list):
        body["system"] = list(system) + [{"type": "text", "text": memory_block}]
    else:
        body["system"] = memory_block
    return body


# ── SSE streaming forward ──

async def _forward_stream(body: dict, headers: dict):
    """Forward to Anthropic with filtered headers, stream SSE back."""
    upstream_headers = filter_request_headers(headers)
    upstream_headers["x-api-key"] = UPSTREAM_KEY
    upstream_headers["anthropic-version"] = headers.get("anthropic-version", "2023-06-01")

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", f"{UPSTREAM_BASE}/v1/messages",
            json=body, headers=upstream_headers,
        ) as resp:
            # Pass through filtered response headers
            yield json.dumps({"__status": resp.status_code, "__headers": filter_response_headers(dict(resp.headers))}) + "\n"
            async for chunk in resp.aiter_bytes():
                yield chunk


# ── Main endpoint ──

@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    headers = dict(request.headers)
    session_key = resolve_session_key(headers)

    # ── Session init: fresh conversation → ask to connect memory ──
    state = _session_state.get(session_key, "uninitialized")
    messages_list = body.get("messages", [])

    if state == "uninitialized" and is_fresh_conversation(messages_list):
        # Inject the AskUserQuestion form as the assistant's first response
        _session_state[session_key] = "pending_confirm"
        form = build_memory_confirm_form()
        return StreamingResponse(
            iter([json.dumps({
                "type": "message_start",
                "message": {"id": "msg_session_init", "role": "assistant", "content": []},
            }) + "\n",
            json.dumps({
                "type": "content_block_start",
                "index": 0,
                "content_block": form,
            }) + "\n",
            json.dumps({
                "type": "content_block_stop",
                "index": 0,
            }) + "\n",
            json.dumps({
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            }) + "\n",
            json.dumps({"type": "message_stop"}) + "\n"]),
            media_type="text/event-stream",
        )

    # ── Session init: pending → parse the answer ──
    if state == "pending_confirm":
        # Look for the tool_result in the last user/tool message
        for m in reversed(messages_list):
            if m.get("role") == "tool":
                answer = extract_memory_confirm_answer(m.get("content"))
                if answer:
                    if "yes" in answer.lower() or "use my memory" in answer.lower():
                        _session_state[session_key] = "initialized"
                    else:
                        _session_state[session_key] = "skipped"
                    break
        # If still pending (no tool_result yet), default to on
        if _session_state.get(session_key) == "pending_confirm":
            _session_state[session_key] = "initialized"

    memory_enabled = _session_state.get(session_key) == "initialized"

    # ── Classify — skip internal CC requests ──
    kind = classify_cc_request(body)
    if kind != "main":
        return StreamingResponse(
            _forward_stream(body, headers),
            media_type="text/event-stream",
        )

    # ── Extract real user text ──
    last_user = None
    for m in reversed(messages_list):
        if m.get("role") == "user":
            last_user = m.get("content")
            break
    user_text = extract_clean_user_text(last_user) if last_user else ""

    # ── Recall + inject memory (only if enabled) ──
    context = ""
    if memory_enabled and user_text:
        context = recall(store, user_text)
    body = _inject_memory(body, context)

    # ── Forward + stream back ──
    async def generate():
        assistant_text = []
        async for chunk in _forward_stream(body, headers):
            text = chunk.decode("utf-8", errors="ignore")
            for line in text.splitlines():
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        continue
                    try:
                        evt = json.loads(payload)
                        if evt.get("type") == "content_block_delta":
                            delta = evt.get("delta", {})
                            if delta.get("type") == "text_delta":
                                assistant_text.append(delta.get("text", ""))
                    except json.JSONDecodeError:
                        pass
            yield chunk

        # Capture the turn (only if memory enabled)
        if memory_enabled and user_text and assistant_text:
            store.add_conversation(session_key, [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": "".join(assistant_text)},
            ])
            # Trigger extraction/aggregation
            _turn_counters[session_key] = _turn_counters.get(session_key, 0) + 1
            if _turn_counters[session_key] % EXTRACT_EVERY_N == 0:
                _run_extraction(session_key)
                _extraction_counters[session_key] = _extraction_counters.get(session_key, 0) + 1
                if _extraction_counters[session_key] % AGGREGATE_EVERY_N == 0:
                    _run_aggregation(session_key)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
