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
import time
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

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
    # auth — never forwarded. The proxy authenticates upstream itself via
    # x-api-key (UPSTREAM_API_KEY); the client's dummy "Bearer <token>" is
    # only for talking to the proxy, not to the real provider.
    "authorization",
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
    """Build the AskUserQuestion tool_use for the memory-confirm step.

    AskUserQuestion's input MUST be wrapped in a top-level `questions` array
    (each item: question/header/options/multiSelect). Claude Code validates the
    tool input against this schema and rejects it with "Invalid tool parameters"
    if `questions` is missing.
    """
    return {
        "type": "tool_use",
        "id": TOOLCALL_PREFIX + "memory_confirm",
        "name": TOOL_NAME,
        "input": {
            "questions": [
                {
                    "question": "Connect this session to your memory?" + SKIP_HINT,
                    "header": "MyMemory",
                    "options": [
                        {"label": MEMORY_YES, "description": "Inject + capture memory for this session"},
                        {"label": MEMORY_NO, "description": "Pass through, no memory"},
                    ],
                    "multiSelect": False,
                }
            ],
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

def _upstream_headers(headers: dict) -> dict:
    """Build forwarded request headers, adding auth + API version."""
    upstream_headers = filter_request_headers(headers)
    upstream_headers["x-api-key"] = UPSTREAM_KEY
    upstream_headers["anthropic-version"] = headers.get("anthropic-version", "2023-06-01")
    return upstream_headers


async def _open_upstream(body: dict, headers: dict):
    """Open a streaming connection to the upstream provider.

    Returns an (httpx.AsyncClient, httpx.Response) pair. The caller owns both
    and MUST close them (resp.aclose() / client.aclose()) when finished.
    """
    client = httpx.AsyncClient(timeout=None)
    req = client.build_request(
        "POST", f"{UPSTREAM_BASE}/v1/messages",
        json=body, headers=_upstream_headers(headers),
    )
    resp = await client.send(req, stream=True)
    return client, resp


async def _relay_upstream_body(client, resp):
    """Close the stream and return the upstream body as a plain Response.

    Used for non-2xx errors and for non-streaming (JSON) requests, so the real
    HTTP status and content-type are preserved instead of being forced to SSE.
    """
    raw = await resp.aread()
    ctype = resp.headers.get("content-type") or "application/json"
    status = resp.status_code
    await resp.aclose()
    await client.aclose()
    return Response(content=raw, status_code=status, media_type=ctype)


async def _forward(body: dict, headers: dict, on_text=None, after_stream=None):
    """Forward a request upstream and return an appropriate Response.

    - upstream non-2xx    → relayed as a real error (status + JSON content-type)
    - upstream non-stream → relayed as plain JSON
    - upstream streaming  → Streamed back as SSE (text/event-stream)

    on_text(fragment) is called per text_delta while streaming.
    after_stream() is called after a successful streaming turn (for capture).
    """
    client, resp = await _open_upstream(body, headers)

    if not (200 <= resp.status_code < 300):
        print(f"[{time.time():.3f}] upstream error HTTP {resp.status_code}")
        return await _relay_upstream_body(client, resp)

    # Non-streaming request → upstream returns a single JSON body.
    if not body.get("stream", False):
        return await _relay_upstream_body(client, resp)

    def _collect_event(raw_event: bytes):
        """Parse one complete SSE event block, feeding text deltas to on_text.

        SSE events can be split across network chunks, so this is only called
        once a full event (terminated by a blank line) has been buffered. This
        avoids truncated-JSON errors and ensures the assistant text is captured
        completely for memory.
        """
        if on_text is None:
            return
        text = raw_event.decode("utf-8", errors="ignore")
        for line in text.splitlines():
            if line.startswith("data: "):
                payload = line[6:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    evt = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if evt.get("type") == "content_block_delta":
                    delta = evt.get("delta", {})
                    if isinstance(delta, dict) and delta.get("type") == "text_delta":
                        t = delta.get("text")
                        if t:
                            on_text(t)

    async def stream():
        buffer = bytearray()
        try:
            async for chunk in resp.aiter_bytes():
                buffer.extend(chunk)
                # Emit every complete SSE event (delimited by a blank line).
                while True:
                    sep = buffer.find(b"\n\n")
                    if sep == -1:
                        break
                    _collect_event(bytes(buffer[:sep]))
                    del buffer[:sep + 2]
                yield chunk
            # Flush any final event at end of stream.
            if buffer:
                _collect_event(bytes(buffer))
                buffer.clear()
        finally:
            await resp.aclose()
            await client.aclose()
        # Only capture after a clean, complete stream.
        if after_stream is not None:
            after_stream()

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── Main endpoint ──

@app.post("/v1/messages")
async def messages(request: Request):
    print(f"[{time.time():.3f}] /v1/messages called")
    body = await request.json()
    headers = dict(request.headers)
    session_key = resolve_session_key(headers)

    # ── Session init: fresh conversation → ask to connect memory ──
    state = _session_state.get(session_key, "uninitialized")
    messages_list = body.get("messages", [])

    if state == "uninitialized" and is_fresh_conversation(messages_list):
        # Inject the AskUserQuestion form as the assistant's first response.
        # Faithful port of buildFormResponse() from claude-code/form.ts:
        #   - proper SSE framing: "event: <type>\ndata: <json>\n\n"
        #   - a thinking block BEFORE the tool_use (DeepSeek requires the
        #     prior assistant turn to carry content[].thinking after a tool call)
        #   - tool_use input streamed via input_json_delta
        _session_state[session_key] = "pending_confirm"
        print(f"[{time.time():.3f}] session={session_key} FRESH -> sending AskUserQuestion form (state=pending_confirm)")
        form = build_memory_confirm_form()
        msg_id = "msg_cc_session_init_" + str(int(time.time() * 1000))
        tool_use_id = TOOLCALL_PREFIX + str(int(time.time() * 1000))
        input_json = json.dumps(form["input"])

        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        events = [
            sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": msg_id, "type": "message", "role": "assistant",
                    "model": "unknown", "content": [],
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }),
            # thinking block must precede tool_use
            sse("content_block_start", {
                "type": "content_block_start", "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            }),
            sse("content_block_delta", {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "[proxy session-init form]"},
            }),
            sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
            # tool_use block
            sse("content_block_start", {
                "type": "content_block_start", "index": 1,
                "content_block": {"type": "tool_use", "id": tool_use_id, "name": TOOL_NAME, "input": {}},
            }),
            sse("content_block_delta", {
                "type": "content_block_delta", "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": input_json},
            }),
            sse("content_block_stop", {"type": "content_block_stop", "index": 1}),
            sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 0},
            }),
            sse("message_stop", {"type": "message_stop"}),
        ]
        return StreamingResponse(
            iter(["".join(events)]),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # ── Session init: pending → parse the answer ──
    if state == "pending_confirm":
        # Look for the tool_result in the last user/tool message
        answered = None
        for m in reversed(messages_list):
            if m.get("role") == "tool":
                content = m.get("content")
                preview = content if isinstance(content, str) else json.dumps(content)[:200]
                answer = extract_memory_confirm_answer(content)
                print(f"[{time.time():.3f}] session={session_key} pending_confirm found tool msg -> answer={answer!r} content={preview!r}")
                if answer:
                    answered = answer
                    if "yes" in answer.lower() or "use my memory" in answer.lower():
                        _session_state[session_key] = "initialized"
                    else:
                        _session_state[session_key] = "skipped"
                    break
        # If still pending (no answer found), default to on
        if _session_state.get(session_key) == "pending_confirm":
            print(f"[{time.time():.3f}] session={session_key} pending_confirm no answer (answered={answered!r}, n_msgs={len(messages_list)}) -> auto-initializing")
            _session_state[session_key] = "initialized"

    memory_enabled = _session_state.get(session_key) == "initialized"

    # ── Classify — skip internal CC requests ──
    kind = classify_cc_request(body)
    if kind != "main":
        return await _forward(body, headers)

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
    assistant_text: list[str] = []

    def _on_text(t: str):
        assistant_text.append(t)

    def _after_stream():
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

    return await _forward(body, headers, on_text=_on_text, after_stream=_after_stream)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
