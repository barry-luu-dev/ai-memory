"""
Claude Code request adapter — parse CC's Anthropic protocol messages.

Design extracted from MemoryProxy/src/common/:
  - user-text-extractor.ts: extract real user text from CC's multi-block content
  - cc-request-classifier.ts: classify requests as main/fork/sidequery
  - user-query-extractor.ts: strip harness noise from user messages

Why this matters: Claude Code wraps user messages in multi-block arrays with
<system-reminder> metadata, internal prompts, and tool outputs. If you feed
all of that into memory extraction, you'll store noise instead of facts.
"""

import re
from typing import Any


# ── Request classification (from cc-request-classifier.ts) ──

def classify_cc_request(body: dict) -> str:
    """
    Classify a Claude Code Anthropic request as "main", "fork", or "sidequery".

    - "main": real user conversation turn → should trigger memory capture
    - "fork": cached variant (SUGGESTION/RECAP/COMPACT) → skip memory
    - "sidequery": background task (TITLE/verify_api_key) → skip memory

    Based on cache_control marker position + tools/thinking heuristics.
    """
    msgs = body.get("messages", [])
    if not isinstance(msgs, list):
        return "main"

    n = len(msgs)
    marker_idx = _find_last_cache_control_index(msgs)

    if marker_idx >= 0:
        if marker_idx == n - 2:
            return "fork"  # skipCacheWrite=true moves marker to n-2
        return "main"

    # No marker: check sidequery heuristics
    tools = body.get("tools", [])
    tools_empty = not isinstance(tools, list) or len(tools) == 0
    thinking = body.get("thinking", {})
    thinking_off = isinstance(thinking, dict) and thinking.get("type") == "disabled"

    if tools_empty and thinking_off:
        return "sidequery"

    return "main"


def _find_last_cache_control_index(msgs: list) -> int:
    """Find the last message index that has a cache_control marker in any content block."""
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and "cache_control" in block:
                return i
    return -1


# ── User text extraction (from user-text-extractor.ts) ──

def extract_last_user_text(content: Any) -> str | None:
    """
    Extract the real user-typed text from Claude Code's content array.

    CC wraps user messages as multi-block arrays. The last type:"text" block
    is the actual user input; earlier blocks are <system-reminder> metadata.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None

    for i in range(len(content) - 1, -1, -1):
        block = content[i]
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            return text
    return None


# ── Harness noise stripping (from user-query-extractor.ts) ──

# Patterns that indicate a Claude Code internal prompt (not user input)
CC_INTERNAL_PATTERNS = [
    r'^\s*\[(?:SUGGESTION|TITLE|SUMMARY|COMPACT|COMPACTION|ANALYSIS|EVAL|RECAP|MEMORY|SIDECHAIN)\s+MODE[:\s]',
    r'^\s*The user stepped away and is coming back\.\s*Recap',
    r'^\s*Your questions have been answered:\s*"',
    r'^\s*\d+\s*\{"parentUuid"',
    r'^\s*\{"parentUuid":\s*"[^"]+","isSidechain"',
    r'^\s*\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[^\]]*\]\[(?:user|assistant|system)\]',
]

# XML wrapper tags that CC injects into user messages (strip them)
XML_WRAPPER_TAGS = [
    "system-reminder", "system_reminder",
    "additional_data", "user_info",
    "open_and_recently_viewed_files", "session",
    "persisted-output", "persisted_output",
    "tool_use_error", "tool-use-error",
    "tool_result", "tool-result",
    "question_answer",
]

# Single-line patterns to drop (tool outputs, file confirmations, etc.)
LINE_DROP_PATTERNS = [
    r'^\s*The file .+ has been (updated|created) successfully.*$',
    r'^\s*File created successfully at:',
    r'^\s*\(Bash completed with no output\)\s*$',
    r'^\s{0,6}\d+\t',  # cat -n tab format
    r'^\s*File .+ has been (updated|created)',
]


def is_internal_prompt(text: str) -> bool:
    """Check if the entire message is a CC internal prompt (not user input)."""
    t = text.strip()
    if not t:
        return False
    return any(re.search(p, t, re.IGNORECASE) for p in CC_INTERNAL_PATTERNS)


def strip_harness_noise(raw_text: str) -> str:
    """
    Strip Claude Code harness noise from user message text.

    Returns the cleaned user input. Returns empty string if the entire
    message was harness noise (caller should skip memory capture).
    """
    # 0) If the whole message is a CC internal prompt, discard entirely
    if is_internal_prompt(raw_text):
        return ""

    # 1) Extract explicit <user_query> blocks (CodeBuddy + some CC templates)
    queries = re.findall(r'<user_query>([\s\S]*?)</user_query>', raw_text, re.IGNORECASE)
    if queries:
        return "\n\n".join(q.strip() for q in queries if q.strip())

    text = raw_text

    # 2a) Strip <question_answer> blocks (session-init forms)
    text = re.sub(
        r'<question_answer[^>]*>[\s\S]*?</question_answer>',
        '', text, flags=re.IGNORECASE
    )

    # 2b) Strip XML wrapper blocks
    for tag in XML_WRAPPER_TAGS:
        text = re.sub(
            rf'<{tag}[^>]*>[\s\S]*?</{tag}>',
            '', text, flags=re.IGNORECASE
        )

    # 2c) Drop single-line tool output patterns
    lines = text.split("\n")
    lines = [
        l for l in lines
        if not any(re.search(p, l, re.IGNORECASE) for p in LINE_DROP_PATTERNS)
    ]
    text = "\n".join(lines)

    # 2d) Strip MEMORY.md-style YAML frontmatter
    text = re.sub(
        r'(?:^|\n)---\s*\n(?:[a-z_][a-z0-9_]*:\s*.*\n)*?'
        r'(?:name|description|metadata|node_type|originSessionId):[\s\S]*?\n---\s*(?:\n|$)',
        '\n', text, flags=re.IGNORECASE
    )

    # 2e) Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def extract_clean_user_text(content: Any) -> str:
    """
    Full pipeline: extract text from CC content blocks, then strip harness noise.

    Returns empty string if there's no real user input (caller should skip memory).
    """
    text = extract_last_user_text(content)
    if not text:
        return ""
    return strip_harness_noise(text)
