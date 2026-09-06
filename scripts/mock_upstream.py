"""Mock Anthropic-compatible upstream that emits scripted SSE.

Used by scripts/smoke_test.sh to verify proxy.py without real API keys.

Emits, in order:
  - a thinking block whose `thinking` is MISSING (tests the thinking-fix patch)
  - a thinking_delta whose `thinking` is null (tests the thinking-fix patch)
  - a text_delta fragment
  - a tool_use block with streamed input_json_delta (tests tool_use capture)
  - a final text_delta

Run:  python scripts/mock_upstream.py   (listens on 127.0.0.1:8098)
"""
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()
HOST = "127.0.0.1"
PORT = 8098


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    print(f"[mock] /v1/messages stream={body.get('stream')}")

    # content_block_start for thinking deliberately OMITS the 'thinking' key,
    # and the thinking_delta carries null — both must be patched to "".
    events = [
        sse("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_mock_1", "type": "message", "role": "assistant",
                "model": "deepseek-chat", "content": [],
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }),
        # thinking block WITHOUT 'thinking' field -> should be patched to ""
        sse("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "thinking"},
        }),
        # thinking_delta with null thinking -> should be patched to ""
        sse("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "thinking_delta", "thinking": None},
        }),
        sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        # real text
        sse("content_block_start", {
            "type": "content_block_start", "index": 1,
            "content_block": {"type": "text", "text": ""},
        }),
        sse("content_block_delta", {
            "type": "content_block_delta", "index": 1,
            "delta": {"type": "text_delta", "text": "Hello "},
        }),
        # tool_use with streamed input
        sse("content_block_start", {
            "type": "content_block_start", "index": 2,
            "content_block": {"type": "tool_use", "id": "toolu_mock_1",
                              "name": "Read", "input": {}},
        }),
        sse("content_block_delta", {
            "type": "content_block_delta", "index": 2,
            "delta": {"type": "input_json_delta",
                      "partial_json": '{"file_path": "/ma'},
        }),
        sse("content_block_delta", {
            "type": "content_block_delta", "index": 2,
            "delta": {"type": "input_json_delta",
                      "partial_json": 'in.py"}'},
        }),
        sse("content_block_stop", {"type": "content_block_stop", "index": 2}),
        # trailing text
        sse("content_block_delta", {
            "type": "content_block_delta", "index": 1,
            "delta": {"type": "text_delta", "text": "world"},
        }),
        sse("content_block_stop", {"type": "content_block_stop", "index": 1}),
        sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 10},
        }),
        sse("message_stop", {"type": "message_stop"}),
    ]
    return StreamingResponse(iter(["".join(events)]), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
