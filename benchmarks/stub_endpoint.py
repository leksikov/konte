"""A local OpenAI-compatible endpoint that can be made to misbehave on cue.

Three of the claims under test are about how the code reacts to an endpoint,
not about how fast an endpoint is:

- what a stalled endpoint costs one query (extraction timeout and retry budget)
- whether a repeated query goes back on the wire at all (extraction cache)
- how many requests one rate-limit response costs (retry granularity: does a
  single 429 resend the answers that already arrived?)

None of those can be provoked against a real endpoint, and all of them are
counts rather than latencies, so they are measured here. Everything else runs
against the configured endpoint for real.

Runs in a background thread of the case process, so a case reads the counters
directly instead of scraping them back over HTTP.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_NAME = "stub-model"


@dataclass
class StubState:
    """Behaviour knobs and the counters a case reads afterwards."""

    #: Seconds to wait before answering, simulating round-trip time.
    latency: float = 0.0
    #: Accept the request and never answer. Used to measure what a stalled
    #: endpoint costs a caller that is waiting on search results.
    stall: bool = False
    #: 1-based request numbers that get one 429 each before succeeding.
    rate_limit_on: set[int] = field(default_factory=set)
    #: Rate limit one particular prompt rather than one particular request.
    #: The client library retries a 429 by itself, so failing a request once
    #: never reaches the caller's own retry logic - the thing worth measuring.
    #: Keying on the prompt means the same chunk can be failed repeatedly,
    #: through the client's budget and into konte's.
    rate_limit_nth_prompt: int | None = None
    #: How many times that prompt is refused. Must exceed the client's own
    #: retry budget for the caller to ever see the failure.
    rate_limit_attempts: int = 3
    #: Canned keyword list handed back for structured-output requests.
    keywords: tuple[str, ...] = ("classification", "heading", "parts")

    requests: int = 0
    rate_limited: int = 0
    completions: int = 0
    distinct_prompts: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _served: set[int] = field(default_factory=set, repr=False)
    _seen: dict[str, int] = field(default_factory=dict, repr=False)
    _target: str | None = field(default=None, repr=False)

    def next_request(self, signature: str = "") -> tuple[int, bool]:
        """Claim a request number and say whether it should be rate limited."""
        with self._lock:
            self.requests += 1
            number = self.requests

            attempts = self._seen.get(signature, 0)
            if signature and attempts == 0:
                self.distinct_prompts += 1
                if self.distinct_prompts == self.rate_limit_nth_prompt:
                    self._target = signature
            if signature:
                self._seen[signature] = attempts + 1

            limit = number in self.rate_limit_on and number not in self._served
            if limit:
                self._served.add(number)
            elif signature and signature == self._target and attempts < self.rate_limit_attempts:
                limit = True

            if limit:
                self.rate_limited += 1
            else:
                self.completions += 1
        return number, limit

    def reset(self) -> None:
        with self._lock:
            self.requests = 0
            self.rate_limited = 0
            self.completions = 0
            self.distinct_prompts = 0
            self._served.clear()
            self._seen.clear()
            self._target = None


def _payload_for(schema: dict, keywords: tuple[str, ...]) -> str:
    """Build a JSON body that satisfies a requested schema.

    Structured output has to actually parse. A reply the client rejects sends
    the caller down its tokenization fallback, which never populates the
    extraction cache - so a broken stub would look exactly like a cache that
    does not work.
    """
    properties = (schema or {}).get("properties", {})
    payload: dict = {}
    for name, spec in properties.items():
        kind = spec.get("type")
        if kind == "array":
            payload[name] = list(keywords)
        elif kind == "integer":
            payload[name] = 1
        elif kind == "number":
            payload[name] = 1.0
        elif kind == "boolean":
            payload[name] = True
        else:
            payload[name] = " ".join(keywords)
    return json.dumps(payload)


_MARKER = re.compile(r"\[\[(\d+)\]\]")


def _marked_contexts(request: dict) -> str | None:
    """Answer a marked-up segment in the protocol it was asked in, if it was.

    Prose would send the build down its per-chunk fallback, leaving the stub
    measuring the path that runs when the protocol fails.
    """
    text = "".join(
        message["content"]
        for message in request.get("messages", [])
        if isinstance(message.get("content"), str)
    )
    positions = sorted({int(marker.group(1)) for marker in _MARKER.finditer(text)})
    if not positions:
        return None
    return "\n\n".join(f"[[{n}]]\nGenerated context for chunk {n}." for n in positions)


def _structured_schema(request: dict) -> dict | None:
    """Return the JSON schema a request asks to be answered in, if any.

    Structured output reaches the wire two different ways depending on client
    and model: as a tool definition, or as ``response_format`` carrying a JSON
    schema. Both are handled so the stub does not silently stop matching when
    the client library changes its default.
    """
    response_format = request.get("response_format") or {}
    if response_format.get("type") == "json_schema":
        return response_format.get("json_schema", {}).get("schema", {})
    tools = request.get("tools") or []
    if tools:
        return tools[0].get("function", {}).get("parameters", {})
    return None


class _Handler(BaseHTTPRequestHandler):
    state: StubState

    def log_message(self, *args) -> None:  # noqa: D102 - silence stdout logging
        pass

    def _send(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            request = {}

        # Identify the prompt, not the request: the same chunk retried is the
        # same signature, which is what lets one chunk be failed repeatedly.
        signature = hashlib.blake2b(
            json.dumps(request.get("messages", []), sort_keys=True).encode(), digest_size=16
        ).hexdigest()
        _, rate_limited = self.state.next_request(signature)

        if self.state.stall:
            # Hold the connection open and answer nothing. The client's own
            # timeout is what ends this, which is exactly the thing measured.
            while True:
                time.sleep(0.5)

        if self.state.latency:
            time.sleep(self.state.latency)

        if rate_limited:
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "1")
            body = json.dumps({"error": {"message": "rate limit", "type": "rate_limit_error"}})
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())
            return

        schema = _structured_schema(request)
        body = _payload_for(schema, self.state.keywords) if schema is not None else None
        if body is None:
            body = _marked_contexts(request)

        if body is not None and request.get("tools"):
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_stub",
                        "type": "function",
                        "function": {
                            "name": request["tools"][0].get("function", {}).get("name", "extract"),
                            "arguments": body,
                        },
                    }
                ],
            }
            finish = "tool_calls"
        else:
            message = {
                "role": "assistant",
                "content": body if body is not None else "Generated context for this chunk.",
            }
            finish = "stop"

        self._send(
            200,
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 0,
                "model": request.get("model", MODEL_NAME),
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )


@contextmanager
def stub_endpoint(**kwargs) -> Iterator[tuple[str, StubState]]:
    """Serve a stub endpoint for the duration of the block.

    Yields ``(base_url, state)``; read the counters off ``state`` afterwards.
    """
    state = StubState(**kwargs)
    handler = type("_BoundHandler", (_Handler,), {"state": state})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}/v1", state
    finally:
        server.shutdown()
        server.server_close()
