#!/usr/bin/env python3
"""Small OpenAI-compatible proxy for one prefill and one decode vLLM server."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from typing import Any
from urllib import request as urllib_request
from urllib import error as urllib_error


def main() -> int:
    args = _parse_args()
    state = {
        "model": args.model,
        "prefill_base_url": args.prefill_base_url.rstrip("/"),
        "decode_base_url": args.decode_base_url.rstrip("/"),
    }

    proxy_server = ThreadingHTTPServer(
        (args.host, args.proxy_port),
        _proxy_sink_handler(),
    )
    proxy_thread = threading.Thread(
        target=proxy_server.serve_forever,
        daemon=True,
    )
    proxy_thread.start()

    try:
        server = ThreadingHTTPServer(
            (args.host, args.port),
            _openai_proxy_handler(state),
        )
        server.serve_forever()
    finally:
        proxy_server.shutdown()
        proxy_server.server_close()
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a vLLM disaggregated prefill proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--proxy-port", type=int, required=True)
    parser.add_argument("--prefill-base-url", required=True)
    parser.add_argument("--decode-base-url", required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def _openai_proxy_handler(state: dict[str, str]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                try:
                    payload = _request_json(f"{state['decode_base_url']}/v1/models")
                except RuntimeError as exc:
                    self._write_json(503, {"error": str(exc)})
                    return
                self._write_json(200, payload)
                return
            self._write_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/completions":
                self._write_json(404, {"error": "not found"})
                return
            try:
                request_payload = self._read_json()
                prefill_payload = dict(request_payload)
                prefill_payload["max_tokens"] = 1
                prefill_payload.setdefault("model", state["model"])
                _request_json(
                    f"{state['prefill_base_url']}/v1/completions",
                    payload=prefill_payload,
                )
                decode_payload = dict(request_payload)
                decode_payload.setdefault("model", state["model"])
                response = _request_json(
                    f"{state['decode_base_url']}/v1/completions",
                    payload=decode_payload,
                )
            except RuntimeError as exc:
                self._write_json(502, {"error": str(exc)})
                return
            self._write_json(200, response)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode() or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON request: {exc}") from exc
            if not isinstance(payload, dict):
                raise RuntimeError("request payload must be a JSON object")
            return payload

        def _write_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _proxy_sink_handler():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_PUT(self) -> None:
            self._consume()
            self._ok()

        def do_POST(self) -> None:
            self._consume()
            self._ok()

        def do_GET(self) -> None:
            self._ok()

        def _consume(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            if length:
                self.rfile.read(length)

        def _ok(self) -> None:
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib_request.Request(url, data=data, headers=headers)
    try:
        with urllib_request.urlopen(request, timeout=60.0) as response:
            body = response.read().decode()
    except (OSError, urllib_error.URLError) as exc:
        raise RuntimeError(f"upstream request failed for {url}: {exc}") from exc
    try:
        decoded = json.loads(body or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"upstream returned invalid JSON for {url}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"upstream returned non-object JSON for {url}")
    return decoded


if __name__ == "__main__":
    raise SystemExit(main())
