"""Local Ollama NDJSON streaming with final citation validation.

Streaming events are provisional. Consumers MUST replace the provisional answer
with the *validated* final result. Never cache, export, or cite partial text as
an accepted answer. No cloud calls and no external RAG framework dependencies.
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from src.prompt_builder import ABSTENTION
from src.ollama_generator import (
    DEFAULT_MODEL, DEFAULT_CONTEXT, DEFAULT_PREDICT, LOCAL_CHAT_URL,
    TEMPLATE_OVERHEAD, _check_answer, _int_between, _safe_result,
    validate_prompt_package,
)

MAX_STREAM_LINE_BYTES = 1_000_000
MAX_STREAM_EVENTS = 12_000
MAX_STREAM_TEXT = 20_000


def _validate_local_endpoint(url: str) -> None:
    parsed = urlparse(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.path != "/api/chat" or parsed.query or parsed.fragment
            or parsed.username is not None or parsed.password is not None):
        raise ValueError("Ollama streaming requests must target local /api/chat")


def _local_stream_json(url: str, payload: dict, timeout: int) -> Iterator[dict]:
    """Read real Ollama line-delimited JSON over HTTP; close on finish/error/cancel."""
    _validate_local_endpoint(url)
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                if len(raw_line) > MAX_STREAM_LINE_BYTES:
                    raise RuntimeError("Oversized Ollama stream frame")
                if not raw_line.strip():
                    continue
                try:
                    frame = json.loads(raw_line.decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as exc:
                    raise RuntimeError("Ollama returned invalid streaming JSON") from exc
                if not isinstance(frame, dict):
                    raise RuntimeError("Ollama returned an invalid stream frame")
                yield frame
    except HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama HTTP {exc.code}: {detail}. Check local model availability and memory."
        ) from exc
    except (URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise RuntimeError("Local Ollama stream failed or disconnected. Check ollama serve.") from exc


def stream_grounded_answer(
    package: dict, *, model: str = DEFAULT_MODEL, num_ctx: int = DEFAULT_CONTEXT,
    max_output_tokens: int = DEFAULT_PREDICT, timeout: int = 600,
    allow_multiple_documents: bool = False,
    sender: Callable[[str, dict, int], Iterator[dict]] | None = None,
) -> Iterator[dict]:
    """Yield cumulative draft text, then one fully validated final result.

    Event schemas:
      {"type":"delta", "text":"full provisional content so far"}
      {"type":"final", "result": validated_generator_result}

    A connection failure or malformed/incomplete stream raises an exception;
    callers must discard all draft text and show a safe error. The model may
    also complete with an unsafe or uncited answer; final result then abstains.
    """
    if not isinstance(model, str) or not model.strip() or len(model) > 128:
        raise ValueError("model must be a nonempty local Ollama model tag")
    _int_between(num_ctx, 1024, 131072, "num_ctx")
    _int_between(max_output_tokens, 16, 8192, "max_output_tokens")
    _int_between(timeout, 1, 3600, "timeout")
    if not validate_prompt_package(package, allow_multiple_documents=allow_multiple_documents):
        yield {"type": "final", "result": _safe_result(
            package, status="skipped_" + package["status"],
            reason="Prompt has no eligible grounded evidence; Ollama was not called",
            answer=ABSTENTION, model=model,
        )}
        return

    total_bytes = sum(len(message["content"].encode("utf-8")) for message in package["messages"])
    if total_bytes + TEMPLATE_OVERHEAD + max_output_tokens > num_ctx:
        raise ValueError(
            f"Conservative context check failed: {total_bytes} UTF-8 text bytes + "
            f"{TEMPLATE_OVERHEAD} template reserve + {max_output_tokens} output tokens "
            f"> num_ctx={num_ctx}. Reduce complete evidence passages or increase context safely."
        )

    payload = {
        "model": model.strip(),
        "messages": package["messages"],
        "stream": True,
        "keep_alive": "60m",
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": num_ctx,
            "num_predict": max_output_tokens,
        },
    }
    transport = sender or _local_stream_json
    chunks = []
    size = 0
    done_frame = None
    stream = iter(transport(LOCAL_CHAT_URL, payload, timeout))
    try:
        for n, frame in enumerate(stream, 1):
            if n > MAX_STREAM_EVENTS:
                raise RuntimeError("Ollama emitted too many stream frames")
            if not isinstance(frame, dict):
                raise RuntimeError("Invalid Ollama stream frame")
            if "error" in frame:
                raise RuntimeError("Ollama reported an error during streaming")
            completed = frame.get("done")
            if type(completed) is not bool:
                raise RuntimeError("Ollama stream frame missing boolean done status")
            message = frame.get("message")
            if message is not None:
                if (not isinstance(message, dict)
                        or message.get("role") not in (None, "assistant")
                        or not isinstance(message.get("content", ""), str)):
                    raise RuntimeError("Ollama streamed a malformed assistant message")
                token = message.get("content", "")
                if token:
                    size += len(token)
                    if size > MAX_STREAM_TEXT:
                        raise RuntimeError("Ollama streamed an oversized answer")
                    chunks.append(token)
                    # The text is strictly unverified until the final event.
                    yield {"type": "delta", "text": "".join(chunks)}
            elif not completed:
                raise RuntimeError("Ollama stream frame has no assistant message")

            if completed:
                done_frame = frame
                break
        if done_frame is None:
            raise RuntimeError("Ollama stream ended without a final completion frame")
        status, reason, answer, cited_ids = _check_answer(
            "".join(chunks), {s["source_id"] for s in package["sources"]}
        )
        if done_frame.get("done_reason") != "stop":
            status = "rejected_incomplete"
            reason = "Ollama stopped without a normal completed response"
            answer, cited_ids = ABSTENTION, []
        yield {"type": "final", "result": _safe_result(
            package, status=status, reason=reason, answer=answer,
            used_ids=cited_ids, model=model, response=done_frame,
        )}
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
