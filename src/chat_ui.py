"""UI-only chat formatting and safe label selection for the local RAG assistant.

No model calls, no third-party UI dependency, and no persistence.
Conversation transcript is UI-only; each new question is independently grounded.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse
from typing import Any

from src.prompt_builder import ABSTENTION
from src.drug_router import resolve_drug_filename

AUTO_LABEL = "Auto-detect from question"
WAITING = (
    "**Retrieving evidence**\n\n"
    "Searching your local DailyMed labels and preparing a grounded answer…"
)
STATUS_PREFIX = "**Working** · "
DRAFT_PREFIX = "**Live draft** · not citation-verified yet\n\n"
STATUS_HTML_IDLE = (
    '<div class="dia-status dia-idle"><span>Ready</span></div>'
)
STATUS_HTML_BUSY = (
    '<div class="dia-status dia-busy">'
    '<span class="dia-spinner" aria-hidden="true"></span>'
    '<span>Searching labels…</span>'
    '</div>'
)
STATUS_HTML_DRAFT = (
    '<div class="dia-status dia-draft">'
    '<span class="dia-spinner" aria-hidden="true"></span>'
    '<span>Streaming draft · citations unverified</span>'
    '</div>'
)
STATUS_HTML_DONE = (
    '<div class="dia-status dia-done"><span>Answer ready · citations checked</span></div>'
)

MAX_CHAT_TURNS = 30


def _content_text(content: object) -> str:
    """Gradio 6 Chatbot returns content blocks; older paths may still use plain strings."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return ""


def _normalize_history(history: list | None) -> list[dict]:
    """Convert Gradio chatbot payloads into plain role/content dicts for UI logic."""
    past: list[dict] = []
    for item in history or []:
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if not isinstance(item, dict):
            continue
        message = {
            "role": item.get("role"),
            "content": _content_text(item.get("content")),
        }
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            message["metadata"] = dict(metadata)
        past.append(message)
    return past


def start_turn(question: str, history: list[dict] | None) -> tuple[str, list[dict], str]:
    """Show user turn + pending indicator immediately; return cleared composer + question."""
    past = _normalize_history(history)
    if not isinstance(question, str) or not question.strip():
        return question if isinstance(question, str) else "", past, ""
    # Never permit a second submission to overwrite an in-progress draft.
    # Keep the next question in the composer for submission after completion.
    if past and _is_pending_assistant(past[-1]):
        return question, past, ""
    question = question.strip()
    # Do not insert invalid payload into the conversation. Backend has a matching check.
    if len(question) > 1000:
        past += [{"role": "user", "content": question[:1000] + "…"},
                 {"role": "assistant", "content": "Please shorten your question to 1,000 characters."}]
        return "", past, ""
    if sum(item.get("role") == "user" for item in past) >= MAX_CHAT_TURNS:
        past += [{"role": "assistant", "content": "Start a new chat to continue. This session has reached the turn limit."}]
        return "", past, ""
    past += [{"role": "user", "content": question}, {"role": "assistant", "content": WAITING}]
    return "", past, question


def check_label_scope(question: str, selected_label: str, service: Any) -> str | None:
    """Prevent selected PDF from silently overriding a different drug named in the question."""
    if not isinstance(selected_label, str) or not selected_label.strip():
        raise ValueError("Choose Auto-detect or one PDF from the source catalog")
    if selected_label == AUTO_LABEL:
        return None  # Backend performs source routing; do not resolve twice.
    named_pdf = resolve_drug_filename(
        question,
        sources_csv=service.sources_csv,
        targets_txt=service.targets_txt,
        filename=None,
        all_drugs=False,
    )
    if named_pdf is not None and named_pdf != selected_label:
        raise ValueError(
            "Your question names a different drug from the PDF selected in settings. "
            "Choose Auto-detect or select the matching label."
        )
    return selected_label


def _markdown_escape(value: object) -> str:
    s = str(value)
    # Prevent model/label text from introducing markdown links or list formatting.
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", s).replace("\n", " ")


def sources_message(run: dict) -> dict | None:
    """One compact, closed-by-default sources panel immediately below the answer."""
    items = run.get("sources") or []
    if not items:
        return None
    text = [
        f"**Source PDF:** `{_markdown_escape(run.get('filename') or 'unknown')}`",
        f"**Retrieved:** {int(run.get('retrieved_count') or 0)} · "
        f"**In prompt:** {int(run.get('included_count') or 0)}",
        "",
    ]
    for item in items:
        cited = "Cited" if item.get("used_in_answer") else "Not cited"
        title = _markdown_escape(item.get("citation") or "[S?]")
        page = _markdown_escape(item.get("page") or "?")
        excerpt = _markdown_escape(item.get("excerpt") or "")
        link = item.get("source_url")
        url = urlparse(link) if isinstance(link, str) else None
        if (url and url.scheme == "https" and url.hostname in {
                "dailymed.nlm.nih.gov", "www.dailymed.nlm.nih.gov"
            } and url.username is None and url.password is None and not url.fragment):
            ref = f"[Original DailyMed label]({link})"
        else:
            ref = "Source link unavailable"
        text += [f"**{title}** · PDF page {page} · {cited} · {ref}",
                 f"> {excerpt}", ""]
    text += ["*Citation IDs are checked for validity, not medical factual support.*"]
    return {
        "role": "assistant",
        "content": "\n".join(text),
        "metadata": {"title": f"📚 Sources · {len(items)} passages", "status": "done"},
    }


def append_result(history: list[dict] | None, run: dict) -> list[dict]:
    """Replace the pending indicator with a validated final answer and sources."""
    past = _normalize_history(history)
    status = str(run.get("status") or "unknown")
    answer = str(run.get("answer") or ABSTENTION)
    if status != "answered":
        reason = str(run.get("reason") or "")
        if reason:
            answer += "\n\n" + "**Status:** " + _markdown_escape(reason)
    if past and _is_pending_assistant(past[-1]):
        past[-1] = {"role": "assistant", "content": answer}
    else:
        past.append({"role": "assistant", "content": answer})
    source_note = sources_message(run)
    if source_note is not None:
        past.append(source_note)
    return past


def _safe_error_hint(exception: Exception) -> str:
    """Map failures to safe operator hints; never echo raw exception text into chat."""
    text = str(exception).lower()
    if any(token in text for token in (
        "out-of-memory", "failed to allocate", "cuda_host", "ggml_backend",
    )):
        return (
            "Local Ollama ran out of memory while loading the model. "
            "Close other apps to free RAM, keep qwen2.5:7b-instruct loaded "
            "(`ollama run qwen2.5:7b-instruct`), then retry."
        )
    if "cannot reach local ollama" in text or "stream failed or disconnected" in text:
        return "Cannot reach local Ollama on 127.0.0.1:11434. Start `ollama serve` and retry."
    if "http 404" in text:
        return "Ollama model tag not found. Run `ollama pull qwen2.5:7b-instruct`."
    if "http 500" in text:
        return (
            "Ollama returned HTTP 500 (often GPU/RAM pressure on Quadro 4 GB). "
            "Free memory, preload the model, then retry."
        )
    if "no pending assistant reply" in text:
        return (
            "Chat UI lost the in-progress reply slot. Click New chat and ask again."
        )
    return "Check the selected drug label, Ollama availability, or memory allocation."


def append_error(history: list[dict] | None, exception: Exception) -> list[dict]:
    """Fail closed without publishing an unvalidated answer or hallucinated sources."""
    return append_result(history, {
        "status": "error", "answer": ABSTENTION,
        "reason": (
            f"Unable to answer safely ({type(exception).__name__}). "
            f"{_safe_error_hint(exception)}"
        ),
        "sources": [],
    })


def reset_chat() -> tuple[str, list[dict], str, str, int]:
    """Clear conversation and return the composer to safe default settings."""
    return "", [], "", AUTO_LABEL, 3


def _is_pending_assistant(item: dict) -> bool:
    text = _content_text(item.get("content"))
    return (item.get("role") == "assistant"
            and (text == WAITING or text.startswith(STATUS_PREFIX)
                 or text.startswith(DRAFT_PREFIX)))


def _replace_pending(history: list[dict] | None, content: str) -> list[dict]:
    past = _normalize_history(history)
    if not past or not _is_pending_assistant(past[-1]):
        raise RuntimeError("No pending assistant reply to update")
    past[-1] = {"role": "assistant", "content": content}
    return past


def apply_stream_event(history: list[dict] | None, event: dict) -> list[dict]:
    """Show progress or unverified draft; finalize only from validated result."""
    if not isinstance(event, dict):
        raise RuntimeError("Invalid stream event")
    kind = event.get("type")
    if kind == "status":
        msg = event.get("message")
        if not isinstance(msg, str):
            raise RuntimeError("Invalid streaming status message")
        return _replace_pending(history, STATUS_PREFIX + _markdown_escape(msg))
    if kind == "delta":
        msg = event.get("text")
        if not isinstance(msg, str):
            raise RuntimeError("Invalid streaming draft")
        return _replace_pending(history, DRAFT_PREFIX + msg)
    if kind == "final":
        result = event.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Invalid final streaming result")
        return append_result(history, result)
    raise RuntimeError("Unrecognized streaming event")


def cancel_turn(history: list[dict] | None) -> list[dict]:
    """Discard any unverified draft after the user stops the streaming run."""
    past = _normalize_history(history)
    if not past or not _is_pending_assistant(past[-1]):
        return past
    return append_result(past, {
        "status": "cancelled",
        "answer": ABSTENTION,
        "reason": "Generation cancelled before citation validation.",
        "sources": [],
    })


def status_html_for_history(history: list[dict] | None) -> str:
    """Map the latest assistant bubble to a compact status-bar state."""
    past = _normalize_history(history)
    if not past:
        return STATUS_HTML_IDLE
    text = _content_text(past[-1].get("content"))
    if text.startswith(DRAFT_PREFIX):
        return STATUS_HTML_DRAFT
    if text == WAITING or text.startswith(STATUS_PREFIX):
        return STATUS_HTML_BUSY
    if past[-1].get("role") == "assistant":
        return STATUS_HTML_DONE
    return STATUS_HTML_IDLE


def stream_chat_turn(history: list[dict], question: str,
                     selected_label: str, evidence_count: float, service: Any):
    """Gradio-independent generator used by the chat callback.

    The final Ollama answer is accepted only after citation validation.
    On any failure, the last provisional message is completely removed.
    Draft banners are removed automatically when the validated final result lands.
    """
    current = _normalize_history(history)
    if not question:
        yield current
        return
    stream = None
    try:
        filename = check_label_scope(question, selected_label, service)
        stream = iter(service.ask_stream(question, filename=filename,
                                         max_sources=int(evidence_count)))
        for event in stream:
            current = apply_stream_event(current, event)
            yield current
            if event.get("type") == "final":
                break
    except (ValueError, OSError, RuntimeError, TypeError, MemoryError) as exc:
        yield append_error(current, exc)
    finally:
        if stream is not None:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
