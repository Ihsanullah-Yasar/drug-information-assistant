"""UI-only chat formatting and safe label selection for the local RAG assistant.

No model calls, no third-party UI dependency, and no persistence.
Conversation transcript is for display in Phase 1; model-history / streaming arrive later.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse
from typing import Any

from src.prompt_builder import ABSTENTION
from src.drug_router import resolve_drug_filename

AUTO_LABEL = "Auto-detect from question"
WAITING = "Searching the drug labels and preparing a grounded answer…"
MAX_CHAT_TURNS = 30


def start_turn(question: str, history: list[dict] | None) -> tuple[str, list[dict], str]:
    """Show user turn + pending indicator immediately; return cleared composer + question."""
    past = [dict(item) for item in (history or [])]
    if not isinstance(question, str) or not question.strip():
        return question if isinstance(question, str) else "", past, ""
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
    named_pdf = resolve_drug_filename(
        question,
        sources_csv=service.sources_csv,
        targets_txt=service.targets_txt,
        filename=None,
        all_drugs=False,
    )
    if selected_label == AUTO_LABEL:
        return None  # The service will resolve and validate the named drug again.
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
    past = [dict(m) for m in (history or [])]
    status = str(run.get("status") or "unknown")
    answer = str(run.get("answer") or ABSTENTION)
    if status != "answered":
        reason = str(run.get("reason") or "")
        if reason:
            answer += "\n\n" + "**Status:** " + _markdown_escape(reason)
    if past and past[-1].get("role") == "assistant" and past[-1].get("content") == WAITING:
        past[-1] = {"role": "assistant", "content": answer}
    else:
        past.append({"role": "assistant", "content": answer})
    source_note = sources_message(run)
    if source_note is not None:
        past.append(source_note)
    return past


def append_error(history: list[dict] | None, exception: Exception) -> list[dict]:
    """Fail closed without publishing an unvalidated answer or hallucinated sources."""
    return append_result(history, {
        "status": "error", "answer": ABSTENTION,
        "reason": (
            f"Unable to answer safely ({type(exception).__name__}). "
            "Check the selected drug label, Ollama availability, or memory allocation."
        ),
        "sources": [],
    })


def reset_chat() -> tuple[str, list[dict], str, str, int]:
    """Clear conversation and return the composer to safe default settings."""
    return "", [], "", AUTO_LABEL, 3
