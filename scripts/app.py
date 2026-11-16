"""Modern Gradio front-end for the local grounded Drug Information Assistant.

Run: python -m scripts.app  (with project .venv activated)
"""
from __future__ import annotations

from pathlib import Path
import sys

from src.prompt_builder import ABSTENTION
from src.rag_pipeline import LocalDrugRAG, available_pdf_labels
from src.chat_ui import (
    AUTO_LABEL,
    STATUS_HTML_BUSY,
    STATUS_HTML_IDLE,
    start_turn,
    reset_chat,
    cancel_turn,
    stream_chat_turn,
    status_html_for_history,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_HEADERS = ["Citation", "PDF label", "Page", "Chunk", "Evidence excerpt", "DailyMed URL", "Cited?"]


def warm_local_ollama(model: str, num_ctx: int) -> None:
    """Load the chat model into Ollama before Gradio accepts questions.

    On 4 GB GPUs, cold-loading Qwen2.5-7B while the BGE embedder is also in
    memory often fails with CUDA_Host OOM. Preloading keeps the runner warm.
    """
    import json
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK only."}],
        "stream": False,
        "keep_alive": "60m",
        "options": {
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": 8,
        },
    }
    request = Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=600) as response:
            response.read(1_000_000)
    except HTTPError as exc:
        detail = exc.read(300).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama warmup failed (HTTP {exc.code}): {detail}. "
            "Free RAM and confirm `ollama serve` is running."
        ) from exc
    except (URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise RuntimeError(
            "Cannot reach local Ollama on 127.0.0.1:11434 for warmup. "
            "Start `ollama serve`, then rerun python -m scripts.app."
        ) from exc


# Intentionally retain the existing helper so original Step 10 tests / scripts work.
def format_run(run: dict) -> tuple[str, str, list[list[str]]]:
    if not isinstance(run, dict):
        raise ValueError("Invalid RAG run")
    status = str(run.get("status", "unknown"))
    source = run.get("filename") or "none selected"
    usage = run.get("usage") or {}
    tokens = usage.get("prompt_eval_count")
    tokens_label = str(tokens) if isinstance(tokens, int) else "not generated"
    details = (
        f"Status: {status} | Source: {source} | "
        f"Retrieved: {run.get('retrieved_count', 0)} | "
        f"Included: {run.get('included_count', 0)} | "
        f"Cited: {len(run.get('citations', []))} | Prompt tokens: {tokens_label}\n"
        f"{run.get('reason', '')}\n"
        "Citation IDs are checked, but medical claims are not automatically fact-checked."
    )
    table = [
        [row["citation"], row["filename"], str(row["page"]), str(row["chunk"]),
         row["excerpt"], row["source_url"], "yes" if row["used_in_answer"] else "no"]
        for row in run.get("sources", [])
    ]
    return str(run.get("answer", ABSTENTION)), details, table


STYLE = r"""
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,500;9..40,650;9..40,750&display=swap');

:root {
  --dia-bg: #f7f7f8;
  --dia-ink: #0d0d0d;
  --dia-muted: #6b6b6b;
  --dia-line: rgba(0,0,0,0.08);
  --dia-composer: #ffffff;
  --dia-composer-border: rgba(0,0,0,0.12);
  --dia-accent: #0f766e;
  --dia-send: #0d0d0d;
  --dia-send-ink: #ffffff;
}

html.dark, body.dark, .dark, .gradio-container.dark {
  --dia-bg: #212121;
  --dia-ink: #ececec;
  --dia-muted: #a0a0a0;
  --dia-line: rgba(255,255,255,0.08);
  --dia-composer: #303030;
  --dia-composer-border: rgba(255,255,255,0.10);
  --dia-accent: #2dd4bf;
  --dia-send: #ffffff;
  --dia-send-ink: #0d0d0d;
}

html, body {
  min-height: 100% !important;
  height: auto !important;
  margin: 0 !important;
  background: var(--dia-bg) !important;
  overflow-x: hidden !important;
  overflow-y: auto !important;
}

.gradio-container {
  width: 100% !important;
  max-width: 100% !important;
  min-height: 100vh !important;
  height: auto !important;
  margin: 0 !important;
  padding: 0 !important;
  font-family: "DM Sans", ui-sans-serif, system-ui, sans-serif !important;
  color: var(--dia-ink) !important;
  background: var(--dia-bg) !important;
  overflow: visible !important;
}

/* Kill Gradio default card chrome */
.gradio-container .main,
.gradio-container .wrap,
.gradio-container .contain,
.gradio-container > .main,
footer { background: transparent !important; box-shadow: none !important; border: 0 !important; overflow: visible !important; }
.gradio-container footer { display: none !important; }

#assistant-app {
  display: flex !important;
  flex-direction: column !important;
  min-height: 100vh !important;
  height: auto !important;
  width: 100% !important;
  background: var(--dia-bg) !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
  gap: 0 !important;
  overflow: visible !important;
}

/* ---- Top bar ---- */
#assistant-header {
  flex: 0 0 52px !important;
  height: 52px !important;
  padding: 0 16px !important;
  margin: 0 !important;
  align-items: center !important;
  border: 0 !important;
  border-bottom: 1px solid var(--dia-line) !important;
  background: var(--dia-bg) !important;
  box-shadow: none !important;
}
#assistant-brand, #assistant-brand * {
  margin: 0 !important;
  font-size: 0.95rem !important;
  font-weight: 650 !important;
  letter-spacing: -0.02em !important;
  color: var(--dia-ink) !important;
}
#assistant-new {
  min-width: auto !important;
  height: 36px !important;
  padding: 0 14px !important;
  border-radius: 999px !important;
  border: 1px solid var(--dia-line) !important;
  background: transparent !important;
  color: var(--dia-ink) !important;
  box-shadow: none !important;
  font-weight: 650 !important;
  letter-spacing: -0.01em !important;
  gap: 6px !important;
}

/* ---- Chat body: grow with content; page scrolls (no nested scroll box) ---- */
#assistant-chat {
  flex: 1 1 auto !important;
  width: 100% !important;
  max-width: 820px !important;
  margin: 0 auto !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 12px 16px 200px !important;
  min-height: calc(100vh - 220px) !important;
  height: auto !important;
  max-height: none !important;
  overflow: visible !important;
}
#assistant-chat > .wrapper,
#assistant-chat .bubble-wrap,
#assistant-chat .scroll-hide,
#assistant-chat [class*="scroll"],
#assistant-chat [class*="panel"],
#assistant-chat > div {
  height: auto !important;
  max-height: none !important;
  min-height: 0 !important;
  overflow: visible !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
#assistant-chat ::-webkit-scrollbar { width: 0 !important; height: 0 !important; }
#assistant-chat .message-wrap {
  max-width: 100% !important;
  padding-inline: 4px !important;
}
#assistant-chat .message,
#assistant-chat .bot,
#assistant-chat .user,
#assistant-chat .prose,
#assistant-chat p,
#assistant-chat li,
#assistant-chat strong {
  font-family: "DM Sans", ui-sans-serif, system-ui, sans-serif !important;
  color: var(--dia-ink) !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
/* ChatGPT-like: almost no bubble chrome */
#assistant-chat .bot .message,
#assistant-chat .bot .bubble-wrap,
#assistant-chat .message.bot,
#assistant-chat .user .message,
#assistant-chat .user .bubble-wrap,
#assistant-chat .message.user {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
}
#assistant-chat .user { opacity: 0.92; }

/* ---- Sticky bottom dock ---- */
#assistant-dock {
  position: fixed !important;
  left: 0 !important;
  right: 0 !important;
  bottom: 0 !important;
  z-index: 40 !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  gap: 6px !important;
  padding: 0 16px 14px !important;
  pointer-events: none;
  background: linear-gradient(180deg, transparent 0%, var(--dia-bg) 28%) !important;
  border: 0 !important;
  box-shadow: none !important;
}
#assistant-dock > * { pointer-events: auto; width: min(768px, 100%); }

#assistant-status {
  min-height: 18px !important;
  padding: 0 !important;
  margin: 0 !important;
  background: transparent !important;
  border: 0 !important;
}
.dia-status {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-height: 18px;
  padding: 0;
  border: 0;
  background: transparent;
  font-size: 0.75rem;
  color: var(--dia-muted);
}
.dia-spinner {
  width: 12px; height: 12px; border-radius: 50%;
  border: 2px solid color-mix(in srgb, var(--dia-accent) 30%, transparent);
  border-top-color: var(--dia-accent);
  animation: dia-spin .7s linear infinite;
}
@keyframes dia-spin { to { transform: rotate(360deg); } }

#assistant-composer {
  width: min(768px, 100%) !important;
  margin: 0 auto !important;
  border: 1px solid var(--dia-composer-border) !important;
  border-radius: 28px !important;
  padding: 10px 12px 8px !important;
  background: var(--dia-composer) !important;
  box-shadow: 0 0 0 1px rgba(0,0,0,0.02), 0 8px 24px rgba(0,0,0,0.18) !important;
}
#assistant-input { border: 0 !important; background: transparent !important; box-shadow: none !important; }
#assistant-input textarea {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 6px 8px !important;
  font-size: 15px !important;
  line-height: 1.45 !important;
  color: var(--dia-ink) !important;
  resize: none !important;
}
#assistant-input textarea::placeholder { color: var(--dia-muted) !important; }
#assistant-composer .form { border: 0 !important; background: transparent !important; }
#assistant-settings { border: 0 !important; background: transparent !important; }
#assistant-settings > button {
  font-size: 12px !important;
  color: var(--dia-muted) !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

#assistant-actions {
  justify-content: flex-end !important;
  gap: 8px !important;
  align-items: center !important;
}
#assistant-send, #assistant-stop {
  width: auto !important;
  min-width: 92px !important;
  max-width: none !important;
  height: 38px !important;
  padding: 0 16px !important;
  border-radius: 999px !important;
  font-size: 13.5px !important;
  font-weight: 650 !important;
  letter-spacing: -0.01em !important;
  box-shadow: none !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 7px !important;
  white-space: nowrap !important;
}
#assistant-send {
  background: var(--dia-send) !important;
  color: var(--dia-send-ink) !important;
  border: 0 !important;
}
#assistant-send:hover {
  filter: brightness(1.08);
}
#assistant-stop {
  background: color-mix(in srgb, #c04545 16%, var(--dia-composer)) !important;
  color: #ffb4b4 !important;
  border: 1px solid rgba(255, 140, 140, 0.35) !important;
}
html:not(.dark) #assistant-stop,
body:not(.dark) #assistant-stop {
  background: #fff1f1 !important;
  color: #9b1c1c !important;
  border-color: rgba(155, 28, 28, 0.25) !important;
}
#assistant-stop:hover {
  filter: brightness(1.05);
}

#assistant-note, #assistant-note * {
  text-align: center !important;
  font-size: 11px !important;
  color: var(--dia-muted) !important;
  margin: 0 !important;
  background: transparent !important;
  border: 0 !important;
}

@media (max-width: 720px) {
  #assistant-chat { padding: 6px 10px 200px !important; max-width: 100% !important; }
  #assistant-dock { padding: 0 10px 10px !important; }
  #assistant-composer { border-radius: 24px !important; }
  #assistant-send, #assistant-stop { min-width: 84px !important; padding: 0 12px !important; }
}
"""


def build_app(*, service: LocalDrugRAG | None = None):
    """Build a Gradio 6 chat layout, without model loading at startup."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is missing. Activate .venv and install: python -m pip install 'gradio>=6,<7'"
        ) from exc

    instance = service or LocalDrugRAG(PROJECT_ROOT)
    labels = available_pdf_labels(instance.sources_csv)

    def begin_turn(question: str, history: list[dict]):
        cleared, past, pending_q = start_turn(question, history)
        if not pending_q:
            return (
                gr.update(value=cleared, interactive=True),
                past,
                pending_q,
                gr.update(visible=True),
                gr.update(visible=False),
                STATUS_HTML_IDLE,
            )
        return (
            gr.update(value=cleared, interactive=False),
            past,
            pending_q,
            gr.update(visible=False),
            gr.update(visible=True),
            STATUS_HTML_BUSY,
        )

    def run_turn(history: list[dict], question: str, selected_label: str, count: float):
        """Yield chat history + live status bar while streaming."""
        if not question:
            yield history, STATUS_HTML_IDLE
            return
        for updated in stream_chat_turn(history, question, selected_label, count, instance):
            yield updated, status_html_for_history(updated)

    def end_turn():
        return gr.update(visible=True), gr.update(visible=False), gr.update(interactive=True)

    def stop_turn(history: list[dict]):
        past = cancel_turn(history)
        return (
            past,
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(interactive=True),
            status_html_for_history(past),
        )

    def new_session():
        msg, chat_hist, pending_q, label, evidence_n = reset_chat()
        return (
            gr.update(value=msg, interactive=True),
            chat_hist,
            pending_q,
            label,
            evidence_n,
            gr.update(visible=True),
            gr.update(visible=False),
            STATUS_HTML_IDLE,
        )

    with gr.Blocks(title="Drug Information Assistant") as demo:
        with gr.Column(elem_id="assistant-app"):
            with gr.Row(elem_id="assistant-header"):
                gr.Markdown("**Drug Information Assistant**", elem_id="assistant-brand")
                new_chat = gr.Button("✎ New chat", elem_id="assistant-new", size="sm")

            chat = gr.Chatbot(
                value=[],
                show_label=False,
                layout="panel",
                height="auto",
                placeholder=(
                    "### How can I help with a drug label?\n"
                    "Ask about warnings, indications, or side effects from your local DailyMed PDFs."
                ),
                examples=[
                    {"text": "What safety warnings are listed for ibuprofen?"},
                    {"text": "What are the side effects of fluoxetine?"},
                    {"text": "What is the acetaminophen liver warning?"},
                    {"text": "What is ampicillin used for?"},
                ],
                buttons=["copy"],
                elem_id="assistant-chat",
                sanitize_html=True,
                autoscroll=True,
            )

            with gr.Column(elem_id="assistant-dock"):
                status = gr.HTML(STATUS_HTML_IDLE, elem_id="assistant-status")
                with gr.Group(elem_id="assistant-composer"):
                    message = gr.Textbox(
                        show_label=False,
                        lines=1,
                        max_lines=6,
                        max_length=1000,
                        placeholder="Ask about a drug label…",
                        elem_id="assistant-input",
                        container=False,
                        autofocus=True,
                    )
                    with gr.Row(elem_id="assistant-actions", equal_height=True):
                        with gr.Column(scale=8, min_width=120):
                            with gr.Accordion("Settings", open=False, elem_id="assistant-settings"):
                                source_label = gr.Dropdown(
                                    choices=[AUTO_LABEL] + labels,
                                    value=AUTO_LABEL,
                                    allow_custom_value=False,
                                    label="Drug label",
                                    filterable=True,
                                    interactive=True,
                                )
                                evidence = gr.Slider(
                                    minimum=1, maximum=5, step=1, value=3,
                                    label="Evidence passages",
                                )
                        stop = gr.Button(
                            "⏹ Stop",
                            variant="secondary",
                            elem_id="assistant-stop",
                            visible=False,
                            scale=0,
                            min_width=92,
                        )
                        send = gr.Button(
                            "➤ Send",
                            variant="primary",
                            elem_id="assistant-send",
                            visible=True,
                            scale=0,
                            min_width=92,
                        )
                gr.Markdown(
                    "Educational only · Not for diagnosis or prescribing · Local 127.0.0.1",
                    elem_id="assistant-note",
                )

        pending = gr.State("")
        submit_inputs = [message, chat]
        submit_outputs = [message, chat, pending, send, stop, status]
        ask_inputs = [chat, pending, source_label, evidence]

        submit_ev = send.click(
            fn=begin_turn, inputs=submit_inputs, outputs=submit_outputs, queue=False,
        )
        send_run = submit_ev.then(
            fn=run_turn, inputs=ask_inputs, outputs=[chat, status], concurrency_limit=1,
        ).then(
            fn=end_turn, inputs=[], outputs=[send, stop, message],
        )

        enter_ev = message.submit(
            fn=begin_turn, inputs=submit_inputs, outputs=submit_outputs, queue=False,
        )
        enter_run = enter_ev.then(
            fn=run_turn, inputs=ask_inputs, outputs=[chat, status], concurrency_limit=1,
        ).then(
            fn=end_turn, inputs=[], outputs=[send, stop, message],
        )

        stop.click(
            fn=stop_turn,
            inputs=[chat],
            outputs=[chat, send, stop, message, status],
            cancels=[send_run, enter_run],
            queue=False,
        )
        new_chat.click(
            fn=new_session,
            inputs=[],
            outputs=[message, chat, pending, source_label, evidence, send, stop, status],
            cancels=[send_run, enter_run],
            queue=False,
        )
    return demo


def build_theme():
    import gradio as gr

    return gr.themes.Base(
        primary_hue="zinc",
        secondary_hue="zinc",
        neutral_hue="zinc",
        font=[gr.themes.GoogleFont("DM Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#f7f7f8",
        body_background_fill_dark="#212121",
        body_text_color="#0d0d0d",
        body_text_color_dark="#ececec",
        block_background_fill="transparent",
        block_background_fill_dark="transparent",
        block_border_width="0px",
        block_shadow="none",
        border_color_primary="rgba(0,0,0,0.08)",
        border_color_primary_dark="rgba(255,255,255,0.08)",
        button_primary_background_fill="#0d0d0d",
        button_primary_background_fill_dark="#ffffff",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#0d0d0d",
    )


def main() -> int:
    try:
        service = LocalDrugRAG(PROJECT_ROOT)
        print(
            f"Warming Ollama model {service.model!r} (num_ctx={service.num_ctx})…",
            flush=True,
        )
        warm_local_ollama(service.model, service.num_ctx)
        print("Ollama model ready. Starting Gradio…", flush=True)
        app = build_app(service=service)
        app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            inbrowser=True,
            theme=build_theme(),
            css=STYLE,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Unable to start local Gradio interface: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
