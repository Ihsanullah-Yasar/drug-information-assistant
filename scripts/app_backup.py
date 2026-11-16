"""Phase 1: modern conversational Gradio front-end for the existing local RAG.

Phase 1 deliberately preserves the synchronous, validated Step 9 generator.
True Ollama token streaming will be introduced in Phase 2.

Run: python -m scripts.app  (with project .venv activated)
"""
from __future__ import annotations

from pathlib import Path
import sys

from src.prompt_builder import ABSTENTION
from src.rag_pipeline import LocalDrugRAG, available_pdf_labels
from src.chat_ui import (
    AUTO_LABEL, start_turn, check_label_scope,
    append_result, append_error, reset_chat,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_HEADERS = ["Citation", "PDF label", "Page", "Chunk", "Evidence excerpt", "DailyMed URL", "Cited?"]

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


# Style layer: use Gradio's theme variables, so both light/dark palettes are supported.
STYLE = r"""
.gradio-container {max-width: 1120px !important; margin: auto !important; padding: 12px 20px 6px !important;}
#assistant-header {padding: 10px 4px 8px !important; align-items: center !important;}
#assistant-brand {font-size: 19px; font-weight: 720; letter-spacing: -.6px; margin: 0 !important;}
#assistant-tagline {font-size: 12px; opacity: .65; margin: 0 !important;}
#assistant-header button {border-radius: 999px !important;}
#assistant-chat {border: 0 !important; background: transparent !important; box-shadow: none !important;}
#assistant-chat .wrap {background: transparent !important;}
#assistant-composer {border: 1px solid var(--border-color-primary) !important; border-radius: 20px !important;
    padding: 12px 14px 7px !important; background: var(--block-background-fill) !important;
    position: sticky !important; bottom: 8px !important; z-index: 10 !important;
    box-shadow: 0 5px 22px rgba(0,0,0,.06) !important;}
#assistant-input {border: 0 !important; background: transparent !important; box-shadow: none !important;}
#assistant-input textarea {background: transparent !important; border: 0 !important; box-shadow: none !important;
    padding: 8px 2px !important; font-size: 15px !important;}
#assistant-composer .form {border: 0 !important; background: transparent !important;}
#assistant-composer button {border-radius: 100px !important;}
#assistant-settings {border: 0 !important; background: transparent !important;}
#assistant-settings > button {font-size: 12px !important; opacity: .75; border: 0 !important;}
#assistant-send {min-width: 100px !important; border-radius: 30px !important;}
#assistant-note {text-align: center; font-size: 11px; opacity: .68; margin: 4px 0 0 !important;}
@media (max-width: 600px) {
 .gradio-container {padding: 6px 10px !important;}
 #assistant-chat {min-height: 52vh !important;}
 #assistant-send {min-width: 72px !important;}
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

    def run_turn(history: list[dict], question: str, selected_label: str, count: float):
        if not question:
            return history
        try:
            filename = check_label_scope(question, selected_label, instance)
            run = instance.ask(question, filename=filename, max_sources=int(count))
            return append_result(history, run)
        except (ValueError, OSError, RuntimeError, TypeError, MemoryError) as exc:
            return append_error(history, exc)

    with gr.Blocks(title="Drug Information Assistant") as demo:
        with gr.Row(elem_id="assistant-header"):
            with gr.Column(scale=6, min_width=230):
                gr.Markdown("💊 **Drug Information Assistant**", elem_id="assistant-brand")
                gr.Markdown("Local · Qwen2.5 7B · Drug-label evidence", elem_id="assistant-tagline")
            with gr.Column(scale=1, min_width=115):
                new_chat = gr.Button("✎ New chat", variant="secondary", size="sm")

        chat = gr.Chatbot(
            value=[],
            show_label=False,
            layout="panel",
            height="min(67vh, 670px)",
            placeholder="### How can I help you?\nAsk a question about a drug in your local DailyMed collection.",
            examples=[
                {"text": "What safety warnings are listed for ibuprofen?"},
                {"text": "What are the side effects of fluoxetine?"},
                {"text": "What is the acetaminophen liver warning?"},
            ],
            buttons=["copy", "copy_all"],
            elem_id="assistant-chat",
            sanitize_html=True,
            autoscroll=True,
        )
        # Both the prompt and ALL settings live in the same input/composer surface.
        with gr.Group(elem_id="assistant-composer"):
            message = gr.Textbox(
                show_label=False,
                lines=2,
                max_lines=6,
                max_length=1000,
                placeholder="Message Drug Information Assistant…",
                elem_id="assistant-input",
                container=False,
                autofocus=True,
            )
            with gr.Row(equal_height=True):
                with gr.Column(scale=5, min_width=185):
                    with gr.Accordion("⚙ Search settings", open=False, elem_id="assistant-settings"):
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
                            label="Complete evidence passages",
                        )
                        gr.Markdown(
                            "Only one drug label is used per answer. "
                            "Select a PDF only when the drug is ambiguous."
                        )
                with gr.Column(scale=1, min_width=110):
                    send = gr.Button("↑ Send", variant="primary", elem_id="assistant-send")

        gr.Markdown(
            "Educational drug-label information only · Not for diagnosis or prescribing · "
            "Citations identify sources, not independently verified claims · "
            "Local access: `127.0.0.1`",
            elem_id="assistant-note",
        )

        # Two-phase event: show user's text / pending message immediately,
        # then run the existing validated *non-streaming* LocalDrugRAG.ask().
        pending = gr.State("")
        submit_inputs = [message, chat]
        submit_outputs = [message, chat, pending]
        ask_inputs = [chat, pending, source_label, evidence]
        submit_ev = send.click(
            fn=start_turn, inputs=submit_inputs, outputs=submit_outputs,
            queue=False,
        )
        submit_ev.then(
            fn=run_turn, inputs=ask_inputs, outputs=[chat],
            concurrency_limit=1,
        )
        enter_ev = message.submit(
            fn=start_turn, inputs=submit_inputs, outputs=submit_outputs,
            queue=False,
        )
        enter_ev.then(
            fn=run_turn, inputs=ask_inputs, outputs=[chat],
            concurrency_limit=1,
        )
        new_chat.click(
            fn=reset_chat, inputs=[], outputs=[message, chat, pending, source_label, evidence], queue=False,
        )
    return demo


def main() -> int:
    try:
        app = build_app()
        app.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True, css=STYLE)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Unable to start local Gradio interface: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
