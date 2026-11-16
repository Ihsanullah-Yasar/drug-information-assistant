"""Step 10: run the local Drug Information Assistant with Gradio.

From the project root:
    python -m pip install 'gradio>=6,<7'
    python -m scripts.app

Open http://127.0.0.1:7860. The app never enables public Gradio sharing.
"""
from __future__ import annotations

from pathlib import Path
import sys

from src.prompt_builder import ABSTENTION
from src.rag_pipeline import LocalDrugRAG, available_pdf_labels

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTO_LABEL = "Auto-detect from question"
TABLE_HEADERS = ["Citation", "PDF label", "Page", "Chunk", "Evidence excerpt", "DailyMed URL", "Cited?"]


def format_run(run: dict) -> tuple[str, str, list[list[str]]]:
    """Return plain text controls, not untrusted HTML/markdown from PDFs."""
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


def build_app(*, service: LocalDrugRAG | None = None):
    """Build the UI without loading models or querying Ollama on startup."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio is missing. Install it with: python -m pip install 'gradio>=6,<7'") from exc

    instance = service or LocalDrugRAG(PROJECT_ROOT)
    labels = available_pdf_labels(instance.sources_csv)

    def respond(question: str, selected_label: str, evidence_count: int):
        try:
            filename = None if selected_label == AUTO_LABEL else selected_label
            result = instance.ask(question, filename=filename, max_sources=int(evidence_count))
            return format_run(result)
        except (ValueError, OSError, RuntimeError, TypeError) as exc:
            return (
                ABSTENTION,
                f"Unable to answer safely: {type(exc).__name__}: {exc}\n"
                "Check the index, local model, memory, or selected drug label. No sources were fabricated.",
                [],
            )

    with gr.Blocks(title="Drug Information Assistant — Local RAG") as demo:
        gr.Markdown(
            "# Drug Information Assistant\n"
            "**100 DailyMed labels · BGE-small embeddings · ChromaDB · Qwen2.5 via Ollama**\n\n"
            "Educational, evidence-grounded answers from your local drug-label collection. "
            "Not for diagnosis, prescribing, or emergency decisions."
        )
        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="Your question", lines=3, max_lines=5,
                    placeholder="What safety warnings are listed for ibuprofen?",
                )
                with gr.Row():
                    label = gr.Dropdown(
                        choices=[AUTO_LABEL] + labels, value=AUTO_LABEL,
                        label="Source label (optional)", allow_custom_value=False,
                        interactive=True,
                    )
                    evidence = gr.Slider(
                        minimum=1, maximum=5, value=3, step=1,
                        label="Maximum complete evidence passages",
                    )
                ask_button = gr.Button("Find answer", variant="primary")
            with gr.Column(scale=2):
                gr.Markdown(
                    "### How to use\n"
                    "1. Name one drug in your question, or choose its PDF label.\n"
                    "2. The app retrieves **Top-5** chunks from that one PDF.\n"
                    "3. It includes up to **3 complete passages** by default.\n"
                    "4. Review the cited excerpts and the original DailyMed label.\n\n"
                    "Questions about multiple drugs are not automatically combined."
                )
        answer = gr.Textbox(
            label="Grounded answer", lines=10, max_lines=18, interactive=False,
        )
        status = gr.Textbox(
            label="Retrieval and validation status", lines=4, max_lines=7,
            interactive=False,
        )
        sources = gr.Dataframe(
            headers=TABLE_HEADERS, datatype="str", interactive=False,
            label="Evidence passages and citations", wrap=True,
        )
        gr.Examples(
            examples=[
                ["What safety warnings are listed for ibuprofen?"],
                ["What warnings are listed for acetaminophen?"],
            ],
            inputs=[question],
        )
        gr.Markdown(
            "**Privacy:** local model and local ChromaDB. This UI is bound to `127.0.0.1` "
            "and does not create a public share link. The DailyMed URLs are source references; "
            "opening them uses your browser. Citations indicate the source document, not "
            "automatic medical verification."
        )
        ask_button.click(
            fn=respond, inputs=[question, label, evidence],
            outputs=[answer, status, sources], concurrency_limit=1,
        )
        question.submit(
            fn=respond, inputs=[question, label, evidence],
            outputs=[answer, status, sources], concurrency_limit=1,
        )
    return demo


def main() -> int:
    try:
        demo = build_app()
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Unable to start local Gradio interface: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
