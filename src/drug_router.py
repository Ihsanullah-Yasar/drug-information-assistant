"""Deterministic drug-name routing for source-grounded RAG retrieval.

This only chooses an exact indexed PDF when the question unambiguously names
one drug from the user's local target list. No fuzzy matching or medical
synonyms are invented; vector retrieval remains in src.retriever.
"""
from __future__ import annotations

import csv
from pathlib import Path
import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9]+", flags=re.ASCII)


def _words(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(unicodedata.normalize("NFKC", value).casefold()))


def _catalog(sources_csv: Path, targets_txt: Path) -> dict[tuple[str, ...], tuple[str, ...]]:
    """Return drug-token -> candidate PDF filenames, using ONLY local sources.

    Filenames are selected by full drug-name tokens at their start, followed by
    a separator, so e.g. `metformin` never selects a different drug by fuzzy
    similarity. There may be one or zero source labels per drug.
    """
    sources_csv, targets_txt = Path(sources_csv), Path(targets_txt)
    if not sources_csv.is_file() or not targets_txt.is_file():
        raise FileNotFoundError(
            "Drug-aware routing needs data/sources.csv and data/target_drugs.txt. "
            "Use --filename to explicitly select a PDF if your catalog is missing."
        )
    with sources_csv.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames or "filename" not in reader.fieldnames:
            raise ValueError("sources.csv must contain a filename column")
        filenames = []
        for row in reader:
            name = row.get("filename")
            if (not isinstance(name, str) or not name.endswith(".pdf")
                    or not name or name != Path(name).name or "\\" in name):
                raise ValueError(f"Invalid PDF filename in sources.csv: {name!r}")
            filenames.append(name)
    if not filenames or len(set(filenames)) != len(filenames):
        raise ValueError("sources.csv is empty or has duplicate PDF filenames")
    targets = [line.strip() for line in targets_txt.read_text(encoding="utf-8-sig").splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    if not targets:
        raise ValueError("target_drugs.txt is empty")
    catalog: dict[tuple[str, ...], tuple[str, ...]] = {}
    for target in targets:
        tokens = _words(target)
        if not tokens or tokens in catalog:
            raise ValueError(f"Invalid or duplicated drug name in target_drugs.txt: {target!r}")
        # Match a full drug prefix, not accidental substrings. Salt/formulation
        # words in the filename can follow the active ingredient.
        prefix = "_".join(tokens) + "_"
        matches = tuple(name for name in filenames if name.casefold().startswith(prefix))
        catalog[tokens] = matches
    return catalog


def named_drugs(question: str, catalog: dict[tuple[str, ...], tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Find whole-name mentions; longest match wins for overlapping aliases."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be nonempty")
    tokens = _words(question)
    found: list[tuple[str, ...]] = []
    occupied: set[int] = set()
    # Longest aliases first, so 'penicillin v' isn't mistaken for 'penicillin'.
    for name in sorted(catalog, key=lambda x: (-len(x), x)):
        size = len(name)
        for start in range(len(tokens) - size + 1):
            span = set(range(start, start + size))
            if not span.intersection(occupied) and tokens[start:start + size] == name:
                occupied.update(span)
                if name not in found:
                    found.append(name)
    return found


def resolve_drug_filename(
    question: str,
    *,
    sources_csv: Path,
    targets_txt: Path,
    filename: str | None = None,
    all_drugs: bool = False,
) -> str | None:
    """Resolve one named target drug to its exact source filename.

    `filename` is explicit user intent. `all_drugs` intentionally opts out of
    auto-scoping, for corpus-wide evaluation or comparative search. Otherwise,
    multi-drug questions are rejected so Step 8 doesn't unknowingly mix drugs.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be nonempty")
    if filename is not None and all_drugs:
        raise ValueError("Use either --filename or --all-drugs, not both")
    if filename is not None:
        return filename  # src.retriever validates PDF filename and source metadata.
    if all_drugs:
        return None
    catalog = _catalog(sources_csv, targets_txt)
    mentions = named_drugs(question, catalog)
    if not mentions:
        return None  # General information-retrieval query with no recognized drug.
    if len(mentions) > 1:
        names = [" ".join(tokens) for tokens in mentions]
        raise ValueError(
            "Multiple drugs mentioned (" + ", ".join(names) + "). "
            "For a single-drug answer, use --filename to choose the intended "
            "label. Use --all-drugs only for an explicitly broad/comparative "
            "retrieval analysis; do not merge unrelated drug facts into one answer."
        )
    drug = mentions[0]
    files = catalog[drug]
    if len(files) != 1:
        raise ValueError(
            f"Drug {' '.join(drug)!r} matched {len(files)} source PDFs. "
            "Choose the correct label with --filename; do not guess a formulation."
        )
    return files[0]
