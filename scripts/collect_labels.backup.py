"""Collect verified DailyMed labels with atomic, resumable source registration."""



import argparse

import csv

from datetime import datetime, timezone

from decimal import Decimal

from email.utils import parsedate_to_datetime

import hashlib

from html.parser import HTMLParser

from http.client import IncompleteRead

import io

import json

import os

from pathlib import Path

import re

import sys

import tempfile

import time

from urllib.error import HTTPError, URLError

from urllib.parse import parse_qs, urlencode, urlparse

from urllib.request import Request, urlopen

from uuid import UUID

import xml.etree.ElementTree as ET



import pymupdf





PROJECT_ROOT = Path(__file__).resolve().parents[1]

API_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

NS = {"s": "urn:hl7-org:v3"}

COLUMNS = ["filename", "product_name", "packager", "label_updated_on", "source_url"]

STATE_FILE = "data/collection_state.json"

REVIEWED_IBUPROFEN_SET_ID = "1a665e64-9f30-be37-4a83-38789f1f1e89"

# Explicitly reviewed identities, not guesses based on filenames.

REVIEWED = {

    "acetaminophen": ("671adabe-5f82-4599-a133-3f67ccb34079", "acetaminophen_325mg_health_pharma.pdf"),

    "ibuprofen": (REVIEWED_IBUPROFEN_SET_ID, "ibuprofen_200mg_haleon_us_holdings_llc.pdf"),

    "naproxen": ("12202ef3-c65e-45a6-88ed-5d945d8fd41c", "naproxen_500mg_remedyrepack_inc.pdf"),

}

HUMAN_CATEGORIES = {"HUMAN OTC DRUG LABEL", "HUMAN PRESCRIPTION DRUG LABEL"}





def normalize(value: str) -> str:

    return value.strip().casefold()





def canonical_set_id(value: str) -> str:

    return str(UUID(value))





def fetch(url: str) -> bytes:

    """GET with four attempts, bounded backoff, and Retry-After support."""

    for attempt in range(4):

        delay = 2 ** attempt

        try:

            request = Request(url, headers={"User-Agent": "DrugInformationAssistant-Course/1.0"})

            with urlopen(request, timeout=45) as response:

                return response.read()

        except HTTPError as error:

            if error.code not in {408, 429, 500, 502, 503, 504} or attempt == 3:

                raise

            retry_after = error.headers.get("Retry-After") if error.headers else None

            if retry_after:

                try:

                    delay = max(delay, float(retry_after))

                except ValueError:

                    until = parsedate_to_datetime(retry_after)

                    if until.tzinfo is None:

                        until = until.replace(tzinfo=timezone.utc)

                    delay = max(delay, (until - datetime.now(timezone.utc)).total_seconds())

            if delay > 60:

                # Never retry earlier than requested or block the batch indefinitely.

                raise RuntimeError(f"HTTP {error.code}: Retry-After {delay:.0f}s exceeds retry budget; retry next run") from error

        except (URLError, TimeoutError, ConnectionError, IncompleteRead):

            if attempt == 3:

                raise

        time.sleep(delay)

    raise RuntimeError("Network attempts exhausted")





def attribute(element: ET.Element, path: str, name: str) -> str:

    node = element.find(path, NS)

    return node.get(name, "") if node is not None else ""





def required(value, field: str) -> str:

    if not isinstance(value, str) or not value.strip() or re.search(r"\bMISSING\b", value, re.I):

        raise ValueError(f"Required field empty or MISSING: {field}")

    return value.strip()





def positive_number(value: str) -> str:

    number = Decimal(value)

    if not number.is_finite() or number <= 0:

        raise ValueError(f"Invalid strength quantity: {value!r}")

    return format(number, "f")





def verify_label(root: ET.Element, candidate: dict, expected_ingredient: str) -> dict:

    """Verify single-ingredient human products using ingredient and active-moiety names."""

    target = normalize(required(expected_ingredient, "expected ingredient"))

    category = attribute(root, "s:code", "displayName")

    if category not in HUMAN_CATEGORIES:

        raise ValueError("human drug label category not verified")

    set_id = canonical_set_id(attribute(root, "s:setId", "root"))

    if set_id != canonical_set_id(candidate["setid"]):

        raise ValueError("Set ID does not match the search result")

    products = root.findall(".//s:subject/s:manufacturedProduct", NS)

    if not products:

        raise ValueError("product details missing")

    evidence, strengths = [], []

    for wrapper in products:

        product = wrapper.find("s:manufacturedProduct", NS)

        if product is None or product.find("s:part", NS) is not None:

            raise ValueError("missing product details or multipart kit")

        actives = [node for node in product.findall("s:ingredient", NS)

                   if node.get("classCode", "").startswith("ACTI")]

        if len(actives) != 1:

            raise ValueError("product does not have exactly one active ingredient")

        active = actives[0]

        actual = required(active.findtext("s:ingredientSubstance/s:name", namespaces=NS), "ingredient")

        moieties = [node.text.strip() for node in active.findall(

            "s:ingredientSubstance/s:activeMoiety/s:activeMoiety/s:name", NS

        ) if node.text and node.text.strip()]

        if target not in {normalize(name) for name in [actual, *moieties]}:

            raise ValueError(f"sole active ingredient {actual!r} is not verified as {expected_ingredient}")

        form = required(attribute(product, "s:formCode", "displayName"), "dosage form")

        routes = sorted({node.get("displayName", "") for node in wrapper.findall(".//s:routeCode", NS)})

        if not routes or not all(routes):

            raise ValueError("route not verified")

        strength = {

            "amount": positive_number(attribute(active, "s:quantity/s:numerator", "value")),

            "unit": required(attribute(active, "s:quantity/s:numerator", "unit"), "strength unit"),

            "basis_amount": positive_number(attribute(active, "s:quantity/s:denominator", "value")),

            "basis_unit": required(attribute(active, "s:quantity/s:denominator", "unit"), "strength denominator unit"),

            "form": form,

        }

        if strength not in strengths:

            strengths.append(strength)

        evidence.append({"ingredient": actual, "active_moieties": moieties, "form": form, "routes": routes})

    strengths.sort(key=lambda s: (Decimal(s["amount"]), s["unit"], Decimal(s["basis_amount"]), s["basis_unit"], s["form"]))

    oral_tablet = all(e["form"].startswith("TABLET") and e["routes"] == ["ORAL"] for e in evidence)

    descriptions = []

    for s in strengths:

        basis = s["form"].lower() if s["basis_unit"] == "1" else s["basis_unit"]

        if s["basis_unit"] == "1" and s["form"].startswith("TABLET"):

            basis = "tablet"

        descriptions.append(f"{s['amount']} {s['unit']} per {s['basis_amount']} {basis}")

    label = {

        "Ingredient": "; ".join(sorted({e["ingredient"] for e in evidence})),

        "Strength": "; ".join(descriptions), "Strengths": strengths,

        "Packager": required(root.findtext("s:author/s:assignedEntity/s:representedOrganization/s:name", namespaces=NS), "packager"),

        "Set ID": set_id,

        "Source page URL": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}",

        "Exact product title (DailyMed API)": candidate.get("title", ""),

        "Verified target": target, "Category": category, "Evidence": evidence,

        "Oral tablet": oral_tablet,

        "Verification": f"{category}; single active ingredient matched to {expected_ingredient} using SPL ingredient/active-moiety data",

    }

    if oral_tablet:

        label["Strength per tablet"] = label["Strength"].replace("per 1 tablet", "per tablet")

    return label





class PageText(HTMLParser):

    """Read visible page text and its label heading without extra dependencies."""

    def __init__(self):

        super().__init__()

        self.parts, self.heading = [], []

        self.in_heading = False

        self.hidden = 0



    def handle_starttag(self, tag, attrs):

        if tag in {"script", "style"}:

            self.hidden += 1

        if tag == "h1":

            self.in_heading = True



    def handle_endtag(self, tag):

        if tag in {"script", "style"}:

            self.hidden = max(0, self.hidden - 1)

        if tag == "h1":

            self.in_heading = False



    def handle_data(self, data):

        if not self.hidden and data.strip():

            self.parts.append(data.strip())

            if self.in_heading:

                self.heading.append(data.strip())





def source_details(label: dict) -> dict:

    """Use DailyMed's displayed Updated date, never a download or current date."""

    parser = PageText()

    parser.feed(fetch(label["Source page URL"]).decode("utf-8", errors="replace"))

    match = re.search(r"\bUpdated\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})", " ".join(parser.parts))

    if not match:

        raise ValueError("DailyMed displayed Updated date missing")

    date = datetime.strptime(match.group(1), "%B %d, %Y")

    label["Label updated on"] = f"{date.strftime('%B')} {date.day}, {date.year}"

    heading = re.sub(r"^Label:\s*", "", " ".join(parser.heading)).strip()
    label["Product name"] = required(
        heading or label.get("Exact product title (DailyMed API)", ""),
        "source page title",
    )
    if not label["Exact product title (DailyMed API)"]:
        label["Exact product title (DailyMed API)"] = label["Product name"]

    return label





def label_by_set_id(set_id: str, target: str, candidate=None) -> dict:

    set_id = canonical_set_id(set_id)

    root = ET.fromstring(fetch(f"{API_BASE}/spls/{set_id}.xml"))

    return source_details(verify_label(root, candidate or {"setid": set_id}, target))





def slug(value: str, preserve_decimals=False) -> str:

    value = value.casefold().replace("%", " percent ").replace("µ", "u").replace("μ", "u")

    if preserve_decimals:

        value = re.sub(r"(?<!\d)\.|\.(?!\d)", "_", value)

    value = re.sub(r"[^a-z0-9.]+" if preserve_decimals else r"[^a-z0-9]+", "_", value).strip("_.")

    if not value:

        raise ValueError("Metadata contains no usable filename characters")

    return value





def build_pdf_filename(label: dict) -> str:

    """Preserve labeled numeric quantities and units for all dosage forms."""

    ingredient = slug(required(label.get("Ingredient"), "Ingredient"))

    packager = slug(required(label.get("Packager"), "Packager"))

    strengths = label.get("Strengths")

    tokens = set()

    if strengths:

        for s in strengths:

            amount = positive_number(s["amount"])

            token = slug(f"{amount}{s['unit']}", preserve_decimals=True)

            if s["basis_unit"] != "1" or Decimal(s["basis_amount"]) != 1:

                token += "_per_" + slug(f"{s['basis_amount']}{s['basis_unit']}", preserve_decimals=True)

            tokens.add((Decimal(amount), token))

    else:

        # Compatibility with the reviewed metadata used by the ibuprofen command.

        value = required(label.get("Strength per tablet"), "Strength per tablet")

        for strength in value.split(";"):

            match = re.fullmatch(r"\s*(\d+(?:\.\d+)?|\.\d+)\s+(.+?)\s+per\s+tablet\s*", strength, re.I)

            if not match:

                raise ValueError(f"Unrecognized strength: {strength!r}")

            amount, unit = match.groups()

            tokens.add((Decimal(amount), slug(f"{positive_number(amount)}{unit}", True)))

    filename = f"{ingredient}_{'_'.join(token for _, token in sorted(tokens))}_{packager}.pdf"

    if len(filename) > 200:

        digest = hashlib.sha256(filename.encode()).hexdigest()[:12]

        filename = f"{ingredient[:40]}_{'_'.join(token for _, token in sorted(tokens))}_{packager[:40]}_{digest}.pdf"

    if len(filename) > 240:

        raise ValueError("Verified strengths produce an overlong Windows filename")

    return filename





def select_label(drug_name: str, excluded_setids=None, max_pages: int = 5) -> dict:
    """Prefer verified oral tablets; try other verified forms if needed.

    max_pages limits network calls for very common names. Increase it explicitly
    if an unresolved drug has a suitable label beyond the first search pages.
    """
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1")

    excluded = set(excluded_setids or ())

    other_candidates, fallbacks, seen, reasons = [], [], set(), []



    def check(candidate):

        try:

            set_id = canonical_set_id(candidate["setid"])

            if set_id in excluded:

                raise ValueError("Set ID already assigned to another target")

            return label_by_set_id(set_id, drug_name, candidate)

        except (OSError, ValueError, RuntimeError, KeyError, ET.ParseError, IncompleteRead) as error:

            reasons.append(f"{candidate.get('setid', 'missing Set ID')}: {error}")

            return None



    page = 1

    while True:

        query = urlencode({"drug_name": drug_name, "name_type": "generic", "pagesize": 100, "page": page})

        result = json.loads(fetch(f"{API_BASE}/spls.json?{query}"))

        for candidate in result["data"]:

            if candidate.get("setid") in seen:

                continue

            seen.add(candidate.get("setid"))

            # Title is a preference hint, never ingredient or form evidence.

            if "TABLET" not in candidate.get("title", "").upper():

                other_candidates.append(candidate)

                continue

            label = check(candidate)

            if label:

                if label["Oral tablet"]:

                    return label

                fallbacks.append(label)

        if page >= min(int(result["metadata"]["total_pages"]), max_pages):

            break

        page += 1

    if fallbacks:

        return fallbacks[0]

    for candidate in other_candidates:

        label = check(candidate)

        if label:

            return label

    raise ValueError(
        f"No verified single-ingredient human label for {drug_name} within "
        f"{max_pages} API page(s). " + ("; ".join(reasons[-3:]) or "No suitable search results")
    )





def atomic_write(path: Path, data: bytes, validator=None) -> None:

    """Validate a temporary file on the destination volume before atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = None

    try:

        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:

            temporary = Path(handle.name)

            handle.write(data)

            handle.flush()

            os.fsync(handle.fileno())

        if validator:

            validator(temporary.read_bytes())

        os.replace(temporary, path)

    finally:

        if temporary is not None:

            temporary.unlink(missing_ok=True)





def load_state(project_root: Path) -> dict:

    path = project_root / STATE_FILE

    if not path.exists():

        return {"version": 1, "targets": {}, "downloads": {}, "errors": {}}

    state = json.loads(path.read_text(encoding="utf-8"))

    if state.get("version") != 1 or any(not isinstance(state.get(key), dict) for key in ("targets", "downloads", "errors")):

        raise ValueError("Invalid collection state; restore a valid progress file")

    return state





def save_state(project_root: Path, state: dict) -> None:

    atomic_write(project_root / STATE_FILE, (json.dumps(state, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))





def local_path(project_root: Path, filename: str) -> Path:
    """Refuse unsafe or non-PDF filenames, particularly on Windows."""
    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or not filename.lower().endswith(".pdf")
        or re.search(r'[<>:"/\\|?*\x00-\x1f]', filename)
        or filename.endswith((" ", "."))
        or filename.lower().split(".")[0] in {
            "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5",
            "com6", "com7", "com8", "com9", "lpt1", "lpt2", "lpt3", "lpt4",
            "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
        }
    ):
        raise ValueError(f"Unsafe local PDF filename: {filename!r}")
    return project_root / "data/documents" / filename



def source_set_id(row: dict) -> str:

    url = urlparse(row["source_url"])

    if url.scheme != "https" or url.hostname not in {"dailymed.nlm.nih.gov", "www.dailymed.nlm.nih.gov"}:

        raise ValueError("Source row is not an official DailyMed URL")

    return canonical_set_id(parse_qs(url.query)["setid"][0])





def read_sources(project_root: Path) -> list[dict]:

    path = project_root / "data/sources.csv"

    if not path.exists():

        return []

    with path.open(encoding="utf-8", newline="") as handle:

        reader = csv.DictReader(handle)

        if reader.fieldnames != COLUMNS:

            raise ValueError("Unexpected sources.csv columns")

        rows = list(reader)

    for row in rows:

        if set(row) != set(COLUMNS) or any(not row[key] for key in COLUMNS):

            raise ValueError("Malformed or incomplete source row")

    return rows





def source_row(label: dict, filename: str) -> dict:

    return dict(zip(COLUMNS, [filename, required(label.get("Product name"), "product name"),

                              required(label.get("Packager"), "packager"),

                              required(label.get("Label updated on"), "label update date"),

                              label["Source page URL"]]))





def register_source(project_root: Path, row: dict) -> None:

    """Append atomically while retaining every byte of existing rows and their dates."""

    path = project_root / "data/sources.csv"

    rows = read_sources(project_root)

    same_name = [r for r in rows if normalize(r["filename"]) == normalize(row["filename"])]

    same_id = [r for r in rows if source_set_id(r) == source_set_id(row)]

    if same_name or same_id:

        if same_name == [row] and same_id == [row]:

            return

        raise ValueError(f"CSV identity or metadata conflict for {row['filename']}")

    original = path.read_bytes() if path.exists() else b""

    newline = "\r\n" if b"\r\n" in original else "\n"

    buffer = io.StringIO(newline="")

    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator=newline)

    if not original:

        writer.writeheader()

    elif not original.endswith((b"\r", b"\n")):

        buffer.write(newline)

    writer.writerow(row)

    atomic_write(path, original + buffer.getvalue().encode("utf-8"))





def choose_filename(label: dict, project_root: Path, state: dict) -> str:

    set_id = label["Set ID"]

    for reviewed_id, filename in REVIEWED.values():

        if reviewed_id == set_id:

            return filename

    rows = [r for r in read_sources(project_root) if source_set_id(r) == set_id]

    if len(rows) > 1:

        raise ValueError("Multiple CSV rows for the selected Set ID")

    if rows:

        return rows[0]["filename"]

    for filename, receipt in state["downloads"].items():

        if receipt["set_id"] == set_id:

            return filename

    occupied = {normalize(r["filename"]) for r in read_sources(project_root)}

    occupied.update(normalize(r["filename"]) for r in state["targets"].values())

    occupied.update(normalize(name) for name in state["downloads"])

    folder = project_root / "data/documents"

    if folder.exists():

        occupied.update(normalize(p.name) for p in folder.iterdir())

    base = build_pdf_filename(label)

    name, number = base, 0

    while normalize(name) in occupied:

        number += 1

        suffix = f"__{set_id}" + (f"_{number}" if number > 1 else "")

        name = f"{Path(base).stem[:190 - len(suffix)]}{suffix}.pdf"

    return name





def pdf_page_count(data: bytes) -> int:
    """Check that a PDF is readable and has text usable by the RAG loader."""
    try:
        with pymupdf.open(stream=data, filetype="pdf") as document:
            if not document.is_pdf or document.needs_pass or document.page_count < 1:
                raise ValueError("Expected a readable PDF with at least one page")
            has_text = any(page.get_text("text").strip() for page in document)
            if not has_text:
                raise ValueError("PDF contains no extractable text; cannot use for text RAG")
            return document.page_count
    except (pymupdf.FileDataError, pymupdf.FileNotFoundError) as error:
        raise ValueError(f"Unreadable PDF: {error}") from error



def download_label(label: dict, *, project_root=None, state=None, filename=None) -> tuple[Path, int]:

    """Reuse only a hash-bound or explicitly reviewed identity, never just a filename."""

    project_root = Path(project_root or PROJECT_ROOT)

    state = state if state is not None else load_state(project_root)

    set_id = canonical_set_id(label["Set ID"])

    filename = filename or choose_filename(label, project_root, state)

    path = local_path(project_root, filename)

    receipt = state["downloads"].get(filename)

    reviewed = (set_id, filename) in REVIEWED.values()

    if receipt and receipt["set_id"] != set_id:

        raise ValueError(f"Download identity conflict: {filename}")

    if path.exists():

        data = path.read_bytes()

        digest = hashlib.sha256(data).hexdigest()

        if receipt:

            if digest != receipt["sha256"]:

                raise ValueError(f"PDF hash changed: {filename}; existing file retained for review")

        elif not reviewed:

            raise ValueError(f"PDF identity unproven: {filename}; no download receipt or reviewed identity")

        count = pdf_page_count(data)

        if not receipt:

            state["downloads"][filename] = {"set_id": set_id, "sha256": digest}

            save_state(project_root, state)

        return path, count

    url = "https://dailymed.nlm.nih.gov/dailymed/downloadpdffile.cfm?" + urlencode({"setId": set_id})

    data = fetch(url)

    count = pdf_page_count(data)

    digest = hashlib.sha256(data).hexdigest()

    if receipt and digest != receipt["sha256"]:

        raise ValueError("Previously verified PDF version changed on DailyMed; refusing to pair old metadata with new bytes")

    # Journal byte identity BEFORE moving the file, so interrupted saves are recoverable.

    state["downloads"][filename] = {"set_id": set_id, "sha256": digest}

    save_state(project_root, state)

    if path.exists():

        raise ValueError(f"Destination appeared during download; not overwritten: {filename}")

    atomic_write(path, data, validator=pdf_page_count)

    return path, count





def download_ibuprofen() -> None:

    label = {"Ingredient": "IBUPROFEN", "Strength per tablet": "200 mg per tablet",

             "Packager": "Haleon US Holdings LLC", "Set ID": REVIEWED_IBUPROFEN_SET_ID}

    existed = (PROJECT_ROOT / "data/documents" / REVIEWED["ibuprofen"][1]).exists()

    path, count = download_label(label)

    print(f"{'Already exists (validated; no download)' if existed else 'Saved'}: {path}")

    print(f"Page count: {count}")





def check_record(target: str, record: dict) -> None:

    label = record["label"]

    if normalize(label["Verified target"]) != normalize(target) or label["Category"] not in HUMAN_CATEGORIES:

        raise ValueError("Saved verification does not match target")

    if canonical_set_id(record["set_id"]) != label["Set ID"] or not label["Evidence"]:

        raise ValueError("Invalid saved label identity/evidence")

    for evidence in label["Evidence"]:

        if normalize(target) not in {normalize(n) for n in [evidence["ingredient"], *evidence["active_moieties"]]}:

            raise ValueError("Saved ingredient/active-moiety evidence does not match target")

    row = record["source_row"]

    if row["filename"] != record["filename"] or source_set_id(row) != record["set_id"]:

        raise ValueError("Saved source row identity conflict")

    for field in COLUMNS:

        required(row.get(field), field)

    datetime.strptime(row["label_updated_on"], "%B %d, %Y")





def collect_target(target: str, project_root: Path, state: dict, max_search_pages: int = 5) -> tuple[Path, int, str]:

    key = normalize(target)

    record = state["targets"].get(key)

    occupied = {r["set_id"] for k, r in state["targets"].items() if k != key}

    occupied.update(sid for k, (sid, _) in REVIEWED.items() if k != key)

    if record is None:

        if key in REVIEWED:

            label = label_by_set_id(REVIEWED[key][0], target)

        else:

            label = select_label(target, occupied, max_pages=max_search_pages)

        if label["Set ID"] in occupied:

            raise ValueError("Set ID is already assigned to a different target")

        filename = choose_filename(label, project_root, state)

        existing = [r for r in read_sources(project_root) if normalize(r["filename"]) == normalize(filename)]

        if existing:

            if len(existing) != 1 or source_set_id(existing[0]) != label["Set ID"]:

                raise ValueError(f"Existing CSV filename points to a different/duplicate source: {filename}")

            row = existing[0]

        else:

            row = source_row(label, filename)

        record = {"set_id": label["Set ID"], "filename": filename, "label": label,

                  "source_row": row, "status": "planned"}

        check_record(target, record)

        state["targets"][key] = record

        save_state(project_root, state)

    check_record(target, record)

    if record["set_id"] in occupied:

        raise ValueError("Set ID is assigned to more than one target")

    existed = local_path(project_root, record["filename"]).exists()

    path, count = download_label(record["label"], project_root=project_root, state=state, filename=record["filename"])

    register_source(project_root, record["source_row"])

    record.update(status="complete", page_count=count)

    state["errors"].pop(key, None)

    save_state(project_root, state)

    return path, count, "reused" if existed else "saved"





def audit_collection(targets: list[str], project_root: Path, state: dict) -> dict:

    """Reopen every counted PDF and cross-check source rows, receipts and identities."""

    rows = read_sources(project_root)

    result = {"verified_targets": 0, "distinct_set_ids": 0, "readable_pdfs": 0,

              "matching_csv_rows": 0, "total_pdf_pages": 0, "unresolved": {}}

    seen_ids, seen_files = set(), set()

    for target in targets:

        key = normalize(target)

        try:

            record = state["targets"].get(key)

            if record is None:

                raise ValueError(state["errors"].get(key, "No verified label recorded"))

            check_record(target, record)

            if record["set_id"] in seen_ids or normalize(record["filename"]) in seen_files:

                raise ValueError("Duplicate Set ID or local filename across targets")

            receipt = state["downloads"].get(record["filename"])

            data = local_path(project_root, record["filename"]).read_bytes()

            if not receipt or receipt["set_id"] != record["set_id"] or hashlib.sha256(data).hexdigest() != receipt["sha256"]:

                raise ValueError("PDF lacks a matching Set ID and SHA-256 receipt")

            pages = pdf_page_count(data)

            matches = [r for r in rows if normalize(r["filename"]) == normalize(record["filename"]) or source_set_id(r) == record["set_id"]]

            if matches != [record["source_row"]]:

                raise ValueError("Missing, duplicate or conflicting source row")

            if record["status"] != "complete":

                raise ValueError(state["errors"].get(key, "Interrupted target; rerun to reconcile"))

            seen_ids.add(record["set_id"])

            seen_files.add(normalize(record["filename"]))

            result["verified_targets"] += 1

            result["readable_pdfs"] += 1

            result["matching_csv_rows"] += 1

            result["total_pdf_pages"] += pages

        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:

            result["unresolved"][target] = str(error)

    result["distinct_set_ids"] = len(seen_ids)

    return result





def run_batch(project_root=None, expected_count=100, limit=None, audit_only=False, max_search_pages=5) -> int:

    project_root = Path(project_root or PROJECT_ROOT)

    targets = [line.strip() for line in (project_root / "data/target_drugs.txt").read_text(encoding="utf-8").splitlines() if line.strip()]

    if len(targets) != expected_count or len({normalize(t) for t in targets}) != expected_count:

        raise ValueError(f"Expected {expected_count} distinct nonempty target names; list left unchanged")

    if limit is not None:
        if limit < 1 or limit > len(targets):
            raise ValueError(f"--limit must be between 1 and {len(targets)}")
        targets = targets[:limit]
    if max_search_pages < 1:
        raise ValueError("--search-pages must be positive")
    state = load_state(project_root)

    if audit_only:
        print("Audit-only mode: no network requests or document modifications.")
    for index, target in enumerate(targets, 1):
        if audit_only:
            continue
        try:
            _, pages, outcome = collect_target(
                target, project_root, state, max_search_pages=max_search_pages
            )

            print(f"[{index}/{len(targets)}] {target} | {outcome} | {pages} pages", flush=True)

        except (OSError, ValueError, RuntimeError, KeyError, TypeError, ET.ParseError, IncompleteRead) as error:

            reason = f"{type(error).__name__}: {error}"

            state["errors"][normalize(target)] = reason

            save_state(project_root, state)

            print(f"[{index}/{len(targets)}] {target} | failed | {reason}", flush=True)

    audit = audit_collection(targets, project_root, state)

    if not audit_only:
        state["audit"] = audit
        save_state(project_root, state)

    print(f"Audit: verified targets {audit['verified_targets']}/{len(targets)}; "

          f"distinct Set IDs {audit['distinct_set_ids']}; readable PDFs {audit['readable_pdfs']}; "

          f"matching CSV rows {audit['matching_csv_rows']}; total PDF pages {audit['total_pdf_pages']}")

    if audit["unresolved"]:

        print("Unresolved targets (rerun the same command to retry):")

        for target, reason in audit["unresolved"].items():

            print(f"  {target}: {reason}")

        return 1

    print("Collection audit passed: every checked target has a distinct verified label, readable PDF and source row.")

    return 0





def main() -> int:

    parser = argparse.ArgumentParser(description=__doc__)

    modes = parser.add_mutually_exclusive_group()

    modes.add_argument("--drug", help="Look up one drug without collecting or registering it")

    modes.add_argument("--download-ibuprofen", action="store_true", help="Download or reuse the reviewed ibuprofen PDF")
    parser.add_argument("--limit", type=int, metavar="N", help="Collect/audit only the first N targets (safe smoke test)")
    parser.add_argument("--audit-only", action="store_true", help="Check local collection without network or downloads")
    parser.add_argument("--search-pages", type=int, default=5, metavar="N", help="Maximum DailyMed API search pages per target (default: 5)")
    args = parser.parse_args()
    if (args.drug is not None or args.download_ibuprofen) and (args.limit is not None or args.audit_only):
        parser.error("--limit and --audit-only are batch-only options")

    try:

        if args.download_ibuprofen:

            download_ibuprofen()

        elif args.drug is not None:

            label = select_label(required(args.drug, "drug name"), max_pages=args.search_pages)

            for field in ("Ingredient", "Exact product title (DailyMed API)", "Strength", "Packager", "Set ID", "Source page URL", "Label updated on", "Verification"):

                print(f"{field}: {label[field]}")

        else:

            return run_batch(limit=args.limit, audit_only=args.audit_only, max_search_pages=args.search_pages)

    except (OSError, ValueError, RuntimeError, KeyError, ET.ParseError, IncompleteRead) as error:

        print(f"Collection failed: {type(error).__name__}: {error}", file=sys.stderr)

        return 1

    return 0





if __name__ == "__main__":

    raise SystemExit(main())


