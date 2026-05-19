"""
version7.py – Hybrid Lab Inventory Assistant with Live Datasheet Ingestion
==========================================================================
Key improvement over v6: add datasheets on-the-fly during the chat session.

New command while chatting:
    /add <file_path>       — ingest a datasheet (.txt, .pdf, .csv, .xlsx)
                             into both the SQLite DB and vector store without
                             restarting.

Supported file types for /add:
  .txt   — chunked and embedded as datasheet context
  .xlsx  — rows treated as inventory items → added to SQLite + vectorstore
  .csv   — same as .xlsx
  .json  — same format as inventory_full.json entries

Pipeline (unchanged from v6):
  1. Extract search keywords + synonyms from query  (1 fast LLM call)
  2. DB keyword search with those terms
  3. Vector search in Chroma
  4. Merge + deduplicate results
  5. Stream answer from gpt-4o-mini

Run:
    python src/version7.py
    python src/version7.py --rebuild
    python src/version7.py --question "I need a USB DAQ"
    python src/version7.py --add path/to/datasheet.xlsx

Requires: OPENAI_API_KEY env var
          python scripts/build_json.py   (generates json/inventory_full.json)
          pip install openai chromadb openpyxl
"""

import base64
import csv
import json
import os
import sqlite3
import sys
from pathlib import Path

import chromadb
from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
FULL_JSON_PATH   = ROOT / "json" / "inventory_full.json"
PROCESSED_DIR    = ROOT / "processed"
VECTORSTORE_DIR  = ROOT / "vectorstore" / "v7_openai"
COLLECTION_NAME  = "lab_inventory_v7"

TOP_K_DB      = 8
TOP_K_VECTOR  = 6
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 60
MIN_TEXT_CHARS = 50

# ---------------------------------------------------------------------------
# Name enrichment — maps substrings found in item names to additional
# search terms injected into search_text and page_content for embedding.
# Two layers:
#   ABBREV_EXPAND : abbreviations that appear IN the name (exact word match)
#   NAME_PATTERNS : substring patterns → extra context (handles shorthand names)
# ---------------------------------------------------------------------------
ABBREV_EXPAND = {
    "DMM":   "digital multimeter voltmeter ammeter",
    "MSO":   "mixed signal oscilloscope scope",
    "DSO":   "digital storage oscilloscope scope",
    "VNA":   "vector network analyzer S-parameter",
    "PNA":   "network analyzer S-parameter",
    "EXA":   "signal analyzer spectrum",
    "PXA":   "signal analyzer spectrum",
    "SDR":   "software defined radio",
    "GNSS":  "GPS global positioning navigation satellite",
    "FPGA":  "programmable logic field-programmable",
    "LCR":   "inductance capacitance resistance meter",
    "BPF":   "band pass filter bandpass",
    "LPF":   "low pass filter",
    "HPF":   "high pass filter",
    "SMA":   "connector coaxial adapter RF",
    "BNC":   "connector coaxial",
    "SP16T": "RF switch 16-port single-pole 16-throw",
    "ZSWA":  "RF switch coaxial",
    "NBFZ":  "band pass filter bandpass Mini-Circuits",
    "TC200": "temperature controller thermocontroller Thorlabs",
    "PSU":   "power supply bench supply",
    "UAV":   "unmanned aerial vehicle drone",
    "FC":    "flight controller autopilot",
    "ESC":   "electronic speed controller motor",
    "IMU":   "inertial measurement unit accelerometer gyroscope",
    "GPS":   "global positioning navigation",
    "PCB":   "printed circuit board",
}

# Substring → extra terms (case-insensitive match against item name)
NAME_PATTERNS: list[tuple[str, str]] = [
    # NI / National Instruments DAQ
    ("Nat In",          "National Instruments NI"),
    ("USB-62",          "DAQ data acquisition multifunction NI National Instruments"),
    ("USB-6212",        "DAQ data acquisition multifunction NI National Instruments"),
    ("USB-6229",        "DAQ data acquisition multifunction NI National Instruments"),
    ("Multifunction",   "DAQ data acquisition IO"),
    # Oscilloscopes
    ("InfiniVision",    "oscilloscope scope Agilent"),
    ("MSO-X",          "oscilloscope mixed signal scope"),
    # Network / signal analyzers
    ("N5230",          "network analyzer VNA S-parameter Agilent"),
    ("N5320",          "network analyzer Agilent"),
    ("N9010",          "signal analyzer spectrum EXA Agilent"),
    # Power supplies
    ("U8001",          "DC power supply bench PSU Agilent"),
    ("U8002",          "DC power supply bench PSU Agilent"),
    ("E3631",          "DC power supply bench triple output PSU Agilent"),
    # RF
    ("NBFZ-780",       "bandpass filter 780MHz RF"),
    ("ZSWA-4",         "RF switch coaxial GHz"),
    # Temperature
    ("TC200",          "temperature controller thermocontroller Thorlabs"),
    # Scales / balances
    ("Balance",        "scale weighing precision gram"),
    ("Counting Scale", "scale weighing balance"),
    # Soldering
    ("Soldering",      "solder iron station"),
    # Drone / UAV components
    ("Pixhawk",        "flight controller autopilot UAV drone"),
    ("Holybro",        "UAV drone flight controller"),
    ("DJI",            "drone UAV quadcopter"),
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a lab inventory assistant. The context you receive contains two types of entries:

  INVENTORY RECORDS — have name, location, and count fields.
  DATASHEET CHUNKS  — labelled [datasheet: <file> chunk <n>], contain raw technical text.

Rules:
- Use inventory records to answer location and availability questions.
- Use datasheet chunks to answer "what is", "what does", and specification questions.
- If datasheet chunks about an item are present, use them to answer — do NOT say the \
item is unknown or not in the inventory just because there is no inventory record.
- Only say an item is unknown if NEITHER inventory records NOR datasheet chunks \
mention it in the context.
- Use exact location names from inventory records.
- No filler phrases.

Answer format:
- Location/availability: "[Name] is located at [location], with [N] unit(s) available. \
[One sentence on what it does.]"
- Multi-device setup: list each item with name, location, count; end with one sentence \
on how they connect.
- What-is / spec question: summarise directly from the datasheet chunks provided.
"""


# ---------------------------------------------------------------------------
# In-memory SQLite with enriched search_text
# ---------------------------------------------------------------------------

def _enrich_name(name: str) -> str:
    """
    Build an enriched search string from an item name by:
    1. Expanding abbreviations that appear as words in the name (ABBREV_EXPAND)
    2. Adding context terms for known product substrings (NAME_PATTERNS)
    """
    extra: list[str] = []
    name_upper = name.upper()
    name_lower = name.lower()

    for abbrev, expansion in ABBREV_EXPAND.items():
        if abbrev.upper() in name_upper:
            extra.append(expansion)

    for pattern, expansion in NAME_PATTERNS:
        if pattern.lower() in name_lower:
            extra.append(expansion)

    return (name + " " + " ".join(extra)).strip() if extra else name


class InventoryDB:
    """In-memory SQLite built from inventory_full.json at startup.
    Adds a search_text column that expands abbreviations for better matching.

    v7: supports live insertion of new items via add_item().
    """

    def __init__(self, json_path: Path):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self._build(json_path)

    def _build(self, json_path: Path):
        self.conn.execute("""
            CREATE TABLE inventory (
                id               INTEGER PRIMARY KEY,
                inventory_number TEXT,
                name             TEXT,
                serial_number    TEXT,
                location         TEXT,
                count            INTEGER DEFAULT 1,
                description      TEXT,
                make             TEXT,
                model            TEXT,
                search_text      TEXT
            )
        """)
        with open(json_path, encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            m = item.get("metadata", {})
            name = m.get("name", "")
            self.conn.execute("""
                INSERT INTO inventory
                (inventory_number, name, serial_number, location,
                 count, description, make, model, search_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                m.get("inventory_number", ""),
                name,
                m.get("serial_number", ""),
                m.get("location", ""),
                m.get("count", 1),
                m.get("description", ""),
                m.get("make", ""),
                m.get("model", ""),
                _enrich_name(name),
            ))
        self.conn.commit()

    # --- v7: live insertion ---------------------------------------------------
    def add_item(self, meta: dict) -> None:
        """Insert a single inventory item from a metadata dict."""
        name = meta.get("name", "")
        self.conn.execute("""
            INSERT INTO inventory
            (inventory_number, name, serial_number, location,
             count, description, make, model, search_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            meta.get("inventory_number", ""),
            name,
            meta.get("serial_number", ""),
            meta.get("location", ""),
            meta.get("count", 1),
            meta.get("description", ""),
            meta.get("make", ""),
            meta.get("model", ""),
            _enrich_name(name),
        ))
        self.conn.commit()

    def add_items(self, items: list[dict]) -> int:
        """Insert multiple inventory items. Returns count added."""
        for meta in items:
            self.add_item(meta)
        return len(items)
    # --------------------------------------------------------------------------

    def search(self, keywords: list[str], limit: int = TOP_K_DB) -> list[dict]:
        """OR-based LIKE search: any keyword matching any field returns the row.
        Multi-word keyword phrases are also split into individual words so
        e.g. 'RF switch matrix' still matches an item named 'SP16T RF Switch'.
        """
        if not keywords:
            return []

        # Build the full term set: original phrases + individual words (≥3 chars)
        seen_terms: set[str] = set()
        all_terms: list[str] = []
        for kw in keywords:
            if kw.lower() not in seen_terms:
                seen_terms.add(kw.lower())
                all_terms.append(kw)
            for word in kw.split():
                if len(word) >= 3 and word.lower() not in seen_terms:
                    seen_terms.add(word.lower())
                    all_terms.append(word)

        conditions, params = [], []
        for term in all_terms:
            conditions.append(
                "(name LIKE ? OR make LIKE ? OR model LIKE ? "
                "OR location LIKE ? OR description LIKE ? OR search_text LIKE ?)"
            )
            w = f"%{term}%"
            params.extend([w] * 6)
        where = " OR ".join(conditions)
        rows = self.conn.execute(
            f"SELECT * FROM inventory WHERE {where} ORDER BY name LIMIT ?",
            params + [limit],
        ).fetchall()
        return [dict(r) for r in rows]

    def format_item(self, r: dict) -> str:
        loc  = r["location"] or "location unspecified"
        line = f"• {r['name']}\n  Location: {loc} | Count: {r['count']}"
        if r["inventory_number"]:
            line += f" | Inventory #: {r['inventory_number']}"
        if r["serial_number"]:
            line += f" | S/N: {r['serial_number']}"
        if r["description"]:
            line += f"\n  Note: {r['description']}"
        return line

    def format_results(self, results: list[dict]) -> str:
        if not results:
            return ""
        return "\n\n".join(self.format_item(r) for r in results)


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

class OpenAIBackend:
    def __init__(self):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("Error: OPENAI_API_KEY environment variable not set.")
            sys.exit(1)
        self.client = OpenAI(api_key=key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in resp.data]

    def extract_keywords(self, query: str) -> list[str]:
        """Return 3-6 search terms including abbreviations and synonyms."""
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=80,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract search keywords from a lab inventory query. "
                        "Return JSON only: {\"keywords\": [\"kw1\", \"kw2\", ...]}\n\n"
                        "Rules:\n"
                        "- Include 3-6 short terms\n"
                        "- Always include both full names AND abbreviations:\n"
                        "  multimeter → [\"multimeter\", \"DMM\"]\n"
                        "  oscilloscope → [\"oscilloscope\", \"MSO\", \"scope\"]\n"
                        "  data acquisition / DAQ → [\"DAQ\", \"data acquisition\", \"NI\"]\n"
                        "  network analyzer → [\"network analyzer\", \"VNA\"]\n"
                        "  power supply → [\"power supply\", \"PSU\"]\n"
                        "  RF filter → [\"filter\", \"BPF\"]\n"
                        "  temperature controller → [\"temperature\", \"TC200\"]\n"
                        "- Include model numbers and manufacturer names if mentioned\n"
                        "- Omit filler: where, how, many, need, find, is, the, a, an"
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        try:
            result  = json.loads(resp.choices[0].message.content)
            keywords = result.get("keywords", [])
            return [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
        except (json.JSONDecodeError, KeyError):
            return []

    def stream_chat(self, user_prompt: str):
        """Yields text chunks, then a final usage dict."""
        stream = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            stream=True,
            stream_options={"include_usage": True},
        )
        usage = {}
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
            if chunk.usage:
                usage = {
                    "input_tokens":  chunk.usage.prompt_tokens,
                    "output_tokens": chunk.usage.completion_tokens,
                }
        yield usage


# ---------------------------------------------------------------------------
# Vectorstore (Chroma) — enriched page_content
# ---------------------------------------------------------------------------

def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + CHUNK_SIZE]))
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_all_docs() -> tuple[list[str], list[dict]]:
    docs, metas = [], []

    with open(FULL_JSON_PATH, encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        content = item.get("page_content", "")
        meta    = item.get("metadata", {})
        name    = meta.get("name", "")
        # Enrich the embedded text with abbreviation expansions
        enriched = _enrich_name(name)
        doc_text = f"{content}\n{enriched}"
        extra    = "  ".join(f"{k}: {v}" for k, v in meta.items() if v)
        docs.append(f"{doc_text}\n{extra}")
        metas.append({
            "source":   "inventory",
            "name":     name,
            "model":    meta.get("model", ""),
            "location": meta.get("location", ""),
            "make":     meta.get("make", ""),
            "inv_num":  meta.get("inventory_number", ""),
            "count":    str(meta.get("count", 1)),  # stored as string; Chroma metadata must be str/int/float
        })

    if PROCESSED_DIR.exists():
        for txt_file in sorted(PROCESSED_DIR.glob("*.txt")):
            text = txt_file.read_text(encoding="utf-8")
            for i, chunk in enumerate(chunk_text(text)):
                docs.append(chunk)
                metas.append({
                    "source": "datasheet",
                    "file":   txt_file.stem,
                    "chunk":  i,
                    "name": "", "model": "", "location": "",
                    "make": "", "inv_num": "", "count": "1",
                })

    return docs, metas


def get_collection(backend: OpenAIBackend, rebuild: bool = False) -> chromadb.Collection:
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Existing vectorstore deleted.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        docs, metas = load_all_docs()
        if not docs:
            print("No documents found. Run:  python scripts/build_json.py")
            sys.exit(1)

        inv_count = sum(1 for m in metas if m["source"] == "inventory")
        ds_count  = len(docs) - inv_count
        print(f"Embedding {len(docs)} chunks "
              f"({inv_count} inventory items, {ds_count} datasheet chunks)...")

        all_embeddings = []
        batch_size = 64
        for i in range(0, len(docs), batch_size):
            batch = docs[i: i + batch_size]
            all_embeddings.extend(backend.embed(batch))
            print(f"  {min(i + batch_size, len(docs))}/{len(docs)}", end="\r", flush=True)
        print()

        collection.add(
            documents=docs,
            embeddings=all_embeddings,
            metadatas=metas,
            ids=[str(i) for i in range(len(docs))],
        )
        print(f"Vectorstore saved → {VECTORSTORE_DIR}\n")
    else:
        print(f"Loaded vectorstore ({collection.count()} chunks).\n")

    return collection


def retrieve_vectors(
    collection: chromadb.Collection,
    backend: OpenAIBackend,
    query: str,
) -> list[dict]:
    """Returns a list of metadata dicts for the top semantic matches."""
    q_vec = backend.embed([query])[0]
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=TOP_K_VECTOR,
        include=["documents", "metadatas"],
    )
    hits = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        hits.append({**meta, "_doc": doc})
    return hits


# ---------------------------------------------------------------------------
# v7: Live datasheet / inventory ingestion
# ---------------------------------------------------------------------------

def _next_chroma_id(collection: chromadb.Collection) -> int:
    """Return the next available integer ID for the Chroma collection."""
    return collection.count()


def _ingest_txt(
    filepath: Path,
    backend: OpenAIBackend,
    collection: chromadb.Collection,
) -> int:
    """Ingest a .txt datasheet: chunk → embed → add to vectorstore.
    Returns the number of chunks added."""
    text = filepath.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    if not chunks:
        return 0

    start_id = _next_chroma_id(collection)
    embeddings = []
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        embeddings.extend(backend.embed(chunks[i: i + batch_size]))

    metas = [
        {
            "source": "datasheet",
            "file":   filepath.stem,
            "chunk":  i,
            "name": "", "model": "", "location": "",
            "make": "", "inv_num": "", "count": "1",
        }
        for i in range(len(chunks))
    ]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metas,
        ids=[str(start_id + i) for i in range(len(chunks))],
    )
    return len(chunks)


def _parse_xlsx_rows(filepath: Path) -> list[dict]:
    """Read an .xlsx file and return a list of inventory metadata dicts.
    Expects a header row with columns matching (case-insensitive):
        name, inventory_number, serial_number, location, count,
        description, make, model
    Missing columns are filled with defaults."""
    try:
        import openpyxl
    except ImportError:
        print("  [error] openpyxl not installed. Run: pip install openpyxl")
        return []

    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if len(rows) < 2:
        return []

    # Normalise header names
    raw_headers = [str(h).strip().lower().replace(" ", "_") if h else "" for h in rows[0]]

    # Map common header variations
    HEADER_ALIASES = {
        "item_name": "name", "item": "name", "equipment": "name", "equipment_name": "name",
        "inv_num": "inventory_number", "inv_#": "inventory_number", "inventory_#": "inventory_number",
        "inv": "inventory_number", "asset_number": "inventory_number", "asset_#": "inventory_number",
        "serial": "serial_number", "s/n": "serial_number", "sn": "serial_number",
        "loc": "location", "room": "location", "building": "location",
        "qty": "count", "quantity": "count", "amount": "count",
        "desc": "description", "notes": "description", "note": "description",
        "manufacturer": "make", "brand": "make", "vendor": "make",
        "model_number": "model", "model_#": "model", "part_number": "model",
    }
    headers = [HEADER_ALIASES.get(h, h) for h in raw_headers]

    VALID_FIELDS = {"name", "inventory_number", "serial_number", "location",
                    "count", "description", "make", "model"}

    items: list[dict] = []
    for row in rows[1:]:
        meta: dict = {}
        for col_idx, val in enumerate(row):
            if col_idx < len(headers) and headers[col_idx] in VALID_FIELDS:
                meta[headers[col_idx]] = val if val is not None else ""
        # Must have at least a name
        if not meta.get("name"):
            continue
        # Ensure count is int
        try:
            meta["count"] = int(meta.get("count", 1))
        except (ValueError, TypeError):
            meta["count"] = 1
        # Fill defaults
        for field in VALID_FIELDS:
            meta.setdefault(field, "")
        items.append(meta)
    return items


def _parse_csv_rows(filepath: Path) -> list[dict]:
    """Read a .csv file and return inventory metadata dicts (same logic as xlsx)."""
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if len(rows) < 2:
        return []

    raw_headers = [h.strip().lower().replace(" ", "_") for h in rows[0]]

    HEADER_ALIASES = {
        "item_name": "name", "item": "name", "equipment": "name", "equipment_name": "name",
        "inv_num": "inventory_number", "inv_#": "inventory_number", "inventory_#": "inventory_number",
        "inv": "inventory_number", "asset_number": "inventory_number", "asset_#": "inventory_number",
        "serial": "serial_number", "s/n": "serial_number", "sn": "serial_number",
        "loc": "location", "room": "location", "building": "location",
        "qty": "count", "quantity": "count", "amount": "count",
        "desc": "description", "notes": "description", "note": "description",
        "manufacturer": "make", "brand": "make", "vendor": "make",
        "model_number": "model", "model_#": "model", "part_number": "model",
    }
    headers = [HEADER_ALIASES.get(h, h) for h in raw_headers]
    VALID_FIELDS = {"name", "inventory_number", "serial_number", "location",
                    "count", "description", "make", "model"}

    items: list[dict] = []
    for row in rows[1:]:
        meta: dict = {}
        for col_idx, val in enumerate(row):
            if col_idx < len(headers) and headers[col_idx] in VALID_FIELDS:
                meta[headers[col_idx]] = val.strip() if val else ""
        if not meta.get("name"):
            continue
        try:
            meta["count"] = int(meta.get("count", 1))
        except (ValueError, TypeError):
            meta["count"] = 1
        for field in VALID_FIELDS:
            meta.setdefault(field, "")
        items.append(meta)
    return items


def _parse_json_items(filepath: Path) -> list[dict]:
    """Read a .json file in the same format as inventory_full.json."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    items = []
    for entry in data:
        if "metadata" in entry:
            m = entry["metadata"]
        else:
            m = entry
        if m.get("name"):
            try:
                m["count"] = int(m.get("count", 1))
            except (ValueError, TypeError):
                m["count"] = 1
            items.append(m)
    return items


def _embed_inventory_items(
    items: list[dict],
    backend: OpenAIBackend,
    collection: chromadb.Collection,
) -> int:
    """Embed inventory items and add them to Chroma. Returns count added."""
    if not items:
        return 0

    docs = []
    metas = []
    for meta in items:
        name = meta.get("name", "")
        enriched = _enrich_name(name)
        extra = "  ".join(f"{k}: {v}" for k, v in meta.items() if v)
        docs.append(f"{enriched}\n{extra}")
        metas.append({
            "source":   "inventory",
            "name":     name,
            "model":    meta.get("model", ""),
            "location": meta.get("location", ""),
            "make":     meta.get("make", ""),
            "inv_num":  meta.get("inventory_number", ""),
            "count":    str(meta.get("count", 1)),
        })

    start_id = _next_chroma_id(collection)
    embeddings = []
    batch_size = 64
    for i in range(0, len(docs), batch_size):
        embeddings.extend(backend.embed(docs[i: i + batch_size]))

    collection.add(
        documents=docs,
        embeddings=embeddings,
        metadatas=metas,
        ids=[str(start_id + i) for i in range(len(docs))],
    )
    return len(docs)


def _append_to_inventory_json(items: list[dict]) -> None:
    """Append new inventory items to FULL_JSON_PATH so they survive restarts."""
    with open(FULL_JSON_PATH, encoding="utf-8") as f:
        existing = json.load(f)
    for meta in items:
        name     = meta.get("name", "")
        location = meta.get("location", "")
        count    = meta.get("count", 1)
        inv_num  = meta.get("inventory_number", "")
        serial   = meta.get("serial_number", "")
        desc     = meta.get("description", "")
        loc_str  = f"located at {location}" if location else "location unspecified"
        parts    = [f"The {name} is {loc_str}.", f"Count: {count}."]
        if inv_num:
            parts.append(f"Inventory #: {inv_num}.")
        if serial:
            parts.append(f"S/N: {serial}.")
        if desc:
            parts.append(desc.rstrip(".") + ".")
        existing.append({
            "page_content": " ".join(parts),
            "metadata": {
                "inventory_number": inv_num,
                "name":             name,
                "make":             meta.get("make", ""),
                "model":            meta.get("model", ""),
                "serial_number":    serial,
                "location":         location,
                "count":            count,
                "description":      desc,
                "source":           "Lab Inventory",
            },
        })
    with open(FULL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def _ocr_pdf(filepath: Path, backend: OpenAIBackend) -> str:
    """Extract text from a PDF: fitz direct extraction with gpt-4o-mini vision fallback.
    Returns the full extracted text as a single string."""
    try:
        import fitz
    except ImportError:
        raise ImportError("pymupdf not installed. Run: pip install pymupdf")

    doc = fitz.open(filepath)
    pages_text = []
    for i, page in enumerate(doc):
        direct = page.get_text().strip()
        if len(direct) >= MIN_TEXT_CHARS:
            pages_text.append(f"=== Page {i + 1} ===\n{direct}")
        else:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            b64 = base64.b64encode(pix.tobytes("png")).decode()
            try:
                resp = backend.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": (
                                "Extract all text from this datasheet page. "
                                "Preserve tables, spec lists, and section headers. "
                                "Output extracted text only — no commentary."
                            )},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": "high",
                            }},
                        ],
                    }],
                    max_tokens=4096,
                )
                pages_text.append(f"=== Page {i + 1} ===\n{resp.choices[0].message.content}")
            except Exception as e:
                pages_text.append(f"=== Page {i + 1} ===\n[OCR failed: {e}]")
        print(f"  page {i + 1}/{len(doc)}", end="\r", flush=True)
    print()
    return "\n\n".join(pages_text)


def ingest_file(
    filepath: Path,
    backend: OpenAIBackend,
    db: InventoryDB,
    collection: chromadb.Collection,
) -> str:
    """
    Ingest a file into both SQLite and the vectorstore at runtime.
    Returns a human-readable status message.

    Supported formats:
      .txt   → datasheet chunks (vector only, no DB rows)
      .xlsx  → inventory rows   (DB + vector)
      .csv   → inventory rows   (DB + vector)
      .json  → inventory items  (DB + vector)
    """
    if not filepath.exists():
        return f"File not found: {filepath}"

    suffix = filepath.suffix.lower()

    if suffix == ".txt":
        n = _ingest_txt(filepath, backend, collection)
        return f"Added {n} datasheet chunks from '{filepath.name}' to vectorstore."

    elif suffix == ".xlsx":
        items = _parse_xlsx_rows(filepath)
        if not items:
            return f"No valid inventory rows found in '{filepath.name}'. Check that it has a header row with at least a 'name' column."
        db_count = db.add_items(items)
        vec_count = _embed_inventory_items(items, backend, collection)
        _append_to_inventory_json(items)
        return (f"Added {db_count} item(s) from '{filepath.name}' "
                f"→ SQLite DB + {vec_count} vector embeddings (persisted to JSON).")

    elif suffix == ".csv":
        items = _parse_csv_rows(filepath)
        if not items:
            return f"No valid inventory rows found in '{filepath.name}'. Check that it has a header row with at least a 'name' column."
        db_count = db.add_items(items)
        vec_count = _embed_inventory_items(items, backend, collection)
        _append_to_inventory_json(items)
        return (f"Added {db_count} item(s) from '{filepath.name}' "
                f"→ SQLite DB + {vec_count} vector embeddings (persisted to JSON).")

    elif suffix == ".json":
        items = _parse_json_items(filepath)
        if not items:
            return f"No valid inventory items found in '{filepath.name}'."
        db_count = db.add_items(items)
        vec_count = _embed_inventory_items(items, backend, collection)
        _append_to_inventory_json(items)
        return (f"Added {db_count} item(s) from '{filepath.name}' "
                f"→ SQLite DB + {vec_count} vector embeddings (persisted to JSON).")

    elif suffix == ".pdf":
        try:
            print(f"  Running OCR on {filepath.name} ...")
            text = _ocr_pdf(filepath, backend)
        except ImportError as e:
            return str(e)
        PROCESSED_DIR.mkdir(exist_ok=True)
        txt_path = PROCESSED_DIR / (filepath.stem + ".txt")
        txt_path.write_text(text, encoding="utf-8")
        print(f"  Saved → {txt_path.name}")
        n = _ingest_txt(txt_path, backend, collection)
        meta = {
            "name":             filepath.stem,
            "location":         "new arrival",
            "count":            1,
            "inventory_number": "",
            "serial_number":    "",
            "description":      "",
            "make":             "",
            "model":            "",
        }
        db.add_item(meta)
        _embed_inventory_items([meta], backend, collection)
        _append_to_inventory_json([meta])
        return (f"OCR'd '{filepath.name}' → '{txt_path.name}' → {n} chunk(s) embedded. "
                f"Added '{meta['name']}' to inventory (location: new arrival, count: 1, persisted to JSON).")

    else:
        return (f"Unsupported file type '{suffix}'. "
                f"Supported: .txt, .pdf, .xlsx, .csv, .json")


# ---------------------------------------------------------------------------
# Hybrid retrieval: merge DB + vector results
# ---------------------------------------------------------------------------

def hybrid_retrieve(
    db: InventoryDB,
    collection: chromadb.Collection,
    backend: OpenAIBackend,
    keywords: list[str],
    query: str,
) -> str:
    """Always run both searches. Merge by inventory_number, DB results first."""
    db_rows    = db.search(keywords) if keywords else []
    vec_hits   = retrieve_vectors(collection, backend, query)

    # Deduplicate: key on inventory_number if available, else name
    seen: set[str] = set()
    merged_parts: list[str] = []

    def dedup_key(name: str, inv: str) -> str:
        return inv if inv else name.lower().strip()

    # DB results (precise matches) come first
    for r in db_rows:
        k = dedup_key(r["name"], r["inventory_number"])
        if k not in seen:
            seen.add(k)
            merged_parts.append(db.format_item(r))

    # Vector results fill in anything DB missed
    for hit in vec_hits:
        if hit.get("source") == "inventory":
            k = dedup_key(hit.get("name", ""), hit.get("inv_num", ""))
            if k not in seen:
                seen.add(k)
                # Reconstruct a simple formatted line from vector metadata
                loc   = hit.get("location", "") or "location unspecified"
                name  = hit.get("name", "")
                model = hit.get("model", "")
                label = name if name else model
                count = hit.get("count", "1")
                inv   = hit.get("inv_num", "")
                line  = f"• {label}\n  Location: {loc} | Count: {count}"
                if inv:
                    line += f" | Inventory #: {inv}"
                merged_parts.append(line)
        else:
            # Datasheet chunk — include raw text for specs context
            merged_parts.append(
                f"[datasheet: {hit.get('file', '')} chunk {hit.get('chunk', '')}]\n"
                + hit.get("_doc", "")
            )

    if not merged_parts:
        return "No matching items found in inventory."

    return "\n\n".join(merged_parts)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def answer_question(
    backend: OpenAIBackend,
    db: InventoryDB,
    collection: chromadb.Collection,
    query: str,
) -> str:
    # Step 1: extract keywords, with fallback for unrecognised terms
    keywords = backend.extract_keywords(query)
    if not keywords:
        _stop = {"what", "is", "where", "how", "many", "need", "find", "the",
                 "a", "an", "are", "do", "we", "have", "i", "can", "tell",
                 "me", "about", "which", "does", "it", "in", "of", "for"}
        keywords = [w.strip("?.,!") for w in query.split()
                    if w.lower().strip("?.,!") not in _stop
                    and len(w.strip("?.,!")) >= 3]
    kw_str = ", ".join(keywords) if keywords else "—"
    print(f"  [keywords: {kw_str}]")

    # Step 2: hybrid retrieval (always both)
    context = hybrid_retrieve(db, collection, backend, keywords, query)

    # Step 3: stream answer
    prompt = f"Inventory context:\n{context}\n\nQuestion: {query}"

    print("\nAssistant: ", end="", flush=True)
    response_parts: list[str] = []
    usage: dict = {}

    for item in backend.stream_chat(prompt):
        if isinstance(item, dict):
            usage = item
        else:
            print(item, end="", flush=True)
            response_parts.append(item)

    if usage:
        print(
            f"\n\n[tokens — in: {usage.get('input_tokens', '?')} "
            f" out: {usage.get('output_tokens', '?')}]"
        )
    print()

    return "".join(response_parts)


def _drain_stdin() -> None:
    """Flush any buffered stdin bytes so a leftover newline from the previous
    response doesn't trigger a phantom empty query."""
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


def chat(backend: OpenAIBackend, db: InventoryDB, collection: chromadb.Collection):
    print("Lab Inventory Assistant [v7 · hybrid search + live ingestion]")
    print("  Commands:  /add <filepath>   — ingest a datasheet or inventory file")
    print("             /status           — show DB & vectorstore stats")
    print("             quit              — exit\n")

    while True:
        _drain_stdin()
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        # --- v7: /add command -------------------------------------------------
        if query.lower().startswith("/add "):
            raw_path = query[5:].strip().strip("'\"")
            if not raw_path:
                print("  Usage: /add <file_path>\n")
                continue
            filepath = Path(raw_path).expanduser().resolve()
            print(f"  Ingesting: {filepath} ...")
            result = ingest_file(filepath, backend, db, collection)
            print(f"  {result}\n")
            continue

        # --- v7: /status command ----------------------------------------------
        if query.lower() == "/status":
            row_count = db.conn.execute("SELECT COUNT(*) FROM inventory").fetchone()[0]
            vec_count = collection.count()
            print(f"  SQLite rows:       {row_count}")
            print(f"  Vectorstore chunks: {vec_count}\n")
            continue

        answer_question(backend, db, collection, query)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    rebuild  = "--rebuild" in sys.argv
    one_shot = None
    add_file = None

    if "--question" in sys.argv:
        idx = sys.argv.index("--question")
        if idx + 1 < len(sys.argv):
            one_shot = sys.argv[idx + 1]

    if "--add" in sys.argv:
        idx = sys.argv.index("--add")
        if idx + 1 < len(sys.argv):
            add_file = sys.argv[idx + 1]

    return rebuild, one_shot, add_file


def main():
    rebuild, one_shot, add_file = parse_args()

    if not FULL_JSON_PATH.exists():
        print(f"Inventory JSON not found: {FULL_JSON_PATH}")
        print("Generate it first:  python scripts/build_json.py")
        sys.exit(1)

    backend    = OpenAIBackend()
    db         = InventoryDB(FULL_JSON_PATH)
    collection = get_collection(backend, rebuild=rebuild)

    # Handle --add flag (ingest before entering chat or one-shot)
    if add_file:
        filepath = Path(add_file).expanduser().resolve()
        print(f"Ingesting: {filepath} ...")
        result = ingest_file(filepath, backend, db, collection)
        print(f"  {result}\n")

    if one_shot:
        answer_question(backend, db, collection, one_shot)
    else:
        chat(backend, db, collection)


if __name__ == "__main__":
    main()