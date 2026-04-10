"""
    python src/version4.py                        # local
    python src/version4.py --backend openai       
    python src/version4.py --backend openai --rebuild       #redo the ocr script
"""

import json
import os
import sys
import time
from abc import ABC, abstractmethod # the langchain RAG tutorial uses this to speed up RAG developmenthttps://reference.langchain.com/python/langchain-classic/chains/api/base
from datetime import datetime
from pathlib import Path

import chromadb


ROOT = Path(__file__).parent.parent
JSON_PATH = ROOT / "json" / "sample.json" # need tof ix absolute paths
PROCESSED_DIR = ROOT / "processed"
VECTORSTORE_ROOT = ROOT / "vectorstore"
TOP_K = 6 # 
CHUNK_SIZE = 500
CHUNK_OVERLAP = 60

LOG_PATH = ROOT / "chat_log.txt"


def log_interaction(backend_name: str, user_input: str, ai_output: str, usage: dict, response_time: float):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] backend={backend_name}\n")
        f.write(f"USER: {user_input}\n")
        f.write(f"AI: {ai_output}\n")
        f.write(f"tokens_in={usage.get('input_tokens', '?')}  tokens_out={usage.get('output_tokens', '?')}  response_time={response_time:.2f}s\n")
        f.write("-" * 80 + "\n")


SYSTEM_PROMPT = """\
You are a lab inventory assistant. When a user asks what they need for a task or setup, follow this  structure:

1. List everything required for the task based on your knowledge (instruments, cables, calibration standards, adapters, etc.).
2. For every item, check the provided inventory context and state one of these things:
   - The exact model and location if found in the context.
   - "Location unknown" if the item is in the inventory but  no location.
   - "Not in inventory" if the item does not appear in the context at all.

Rules:
- Do not ever skip step 1. Always reason about  the task requires before checking inventory.
- "Location unknown" and "Not in inventory" are different do not switch them.
- Only use locations stated verbatim in the context. Never guess a location.
- No verbose phrases like "you may want to verify" or "typically used with".
- Be concise. Use a bullet list.\
"""



class Backend(ABC):
    name: str
    vectorstore_dir: Path
    collection_name: str

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, return list of float vectors."""
        ...

    @abstractmethod
    def stream_chat(self, prompt: str):
        """Yield text chunks from the LLM. Last yielded item is a dict with usage stats."""
        ...


class OllamaBackend(Backend):
    name = "ollama"
    vectorstore_dir = VECTORSTORE_ROOT / "ollama"
    collection_name = "lab_inventory_ollama"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        print("Loading local embedding model (sentence-transformers)...")
        self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Done.\n")

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._embed_model.encode(texts, show_progress_bar=False, batch_size=32)
        return [v.tolist() for v in vecs]

    def stream_chat(self, prompt: str):
        import ollama
        try:
            stream = ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={"stop": ["\nUser:", "\nYou:"]},
            )
            usage = {}
            # DEBUG
            for part in stream:
                if part.get("done"):
                    usage = {
                        "input_tokens": part.get("prompt_eval_count", "?"),
                        "output_tokens": part.get("eval_count", "?"),
                    }
                else:
                    yield part["message"]["content"] # retrn the message and wait for more, yield functi

            yield usage
        except Exception as e:
            yield f"\n[Ollama error: {e}]\nMake sure Ollama is running:  ollama serve"
            yield {}


class OpenAIBackend(Backend):
    name = "openai"
    vectorstore_dir = VECTORSTORE_ROOT / "openai"
    collection_name = "lab_inventory_openai"

    def __init__(self):
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("Error")
            sys.exit(1)
        self._client = OpenAI(api_key=key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # right now its about 0,0002C per 1k tokens
        resp = self._client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in resp.data]

    def stream_chat(self, prompt: str):
        try:
            stream = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True}, #debug, token count
            )
            usage = {}
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
                if chunk.usage:
                    usage = {
                        "input_tokens": chunk.usage.prompt_tokens,
                        "output_tokens": chunk.usage.completion_tokens,
                    }
            yield usage
        except Exception as e:
            yield f"\n[OpenAI error: {e}]"
            yield {}


def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + CHUNK_SIZE]))
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_all_docs() -> tuple[list[str], list[dict]]:
    docs, metas = [], []

    #  JSON inventory
    with open(JSON_PATH, encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        content = item.get("page_content", "")
        meta = item.get("metadata", {})
        extra = "  ".join(f"{k}: {v}" for k, v in meta.items())
        docs.append(f"{content}\n{extra}")
        metas.append({
            "source": "inventory",
            "model": meta.get("model", ""),
            "location": meta.get("location", ""),
            "category": meta.get("category", ""),
            "make": meta.get("make", ""),
        })

    # processed datasheets
    if PROCESSED_DIR.exists():
        for txt_file in sorted(PROCESSED_DIR.glob("*.txt")):
            text = txt_file.read_text(encoding="utf-8")
            for i, chunk in enumerate(chunk_text(text)):
                docs.append(chunk)
                metas.append({"source": "datasheet", "file": txt_file.stem, "chunk": i})

    return docs, metas



def build_keyword_map() -> dict[str, list[dict]]:
    """
    mapping of lowercase search terms to lists of Chroma where conditions.
    """
    terms: dict[str, list[dict]] = {}

    with open(JSON_PATH, encoding="utf-8") as f:
        items = json.load(f)

    for item in items:
        meta = item.get("metadata", {})
        model = meta.get("model", "")
        category = meta.get("category", "")
        make = meta.get("make", "")

        if model and model.lower() != "unknown":
            # match inventory entry / any datasheet file with same name
            terms.setdefault(model.lower(), []).extend([
                {"model": model},
                {"file": model},
            ])
        if category:
            terms.setdefault(category.lower(), []).append({"category": category})
        if make:
            terms.setdefault(make.lower(), []).append({"make": make})

    # datasheet filenames as additional terms (for example things like "keithley"
    if PROCESSED_DIR.exists():
        for txt_file in PROCESSED_DIR.glob("*.txt"):
            stem = txt_file.stem
            terms.setdefault(stem.lower(), []).append({"file": stem})

    return terms


def extract_filter(query: str, keyword_map: dict[str, list[dict]]) -> dict | None:
    """
    Find keyword matches in the query and return a Chroma where filter, or None
    if no matches (falls back to unfiltered search).
    """
    q = query.lower()
    conditions = []
    for term, conds in keyword_map.items():
        if term in q:
            conditions.extend(conds)

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$or": conditions}




def build_vectorstore(collection: chromadb.Collection, backend: Backend):
    docs, metas = load_all_docs()
    if not docs:
        print("No documents found. Add items to json/sample.json or run OCR first.")
        return

    inv_count = sum(1 for m in metas if m["source"] == "inventory")
    ds_count = len(docs) - inv_count
    print(f"Embedding {len(docs)} chunks ({inv_count} inventory, {ds_count} datasheet) via {backend.name}...")

    # Embed in batches to avoid large API calls at once
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        all_embeddings.extend(backend.embed(batch))
        print(f"  {min(i + batch_size, len(docs))}/{len(docs)}", end="\r", flush=True)
    print()

    collection.add(
        documents=docs,
        embeddings=all_embeddings,
        metadatas=metas,
        ids=[str(i) for i in range(len(docs))],
    )
    print(f"Vectorstore saved to {backend.vectorstore_dir}\n")


def get_collection(backend: Backend, rebuild: bool = False) -> chromadb.Collection:
    backend.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(backend.vectorstore_dir))

    if rebuild:
        try:
            client.delete_collection(backend.collection_name)
            print("Existing vectorstore deleted.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        backend.collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        build_vectorstore(collection, backend)
    else:
        print(f"Loaded [{backend.name}] vectorstore ({collection.count()} chunks).\n")

    return collection




def retrieve(
    collection: chromadb.Collection,
    backend: Backend,
    query: str,
    keyword_map: dict,
) -> list[tuple[str, dict]]:
    q_vec = backend.embed([query])[0]
    where = extract_filter(query, keyword_map)

    try:
        results = collection.query(
            query_embeddings=[q_vec],
            n_results=TOP_K,
            where=where,
            include=["documents", "metadatas"],
        )
    except Exception:
        # Filter matched nothing — fall back to unfiltered search
        results = collection.query(
            query_embeddings=[q_vec],
            n_results=TOP_K,
            include=["documents", "metadatas"],
        )

    if where:
        label = f"filtered ({', '.join(str(w) for w in ([where] if '$or' not in where else where['$or'])[:2])}...)"
        print(f"  [retrieval: {label}]")
    else:
        print(f"  [retrieval: unfiltered]")

    return list(zip(results["documents"][0], results["metadatas"][0]))


def build_prompt(query: str, chunks: list[tuple[str, dict]]) -> str:
    ctx_parts = []
    for doc, meta in chunks:
        if meta.get("source") == "inventory":
            label = f"inventory: {meta.get('model', '')}"
        else:
            label = f"datasheet: {meta.get('file', '')} chunk {meta.get('chunk', '')}"
        ctx_parts.append(f"[{label}]\n{doc}")

    context = "\n\n---\n\n".join(ctx_parts)
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nUser: {query}\nAssistant:"




def chat(collection: chromadb.Collection, backend: Backend):
    keyword_map = build_keyword_map()
    print(f"Lab Inventory Assistant [{backend.name}] — type 'quit' to exit\n")
    while True:
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

        chunks = retrieve(collection, backend, query, keyword_map)
        prompt = build_prompt(query, chunks)

        print("\nAssistant: ", end="", flush=True)
        usage = {}
        ai_output_parts = []
        t_start = time.monotonic()
        for item in backend.stream_chat(prompt):
            if isinstance(item, dict):
                usage = item
            else:
                print(item, end="", flush=True)
                ai_output_parts.append(item)
        response_time = time.monotonic() - t_start

        if usage:
            print(f"\n\n[tokens — in: {usage.get('input_tokens', '?')}  out: {usage.get('output_tokens', '?')}  time: {response_time:.2f}s]")
        print()

        log_interaction(backend.name, query, "".join(ai_output_parts), usage, response_time)




def parse_args() -> tuple[str, bool]:
    backend_name = "ollama"
    rebuild = False
    args = sys.argv[1:]
    if "--backend" in args:
        idx = args.index("--backend")
        if idx + 1 < len(args):
            backend_name = args[idx + 1]
    if "--rebuild" in args:
        rebuild = True
    return backend_name, rebuild


def main():
    backend_name, rebuild = parse_args()

    if backend_name == "openai":
        backend = OpenAIBackend()
    elif backend_name == "ollama":
        backend = OllamaBackend()
    else:
        print(f"Unknown backend '{backend_name}'. Choose: ollama, openai")
        sys.exit(1)

    collection = get_collection(backend, rebuild=rebuild)
    chat(collection, backend)


if __name__ == "__main__":
    main()
