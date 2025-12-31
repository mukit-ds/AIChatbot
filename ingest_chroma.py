import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

load_dotenv()

RAW_JSONL = Path("kb_out/raw_docs.jsonl")

CHROMA_DIR = os.getenv("CHROMA_DIR", "kb_chroma")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment/.env")


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Simple char-based chunker with overlap.
    Good enough for production v1; you can switch to token-based later.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap, 0)

    return chunks


def chunk_params(kb_type: str) -> Tuple[int, int]:
    # Tune per your 3 knowledge types
    if kb_type == "company":
        return 550, 120

    if kb_type == "policies":
        return 1400, 180
    # blogs
    return 1600, 220


def stable_chunk_id(base_id: str, chunk_index: int, chunk_text: str) -> str:
    h = hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()[:12]
    return f"{base_id}::chunk{chunk_index:03d}::{h}"


def main():
    if not RAW_JSONL.exists():
        raise FileNotFoundError(f"Not found: {RAW_JSONL.resolve()}")

    # Persistent local chroma
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # OpenAI embeddings
    embed_fn = OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBED_MODEL,
    )

    # 3 collections (recommended)
    col_map = {
        "company": client.get_or_create_collection("company_kb", embedding_function=embed_fn),
        "policies": client.get_or_create_collection("policy_kb", embedding_function=embed_fn),
        "blogs": client.get_or_create_collection("blogs_kb", embedding_function=embed_fn),
    }

    added = {"company": 0, "policies": 0, "blogs": 0}

    # Read raw docs and chunk + add
    with RAW_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            kb_type = doc["metadata"]["kb_type"]
            if kb_type not in col_map:
                continue

            text = doc["text"]
            base_id = doc["id"]

            csize, ov = chunk_params(kb_type)
            chunks = chunk_text(text, chunk_size=csize, overlap=ov)

            ids, docs_texts, metas = [], [], []
            for i, ch in enumerate(chunks, start=1):
                cid = stable_chunk_id(base_id, i, ch)
                ids.append(cid)
                docs_texts.append(ch)

                md = dict(doc["metadata"])
                md.update({
                    "raw_id": base_id,
                    "chunk_index": i,
                    "chunk_size": csize,
                    "overlap": ov,
                })
                metas.append(md)

            if ids:
                col_map[kb_type].add(ids=ids, documents=docs_texts, metadatas=metas)
                added[kb_type] += len(ids)

    print("✅ Chroma ingestion complete")
    print(f"- CHROMA_DIR: {Path(CHROMA_DIR).resolve()}")
    print("- Added chunks:", added)

    # Quick stats
    for t, col in col_map.items():
        print(f"- Collection {col.name}: {col.count()} items")


if __name__ == "__main__":
    main()
