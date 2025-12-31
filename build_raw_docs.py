import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from docx import Document

KB_SOURCES_DIR = Path("kb_sources")
OUT_DIR = Path("kb_out")
OUT_JSONL = OUT_DIR / "raw_docs.jsonl"


def slug_to_title(filename: str) -> str:
    # "buy-commercial-property-dubai-guide.docx" -> "Buy Commercial Property Dubai Guide"
    name = Path(filename).stem
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name.title()


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")  # non-breaking space
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_docx_blocks(docx_path):
    """
    Detect headings more aggressively.
    Treat short uppercase / title-like lines as headings too.
    """
    doc = Document(str(docx_path))
    blocks = []

    for p in doc.paragraphs:
        text = (p.text or "").strip()
        if not text:
            continue

        style = (p.style.name or "").lower() if p.style else ""
        is_heading = "heading" in style

        # 🔥 EXTRA HEADING DETECTION (CRITICAL FOR TEAM LISTS)
        if not is_heading:
            if (
                len(text) <= 40
                and text.istitle()
                and not text.endswith(".")
            ):
                is_heading = True

        blocks.append({
            "type": "heading" if is_heading else "para",
            "text": text
        })

    return blocks

def blocks_to_sections(blocks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Groups paragraphs under the latest heading.
    Produces sections: {section_title, content}
    If no heading exists, section_title = "General"
    """
    sections = []
    current_title = "General"
    current_lines = []

    def flush():
        nonlocal current_lines, current_title
        if current_lines:
            sections.append({
                "section_title": current_title,
                "content": "\n".join(current_lines).strip()
            })
            current_lines = []

    for b in blocks:
        if b["type"] == "heading":
            flush()
            current_title = b["text"]
        else:
            current_lines.append(b["text"])

    flush()
    return sections


def detect_kb_type(path: Path) -> str:
    # kb_sources/company/... -> company
    # kb_sources/policies/... -> policies
    # kb_sources/blogs/... -> blogs
    parts = [p.lower() for p in path.parts]
    if "company" in parts:
        return "company"
    if "policies" in parts:
        return "policies"
    if "blogs" in parts:
        return "blogs"
    return "unknown"


def build_documents() -> List[Dict[str, Any]]:
    docs = []
    for docx_path in KB_SOURCES_DIR.rglob("*.docx"):
        kb_type = detect_kb_type(docx_path)
        title = slug_to_title(docx_path.name)

        blocks = read_docx_blocks(docx_path)
        sections = blocks_to_sections(blocks)

        # If sections are empty (rare), fallback
        if not sections:
            sections = [{"section_title": "General", "content": ""}]

        for i, sec in enumerate(sections):
            text = clean_text(sec["content"])
            if not text:
                continue

            docs.append({
                "id": f"{kb_type}::{docx_path.name}::sec{i+1}",
                "text": text,
                "metadata": {
                    "kb_type": kb_type,
                    "source_file": docx_path.name,
                    "source_path": str(docx_path.as_posix()),
                    "title": title,
                    "section": sec["section_title"],
                }
            })

    return docs


def main():
    if not KB_SOURCES_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {KB_SOURCES_DIR.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    docs = build_documents()

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("✅ Done")
    print(f"- Source folder: {KB_SOURCES_DIR.resolve()}")
    print(f"- Output file:   {OUT_JSONL.resolve()}")
    print(f"- Total docs (sections): {len(docs)}")

    # Quick breakdown by type
    counts = {}
    for d in docs:
        t = d["metadata"]["kb_type"]
        counts[t] = counts.get(t, 0) + 1
    print("- Breakdown:", counts)


if __name__ == "__main__":
    main()
