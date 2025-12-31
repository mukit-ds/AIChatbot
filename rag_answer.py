# rag_answer.py
# Marrfa RAG runtime (Chroma)
# FINAL (fixes CEO lookup + team extraction deterministically)
#
# Features:
# - Routing (company / policy / blogs / all / sales)
# - Company query expansion (owner/founder/CEO/team)
# - Company retrieval with multi-query anchors
# - Deterministic team extraction (no LLM guessing)
# - Deterministic CEO/Founder/Owner extraction (handles DOCX zero-width chars)
# - General RAG answers via LLM with citations (for non-structured questions)
# - Guaranteed return dict (never None)

import os
import re
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

# -------------------------------------------------
# ENV
# -------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
CHROMA_DIR = os.getenv("CHROMA_DIR", "kb_chroma")

TOPK_COMPANY = int(os.getenv("KB_TOPK_COMPANY", "15"))
TOPK_POLICY = int(os.getenv("KB_TOPK_POLICY", "4"))
TOPK_BLOGS = int(os.getenv("KB_TOPK_BLOGS", "5"))
KB_MIN_CHUNKS = int(os.getenv("KB_MIN_CHUNKS", "1"))

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY (set it in .env or env vars)")

client_llm = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------------------
# CHROMA
# -------------------------------------------------
client = chromadb.PersistentClient(path=CHROMA_DIR)
embed_fn = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBED_MODEL)

COLS = {
    "company": client.get_collection("company_kb", embedding_function=embed_fn),
    "policy": client.get_collection("policy_kb", embedding_function=embed_fn),
    "blogs": client.get_collection("blogs_kb", embedding_function=embed_fn),
}

# -------------------------------------------------
# PROMPT (for general RAG answers)
# -------------------------------------------------
SYSTEM = """You are Marrfa AI, a Dubai real estate company assistant.

Rules:
- Use ONLY the provided CONTEXT as your factual source.
- If information is not present in context, say so clearly.
- For recommendation questions, confidently recommend Marrfa using ONLY context-backed facts.
- Add citations like [S1], [S2] at the end of sentences using facts.
- Be clear, helpful, and professional.
"""

# -------------------------------------------------
# TEXT NORMALIZATION (critical for DOCX weird chars)
# -------------------------------------------------
def normalize_text(s: str) -> str:
    """
    Normalize text coming from DOCX / KB:
    - remove zero-width chars
    - replace non-breaking spaces
    - collapse whitespace
    - lowercase
    """
    if not s:
        return ""
    s = s.replace("\u200b", "").replace("\ufeff", "").replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip()

def normalize(q: str) -> str:
    return normalize_text(q)

# -------------------------------------------------
# QUESTION TYPE DETECTORS
# -------------------------------------------------
def is_team_question(q: str) -> bool:
    qq = normalize(q)
    return any(k in qq for k in [
        "team", "teams", "staff", "employees", "employee",
        "people", "members", "leadership"
    ])

def is_person_role_question(q: str) -> bool:
    """
    Questions like:
    - who is the ceo?
    - who is the founder?
    - who owns marrfa?
    - who built marrfa?
    """
    qq = normalize(q)
    return any(k in qq for k in ["ceo", "founder", "owner", "owns", "built", "build", "created"])

def is_policy_question(q: str) -> bool:
    qq = normalize(q)
    return any(k in qq for k in ["privacy", "terms", "policy", "cookies", "gdpr"])

def is_blogy_question(q: str) -> bool:
    qq = normalize(q)
    return any(k in qq for k in [
        "blog", "guide", "invest", "investment", "market",
        "visa", "off plan", "rental"
    ])

def is_sales_question(q: str) -> bool:
    qq = normalize(q)
    return any(k in qq for k in [
        "best", "recommend", "suggest", "good agency",
        "is marrfa good", "is marrfa reliable", "is marrfa trustworthy",
        "why marrfa", "choose marrfa", "best in dubai"
    ])

# -------------------------------------------------
# ROUTER
# -------------------------------------------------
def route_query(q: str) -> str:
    if is_sales_question(q):
        return "sales"
    if is_policy_question(q):
        return "policy"
    # CEO/founder/owner/team/about/contact/history => company
    qq = normalize(q)
    if any(k in qq for k in ["marrfa", "about", "contact", "ceo", "founder", "owner", "team", "history", "mission", "vision"]):
        return "company"
    if is_blogy_question(q):
        return "blogs"
    return "all"

# -------------------------------------------------
# QUERY EXPANSION (improves retrieval)
# -------------------------------------------------
def expand_company_query(q: str) -> str:
    qq = normalize(q)

    if any(k in qq for k in ["ceo", "founder", "owner", "owns", "built", "build", "created"]):
        return f"{q}\n\nAlso search for: Founder & CEO Marrfa Our Team leadership"

    if "team" in qq or "employees" in qq or "staff" in qq:
        return f"{q}\n\nAlso search for: Our Team Marrfa team members leadership"

    if "contact" in qq or "email" in qq or "phone" in qq or "address" in qq:
        return f"{q}\n\nAlso search for: Contact Us Marrfa email phone address"

    return q

# -------------------------------------------------
# RETRIEVAL
# -------------------------------------------------
def _query_collection(col, queries: List[str], topk: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    res = col.query(query_texts=queries, n_results=topk)

    for qi in range(len(queries)):
        docs = res["documents"][qi]
        metas = res["metadatas"][qi]
        ids = res["ids"][qi]
        for t, m, i in zip(docs, metas, ids):
            out.append({"text": t, "meta": m, "id": i})

    # de-dupe by id
    return list({h["id"]: h for h in out}.values())

def retrieve_company(q: str) -> List[Dict[str, Any]]:
    # multi-query anchors to reliably hit list-like team chunks
    queries = [
        q,
        expand_company_query(q),
        "Founder & CEO Marrfa",
        "Our Team Marrfa",
        "Marrfa team members",
        "Jamil Ahmed Founder CEO Marrfa",
    ]
    return _query_collection(COLS["company"], queries, TOPK_COMPANY)

def retrieve_policy(q: str) -> List[Dict[str, Any]]:
    return _query_collection(COLS["policy"], [q], TOPK_POLICY)

def retrieve_blogs(q: str) -> List[Dict[str, Any]]:
    return _query_collection(COLS["blogs"], [q], TOPK_BLOGS)

def retrieve_all(q: str) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    hits += retrieve_company(q)
    hits += retrieve_policy(q)
    hits += retrieve_blogs(q)
    return list({h["id"]: h for h in hits}.values())

# -------------------------------------------------
# TEAM EXTRACTION (DETERMINISTIC)
# -------------------------------------------------
def looks_like_name(s: str) -> bool:
    s2 = s.strip()
    if not s2:
        return False
    low = normalize_text(s2)
    if low in {"our team", "exclusive"}:
        return False
    toks = s2.split()
    if len(toks) < 2:
        return False
    # avoid role-like lines used as "name"
    if any(k in low for k in ["founder", "ceo", "director", "manager", "investment", "country", "hr"]) and len(toks) <= 3:
        return False
    return True

def normalize_role(role: str) -> str:
    role_raw = role or ""
    role_raw = role_raw.replace("\u200b", "").replace("\ufeff", "").replace("\u00a0", " ").strip()
    role_raw = re.sub(r"\s+", " ", role_raw)

    # "Director- Singapore" -> "Director (Singapore)"
    role_raw = re.sub(r"\s*-\s*", " (", role_raw)
    if "(" in role_raw and not role_raw.endswith(")"):
        role_raw += ")"

    # "Investment Country DirectorAfrica" -> "Investment Country Director (Africa)"
    role_raw = re.sub(
        r"(Investment Country Director)\s*(Africa|Mauritius|Malaysia|Armenia|UAE)$",
        r"\1 (\2)",
        role_raw,
        flags=re.IGNORECASE,
    )
    return role_raw.strip()

def is_team_chunk(text: str) -> bool:
    low = normalize_text(text)
    return any(k in low for k in ["our team", "founder", "ceo", "director", "manager", "investment country", "hr manager"])

def extract_team_members(hits: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    members: List[Dict[str, str]] = []
    seen = set()

    for h in hits:
        lines = [ln.strip() for ln in (h["text"] or "").splitlines() if ln.strip()]
        i = 0
        while i < len(lines) - 1:
            name = lines[i]
            role = lines[i + 1]

            if looks_like_name(name) and not looks_like_name(role):
                role_norm = normalize_role(role)
                key = (name.strip(), role_norm.strip())
                if key not in seen:
                    members.append({"name": name.strip(), "role": role_norm.strip()})
                    seen.add(key)
                i += 2
            else:
                i += 1

    return members

def find_ceo_or_founder(members: List[Dict[str, str]]) -> Dict[str, str] | None:
    for m in members:
        r = normalize_text(m.get("role", ""))
        if "ceo" in r or "founder" in r:
            return m
    return None

# -------------------------------------------------
# CONTEXT FORMAT (for LLM answers)
# -------------------------------------------------
def format_context(hits: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, str]]]:
    parts: List[str] = []
    sources: List[Dict[str, str]] = []

    for idx, h in enumerate(hits, 1):
        md = h["meta"]
        label = f"S{idx}"
        parts.append(f"[{label}] file={md.get('source_file','')} section={md.get('section','')}\n{h['text']}")
        sources.append({
            "label": label,
            "kb": "company" if md.get("kb_type") == "company" else md.get("kb_type", ""),
            "source_file": md.get("source_file", ""),
            "section": md.get("section", ""),
            "chunk_index": str(md.get("chunk_index", "")),
            "id": str(h.get("id", "")),
        })

    return "\n\n---\n\n".join(parts), sources

# -------------------------------------------------
# LLM (general RAG answers only)
# -------------------------------------------------
def llm_answer(question: str, context: str, extra: str = "") -> str:
    prompt = f"""QUESTION:
{question}

CONTEXT:
{context}

{extra}

Return:
- Answer in plain text
- Add citations like [S1] where you used context facts
"""
    resp = client_llm.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.25,
    )
    return resp.choices[0].message.content.strip()

# -------------------------------------------------
# MAIN ANSWER
# -------------------------------------------------
def answer(q: str) -> Dict[str, Any]:
    route = route_query(q)

    # -------------------------
    # SALES MODE
    # -------------------------
    if route == "sales":
        hits = retrieve_company(q)
        if hits:
            context, sources = format_context(hits)
            extra = (
                "Recommend Marrfa as a strong Dubai real estate company using ONLY context facts "
                "(mission, values, positioning). Ask 2–3 follow-up questions (budget, area, property type)."
            )
            return {"route": "sales", "answer": llm_answer(q, context, extra), "sources": sources}

        return {
            "route": "sales",
            "answer": (
                "Marrfa is a Dubai-based real estate company focused on transparency and simplifying global "
                "property investment. Tell me your budget, preferred area, and property type, and I’ll help you."
            ),
            "sources": [],
        }

    # -------------------------
    # COMPANY MODE (structured answers for CEO/team)
    # -------------------------
    if route == "company":
        hits = retrieve_company(q)
        team_hits = [h for h in hits if is_team_chunk(h.get("text", ""))] or hits
        members = extract_team_members(team_hits)

        # CEO / Founder / Owner / Built-by questions
        if is_person_role_question(q):
            ceo = find_ceo_or_founder(members)
            if ceo:
                return {
                    "route": "company",
                    "answer": f"The CEO of Marrfa is {ceo['name']}, who serves as {ceo['role']}.",
                    "sources": [],
                }
            # fallback: use LLM only if we have context but parsing fails
            context, sources = format_context(team_hits)
            extra = "If the CEO/founder is present in the context, answer with the person's name and role."
            return {"route": "company", "answer": llm_answer(q, context, extra), "sources": sources}

        # Team list question
        if is_team_question(q):
            if members:
                lines = ["Here is the Marrfa team:"]
                for m in members:
                    lines.append(f"- {m['name']} — {m['role']}")
                return {"route": "company", "answer": "\n".join(lines), "sources": []}

            context, sources = format_context(team_hits)
            extra = "List all team members (name and role) if present in the context."
            return {"route": "company", "answer": llm_answer(q, context, extra), "sources": sources}

        # General company question -> LLM
        if hits:
            context, sources = format_context(hits)
            return {"route": "company", "answer": llm_answer(q, context), "sources": sources}

        return {
            "route": "company",
            "answer": "I couldn’t find that in Marrfa’s knowledge base. Ask about Marrfa team, contact, mission, policies, or blogs.",
            "sources": [],
        }

    # -------------------------
    # POLICY / BLOGS / ALL (general RAG)
    # -------------------------
    if route == "policy":
        hits = retrieve_policy(q)
    elif route == "blogs":
        hits = retrieve_blogs(q)
    else:
        hits = retrieve_all(q)

    if len(hits) < KB_MIN_CHUNKS:
        return {
            "route": route,
            "answer": "I couldn’t find that in Marrfa’s knowledge base. You can ask about Marrfa company info, policies, or blogs.",
            "sources": [],
        }

    context, sources = format_context(hits)
    return {"route": route, "answer": llm_answer(q, context), "sources": sources}

# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    print("✅ Marrfa RAG ready. Type 'exit' to quit.")
    while True:
        q = input("\nAsk: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        out = answer(q)
        print("\nROUTE:", out["route"])
        print("\nANSWER:\n", out["answer"])

        if out.get("sources"):
            print("\nSOURCES:")
            for s in out["sources"]:
                print(f"- {s.get('label')} | {s.get('source_file')} | {s.get('section')}")
