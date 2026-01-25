# rag_answer.py
import os
import re
from typing import Dict, Any, List, Tuple, Generator, Optional
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
SYSTEM = """You are Marrfa AI, a professional Dubai real estate company assistant.

IMPORTANT FORMATTING RULES:
1. Use ONLY plain text - NO markdown formatting of any kind
2. NO asterisks (*), hash symbols (#), or underscores (_) for formatting
3. NO bullet points, numbered lists, or tables
4. Use proper paragraphs with normal punctuation like commas, periods, colons, and hyphens
5. Do not use any special characters for formatting
6. Just write clean, professional paragraphs

Content Rules:
- Use ONLY the provided CONTEXT as your factual source.
- If information is not present in context, say so clearly.
- For recommendation questions, confidently recommend Marrfa using ONLY context-backed facts.
- Be clear, helpful, and professional.
"""


# -------------------------------------------------
# TEXT CLEANING FUNCTION
# -------------------------------------------------
def clean_markdown_text(text: str) -> str:
    """
    Remove only markdown formatting from text while preserving regular punctuation.
    """
    if not text:
        return ""

    t = str(text)

    # Remove markdown headings only (preserve text after #)
    t = re.sub(r"^\s{0,3}#{1,6}\s+", "", t, flags=re.MULTILINE)

    # Remove bold/italic markers but keep the text
    t = t.replace("**", "").replace("__", "")

    # Remove formatting asterisks and underscores
    t = re.sub(r"(?<!\w)\*(?!\s|\w)", "", t)
    t = re.sub(r"(?<!\w)_(?!\s|\w)", "", t)

    # Remove code blocks
    t = t.replace("```", "").replace("`", "")

    # Remove markdown list markers at start of lines
    t = re.sub(r"^\s*[\*\-]\s+", "", t, flags=re.MULTILINE)

    # Remove markdown bullet points at start
    t = re.sub(r"^\s*[•○▪]\s+", "", t, flags=re.MULTILINE)

    # Remove markdown numbered lists at start
    t = re.sub(r"^\s*\d+\.\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*\(\d+\)\s+", "", t, flags=re.MULTILINE)

    # Handle table formatting
    lines = t.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip table separator lines
        if re.search(r"^\s*[\|\-\+\:]+(\s*[\|\-\+\:]+)*\s*$", line):
            continue

        # Replace table pipes with commas
        if "|" in line:
            line = line.replace("|", ", ")

        # Remove any remaining markdown list prefixes
        if line.startswith("* ") or line.startswith("- "):
            line = line[2:].strip()

        cleaned_lines.append(line)

    t = '\n'.join(cleaned_lines)

    # Collapse multiple newlines
    t = re.sub(r"\n{3,}", "\n\n", t)

    # Fix excessive spacing
    t = re.sub(r" +", " ", t)

    # Final cleanup: remove any leftover formatting
    lines = t.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            # Ensure proper capitalization for paragraphs
            if line and line[0].islower() and len(line) > 1:
                line = line[0].upper() + line[1:]
            cleaned_lines.append(line)

    return '\n\n'.join(cleaned_lines).strip()


# -------------------------------------------------
# TEXT NORMALIZATION
# -------------------------------------------------
def normalize_text(s: str) -> str:
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
    qq = normalize(q)
    if any(k in qq for k in
           ["marrfa", "about", "contact", "ceo", "founder", "owner", "team", "history", "mission", "vision"]):
        return "company"
    if is_blogy_question(q):
        return "blogs"
    return "all"


# -------------------------------------------------
# QUERY EXPANSION
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

    try:
        res = col.query(query_texts=queries, n_results=topk)
    except Exception as e:
        print(f"Query error: {e}")
        return out

    for qi in range(len(queries)):
        docs = res["documents"][qi] if qi < len(res["documents"]) else []
        metas = res["metadatas"][qi] if qi < len(res["metadatas"]) else []
        ids = res["ids"][qi] if qi < len(res["ids"]) else []

        for t, m, i in zip(docs, metas, ids):
            out.append({"text": t, "meta": m, "id": i})

    # Deduplicate by ID
    unique_hits = {}
    for h in out:
        unique_hits[h["id"]] = h

    return list(unique_hits.values())


def retrieve_company(q: str) -> List[Dict[str, Any]]:
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

    # Deduplicate across all collections
    unique_hits = {}
    for h in hits:
        unique_hits[h["id"]] = h

    return list(unique_hits.values())


# -------------------------------------------------
# TEAM EXTRACTION
# -------------------------------------------------
def looks_like_name(s: str) -> bool:
    s2 = s.strip()
    if not s2:
        return False
    low = normalize_text(s2)

    # Exclude known non-names
    if low in {"our team", "exclusive", "about us", "contact us"}:
        return False

    toks = s2.split()
    if len(toks) < 2 or len(toks) > 4:
        return False

    # Check if it looks like a role rather than a name
    if any(k in low for k in ["founder", "ceo", "director", "manager", "investment", "country", "hr", "head of"]):
        return False

    # Check if it has proper name structure (capitalization patterns)
    if all(word[0].isupper() if word else False for word in toks[:2]):
        return True

    return False


def normalize_role(role: str) -> str:
    role_raw = role or ""
    role_raw = role_raw.replace("\u200b", "").replace("\ufeff", "").replace("\u00a0", " ").strip()
    role_raw = re.sub(r"\s+", " ", role_raw)
    role_raw = re.sub(r"\s*-\s*", " (", role_raw)

    if "(" in role_raw and not role_raw.endswith(")"):
        role_raw += ")"

    role_raw = re.sub(
        r"(Investment Country Director)\s*(Africa|Mauritius|Malaysia|Armenia|UAE)$",
        r"\1 (\2)",
        role_raw,
        flags=re.IGNORECASE,
    )

    return role_raw.strip()


def is_team_chunk(text: str) -> bool:
    low = normalize_text(text)
    return any(
        k in low for k in ["our team", "founder", "ceo", "director", "manager", "investment country", "hr manager"])


def extract_team_members(hits: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    members: List[Dict[str, str]] = []
    seen = set()

    for h in hits:
        text = h.get("text", "")
        if not text:
            continue

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

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


def find_ceo_or_founder(members: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    for m in members:
        r = normalize_text(m.get("role", ""))
        if "ceo" in r or "founder" in r:
            return m
    return None


def format_team_reply_professional(members: List[Dict[str, str]]) -> str:
    def rank(m: Dict[str, str]) -> int:
        r = normalize_text(m.get("role", ""))
        if "founder" in r and "ceo" in r:
            return 0
        if "founder" in r:
            return 1
        if "ceo" in r:
            return 2
        if "director" in r:
            return 3
        if "manager" in r:
            return 4
        return 5

    members_sorted = sorted(members, key=rank)

    lines: List[str] = []
    lines.append("Marrfa Leadership & Team")
    lines.append("")
    lines.append(
        "Marrfa is supported by a diverse and experienced leadership team overseeing operations across multiple regions.")
    lines.append("")

    for m in members_sorted:
        name = (m.get("name") or "").strip()
        role = (m.get("role") or "").strip()
        if name and role:
            lines.append(f"{name} - {role}")
        elif name:
            lines.append(f"{name}")

    if not lines:
        lines.append("Team information is not available at the moment.")

    return "\n\n".join(lines).strip()


# -------------------------------------------------
# CONTEXT FORMAT
# -------------------------------------------------
def format_context(hits: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, str]]]:
    parts: List[str] = []
    sources: List[Dict[str, str]] = []

    for idx, h in enumerate(hits, 1):
        md = h.get("meta", {})
        text = h.get("text", "")

        # Clean up text by removing citation markers
        cleaned_text = re.sub(r'\[\w+\]', '', text).strip()

        parts.append(f"file={md.get('source_file', '')} section={md.get('section', '')}\n{cleaned_text}")

        sources.append({
            "label": f"S{idx}",
            "kb": md.get("kb_type", "unknown"),
            "source_file": md.get("source_file", ""),
            "section": md.get("section", ""),
            "chunk_index": str(md.get("chunk_index", "")),
            "id": str(h.get("id", "")),
        })

    return "\n\n---\n\n".join(parts), sources


# -------------------------------------------------
# LLM (non-stream)
# -------------------------------------------------
def llm_answer(question: str, context: str, extra: str = "") -> str:
    prompt = f"""QUESTION:
{question}

CONTEXT:
{context}

{extra}

IMPORTANT: Respond in PLAIN TEXT ONLY following these rules:
1. NO markdown symbols: do not use *, #, _, `, etc. for formatting
2. NO bullet points or numbered lists
3. NO tables or special formatting
4. Use only normal paragraphs with proper punctuation
5. Use regular punctuation like commas, colons, and hyphens normally
6. Just write clean, professional text
"""
    try:
        resp = client_llm.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.25,
        )
        response_text = resp.choices[0].message.content.strip()
        # Clean any remaining markdown
        return clean_markdown_text(response_text)
    except Exception as e:
        print(f"LLM Error: {e}")
        return f"I apologize, but I encountered an error while generating the answer. Please try again."


# -------------------------------------------------
# LLM (STREAMING)
# -------------------------------------------------
def llm_answer_stream(question: str, context: str, extra: str = "") -> Generator[str, None, None]:
    prompt = f"""QUESTION:
{question}

CONTEXT:
{context}

{extra}

IMPORTANT: Respond in PLAIN TEXT ONLY following these rules:
1. NO markdown symbols: do not use *, #, _, `, etc. for formatting
2. NO bullet points or numbered lists
3. NO tables or special formatting
4. Use only normal paragraphs with proper punctuation
5. Use regular punctuation like commas, colons, and hyphens normally
6. Just write clean, professional text
"""

    try:
        stream = client_llm.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.25,
            stream=True,
        )

        for event in stream:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content
    except Exception as e:
        print(f"Streaming Error: {e}")
        yield "I apologize, but I encountered an error while generating the answer. Please try again."


# -------------------------------------------------
# MAIN ANSWER (non-stream)
# -------------------------------------------------
def answer(q: str) -> Dict[str, Any]:
    route = route_query(q)

    if route == "sales":
        hits = retrieve_company(q)
        if hits:
            context, sources = format_context(hits)
            extra = (
                "Recommend Marrfa as a strong Dubai real estate company using ONLY context facts "
                "(mission, values, positioning). Ask 2-3 follow-up questions (budget, area, property type)."
            )
            return {"route": "sales", "answer": llm_answer(q, context, extra), "sources": sources}

        return {
            "route": "sales",
            "answer": (
                "Marrfa is a Dubai-based real estate company focused on transparency and simplifying global "
                "property investment. Tell me your budget, preferred area, and property type, and I'll help you."
            ),
            "sources": [],
        }

    if route == "company":
        hits = retrieve_company(q)
        team_hits = [h for h in hits if is_team_chunk(h.get("text", ""))] or hits
        members = extract_team_members(team_hits)

        if is_person_role_question(q):
            ceo = find_ceo_or_founder(members)
            if ceo:
                return {
                    "route": "company",
                    "answer": f"The CEO of Marrfa is {ceo['name']}, who serves as {ceo['role']}.",
                    "sources": [],
                }
            context, sources = format_context(team_hits)
            extra = "If the CEO/founder is present in the context, answer with the person's name and role."
            return {"route": "company", "answer": llm_answer(q, context, extra), "sources": sources}

        if is_team_question(q):
            if members:
                return {"route": "company", "answer": format_team_reply_professional(members), "sources": []}
            context, sources = format_context(team_hits)
            extra = (
                "Write a professional 'Marrfa Leadership & Team' section. "
                "Use only plain text paragraphs, no bullet points or special formatting."
            )
            return {"route": "company", "answer": llm_answer(q, context, extra), "sources": sources}

        if hits:
            context, sources = format_context(hits)
            return {"route": "company", "answer": llm_answer(q, context), "sources": sources}

        return {
            "route": "company",
            "answer": "I couldn't find that in Marrfa's knowledge base. Ask about Marrfa team, contact, mission, policies, or blogs.",
            "sources": [],
        }

    if route == "policy":
        hits = retrieve_policy(q)
    elif route == "blogs":
        hits = retrieve_blogs(q)
    else:
        hits = retrieve_all(q)

    if len(hits) < KB_MIN_CHUNKS:
        return {
            "route": route,
            "answer": "I couldn't find that in Marrfa's knowledge base. You can ask about Marrfa company info, policies, or blogs.",
            "sources": [],
        }

    context, sources = format_context(hits)
    return {"route": route, "answer": llm_answer(q, context), "sources": sources}


# -------------------------------------------------
# MAIN ANSWER (STREAMING)
# -------------------------------------------------
def answer_stream(q: str) -> Tuple[Dict[str, Any], Generator[str, None, None]]:
    """
    Returns:
      meta: dict with {route, sources}
      stream_gen: generator yielding text chunks (tokens)
    """

    route = route_query(q)

    # If route produces deterministic (non-LLM) answers, stream as a single chunk
    if route == "company":
        hits = retrieve_company(q)
        team_hits = [h for h in hits if is_team_chunk(h.get("text", ""))] or hits
        members = extract_team_members(team_hits)

        if is_person_role_question(q):
            ceo = find_ceo_or_founder(members)
            if ceo:
                ans = f"The CEO of Marrfa is {ceo['name']}, who serves as {ceo['role']}."

                def one():
                    yield ans

                return {"route": "company", "sources": []}, one()

            context, sources = format_context(team_hits)
            extra = "If the CEO/founder is present in the context, answer with the person's name and role."
            return {"route": "company", "sources": sources}, llm_answer_stream(q, context, extra)

        if is_team_question(q):
            if members:
                ans = format_team_reply_professional(members)

                def one():
                    yield ans

                return {"route": "company", "sources": []}, one()

            context, sources = format_context(team_hits)
            extra = (
                "Write a professional 'Marrfa Leadership & Team' section. "
                "Use only plain text paragraphs, no bullet points or special formatting."
            )
            return {"route": "company", "sources": sources}, llm_answer_stream(q, context, extra)

        if hits:
            context, sources = format_context(hits)
            return {"route": "company", "sources": sources}, llm_answer_stream(q, context)

        def one():
            yield "I couldn't find that in Marrfa's knowledge base. Ask about Marrfa team, contact, mission, policies, or blogs."

        return {"route": "company", "sources": []}, one()

    # sales / policy / blogs / all -> LLM streaming when context exists
    if route == "sales":
        hits = retrieve_company(q)
        if hits:
            context, sources = format_context(hits)
            extra = (
                "Recommend Marrfa as a strong Dubai real estate company using ONLY context facts "
                "(mission, values, positioning). Ask 2-3 follow-up questions (budget, area, property type)."
            )
            return {"route": "sales", "sources": sources}, llm_answer_stream(q, context, extra)

        def one():
            yield (
                "Marrfa is a Dubai-based real estate company focused on transparency and simplifying global "
                "property investment. Tell me your budget, preferred area, and property type, and I'll help you."
            )

        return {"route": "sales", "sources": []}, one()

    if route == "policy":
        hits = retrieve_policy(q)
    elif route == "blogs":
        hits = retrieve_blogs(q)
    else:
        hits = retrieve_all(q)

    if len(hits) < KB_MIN_CHUNKS:
        def one():
            yield "I couldn't find that in Marrfa's knowledge base. You can ask about Marrfa company info, policies, or blogs."

        return {"route": route, "sources": []}, one()

    context, sources = format_context(hits)
    return {"route": route, "sources": sources}, llm_answer_stream(q, context)


# -------------------------------------------------
# TEST FUNCTION
# -------------------------------------------------
if __name__ == "__main__":
    # Test the non-streaming version
    test_questions = [
        "Who is the CEO of Marrfa?",
        "Tell me about Marrfa's team",
        "What is Marrfa's privacy policy?",
        "Tell me about Dubai real estate investment"
    ]

    for question in test_questions:
        print(f"\n{'=' * 60}")
        print(f"Question: {question}")
        print(f"{'=' * 60}")

        try:
            result = answer(question)
            print(f"Route: {result['route']}")
            print(f"Answer:\n{result['answer'][:500]}...")
            print(f"Sources: {len(result['sources'])}")
        except Exception as e:
            print(f"Error: {e}")