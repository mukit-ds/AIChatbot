# app.py
import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple, Generator

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

import rag_answer

# Optional LLM router (hybrid approach)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(title="Marrfa AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set your frontend domain(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = ""
    session_id: Optional[str] = None
    is_logged_in: bool = False


# ============================================================
# Helpers
# ============================================================
STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "for", "on", "with", "at", "from", "by",
    "is", "it", "this", "that", "as", "are", "be", "you", "your", "do", "does", "can", "should",
    "what", "why", "how", "when", "where", "vs", "about", "into", "than", "then", "also",
}


def _norm(s: str) -> str:
    return (s or "").lower().strip()


def _tokenize(text: str) -> List[str]:
    text = _norm(text)
    toks = re.findall(r"[a-z0-9]+", text)
    return [t for t in toks if t not in STOPWORDS and len(t) >= 3]


def _build_ngrams(tokens: List[str], n: int) -> List[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i + n]) for i in range(0, len(tokens) - n + 1)]


def sse_event(data: Dict[str, Any], event: str = "message") -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ============================================================
# Blog keyword loader (from raw_docs.jsonl)
# ============================================================
def load_blog_keywords_from_raw_docs(raw_docs_path: str) -> Dict[str, Any]:
    """
    Builds blog keyword/phrase sets from raw_docs.jsonl (your blogs dump).
    Uses titles and title n-grams so routing can recognize blog topics.
    """
    keywords: set[str] = set()
    phrases: set[str] = set()
    titles: set[str] = set()

    if not raw_docs_path or not os.path.exists(raw_docs_path):
        return {"keywords": set(), "phrases": set(), "titles": set()}

    try:
        with open(raw_docs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                md = obj.get("metadata", {}) or {}
                # IMPORTANT: your blogs entries should have kb_type == "blogs"
                if str(md.get("kb_type", "")).lower() != "blogs":
                    continue

                title = md.get("title") or ""
                title_n = _norm(title)
                if not title_n:
                    continue

                titles.add(title_n)

                toks = _tokenize(title_n)
                for t in toks:
                    keywords.add(t)

                for bi in _build_ngrams(toks, 2):
                    phrases.add(bi)
                for tri in _build_ngrams(toks, 3):
                    phrases.add(tri)

        # Add a small set of domain "blog intent" markers
        phrases |= {
            "golden visa", "investor visa", "residency visa",
            "right time", "best time",
            "advantages", "disadvantages",
            "commercial property",
            "how to buy", "buying guide", "step by step",
            "rental yield", "capital appreciation", "market trends",
            "expo 2020", "vision 2040",
        }
        keywords |= {
            "visa", "golden", "residency",
            "invest", "investment", "roi", "yield",
            "advantages", "disadvantages",
            "commercial", "fees", "charges", "tax", "taxes",
            "guide", "steps", "process",
            "market", "trends",
            "expo", "vision",
        }

        return {"keywords": keywords, "phrases": phrases, "titles": titles}
    except Exception:
        return {"keywords": set(), "phrases": set(), "titles": set()}


RAW_DOCS_PATH = os.getenv("RAW_DOCS_PATH", "raw_docs.jsonl")
BLOG_KB = load_blog_keywords_from_raw_docs(RAW_DOCS_PATH)
BLOG_TITLE_SET = BLOG_KB["titles"]
BLOG_KEYWORDS_FROM_DOCS = BLOG_KB["keywords"]
BLOG_PHRASES_FROM_DOCS = BLOG_KB["phrases"]

# ============================================================
# Routing: rules + LLM fallback (hybrid)
# ============================================================
GREETING_PATTERNS = {
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "greetings", "hi there", "hey there", "how are you", "how's it going",
    "what's up", "sup", "yo", "hello there", "hiya", "hola", "bonjour"
}

POLICY_KEYWORDS = {
    "policy", "policies", "terms", "conditions", "privacy", "legal", "refund",
    "compliance", "rules", "regulations", "agreement", "disclaimer", "privacy policy"
}

COMPANY_KEYWORDS = {
    "marrfa", "marffa", "marfa", "ceo", "founder", "owner", "team", "about us",
    "contact", "address", "phone", "email", "office", "who are you"
}

# Informational/blog markers (plus title-derived phrases/keywords from raw_docs.jsonl)
BLOG_INFO_MARKERS = {
    "right time", "is it right time", "good time", "best time",
    "should i", "is it worth", "worth it",
    "why invest", "why buy",
    "advantages", "disadvantages", "pros and cons", "risks",
    "visa", "golden visa", "residency", "investor visa",
    "agent", "agents", "real estate agent", "broker", "brokers",
    "make good money", "salary", "commission", "earn",
    "market", "market trends", "forecast", "outlook",
    "roi", "yield", "rental income", "capital appreciation",
    "true cost", "fees", "charges", "tax", "taxes",
    "process", "steps", "how to", "guide", "step by step",
    "expo 2020", "vision 2040",
}

GENERIC_LOCS = {"dubai", "uae", "united arab emirates"}

AREAS = [
    "dubai marina",
    "palm jumeirah",
    "downtown dubai",
    "downtown",
    "business bay",
    "jumeirah village circle",
    "jvc",
    "jumeirah village triangle",
    "jvt",
    "dubai hills",
    "dubai hills estate",
    "dubai creek harbour",
    "dubai creek harbor",
    "dubai harbour",
    "dubai harbor",
    "bluewaters",
    "bluewaters island",
    "city walk",
    "al barari",
    "emirates hills",
    "arabian ranches",
    "damac hills",
    "damac hills 2",
    "dubai south",
    "mbr city",
    "meydan",
    "arjan",
    "jlt",
    "jumeirah lake towers",
    "al furjan",
    "motor city",
    "sports city",
    "dubai silicon oasis",
    "dubailand",
    "international city",
    "deira",
    "bur dubai",
    "jumeirah",
    "umm suqeim",
    "al sufouh",
    "dubai media city",
    "dubai internet city",
    "al quoz",
    "the greens",
    "the views",
    "springs",
    "meadows",
    "lakes",
    "jbr",
    "jumeirah beach residence",
]


def parse_query_to_filters(query: str) -> Dict[str, Any]:
    q_raw = (query or "").strip()
    q = _norm(q_raw)
    if not q:
        return {}

    filters: Dict[str, Any] = {}

    # Desired result count: only when number followed by "properties/listings/results..."
    mcount = re.search(
        r"\bshow\s+me\s+(\d{1,2})\s*(?:properties|property|listings|listing|homes|home|options|option|results|result)\b",
        q,
    )
    if not mcount:
        mcount = re.search(
            r"\b(\d{1,2})\s*(?:properties|property|listings|listing|homes|home|options|option|results|result)\b",
            q,
        )
    if mcount:
        try:
            n = int(mcount.group(1))
            if 1 <= n <= 50:
                filters["desired_count"] = n
        except Exception:
            pass

    # Location: prefer known Dubai areas
    for area in AREAS:
        if area in q:
            filters["search_query"] = area
            break

    # Fallback: "in <location>"
    if "search_query" not in filters:
        mloc = re.search(r"\b(?:in|at|within|around|near)\s+([a-z][a-z\s\-]{2,40})", q)
        if mloc:
            loc = mloc.group(1).strip()
            loc = re.split(
                r"\b(under|below|over|above|between|with|for|budget|around|approx|about|studio|\d+\s*bed)\b",
                loc
            )[0].strip()
            if loc:
                filters["search_query"] = loc

    # Developer (client-side filter)
    mdev = re.search(r"\b(?:projects?\s+by|by|from)\s+([a-z0-9&\.\- ]{2,40})\b", q)
    if mdev:
        dev = mdev.group(1).strip()
        dev = re.split(r"\b(in|under|below|over|above|between|with)\b", dev)[0].strip()
        if dev:
            filters["developer_name"] = dev

    # Property type
    type_map = {
        "villa": "Villa", "villas": "Villa",
        "townhouse": "Townhouse", "townhouses": "Townhouse",
        "apartment": "Apartment", "apartments": "Apartment",
        "flat": "Apartment", "flats": "Apartment",
        "penthouse": "Penthouse", "penthouses": "Penthouse",
        "duplex": "Duplex", "duplexes": "Duplex",
        "studio": "Apartment",
        "commercial": "Commercial",
    }
    for k, v in type_map.items():
        if re.search(rf"\b{re.escape(k)}\b", q):
            filters["unit_types"] = [v]
            break

    # Bedrooms
    if re.search(r"\bstudio\b", q):
        filters["unit_bedrooms"] = "Studio"
    else:
        mb = re.search(r"\b(\d+)\s*(?:bed|beds|bedroom|bedrooms|br)\b", q)
        if mb:
            try:
                n = int(mb.group(1))
                if 1 <= n <= 10:
                    # Canonical string expected by API
                    filters["unit_bedrooms"] = f"{n} bedroom"
            except Exception:
                pass

    # Price parsing (AED)
    def _word_unit_to_multiplier(word: str) -> int:
        word = (word or "").lower()
        if word in ("k", "thousand"):
            return 1_000
        if word in ("m", "million"):
            return 1_000_000
        if word in ("b", "billion"):
            return 1_000_000_000
        return 1

    def _parse_amount_token(num_s: str, unit_s: str) -> Optional[int]:
        try:
            n = float(num_s)
        except Exception:
            return None
        return int(n * _word_unit_to_multiplier(unit_s))

    # If user uses foreign currency, return warning
    mcur = re.search(r"(\d+(?:\.\d+)?)\s*(usd|eur|gbp|inr|sar|qar|omr|kwd|bhd)\b", q, re.IGNORECASE)
    if mcur:
        return {"foreign_currency": True, "amount": mcur.group(1), "currency": mcur.group(2).upper()}

    cleaned = re.sub(r"\b\d+\s*(bed|beds|bedroom|bedrooms|br)\b", "", q)
    amt_re = r"(\d+(?:\.\d+)?)\s*(k|m|b|thousand|million|billion)?\s*(?:aed|dhs|dirham|dirhams)?"

    # between X and Y
    m = re.search(rf"\bbetween\s+{amt_re}\s+and\s+{amt_re}\b", cleaned)
    if m:
        low = _parse_amount_token(m.group(1), m.group(2) or "")
        high = _parse_amount_token(m.group(3), m.group(4) or "")
        if low is not None:
            filters["unit_price_from"] = low
        if high is not None:
            filters["unit_price_to"] = high

    # under/below
    m = re.search(rf"\b(under|below|less than|upto|up to)\s+{amt_re}\b", cleaned)
    if m:
        amt = _parse_amount_token(m.group(2), m.group(3) or "")
        if amt is not None:
            filters["unit_price_to"] = amt

    # over/above
    m = re.search(rf"\b(over|above|more than)\s+{amt_re}\b", cleaned)
    if m:
        amt = _parse_amount_token(m.group(2), m.group(3) or "")
        if amt is not None:
            filters["unit_price_from"] = amt

    # around/budget is
    if "unit_price_from" not in filters and "unit_price_to" not in filters:
        m = re.search(
            rf"\b(budget\s*(?:is|=)|my\s+budget\s+is|around|about|approx(?:\.|imately)?|near)\s+{amt_re}\b",
            cleaned
        )
        if m:
            amt = _parse_amount_token(m.group(2), m.group(3) or "")
            if amt is not None:
                filters["unit_price_to"] = amt

    return filters


def clean_reply_text(text: str) -> str:
    """
    Remove only markdown formatting from LLM output while preserving regular punctuation.
    Ensures fresh professional paragraph-only responses.
    """
    if not text:
        return ""

    t = str(text)

    # Remove markdown headings only (preserve the text after #)
    t = re.sub(r"^\s{0,3}#{1,6}\s+", "", t, flags=re.MULTILINE)

    # Remove bold/italic markers but keep the text
    t = t.replace("**", "").replace("__", "")

    # Remove asterisks and underscores only when they're used for formatting
    # (not when they're part of regular text like in *this example*)
    # Remove standalone formatting asterisks
    t = re.sub(r"(?<!\w)\*(?!\s|\w)", "", t)
    t = re.sub(r"(?<!\w)_(?!\s|\w)", "", t)

    # Remove code blocks
    t = t.replace("```", "").replace("`", "")

    # Remove markdown list markers but preserve regular hyphens and numbers
    # Only remove when at start of line followed by space
    t = re.sub(r"^\s*[\*\-]\s+", "", t, flags=re.MULTILINE)

    # Remove markdown bullet points (• ○ ▪) when at start
    t = re.sub(r"^\s*[•○▪]\s+", "", t, flags=re.MULTILINE)

    # Remove markdown numbered lists (1., 2., etc.)
    t = re.sub(r"^\s*\d+\.\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*\(\d+\)\s+", "", t, flags=re.MULTILINE)

    # Remove table markers (preserve regular | characters in text)
    # Only remove when they look like table formatting
    lines = t.split('\n')
    cleaned_lines = []
    for line in lines:
        # Check if line looks like a table row (contains | with spaces)
        if re.search(r"\s*\|\s*", line):
            # Check if it's a table separator line
            if re.search(r"^\s*[\|\-\+\:]+(\s*[\|\-\+\:]+)*\s*$", line):
                # Skip table separator lines
                continue
            # Replace | with commas for readability
            line = line.replace("|", ", ")
        cleaned_lines.append(line.strip())

    t = '\n'.join(cleaned_lines)

    # Collapse multiple newlines
    t = re.sub(r"\n{3,}", "\n\n", t)

    # Ensure proper paragraph formatting
    lines = t.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            # Remove any remaining formatting asterisks
            if line.startswith("* ") or line.startswith("- "):
                line = line[2:].strip()
            cleaned_lines.append(line)

    t = '\n\n'.join(cleaned_lines)

    return t.strip()

def is_property_specific_query(query: str) -> bool:
    """
    True only when user asks about ONE named property:
      - Tell me about Park Horizon
      - What is The Cove II?
    Not: show me apartments in Dubai Marina / 2 bed apartments
    """
    q = (query or "").lower().strip()
    if not q:
        return False

    property_phrases = [
        "tell me about",
        "tell me more about",
        "information about",
        "details about",
        "describe",
        "what is",
        "what's",
    ]

    # If remainder contains these, it's not "specific name"
    search_markers = {
        "property", "properties", "listing", "listings",
        "apartment", "apartments", "flat", "flats",
        "villa", "villas", "townhouse", "townhouses",
        "penthouse", "penthouses", "studio", "commercial",
        "in", "at", "near", "within", "around",
        "under", "below", "less than", "over", "above", "between",
        "budget", "aed", "dhs", "dirham", "million", "k", "m",
        "bed", "beds", "bedroom", "bedrooms", "br",
        "ready", "off plan", "new launch",
        "show me", "find", "search", "list",
    }

    for phrase in property_phrases:
        if phrase in q:
            remainder = q.split(phrase, 1)[1].strip()
            remainder = re.sub(r"^[\s:,-]+", "", remainder)
            remainder = re.sub(r"[\?\!\.]+$", "", remainder).strip()

            if not remainder or len(remainder) < 3:
                return False
            if len(remainder) > 60:
                return False

            for mk in search_markers:
                if mk in {"less than", "off plan"}:
                    if mk in remainder:
                        return False
                else:
                    if re.search(rf"\b{re.escape(mk)}\b", remainder):
                        return False

            return True

    return False


def is_property_search_query(query_text: str) -> bool:
    """
    PROPERTY search means: user wants listings or listing filters.
    IMPORTANT: allow "Show me properties in Dubai." as PROPERTY search.
    """
    q = (query_text or "").strip()
    if not q:
        return False
    qq = q.lower()

    # Explicit listing verbs -> PROPERTY
    if re.search(r"\b(show\s+me|find|search|list|give\s+me)\b", qq):
        # Example: "Show me properties in Dubai." -> PROPERTY
        return True

    # Parse filters but do NOT treat "in Dubai" alone as property search.
    try:
        filters = parse_query_to_filters(q)
    except Exception:
        filters = {}

    if filters.get("desired_count"):
        return True

    # strong filter signals
    if filters.get("unit_price_from") is not None or filters.get("unit_price_to") is not None:
        return True
    if filters.get("unit_bedrooms"):
        return True
    if filters.get("unit_types"):
        return True
    if (filters.get("developer_name") or "").strip():
        return True

    # Location: only if it's NOT generic
    loc = (filters.get("search_query") or "").strip().lower()
    if loc and loc not in GENERIC_LOCS:
        return True

    # keyword-based signals
    if re.search(r"\b(under|below|over|above|between|budget|aed|dhs|dirham|million|\bk\b|\bm\b)\b", qq):
        return True
    if re.search(r"\b(studio|bed|beds|bedroom|bedrooms|br)\b", qq):
        return True

    return False


def looks_like_blog_question(query_text: str) -> bool:
    """
    Uses:
      - blog titles/phrases/keywords from raw_docs.jsonl
      - generic info markers ("how to", "advantages", "visa", etc.)
    """
    qq = _norm(query_text)
    if not qq:
        return False

    # direct title substring match
    for t in BLOG_TITLE_SET:
        if t and t in qq:
            return True

    # phrase match from titles + curated
    for ph in BLOG_PHRASES_FROM_DOCS:
        if ph and ph in qq:
            return True

    # keyword overlap (>=2) from titles
    toks = set(_tokenize(qq))
    if toks and (len(toks & BLOG_KEYWORDS_FROM_DOCS) >= 2):
        return True

    # generic info markers
    for m in BLOG_INFO_MARKERS:
        if m in qq:
            return True

    # question words + informational topic
    if re.search(r"\b(what|why|how|when|where)\b", qq) and any(
            k in qq for k in ["buy", "invest", "visa", "agent", "market", "property", "real estate", "commercial"]
    ):
        return True

    return False


# ----------------------------
# LLM router (only for ambiguous)
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ROUTER_MODEL = os.getenv("OPENAI_ROUTER_MODEL", "gpt-4o-mini")
_router_client = None

if OpenAI is not None and OPENAI_API_KEY:
    try:
        _router_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        _router_client = None


def llm_route_query(query: str) -> Dict[str, Any]:
    """
    Returns: {"route": "PROPERTY"|"BLOG"|"COMPANY"|"POLICY", "confidence": float, "reason": str}
    """
    if _router_client is None:
        return {"route": "BLOG", "confidence": 0.0, "reason": "router_unavailable"}

    sys = (
        "You are an intent router for a Dubai real estate assistant.\n"
        "Return JSON only. No markdown.\n"
        "Routes:\n"
        "- PROPERTY: user wants property listings/search/filter results.\n"
        "- BLOG: user asks informational advice/guide/market/visa/investment topic.\n"
        "- POLICY: terms, privacy, legal.\n"
        "- COMPANY: Marrfa company/team/contact/about.\n"
        "Decision rule:\n"
        "Choose PROPERTY only if the user is asking to see listings OR provides filters "
        "(budget/price, bedrooms, location, property type, ready/off-plan).\n"
        "Otherwise prefer BLOG.\n"
        "Output schema: {\"route\":\"BLOG\",\"confidence\":0.0-1.0,\"reason\":\"...\"}"
    )

    try:
        r = _router_client.chat.completions.create(
            model=OPENAI_ROUTER_MODEL,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": f"Query: {query}"},
            ],
            temperature=0,
        )
        text = (r.choices[0].message.content or "").strip()

        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return {"route": "BLOG", "confidence": 0.0, "reason": "router_bad_output"}

        obj = json.loads(m.group(0))
        route = str(obj.get("route", "BLOG")).upper()
        conf = float(obj.get("confidence", 0.0) or 0.0)
        reason = str(obj.get("reason", ""))[:200]

        if route not in {"PROPERTY", "BLOG", "COMPANY", "POLICY"}:
            route = "BLOG"
        conf = max(0.0, min(1.0, conf))

        return {"route": route, "confidence": conf, "reason": reason}
    except Exception:
        return {"route": "BLOG", "confidence": 0.0, "reason": "router_exception"}


def classify_intent_hybrid(query_text: str) -> Dict[str, Any]:
    q = _norm(query_text)
    if not q:
        return {"intent": "COMPANY", "method": "empty_fallback"}

    def is_greeting_query(q: str) -> bool:
        q = (q or "").lower().strip()
        if not q:
            return False

        # Only treat as greeting if it's basically just a greeting (short / no business intent)
        # This prevents "yo" inside "your" from triggering.
        tokens = re.findall(r"[a-z']+", q)

        greeting_words = {"hello", "hi", "hey", "hiya", "hola", "bonjour", "sup"}
        greeting_phrases = {
            "good morning", "good afternoon", "good evening",
            "how are you", "how's it going", "what's up", "hello there", "hey there", "hi there"
        }

        joined = " ".join(tokens)

        # phrase match (word-boundary safe via tokenization)
        if any(p in joined for p in greeting_phrases):
            return True

        # single-word greeting match
        if len(tokens) <= 3 and any(t in greeting_words for t in tokens):
            return True

        return False

    # inside classify_intent_hybrid(query_text):
    if is_greeting_query(query_text):
        return {"intent": "GREETING", "method": "keyword"}

    if any(w in q for w in POLICY_KEYWORDS):
        return {"intent": "POLICY", "method": "keyword"}

    if any(w in q for w in COMPANY_KEYWORDS):
        return {"intent": "COMPANY", "method": "keyword"}

    if is_property_specific_query(query_text):
        return {"intent": "PROPERTY", "method": "property_specific"}

    # IMPORTANT ORDER:
    # 1) BLOG if informational AND not a clear listings search
    if looks_like_blog_question(query_text) and not is_property_search_query(query_text):
        return {"intent": "BLOG", "method": "hard_blog"}

    # 2) PROPERTY if clear listings search
    if is_property_search_query(query_text):
        return {"intent": "PROPERTY", "method": "hard_property_search"}

    # 3) Ambiguous -> LLM router
    rr = llm_route_query(query_text)

    # Guardrail: if filters exist, force PROPERTY
    if is_property_search_query(query_text) and rr.get("route") != "PROPERTY":
        return {"intent": "PROPERTY", "method": f"llm_override_property({rr.get('reason', '')})"}

    route = rr.get("route", "BLOG")
    conf = rr.get("confidence", 0.0)

    # Low confidence: prefer BLOG (safer than showing random listings)
    if conf < 0.6:
        return {"intent": "BLOG", "method": f"llm_low_conf({rr.get('reason', '')})"}

    if route == "PROPERTY":
        return {"intent": "PROPERTY", "method": f"llm({rr.get('reason', '')})"}
    if route == "POLICY":
        return {"intent": "POLICY", "method": f"llm({rr.get('reason', '')})"}
    if route == "COMPANY":
        return {"intent": "COMPANY", "method": f"llm({rr.get('reason', '')})"}

    return {"intent": "BLOG", "method": f"llm({rr.get('reason', '')})"}


# ============================================================
# Marrfa property API client
# ============================================================
MARRFA_PROPERTIES_URL = os.getenv("MARRFA_PROPERTIES_URL", "https://apiv2.marrfa.com/properties")
_session = requests.Session()


def _maybe_csv(val: Any) -> Any:
    if isinstance(val, (list, tuple, set)):
        return ",".join(str(x) for x in val)
    return val


def _extract_url(x: Any) -> Optional[str]:
    if not x:
        return None
    if isinstance(x, str):
        s = x.strip()
        if s.startswith(("http://", "https://")):
            return s
        if s.startswith("{") and s.endswith("}") and '"url"' in s:
            try:
                obj = json.loads(s)
                u = obj.get("url")
                if isinstance(u, str) and u.startswith(("http://", "https://")):
                    return u
            except Exception:
                return None
        return None
    if isinstance(x, dict):
        for k in ("url", "image", "src"):
            u = x.get(k)
            if isinstance(u, str) and u.startswith(("http://", "https://")):
                return u
        return None
    if isinstance(x, list) and x:
        return _extract_url(x[0])
    return None


def search_properties(filters: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    params: Dict[str, Any] = {}
    for key in ["search_query", "unit_types", "unit_bedrooms", "unit_price_from", "unit_price_to", "page", "per_page"]:
        if filters.get(key) is not None:
            params[key] = _maybe_csv(filters[key])

    try:
        resp = _session.get(MARRFA_PROPERTIES_URL, params=params, timeout=12)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return [], []

    items = data.get("items", []) or data.get("data", []) or []
    properties_full: List[Dict[str, Any]] = items

    properties_min: List[Dict[str, Any]] = []
    for p in items:
        try:
            pid = p.get("id")

            completion = p.get("completion_datetime") or p.get("completion_date")
            completion_year = str(completion)[:4] if completion else None

            min_price = p.get("min_price_aed") or p.get("min_price")
            max_price = p.get("max_price_aed") or p.get("max_price")

            price_from = float(min_price) if min_price not in (None, 0, "0") else None
            price_to = float(max_price) if max_price not in (None, 0, "0") else None

            cover_url = None
            for k in ("cover_image", "cover_image_url", "thumbnail", "thumbnail_url", "cover_image_path"):
                if p.get(k):
                    cover_url = _extract_url(p.get(k))
                    if cover_url:
                        break

            listing_url = f"https://www.marrfa.com/propertylisting/{pid}" if pid is not None else None

            properties_min.append({
                "id": pid,
                "title": p.get("name") or p.get("title") or "Untitled property",
                "location": p.get("area") or p.get("location") or "Dubai",
                "developer": p.get("developer") or "",
                "completion_year": completion_year,
                "price_from": price_from,
                "price_to": price_to,
                "currency": p.get("price_currency") or "AED",
                "cover_image": cover_url,
                "listing_url": listing_url,
            })
        except Exception:
            continue

    return properties_min, properties_full


def extract_property_name(query: str) -> str:
    q = (query or "").lower()
    phrases_to_remove = [
        "tell me about", "tell me more about", "information about", "details about",
        "what is", "what's", "describe",
        "please", "can you", "could you",
    ]
    for phrase in phrases_to_remove:
        q = q.replace(phrase, " ")
    q = re.sub(r"\s+", " ", q).strip()
    if q:
        return " ".join(w.capitalize() for w in q.split())
    return ""


def handle_specific_property_query(query_text: str) -> Dict[str, Any]:
    property_name = extract_property_name(query_text)

    if not property_name:
        return {
            "reply": "I didn't catch the property name. Please tell me which property you mean.",
            "properties": [],
            "properties_full": [],
            "property_images": [],
            "total": 0,
            "filters_used": {"intent": "PROPERTY", "search_type": "specific"},
        }

    filters = {"search_query": property_name, "page": 1, "per_page": 10}
    properties_min, properties_full = search_properties(filters)

    # If no direct results, broaden and filter by name
    if not properties_min:
        filters["search_query"] = "dubai"
        properties_min, properties_full = search_properties(filters)

        filtered_min, filtered_full = [], []
        for pm, pf in zip(properties_min, properties_full):
            if property_name.lower() in (pm.get("title", "").lower()):
                filtered_min.append(pm)
                filtered_full.append(pf)
        properties_min, properties_full = filtered_min, filtered_full

    if not properties_min:
        return {
            "reply": (
                f"I couldn't find details for '{property_name}'. "
                f"The name might be different. If you share the exact listing name or link, I can try again."
            ),
            "properties": [],
            "properties_full": [],
            "property_images": [],
            "total": 0,
            "filters_used": {"intent": "PROPERTY", "search_type": "specific", "property_name": property_name},
        }

    prop = properties_min[0]
    prop_full = properties_full[0] if properties_full else {}

    # Images (up to 10)
    images: List[str] = []
    gallery = prop_full.get("gallery_images")
    if isinstance(gallery, str):
        try:
            gallery = json.loads(gallery)
        except Exception:
            gallery = []
    if isinstance(gallery, list):
        for img in gallery:
            u = _extract_url(img)
            if u and u not in images:
                images.append(u)
            if len(images) >= 10:
                break

    cover_url = prop.get("cover_image")
    if cover_url and cover_url not in images:
        images.insert(0, cover_url)
    images = images[:10]

    title = prop.get("title", "Unknown property")
    location = prop.get("location", "Dubai")
    developer = prop.get("developer", "Unknown developer")
    completion = prop.get("completion_year") or "N/A"

    price_from = prop.get("price_from")
    price_to = prop.get("price_to")
    price_info = "Price not available"
    if price_from is not None and price_to is not None:
        price_info = f"AED {price_from:,.0f} - {price_to:,.0f}"
    elif price_from is not None:
        price_info = f"From AED {price_from:,.0f}"
    elif price_to is not None:
        price_info = f"Up to AED {price_to:,.0f}"

    description = prop_full.get("description") or prop_full.get("short_description") or ""
    if not description:
        description = (
            f"{title} is a premium development in {location} by {developer}. "
            f"Completion is expected in {completion}."
        )

    reply = (
        f"Here are the details for {title}:\n\n"
        f"Location: {location}\n"
        f"Developer: {developer}\n"
        f"Completion: {completion}\n"
        f"Price: {price_info}\n\n"
        f"{description}"
    )

    # Clean the reply
    cleaned_reply = clean_reply_text(reply)

    return {
        "reply": cleaned_reply,
        "properties": properties_min[:1],
        "properties_full": properties_full[:1],
        "property_images": images,
        "total": 1,
        "filters_used": {"intent": "PROPERTY", "search_type": "specific", "property_name": property_name},
    }


def handle_property_query(query_text: str) -> Dict[str, Any]:
    # Specific property first
    if is_property_specific_query(query_text):
        return handle_specific_property_query(query_text)

    filters = parse_query_to_filters(query_text)

    if filters.get("foreign_currency"):
        amount = filters.get("amount")
        currency = filters.get("currency")
        return {
            "reply": (
                f"You specified {amount} {currency}. For accurate Dubai property search, "
                f"please convert to AED and try again (example: properties under 2M AED)."
            ),
            "properties": [],
            "properties_full": [],
            "total": 0,
            "filters_used": {**filters, "intent": "PROPERTY", "currency_warning": True},
        }

    # Default: Dubai (general search) — supports: "Show me properties in Dubai."
    filters.setdefault("search_query", "dubai")
    filters["page"] = 1
    filters["per_page"] = min(15, int(filters.get("desired_count", 15)))

    properties_min, properties_full = search_properties(filters)

    # Client-side filtering (tighten results reliably)
    def _ci_contains(hay: str, needle: str) -> bool:
        return needle.lower() in (hay or "").lower()

    # Developer filter
    dev_filter = (filters.get("developer_name") or "").strip()
    if dev_filter:
        keep_ids = []
        for p in properties_full:
            if _ci_contains(str(p.get("developer") or ""), dev_filter):
                keep_ids.append(p.get("id"))
        properties_full = [p for p in properties_full if p.get("id") in keep_ids]
        properties_min = [p for p in properties_min if p.get("id") in keep_ids]

    # Location filter: if specific (not generic)
    loc_filter = (filters.get("search_query") or "").strip()
    if loc_filter and loc_filter.lower() not in GENERIC_LOCS:
        keep_ids = []
        for p in properties_full:
            if _ci_contains(str(p.get("area") or p.get("location") or ""), loc_filter):
                keep_ids.append(p.get("id"))
        if keep_ids:
            properties_full = [p for p in properties_full if p.get("id") in keep_ids]
            properties_min = [p for p in properties_min if p.get("id") in keep_ids]

    # Bedrooms fallback filter (if API is broad)
    bed_filter = filters.get("unit_bedrooms")
    if bed_filter:
        desired = _norm(str(bed_filter))
        keep_ids = []
        for p in properties_full:
            candidates = [
                p.get("unit_bedrooms"),
                p.get("bedrooms"),
                p.get("bedroom"),
                p.get("unit_bedroom"),
            ]
            hit = False
            for c in candidates:
                if c is None:
                    continue
                c_norm = _norm(str(c))
                if desired == "studio":
                    if "studio" in c_norm:
                        hit = True
                else:
                    m = re.search(r"(\d+)", desired)
                    if m and m.group(1) in c_norm:
                        hit = True
            if hit:
                keep_ids.append(p.get("id"))
        if keep_ids:
            properties_full = [p for p in properties_full if p.get("id") in keep_ids]
            properties_min = [p for p in properties_min if p.get("id") in keep_ids]

    total = len(properties_min)

    if total == 0:
        return {
            "reply": (
                "Sorry - I can't find any properties that match those criteria right now. "
                "Try changing the budget, area, property type, or bedrooms "
                "(example: '2 bed apartment in Dubai Marina under 2M AED')."
            ),
            "properties": [],
            "properties_full": [],
            "total": 0,
            "filters_used": {**filters, "intent": "PROPERTY"},
        }

    show_n = min(int(filters.get("desired_count", 15)), total)
    loc = str(filters.get("search_query", "Dubai")).title()

    reply = (
        f"I found {show_n} properties in {loc} that match your criteria. "
        f"Please review the options below."
    )

    # Clean the reply
    cleaned_reply = clean_reply_text(reply)

    return {
        "reply": cleaned_reply,
        "properties": properties_min[:show_n],
        "properties_full": properties_full[:show_n],
        "total": total,
        "filters_used": {**filters, "intent": "PROPERTY"},
    }


# ============================================================
# Endpoints
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok", "service": "marrfa-ai"}


@app.post("/chat")
def chat(req: ChatRequest):
    query = (req.query or "").strip()
    intent_obj = classify_intent_hybrid(query)
    intent = intent_obj["intent"]

    if intent == "GREETING":
        return JSONResponse(content={
            "reply": (
                "Welcome to Marrfa AI. I can help with Marrfa information, policies, blog insights, "
                "and Dubai property listings. How may I help you today?"
            ),
            "route": "greeting",
            "intent": "GREETING",
            "properties": [],
            "properties_full": [],
            "total": 0,
            "sources": [],
            "filters_used": {"intent": "GREETING", "method": intent_obj.get("method")},
        })

    if intent == "PROPERTY":
        out = handle_property_query(query)
        # Clean the reply text
        cleaned_reply = clean_reply_text(out.get("reply", ""))
        return JSONResponse(content={
            "reply": cleaned_reply,
            "route": "property",
            "intent": "PROPERTY",
            "properties": out.get("properties", []),
            "properties_full": out.get("properties_full", []),
            "property_images": out.get("property_images", []),
            "total": out.get("total", 0),
            "sources": [],
            "filters_used": out.get("filters_used", {}),
        })

    # BLOG / COMPANY / POLICY -> RAG KB
    rag_out = rag_answer.answer(query)
    reply = rag_out.get("answer", "") or ""
    if not reply.strip():
        reply = "I couldn't find information for that query. Please try rephrasing."

    # Clean the RAG reply
    cleaned_reply = clean_reply_text(reply)

    return JSONResponse(content={
        "reply": cleaned_reply,
        "route": rag_out.get("route", "rag"),
        "intent": intent,
        "properties": [],
        "properties_full": [],
        "total": 0,
        "sources": rag_out.get("sources", []),
        "filters_used": {"intent": intent, "method": intent_obj.get("method")},
    })


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    query = (req.query or "").strip()
    intent_obj = classify_intent_hybrid(query)
    intent = intent_obj["intent"]

    def gen() -> Generator[str, None, None]:
        yield sse_event({"type": "start", "intent": intent, "query": query}, event="start")
        yield sse_event({"type": "loading", "message": "Processing your request..."}, event="loading")

        if intent == "GREETING":
            final = {
                "reply": (
                    "Welcome to Marrfa AI. I can help with Marrfa information, policies, blog insights, "
                    "and Dubai property listings. How may I help you today?"
                ),
                "route": "greeting",
                "intent": "GREETING",
                "properties": [],
                "properties_full": [],
                "total": 0,
                "sources": [],
                "filters_used": {"intent": "GREETING", "method": intent_obj.get("method")},
            }
            yield sse_event({"type": "final", **final}, event="final")
            yield sse_event({"type": "done"}, event="done")
            return

        if intent == "PROPERTY":
            yield sse_event({"type": "loading", "message": "Searching properties..."}, event="loading")
            out = handle_property_query(query)
            # Clean the reply text
            cleaned_reply = clean_reply_text(out.get("reply", ""))
            final = {
                "reply": cleaned_reply,
                "route": "property",
                "intent": "PROPERTY",
                "properties": out.get("properties", []),
                "properties_full": out.get("properties_full", []),
                "property_images": out.get("property_images", []),
                "total": out.get("total", 0),
                "sources": [],
                "filters_used": out.get("filters_used", {}),
            }
            yield sse_event({"type": "final", **final}, event="final")
            yield sse_event({"type": "done"}, event="done")
            return

        yield sse_event({"type": "loading", "message": "Searching knowledge base..."}, event="loading")

        try:
            meta, stream_gen = rag_answer.answer_stream(query)
            yield sse_event({"type": "content_start"}, event="content_start")

            full_text = ""
            chunk_count = 0
            for chunk in stream_gen:
                if chunk:
                    full_text += chunk
                    chunk_count += 1
                    yield sse_event({"type": "delta", "delta": chunk}, event="delta")

            if chunk_count == 0 or not full_text.strip():
                full_text = "I couldn't find information for that query. Please try rephrasing."

            # Clean the accumulated text
            cleaned_text = clean_reply_text(full_text)

            final = {
                "reply": cleaned_text,
                "route": meta.get("route", "rag"),
                "intent": intent,
                "properties": [],
                "properties_full": [],
                "total": 0,
                "sources": meta.get("sources", []),
                "filters_used": {"intent": intent, "method": intent_obj.get("method")},
            }
            yield sse_event({"type": "final", **final}, event="final")
        except Exception as e:
            msg = f"An error occurred while processing your request: {str(e)}"
            yield sse_event({"type": "error", "message": msg}, event="error")
            final = {
                "reply": msg,
                "route": "error",
                "intent": intent,
                "properties": [],
                "properties_full": [],
                "total": 0,
                "sources": [],
                "filters_used": {"intent": intent, "method": intent_obj.get("method")},
            }
            yield sse_event({"type": "final", **final}, event="final")
        finally:
            yield sse_event({"type": "done"}, event="done")

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )