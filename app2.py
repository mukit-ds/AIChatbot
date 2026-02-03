import os
import re
import json
import datetime
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


def clean_reply_text(text: str) -> str:
    """Sanitize model output to professional plain text (removes markdown symbols)."""
    if not text:
        return ""

    t = str(text)

    # Remove markdown headings like ### Title
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.MULTILINE)

    # Remove bold/italic markers
    t = t.replace("**", "").replace("__", "")
    t = t.replace("`", "")
    t = re.sub(r"(?<!\w)\*(?!\s)", "", t)
    t = re.sub(r"(?<!\w)_(?!\s)", "", t)

    # Convert leading markdown bullets to hyphen bullets
    t = re.sub(r"^\s*\*\s+", "- ", t, flags=re.MULTILINE)

    # Collapse excessive blank lines
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def is_greeting_query(query_text: str) -> bool:
    """Return True only if the message is essentially a greeting."""
    q = (query_text or "").lower().strip()
    if not q:
        return False

    # Tokenize to avoid substring bugs: "hi" in "highest", "yo" in "your"
    tokens = re.findall(r"[a-z']+", q)
    if not tokens:
        return False

    joined = " ".join(tokens)
    greeting_words = {"hello", "hi", "hey", "hiya", "hola", "bonjour", "sup", "yo"}
    greeting_phrases = {
        "good morning", "good afternoon", "good evening",
        "how are you", "how's it going", "what's up",
        "hello there", "hey there", "hi there"
    }

    if any(p in joined for p in greeting_phrases):
        return True

    # Only treat as greeting if it's short and starts with a greeting word
    if len(tokens) <= 3 and tokens[0] in greeting_words:
        return True

    return False


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

        # Add a small set of domain “blog intent” markers
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
    "dubai marina", "palm jumeirah", "downtown dubai", "downtown", "business bay",
    "jumeirah village circle", "jvc", "jumeirah village triangle", "jvt",
    "dubai hills", "dubai hills estate", "dubai creek harbour", "dubai creek harbor",
    "dubai harbour", "dubai harbor", "bluewaters", "bluewaters island", "city walk",
    "al barari", "emirates hills", "arabian ranches", "damac hills", "damac hills 2",
    "dubai south", "mbr city", "meydan", "arjan", "jlt", "jumeirah lake towers",
    "al furjan", "motor city", "sports city", "dubai silicon oasis", "dubailand",
    "international city", "deira", "bur dubai", "jumeirah", "umm suqeim", "al sufouh",
    "dubai media city", "dubai internet city", "al quoz", "the greens", "the views",
    "springs", "meadows", "lakes", "jbr", "jumeirah beach residence",
]


def parse_query_to_filters(query: str) -> Dict[str, Any]:
    q_raw = (query or "").strip()
    q = _norm(q_raw)
    if not q:
        return {}

    filters: Dict[str, Any] = {}

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

    for area in AREAS:
        if area in q:
            filters["search_query"] = area
            break

    if "search_query" not in filters:
        mloc = re.search(r"\b(?:in|at|within|around|near)\s+([a-z][a-z\s\-]{2,40})", q)
        if mloc:
            loc = mloc.group(1).strip()
            loc = re.split(
                r"\b(under|below|over|above|between|with|for|budget|around|approx|about|studio|\d+\s*(?:bed|beds|bedroom|bedrooms|br|room|rooms))\b",
                loc
            )[0].strip()
            if loc:
                filters["search_query"] = loc

    mdev = re.search(r"\b(?:projects?\s+by|by|from)\s+([a-z0-9&\.\- ]{2,40})\b", q)
    if mdev:
        dev = mdev.group(1).strip()
        dev = re.split(r"\b(in|under|below|over|above|between|with)\b", dev)[0].strip()
        if dev:
            filters["developer_name"] = dev

    type_map = {
        "villa": "Villa", "villas": "Villa",
        "townhouse": "Townhouse", "townhouses": "Townhouse",
        # keep original mapping to API values; we handle apartments via unit_blocks filter
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

    # -------------------------
    # Completion / Ready / Off-plan parsing (robust)
    # Supports:
    # - completed in 2026 / completion 2026 / handover 2026 / delivered by 2026
    # - completed before 2026 / after 2026
    # - completed between 2026 and 2028
    # - ready / fully completed / ready to move / delivered / handed over
    # - off-plan / under construction / new launch
    # -------------------------

    # Status flags
    if re.search(
            r"\boff\s*[-]?\s*plan\b|\boffplan\b|\bunder\s+construction\b|\bnew\s+launch\b|\bpre\s*[-]?\s*launch\b|\bupcoming\b|\bin\s+development\b",
            q):
        filters["project_status"] = "OFF_PLAN"
    elif re.search(
            r"\bready\b|\bready\s+to\s+move\b|\bready\s+to\s+sell\b|\bready\s+to\s+rent\b|\bcompleted\b|\bfully\s+completed\b|\bfinish(?:ed)?\b|\bdelivered\b|\bhand\s*[-]?\s*over\b|\bhandover\b",
            q):
        filters["project_status"] = "READY"

    # Completion year/range
    year_re = r"(20\d{2})"
    trigger_re = r"(?:completion|completed|handover|hand\s*[-]?\s*over|delivered|finish(?:ed)?|ready)"
    # between 2026 and 2028 (with trigger anywhere)
    mb = re.search(rf"\bbetween\s+{year_re}\s*(?:and|to|&|-)\s*{year_re}\b", q)
    if mb and re.search(trigger_re, q):
        y1 = int(mb.group(1));
        y2 = int(mb.group(2))
        filters["completion_year_min"] = min(y1, y2)
        filters["completion_year_max"] = max(y1, y2)
    else:
        # before / until
        mmax = re.search(rf"\b(?:before|until|up\s+to|<=)\s*{year_re}\b", q)
        if mmax and re.search(trigger_re, q):
            filters["completion_year_max"] = int(mmax.group(1))
        # after / from
        mmin = re.search(rf"\b(?:after|from|>=)\s*{year_re}\b", q)
        if mmin and re.search(trigger_re, q):
            filters["completion_year_min"] = int(mmin.group(1))

        # exact year near trigger
        mex1 = re.search(rf"\b{trigger_re}\s*(?:in|by|at|on)?\s*{year_re}\b", q)
        mex2 = re.search(rf"\b{year_re}\s*(?:completion|handover|delivery|delivered|ready)\b", q)
        if mex1:
            filters["completion_year"] = int(mex1.group(1))
        elif mex2:
            filters["completion_year"] = int(mex2.group(1))

    # -------------------------
    # Bedrooms / Rooms parsing (robust)
    # Treat "rooms" as "bedrooms"
    # Supports:
    # - 2 bed / 2 bedroom / 2 rooms
    # - studio
    # - more than 2 bedrooms / above 3 rooms
    # - at least 5 bedrooms / minimum 4 rooms
    # - up to 3 bedrooms / max 4 rooms
    # - between 2 and 5 bedrooms
    # - 2+ bedrooms / 3+ rooms
    # -------------------------
    if re.search(r"\bstudio\b", q):
        filters["unit_bedrooms"] = "Studio"
        filters["bedrooms_min"] = 0
        filters["bedrooms_max"] = 0
    else:
        bed_token = r"(?:bed|beds|bedroom|bedrooms|br|room|rooms)"

        # between 2 and 5 bedrooms / rooms
        mbetween = re.search(rf"\bbetween\s+(\d+)\s*(?:and|to|&|-)\s*(\d+)\s*{bed_token}\b", q)
        if mbetween:
            a = int(mbetween.group(1))
            b = int(mbetween.group(2))
            lo, hi = (a, b) if a <= b else (b, a)
            filters["bedrooms_min"] = lo
            filters["bedrooms_max"] = hi

        # 2+ bedrooms / 2+ rooms
        if "bedrooms_min" not in filters and "bedrooms_max" not in filters:
            mplus = re.search(rf"\b(\d+)\s*\+\s*{bed_token}\b", q)
            if mplus:
                filters["bedrooms_min"] = int(mplus.group(1))

        # min / at least / >= / more than / above / over
        if "bedrooms_min" not in filters:
            mmin = re.search(rf"\b(?:min(?:imum)?|at\s+least|>=|more\s+than|over|above)\s*(\d+)\s*{bed_token}\b", q)
            if mmin:
                n = int(mmin.group(1))
                # "more than 2" means 3+
                if re.search(r"\b(more\s+than|over|above)\b", mmin.group(0)):
                    n += 1
                filters["bedrooms_min"] = n

        # max / up to / <= / less than / under / below
        if "bedrooms_max" not in filters:
            mmax = re.search(rf"\b(?:max(?:imum)?|up\s+to|<=|less\s+than|under|below)\s*(\d+)\s*{bed_token}\b", q)
            if mmax:
                n = int(mmax.group(1))
                # "less than 3" means <= 2
                if re.search(r"\b(less\s+than|under|below)\b", mmax.group(0)):
                    n = max(0, n - 1)
                filters["bedrooms_max"] = n

        # exact: 2 bed / 2 rooms
        if "bedrooms_min" not in filters and "bedrooms_max" not in filters:
            mexact = re.search(rf"\b(\d+)\s*{bed_token}\b", q)
            if mexact:
                n = int(mexact.group(1))
                if 0 <= n <= 10:
                    filters["unit_bedrooms"] = f"{n} bedroom" if n != 0 else "Studio"
                    filters["bedrooms_min"] = n
                    filters["bedrooms_max"] = n

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

    # -------------------------
    # Completion / Status parsing (ready / off-plan / year)
    # -------------------------
    completion_keywords = r"(?:complete|completed|completion|handover|hand-over|deliver|delivered|delivery|finish|finished|ready)"

    # between 2026 and 2028
    mbetween = re.search(r"\bbetween\s+(19\d{2}|20\d{2}|21\d{2})\s*(?:and|to|&|-)\s*(19\d{2}|20\d{2}|21\d{2})\b", q)
    if mbetween and re.search(completion_keywords, q):
        a, b = int(mbetween.group(1)), int(mbetween.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        filters["completion_year_min"] = lo
        filters["completion_year_max"] = hi

    # completed at/in/by 2026 (or any year mentioned alongside completion keywords)
    myear = re.search(r"\b(19\d{2}|20\d{2}|21\d{2})\b", q)
    if myear and re.search(completion_keywords, q):
        y = int(myear.group(1))
        filters.setdefault("completion_year_min", y)
        filters.setdefault("completion_year_max", y)

    # before / after
    mbefore = re.search(r"\b(?:before|earlier\s+than|until|up\s+to)\s+(19\d{2}|20\d{2}|21\d{2})\b", q)
    if mbefore and re.search(completion_keywords, q):
        filters["completion_year_max"] = int(mbefore.group(1))

    mafter = re.search(r"\b(?:after|later\s+than|from)\s+(19\d{2}|20\d{2}|21\d{2})\b", q)
    if mafter and re.search(completion_keywords, q):
        filters["completion_year_min"] = int(mafter.group(1))

    # Ready vs Off-plan intent
    if re.search(r"\b(off[-\s]?plan|offplan|under\s+construction|new\s+launch|pre[-\s]?launch|upcoming|in\s+development)\b", q):
        filters["status_intent"] = "OFF_PLAN"
    elif re.search(r"\b(ready\s+to\s+move|ready\s+to\s+sell|ready\s+to\s+rent|fully\s+completed|ready|delivered|handed\s+over|hand\s+over|completed|finished)\b", q):
        filters["status_intent"] = "READY"

    # -------------------------
    # Budget parsing (robust)
    # -------------------------

    # If user gives foreign currency, keep your existing behavior
    mcur = re.search(r"(\d+(?:\.\d+)?)\s*(usd|eur|gbp|inr|sar|qar|omr|kwd|bhd)\b", q, re.IGNORECASE)
    if mcur:
        return {"foreign_currency": True, "amount": mcur.group(1), "currency": mcur.group(2).upper()}

    # Clean punctuation noise: "at 5M,," -> "at 5m"
    cleaned = q.lower()
    cleaned = cleaned.replace(",", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # If query includes completion filters, don't treat 4-digit years as AED budgets
    if ("completion_year_min" in filters or "completion_year_max" in filters or filters.get("status_intent")):
        cleaned = re.sub(r"\b(completed|completion|handover|hand\s*over|delivered|delivery|ready|finished)\b\s*(?:in|at|by|on|for)?\s*(19\d{2}|20\d{2}|21\d{2})\b", " ", cleaned)
        cleaned = re.sub(r"\b(?:in|at|by|on|for)\s*(19\d{2}|20\d{2}|21\d{2})\b", " ", cleaned)
        cleaned = re.sub(r"\bbetween\s+(19\d{2}|20\d{2}|21\d{2})\s*(?:and|to|&|-)\s*(19\d{2}|20\d{2}|21\d{2})\b", " ", cleaned)
        cleaned = re.sub(r"\b(before|after|until|up\s+to|from)\s+(19\d{2}|20\d{2}|21\d{2})\b", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Remove bedroom mentions so they don't confuse price parsing
    cleaned = re.sub(r"\bbetween\s+\d+\s*(?:and|to|&|-)\s*\d+\s*(bed|beds|bedroom|bedrooms|br|room|rooms)\b", " ",
                     cleaned)
    cleaned = re.sub(
        r"\b(?:min(?:imum)?|at\s+least|>=|more\s+than|over|above|max(?:imum)?|up\s+to|<=|less\s+than|under|below)\s*\d+\s*(bed|beds|bedroom|bedrooms|br|room|rooms)\b",
        " ", cleaned)
    cleaned = re.sub(r"\b\d+\s*\+\s*(bed|beds|bedroom|bedrooms|br|room|rooms)\b", " ", cleaned)
    cleaned = re.sub(r"\b\d+\s*(bed|beds|bedroom|bedrooms|br|room|rooms)\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # number + optional unit + optional AED tokens
    # examples matched:
    # 2m, 2.5 m, 2000k, 2 million, 2m aed, 2m dhs
    amt_re = r"(\d+(?:\.\d+)?)\s*(k|m|b|thousand|million|billion)?\s*(?:aed|dhs|dirham|dirhams)?"

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

    # 1) BETWEEN patterns (supports: between 2 & 5m, between 2-5m, between 2 to 5m)
    m = re.search(rf"\bbetween\s+{amt_re}\s*(?:and|to|&|-)\s*{amt_re}\b", cleaned)
    if m:
        low = _parse_amount_token(m.group(1), m.group(2) or "")
        high = _parse_amount_token(m.group(3), m.group(4) or "")
        if low is not None:
            filters["unit_price_from"] = low
        if high is not None:
            filters["unit_price_to"] = high
        return filters

    # 2) RANGE patterns like "2-5m", "2 to 5m", "2 & 5m", "from 2m to 5m"
    m = re.search(rf"\bfrom\s+{amt_re}\s*(?:to|-|&|and)\s*{amt_re}\b", cleaned)
    if not m:
        m = re.search(rf"\b{amt_re}\s*(?:to|-|&|and)\s*{amt_re}\b", cleaned)
    if m:
        low = _parse_amount_token(m.group(1), m.group(2) or "")
        high = _parse_amount_token(m.group(3), m.group(4) or "")
        if low is not None and high is not None:
            filters["unit_price_from"] = low
            filters["unit_price_to"] = high
            return filters

    # 3) UNDER / BELOW / LESS THAN / UP TO
    m = re.search(rf"\b(under|below|less than|upto|up to)\s+{amt_re}\b", cleaned)
    if m:
        amt = _parse_amount_token(m.group(2), m.group(3) or "")
        if amt is not None:
            filters["unit_price_to"] = amt
            return filters

    # 4) ABOVE / OVER / MORE THAN / GREATER THAN
    m = re.search(rf"\b(over|above|more than|greater than)\s+{amt_re}\b", cleaned)
    if m:
        amt = _parse_amount_token(m.group(2), m.group(3) or "")
        if amt is not None:
            filters["unit_price_from"] = amt
            return filters

    # 5) Single-budget phrases: "in 2m", "at 5m", "for 3m", "budget 2m"
    m = re.search(rf"\b(in|at|for|around|about|approx|approximately|budget|price)\s+{amt_re}\b", cleaned)
    if m:
        num_s = m.group(2)
        unit_s = (m.group(3) or "")
        # Skip 4-digit years like "at 2026" when completion filters exist and no unit (k/m/b)
        if (unit_s == "") and re.fullmatch(r"(19\d{2}|20\d{2}|21\d{2})", num_s) and ("completion_year_min" in filters or "completion_year_max" in filters or filters.get("status_intent")):
            pass
        else:
            amt = _parse_amount_token(num_s, unit_s)
            if amt is not None:
                filters["unit_price_to"] = amt
                return filters

    # 6) Bare amount at end: "apartments 2m", "villas 10m"
    m = re.search(rf"(?:^|\s){amt_re}(?:\s|$)", cleaned)
    if m:
        num_s = m.group(1)
        unit_s = (m.group(2) or "")
        if (unit_s == "") and re.fullmatch(r"(19\d{2}|20\d{2}|21\d{2})", num_s) and ("completion_year_min" in filters or "completion_year_max" in filters or filters.get("status_intent")):
            pass
        else:
            amt = _parse_amount_token(num_s, unit_s)
            if amt is not None:
                filters.setdefault("unit_price_to", amt)

    return filters


def is_property_specific_query(query: str) -> bool:
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

    search_markers = {
        "property", "properties", "listing", "listings",
        "apartment", "apartments", "flat", "flats",
        "villa", "villas", "townhouse", "townhouses",
        "penthouse", "penthouses", "studio", "commercial",
        "in", "at", "near", "within", "around",
        "under", "below", "less than", "over", "above", "between",
        "budget", "aed", "dhs", "dirham", "million", "k", "m",
        "bed", "beds", "bedroom", "bedrooms", "br", "room", "rooms",
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
    q = (query_text or "").strip()
    if not q:
        return False
    qq = q.lower()

    if re.search(r"\b(show\s+me|find|search|list|give\s+me)\b", qq):
        return True

    INFO_PATTERNS = [
        "what is", "what are", "how to", "guide", "impact", "effect", "affect", "influence",
        "explain", "why", "pros", "cons", "benefits", "rules", "tax", "visa", "golden visa",
        "expo", "vision 2040", "rental returns", "roi", "yield", "yields"
    ]
    info_like = any(p in qq for p in INFO_PATTERNS)

    try:
        filters = parse_query_to_filters(q)
    except Exception:
        filters = {}

    if filters.get("desired_count"):
        return True
    if filters.get("unit_price_from") is not None or filters.get("unit_price_to") is not None:
        return True
    if filters.get("unit_bedrooms"):
        return True
    if filters.get("bedrooms_min") is not None or filters.get("bedrooms_max") is not None:
        return True
    if filters.get("completion_year") is not None or filters.get("completion_year_min") is not None or filters.get(
            "completion_year_max") is not None:
        return True
    if filters.get("project_status"):
        return True
    if filters.get("unit_types") and not info_like:
        return True
    if (filters.get("developer_name") or "").strip():
        return True

    loc = (filters.get("search_query") or "").strip().lower()
    if loc and (loc not in GENERIC_LOCS) and (not info_like):
        return True

    if re.search(r"\b(under|below|over|above|between|budget|aed|dhs|dirham|million|\bk\b|\bm\b)\b", qq):
        return True
    if re.search(r"\b(studio|bed|beds|bedroom|bedrooms|br|room|rooms)\b", qq):
        return True

    if "rental return" in q or "rental returns" in q or "yield" in q or "roi" in q:
        return False

    return False


def looks_like_blog_question(query_text: str) -> bool:
    qq = _norm(query_text)
    if not qq:
        return False

    for t in BLOG_TITLE_SET:
        if t and t in qq:
            return True

    for ph in BLOG_PHRASES_FROM_DOCS:
        if ph and ph in qq:
            return True

    toks = set(_tokenize(qq))
    if toks and (len(toks & BLOG_KEYWORDS_FROM_DOCS) >= 2):
        return True

    for m in BLOG_INFO_MARKERS:
        if m in qq:
            return True

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

    if is_greeting_query(query_text):
        return {"intent": "GREETING", "method": "keyword"}

    if any(w in q for w in POLICY_KEYWORDS):
        return {"intent": "POLICY", "method": "keyword"}

    if any(w in q for w in COMPANY_KEYWORDS):
        return {"intent": "COMPANY", "method": "keyword"}

    if is_property_specific_query(query_text):
        return {"intent": "PROPERTY", "method": "property_specific"}

    if looks_like_blog_question(query_text) and not is_property_search_query(query_text):
        return {"intent": "BLOG", "method": "hard_blog"}

    if is_property_search_query(query_text):
        return {"intent": "PROPERTY", "method": "hard_property_search"}

    rr = llm_route_query(query_text)

    if is_property_search_query(query_text) and rr.get("route") != "PROPERTY":
        return {"intent": "PROPERTY", "method": f"llm_override_property({rr.get('reason', '')})"}

    route = rr.get("route", "BLOG")
    conf = rr.get("confidence", 0.0)

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


# ----------------------------
# Property details & unit_blocks helpers
# ----------------------------
_details_cache: Dict[int, Dict[str, Any]] = {}


def fetch_property_details(property_id: int) -> Dict[str, Any]:
    if property_id in _details_cache:
        return _details_cache[property_id]

    try:
        resp = _session.get(f"{MARRFA_PROPERTIES_URL.rstrip('/')}/{property_id}", timeout=12)
        resp.raise_for_status()
        data = resp.json() or {}
    except Exception:
        data = {}

    _details_cache[property_id] = data
    return data


def _canon_type(s: Any) -> str:
    t = _norm(str(s or ""))
    if t.endswith("s") and len(t) > 3:
        t = t[:-1]
    return t


def unit_blocks_has_type(unit_blocks: Any, requested_types: List[str]) -> bool:
    if not unit_blocks or not isinstance(unit_blocks, list):
        return False

    wanted = {_canon_type(x) for x in (requested_types or []) if x}
    if not wanted:
        return False

    for b in unit_blocks:
        if not isinstance(b, dict):
            continue
        cand = []
        if b.get("normalized_type"):
            cand.append(b.get("normalized_type"))
        if b.get("unit_type"):
            cand.append(b.get("unit_type"))
        if b.get("type"):
            cand.append(b.get("type"))
        for c in cand:
            if _canon_type(c) in wanted:
                return True

    return False


def requested_unit_block_types(filters: Dict[str, Any], query_text: str) -> List[str]:
    q = _norm(query_text)
    if re.search(r"\b(apartment|apartments|flat|flats|studio)\b", q):
        return ["apartment"]
    if re.search(r"\b(villa|villas)\b", q):
        return ["villa"]
    if re.search(r"\b(penthouse|penthouses)\b", q):
        return ["penthouse"]

    ut = filters.get("unit_types")
    if isinstance(ut, list) and ut:
        u = _norm(str(ut[0]))
        if u == "apartment":
            return ["apartment"]
        if u == "villa":
            return ["villa"]
        if u == "penthouse":
            return ["penthouse"]
    return []


# ============================================================
# ✅ Added: strict unit_blocks budget/bedroom helpers
# ============================================================
def _to_float(x: Any) -> Optional[float]:
    if x is None or x == "" or x == 0 or x == "0":
        return None
    try:
        return float(x)
    except Exception:
        return None


def _parse_bed_range(filters: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """
    Returns (bedrooms_min, bedrooms_max) where:
      - Studio => (0, 0)
      - Exact N => (N, N)
      - Range   => (min, max)
      - Only min => (min, None)
      - Only max => (None, max)
    """
    # Studio handling
    ub = filters.get("unit_bedrooms")
    if isinstance(ub, str) and "studio" in ub.lower():
        return 0, 0

    bmin = filters.get("bedrooms_min")
    bmax = filters.get("bedrooms_max")

    def _to_int(x: Any) -> Optional[int]:
        try:
            if x is None or x == "":
                return None
            return int(float(x))
        except Exception:
            return None

    bmin_i = _to_int(bmin)
    bmax_i = _to_int(bmax)

    if bmin_i is None and bmax_i is None:
        # fallback to unit_bedrooms if older callers set only that
        if isinstance(ub, str):
            m = re.search(r"(\d+)", ub)
            if m:
                n = _to_int(m.group(1))
                if n is not None:
                    return n, n
        return None, None

    return bmin_i, bmax_i


def _block_matches_bedrooms(block: Dict[str, Any], bed_min: Optional[int], bed_max: Optional[int]) -> bool:
    if bed_min is None and bed_max is None:
        return True

    b = block.get("bedrooms")
    if b is None:
        b = block.get("unit_bedrooms")
    if b is None:
        b = block.get("units_bedrooms")

    # Studio blocks sometimes store 0 or "studio"
    s = str(b or "").strip().lower()
    if "studio" in s:
        b_i = 0
    else:
        m = re.search(r"(\d+)", s)
        if not m:
            return False
        try:
            b_i = int(m.group(1))
        except Exception:
            return False

    if bed_min is not None and b_i < bed_min:
        return False
    if bed_max is not None and b_i > bed_max:
        return False
    return True


def _block_type(block: Dict[str, Any]) -> str:
    return _canon_type(block.get("normalized_type") or block.get("unit_type") or "")


def _block_price_range_aed(block: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    lo = _to_float(block.get("units_price_from_aed"))
    hi = _to_float(block.get("units_price_to_aed"))
    if lo is None and hi is None:
        lo = _to_float(block.get("units_price_from"))
        hi = _to_float(block.get("units_price_to"))
    if lo is None and hi is not None:
        lo = hi
    if hi is None and lo is not None:
        hi = lo
    return lo, hi


def _block_matches_budget(
        lo: Optional[float],
        hi: Optional[float],
        budget_from: Optional[int],
        budget_to: Optional[int]
) -> bool:
    """
    STRICT budget rules (fixes "wrong number"):
    - under X  => require hi <= X
    - above X  => require lo >= X
    - between A,B => require overlap (any unit in range)
    """
    if budget_from is None and budget_to is None:
        return True
    if lo is None and hi is None:
        return False

    if hi is None:
        hi = lo
    if lo is None:
        lo = hi

    if lo is None or hi is None:
        return False

    # strict under
    if budget_from is None and budget_to is not None:
        return hi <= float(budget_to)

    # strict above
    if budget_from is not None and budget_to is None:
        return lo >= float(budget_from)

    # between => overlap
    if budget_from is not None and budget_to is not None:
        a = float(min(budget_from, budget_to))
        b = float(max(budget_from, budget_to))
        return (lo <= b) and (hi >= a)

    return True


def _best_price_range_from_unit_blocks(
        unit_blocks: Any,
        req_types: List[str],
        bed_min: Optional[int],
        bed_max: Optional[int],
        budget_from: Optional[int],
        budget_to: Optional[int]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute a (min,max) price range from unit_blocks *only for matching blocks*.
    This also allows us to override the displayed card range so it matches the filter.
    """
    if not isinstance(unit_blocks, list) or not unit_blocks:
        return None, None

    wanted = {_canon_type(x) for x in (req_types or []) if x}
    mins: List[float] = []
    maxs: List[float] = []

    for b in unit_blocks:
        if not isinstance(b, dict):
            continue

        if wanted and _block_type(b) not in wanted:
            continue

        if not _block_matches_bedrooms(b, bed_min, bed_max):
            continue

        lo, hi = _block_price_range_aed(b)
        if not _block_matches_budget(lo, hi, budget_from, budget_to):
            continue

        if lo is not None:
            mins.append(lo)
        if hi is not None:
            maxs.append(hi)

    if not mins and not maxs:
        return None, None

    lo = min(mins) if mins else None
    hi = max(maxs) if maxs else None
    return lo, hi


def _minify_property_item(p: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

        return {
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
        }
    except Exception:
        return None


# ============================================================
# Completion year + Ready/Off-plan filtering helpers
# ============================================================
def _extract_completion_year(p: Dict[str, Any]) -> Optional[int]:
    # Prefer explicit field if present
    for k in ("completion_year", "handover_year", "delivery_year"):
        v = p.get(k)
        try:
            if v is not None and str(v).strip():
                y = int(str(v)[:4])
                if 2000 <= y <= 2100:
                    return y
        except Exception:
            pass

    # Try datetime/date fields
    for k in ("completion_datetime", "completion_date", "handover_date", "delivery_date"):
        v = p.get(k)
        if not v:
            continue
        s = str(v)
        m = re.search(r"(20\d{2})", s)
        if m:
            try:
                y = int(m.group(1))
                if 2000 <= y <= 2100:
                    return y
            except Exception:
                pass

    return None


def _matches_completion_filters(p: Dict[str, Any], filters: Dict[str, Any], strict_unknown_year: bool = True) -> bool:
    """Completion year filtering.
    Note: the list endpoint often omits completion fields for many items.
    - strict_unknown_year=True  => if user asked for completion constraints and year is missing, reject.
    - strict_unknown_year=False => if year is missing, keep the item and let detail-level filtering decide.
    """
    y = _extract_completion_year(p)
    if y is None:
        if strict_unknown_year and any(k in filters for k in ("completion_year", "completion_year_min", "completion_year_max")):
            return False
        return True

    if "completion_year" in filters:
        try:
            return int(filters["completion_year"]) == y
        except Exception:
            return True

    y_min = filters.get("completion_year_min")
    y_max = filters.get("completion_year_max")
    try:
        y_min = int(y_min) if y_min is not None else None
    except Exception:
        y_min = None
    try:
        y_max = int(y_max) if y_max is not None else None
    except Exception:
        y_max = None

    if y_min is not None and y < y_min:
        return False
    if y_max is not None and y > y_max:
        return False
    return True


def _norm_status(s: Any) -> str:
    t = _norm(str(s or ""))
    t = t.replace("-", "_").replace(" ", "_")
    t = re.sub(r"_+", "_", t).strip("_")
    return t


def _extract_status_token(p: Dict[str, Any]) -> str:
    # Try multiple keys because API payloads vary
    for k in ("status", "sale_status", "project_status", "property_status", "construction_status"):
        if p.get(k) is not None:
            st = _norm_status(p.get(k))
            if st:
                return st
    return ""


def _matches_status_filters(p: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    want = filters.get("project_status")
    if not want:
        return True

    want = str(want).upper().strip()
    st = _extract_status_token(p)

    # If unknown but user demanded status, fail (strict)
    if not st:
        return False

    # READY bucket
    ready_tokens = {
        "ready", "completed", "complete", "handover", "handed_over", "delivered", "finished",
        "ready_to_move", "ready_to_move_in", "ready_to_sell", "ready_to_rent"
    }
    # OFF-PLAN bucket
    offplan_tokens = {
        "off_plan", "offplan", "under_construction", "underconstruction",
        "new_launch", "newlaunch", "pre_launch", "prelaunch", "upcoming",
        "in_development", "development", "launched"
    }

    if want == "READY":
        return any(tok in st for tok in ready_tokens) or st in ready_tokens
    if want == "OFF_PLAN":
        return any(tok in st for tok in offplan_tokens) or st in offplan_tokens

    return True



# ============================================================
# Completion year + Ready/Off-plan filtering helpers
# ============================================================
def _extract_completion_year(p: Dict[str, Any]) -> Optional[int]:
    """
    Best-effort extraction of completion/handover year from both list payloads and /properties/{id} payloads.
    """
    if not isinstance(p, dict):
        return None

    # explicit year fields
    for k in ("completion_year", "handover_year", "delivery_year", "completionYear", "handoverYear", "deliveryYear"):
        v = p.get(k)
        try:
            if v is not None and str(v).strip():
                y = int(str(v)[:4])
                if 1900 <= y <= 2100:
                    return y
        except Exception:
            pass

    # datetime/date fields
    for k in ("completion_datetime", "completion_date", "handover_date", "delivery_date", "completionDatetime"):
        v = p.get(k)
        if not v:
            continue
        s = str(v)
        m = re.search(r"(19\d{2}|20\d{2}|21\d{2})", s)
        if m:
            try:
                y = int(m.group(1))
                if 1900 <= y <= 2100:
                    return y
            except Exception:
                pass

    return None


def _matches_completion_filters(p: Dict[str, Any], filters: Dict[str, Any], strict_unknown_year: bool = True) -> bool:
    """
    strict_unknown_year:
      - False: if completion year is missing, do NOT reject (useful at list-level)
      - True:  if completion year is missing, reject when user asked for year constraints (detail-level)
    """
    y = _extract_completion_year(p)

    # exact year
    if "completion_year" in filters and filters.get("completion_year") is not None:
        try:
            target = int(filters["completion_year"])
        except Exception:
            return True
        if y is None:
            return (not strict_unknown_year)
        return y == target

    # range
    y_min = filters.get("completion_year_min")
    y_max = filters.get("completion_year_max")
    if y_min is None and y_max is None:
        return True

    if y is None:
        return (not strict_unknown_year)

    try:
        if y_min is not None and y < int(y_min):
            return False
    except Exception:
        pass
    try:
        if y_max is not None and y > int(y_max):
            return False
    except Exception:
        pass
    return True



def _matches_status_filters(p: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """
    Uses filters["project_status"] from parse_query_to_filters.
    If payload has explicit status, use it. Else infer using completion year vs current year.

    READY intent keywords: ready / completed / delivered / handed over
    OFF_PLAN intent keywords: off-plan / under construction / upcoming
    """
    status_intent = (filters.get("project_status") or "").upper()
    if status_intent not in {"READY", "OFF_PLAN"}:
        return True

    # Try explicit fields first
    explicit = _norm(str(p.get("status") or p.get("project_status") or p.get("sale_status") or ""))
    # Common payload variants
    if explicit:
        if status_intent == "READY":
            if any(x in explicit for x in ["ready", "completed", "complete", "delivered", "handover", "handed", "secondary"]):
                return True
        if status_intent == "OFF_PLAN":
            if any(x in explicit for x in ["off", "plan", "construction", "primary", "new", "launch", "upcoming"]):
                return True
        # if explicit exists but doesn't match known tokens, fall back to year inference below

    # Infer using completion year
    y = _extract_completion_year(p)
    if y is None:
        # if asked READY/OFF_PLAN but we can't infer, keep it (avoid over-filtering)
        return True

    current_year = datetime.datetime.now().year
    if status_intent == "READY":
        return y <= current_year
    if status_intent == "OFF_PLAN":
        return y > current_year

    return True


def _extract_any_bedrooms_value(p: Dict[str, Any]) -> Optional[int]:
    """
    Extract bedrooms from either:
      - top-level fields (bedrooms, unit_bedrooms, etc.)
      - unit_blocks bedrooms if present (takes min bedrooms found)
    """
    if not isinstance(p, dict):
        return None

    for k in ("bedrooms", "bedroom", "unit_bedrooms", "unit_bedroom"):
        v = p.get(k)
        try:
            if v is not None and str(v).strip():
                return int(float(v))
        except Exception:
            pass

    blocks = p.get("unit_blocks")
    if isinstance(blocks, list) and blocks:
        vals = []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            for k in ("bedrooms", "unit_bedrooms"):
                v = b.get(k)
                try:
                    if v is not None and str(v).strip():
                        vals.append(int(float(v)))
                except Exception:
                    continue
        if vals:
            return min(vals)

    return None


def _matches_bedroom_range(p: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """
    Applies bedrooms_min/bedrooms_max if present (rooms treated as bedrooms in parser).
    If unit_bedrooms exact is already used, this is usually redundant, but helps for list endpoint payload variance.
    """
    bmin = filters.get("bedrooms_min")
    bmax = filters.get("bedrooms_max")
    if bmin is None and bmax is None:
        return True

    b = _extract_any_bedrooms_value(p)
    if b is None:
        return True  # don't drop if unknown

    try:
        if bmin is not None and b < int(bmin):
            return False
    except Exception:
        pass
    try:
        if bmax is not None and b > int(bmax):
            return False
    except Exception:
        pass

    return True

def search_properties(filters: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    params: Dict[str, Any] = {}

    # The Marrfa list endpoint returns 0 for unit_types=Apartment.
    # We skip passing that and later filter via unit_blocks from /properties/{id}.
    skip_unit_types = False
    ut = filters.get("unit_types")
    if isinstance(ut, list) and len(ut) == 1 and str(ut[0]).strip().lower() == "apartment":
        skip_unit_types = True

    for key in ["search_query", "unit_types", "unit_bedrooms", "unit_price_from", "unit_price_to", "page", "per_page"]:
        if filters.get(key) is None:
            continue
        if key == "unit_types" and skip_unit_types:
            continue
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
    filtered_full: List[Dict[str, Any]] = []

    for p in items:
        if not isinstance(p, dict):
            continue

        # Apply completion/status filters at list-level (fast)
        if not _matches_completion_filters(p, filters, strict_unknown_year=False):
            continue
        if not _matches_status_filters(p, filters):
            continue

        pm = _minify_property_item(p)
        if pm:
            properties_min.append(pm)
            filtered_full.append(p)

    return properties_min, filtered_full


# ============================================================
# ✅ Replaced: collect_properties_by_unit_blocks (strict budget + overrides displayed price range)
# ============================================================
def collect_properties_by_unit_blocks(filters: Dict[str, Any], req_types: List[str], desired: int) -> Tuple[
    List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Scan list pages, fetch /properties/{id}, and enforce:
      - unit type (villa/penthouse/apartment)
      - bedrooms (if requested)
      - budget (STRICT under/above)
    using unit_blocks as source of truth.
    """
    desired = max(1, min(50, int(desired or 15)))

    # --- parse budget ---
    budget_from = filters.get("unit_price_from")
    budget_to = filters.get("unit_price_to")
    try:
        budget_from = int(float(budget_from)) if budget_from is not None else None
    except Exception:
        budget_from = None
    try:
        budget_to = int(float(budget_to)) if budget_to is not None else None
    except Exception:
        budget_to = None

    bed_min, bed_max = _parse_bed_range(filters)

    collected_min: List[Dict[str, Any]] = []
    collected_full: List[Dict[str, Any]] = []
    seen: set[int] = set()

    per_page_scan = 25
    max_pages = 8  # keep faster to reduce timeouts
    max_detail_calls = 160  # hard cap to avoid long requests
    detail_calls = 0

    # IMPORTANT: remove price filters from list scanning (we enforce from unit_blocks)
    scan_filters = dict(filters)
    scan_filters.pop("unit_price_from", None)
    scan_filters.pop("unit_price_to", None)

    for page in range(1, max_pages + 1):
        page_filters = dict(scan_filters)
        page_filters["page"] = page
        page_filters["per_page"] = per_page_scan

        props_min, props_full = search_properties(page_filters)
        if not props_full:
            continue

        for item, pm in zip(props_full, props_min):
            pid = item.get("id")
            try:
                pid_int = int(pid)
            except Exception:
                continue
            if pid_int in seen:
                continue
            seen.add(pid_int)

            if detail_calls >= max_detail_calls:
                return collected_min[:desired], collected_full[:desired]

            detail = fetch_property_details(pid_int) or {}
            detail_calls += 1

            # Apply completion/status constraints using detail first
            basis = detail if isinstance(detail, dict) and detail else item
            if not _matches_completion_filters(basis, filters):
                continue
            if not _matches_status_filters(basis, filters):
                continue
            if not _matches_bedroom_range(basis, filters):
                continue

            # Apply completion/status filters using detail first (strict)
            basis = detail if isinstance(detail, dict) and detail else item
            if not _matches_completion_filters(basis, filters):
                continue
            if not _matches_status_filters(basis, filters):
                continue

            blocks = (detail.get("unit_blocks") or item.get("unit_blocks") or [])
            if not unit_blocks_has_type(blocks, req_types):
                continue

            # STRICT: require at least one matching block that also matches budget+bedrooms
            lo, hi = _best_price_range_from_unit_blocks(blocks, req_types, bed_min, bed_max, budget_from, budget_to)
            if lo is None and hi is None:
                continue

            # Use pm (already minified), but override displayed price range so it matches filter
            out_pm = dict(pm) if isinstance(pm, dict) else (_minify_property_item(item) or {"id": pid_int})
            if lo is not None:
                out_pm["price_from"] = lo
            if hi is not None:
                out_pm["price_to"] = hi

            collected_min.append(out_pm)
            collected_full.append(detail if isinstance(detail, dict) and detail else item)

            if len(collected_min) >= desired:
                return collected_min[:desired], collected_full[:desired]

    return collected_min[:desired], collected_full[:desired]


# ============================================================
# ✅ NEW: FAST candidate-based unit_blocks filtering (prevents timeouts)
# ============================================================
def collect_properties_from_candidates_unit_blocks(
        filters: Dict[str, Any],
        req_types: List[str],
        desired: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    FAST path:
    - Use list endpoint with current filters (including price/location/bed)
    - Then validate unit_blocks by fetching details only for returned items
    This prevents timeouts for "apartments" queries.
    """
    desired = max(1, min(50, int(desired or 15)))

    # Budget parse
    budget_from = filters.get("unit_price_from")
    budget_to = filters.get("unit_price_to")
    try:
        budget_from = int(float(budget_from)) if budget_from is not None else None
    except Exception:
        budget_from = None
    try:
        budget_to = int(float(budget_to)) if budget_to is not None else None
    except Exception:
        budget_to = None

    bed_min, bed_max = _parse_bed_range(filters)

    # 1) get candidates (already narrowed)
    candidates_min, candidates_full = search_properties(filters)
    if not candidates_full:
        return [], []

    collected_min: List[Dict[str, Any]] = []
    collected_full: List[Dict[str, Any]] = []
    seen: set[int] = set()

    # keep it bounded
    max_candidates_to_check = min(len(candidates_full), 60)

    for idx in range(max_candidates_to_check):
        item = candidates_full[idx]
        pm = candidates_min[idx] if idx < len(candidates_min) else (_minify_property_item(item) or {})

        pid = item.get("id") or (pm.get("id") if isinstance(pm, dict) else None)
        try:
            pid_int = int(pid)
        except Exception:
            continue

        if pid_int in seen:
            continue
        seen.add(pid_int)

        detail = fetch_property_details(pid_int) or {}

        # Apply completion/status filters using detail first (strict)
        basis = detail if isinstance(detail, dict) and detail else item
        if not _matches_completion_filters(basis, filters):
            continue
        if not _matches_status_filters(basis, filters):
            continue

        blocks = (detail.get("unit_blocks") or item.get("unit_blocks") or [])

        # Must have requested type
        if not unit_blocks_has_type(blocks, req_types):
            continue

        # Must match bed/budget using unit_blocks truth
        lo, hi = _best_price_range_from_unit_blocks(blocks, req_types, bed_min, bed_max, budget_from, budget_to)
        if lo is None and hi is None:
            continue

        # Override displayed price to matching range
        out_pm = dict(pm) if isinstance(pm, dict) else (_minify_property_item(item) or {"id": pid_int})
        if lo is not None:
            out_pm["price_from"] = lo
        if hi is not None:
            out_pm["price_to"] = hi

        collected_min.append(out_pm)
        collected_full.append(detail if isinstance(detail, dict) and detail else item)

        if len(collected_min) >= desired:
            return collected_min[:desired], collected_full[:desired]

    return collected_min[:desired], collected_full[:desired]


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
            "reply": "I didn’t catch the property name. Please tell me which property you mean.",
            "properties": [],
            "properties_full": [],
            "property_images": [],
            "total": 0,
            "filters_used": {"intent": "PROPERTY", "search_type": "specific"},
        }

    filters = {"search_query": property_name, "page": 1, "per_page": 10}
    properties_min, properties_full = search_properties(filters)

    # If user said "tell me about X apartment/villa/penthouse" (rare), filter by unit_blocks.
    req_types = requested_unit_block_types(filters, query_text)
    if req_types and properties_full:
        filtered_min: List[Dict[str, Any]] = []
        filtered_full: List[Dict[str, Any]] = []
        for pm, pf in zip(properties_min, properties_full):
            pid = pm.get("id") or pf.get("id")
            try:
                pid_int = int(pid)
            except Exception:
                continue
            detail = fetch_property_details(pid_int)
            blocks = detail.get("unit_blocks") or pf.get("unit_blocks") or []
            if unit_blocks_has_type(blocks, req_types):
                filtered_min.append(pm)
                filtered_full.append(detail if detail else pf)
        properties_min, properties_full = filtered_min, filtered_full

    if not properties_min:
        return {
            "reply": (
                f"I couldn’t find details for '{property_name}'. "
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
        price_info = f"AED {price_from:,.0f} – {price_to:,.0f}"
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
        f"• Location: {location}\n"
        f"• Developer: {developer}\n"
        f"• Completion: {completion}\n"
        f"• Price: {price_info}\n\n"
        f"{description}"
    )

    return {
        "reply": reply,
        "properties": [prop],
        "properties_full": [prop_full],
        "property_images": images,
        "total": 1,
        "filters_used": {"intent": "PROPERTY", "search_type": "specific", "property_name": property_name},
    }


def handle_property_query(query_text: str) -> Dict[str, Any]:
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

    # If query is only a unit-type request (apartments/villas/penthouses) with no location,
    # don't force "dubai" (lets API return broader pool). Otherwise keep existing default.
    req_types = requested_unit_block_types(filters, query_text)
    has_loc = bool(filters.get("search_query"))
    has_other_filters = bool(
        filters.get("unit_bedrooms")
        or filters.get("unit_price_from") is not None
        or filters.get("unit_price_to") is not None
        or (filters.get("developer_name") or "").strip()
    )
    if not has_loc and (not has_other_filters) and req_types:
        # no default search_query
        pass
    else:
        filters.setdefault("search_query", "dubai")

    filters["page"] = 1
    filters["per_page"] = min(15, int(filters.get("desired_count", 15)))

    # ✅ UPDATED: fast path first, then fallback scanning
    if req_types:
        properties_min, properties_full = collect_properties_from_candidates_unit_blocks(
            filters, req_types, desired=int(filters.get("per_page", 15))
        )
        if not properties_min:
            properties_min, properties_full = collect_properties_by_unit_blocks(
                filters, req_types, desired=int(filters.get("per_page", 15))
            )
    else:
        properties_min, properties_full = search_properties(filters)

    # Client-side filtering (keep your existing logic)
    def _ci_contains(hay: str, needle: str) -> bool:
        return needle.lower() in (hay or "").lower()

    dev_filter = (filters.get("developer_name") or "").strip()
    if dev_filter and properties_full:
        keep_ids = []
        for p in properties_full:
            if _ci_contains(str(p.get("developer") or ""), dev_filter):
                keep_ids.append(p.get("id"))
        properties_full = [p for p in properties_full if p.get("id") in keep_ids]
        properties_min = [p for p in properties_min if p.get("id") in keep_ids]

    loc_filter = (filters.get("search_query") or "").strip()
    if loc_filter and loc_filter.lower() not in GENERIC_LOCS and properties_full:
        keep_ids = []
        for p in properties_full:
            if _ci_contains(str(p.get("area") or p.get("location") or ""), loc_filter):
                keep_ids.append(p.get("id"))
        if keep_ids:
            properties_full = [p for p in properties_full if p.get("id") in keep_ids]
            properties_min = [p for p in properties_min if p.get("id") in keep_ids]

    bed_filter = filters.get("unit_bedrooms")
    if bed_filter and properties_full:
        desired_b = _norm(str(bed_filter))
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
                if desired_b == "studio":
                    if "studio" in c_norm:
                        hit = True
                else:
                    m = re.search(r"(\d+)", desired_b)
                    if m and m.group(1) in c_norm:
                        hit = True
            if hit:
                keep_ids.append(p.get("id"))
        if keep_ids:
            properties_full = [p for p in properties_full if p.get("id") in keep_ids]
            properties_min = [p for p in properties_min if p.get("id") in keep_ids]


    # -------------------------
    # ✅ Completion year + status (ready/off-plan) + bedroom range filtering (client-side)
    # This is required because the Marrfa list endpoint does NOT reliably filter by completion year/status.
    # We apply it for BOTH req_types and non-req_types paths.
    # -------------------------
    if properties_full:
        keep_ids = []
        for p in properties_full:
            basis = p
            if not _matches_completion_filters(basis, filters):
                continue
            if not _matches_status_filters(basis, filters):
                continue
            if not _matches_bedroom_range(basis, filters):
                continue
            keep_ids.append(p.get("id"))

        if keep_ids:
            properties_full = [p for p in properties_full if p.get("id") in keep_ids]
            properties_min = [p for p in properties_min if p.get("id") in keep_ids]
        else:
            properties_full = []
            properties_min = []
    total = len(properties_min)

    if total == 0:
        return {
            "reply": (
                "Sorry — I can’t find any properties that match those criteria right now. "
                "Try changing the budget, area, property type, or bedrooms "
                "(example: ‘2 bed apartment in Dubai Marina under 2M AED’)."
            ),
            "properties": [],
            "properties_full": [],
            "total": 0,
            "filters_used": {**filters, "intent": "PROPERTY"},
        }

    show_n = min(int(filters.get("desired_count", 15)), total)
    loc = str(filters.get("search_query", "Dubai") or "Dubai").title()

    reply = (
        f"I found {show_n} properties in {loc} that match your criteria. "
        f"Please review the options below."
    )

    return {
        "reply": reply,
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
        if (out.get("total", 0) == 0) and looks_like_blog_question(query):
            rag_out = rag_answer.answer(query)
            reply = clean_reply_text(rag_out.get("answer", "") or "")
            if not reply.strip():
                reply = "I couldn't find that in Marrfa's knowledge base. Please try rephrasing."
            return JSONResponse(content={
                "reply": reply,
                "route": rag_out.get("route", "rag"),
                "intent": "BLOG",
                "properties": [],
                "properties_full": [],
                "total": 0,
                "sources": rag_out.get("sources", []),
                "filters_used": {"intent": "BLOG", "method": intent_obj.get("method")},
            })

        return JSONResponse(content={
            "reply": clean_reply_text(out.get("reply", "")),
            "route": "property",
            "intent": "PROPERTY",
            "properties": out.get("properties", []),
            "properties_full": out.get("properties_full", []),
            "property_images": out.get("property_images", []),
            "total": out.get("total", 0),
            "sources": [],
            "filters_used": out.get("filters_used", {}),
        })

    rag_out = rag_answer.answer(query)
    reply = clean_reply_text(rag_out.get("answer", "") or "")
    if not reply.strip():
        reply = "I couldn’t find information for that query. Please try rephrasing."

    return JSONResponse(content={
        "reply": reply,
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

            if (out.get("total", 0) == 0) and looks_like_blog_question(query):
                yield sse_event({"type": "loading", "message": "Searching blog knowledge..."}, event="loading")
                meta, stream_gen = rag_answer.answer_stream(query)
                yield sse_event({"type": "content_start"}, event="content_start")
                full_text = ""
                for chunk in stream_gen:
                    if chunk:
                        full_text += chunk
                        yield sse_event({"type": "delta", "delta": chunk}, event="delta")
                full_text = clean_reply_text(full_text)
                if not full_text.strip():
                    full_text = "I couldn't find that in Marrfa's knowledge base. Please try rephrasing."
                final = {
                    "reply": full_text,
                    "route": meta.get("route", "rag"),
                    "intent": "BLOG",
                    "properties": [],
                    "properties_full": [],
                    "total": 0,
                    "sources": meta.get("sources", []),
                    "filters_used": {"intent": "BLOG", "method": intent_obj.get("method")},
                }
                yield sse_event({"type": "final", **final}, event="final")
                yield sse_event({"type": "done"}, event="done")
                return

            final = {
                "reply": clean_reply_text(out.get("reply", "")),
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
                full_text = "I couldn’t find information for that query. Please try rephrasing."

            final = {
                "reply": clean_reply_text(full_text),
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
