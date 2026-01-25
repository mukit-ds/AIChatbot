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

app = FastAPI(title="Marrfa AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set frontend domain(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = ""
    session_id: Optional[str] = None
    is_logged_in: bool = False


# ----------------------------
# Intent classification (simple keyword-based)
# ----------------------------
GREETING_PATTERNS = {
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "greetings", "hi there", "hey there", "how are you", "how's it going",
    "what's up", "sup", "yo", "hello there", "hiya", "hola", "bonjour"
}

POLICY_KEYWORDS = {
    "policy", "policies", "terms", "conditions", "privacy", "legal", "refund",
    "compliance", "rules", "regulations", "agreement", "disclaimer", "privacy policy"
}

BLOG_KEYWORDS = {
    "visa", "golden visa", "residency", "blog", "article", "guide", "how to",
    "tips", "investment guide", "vision 2040", "market trends", "true cost",
    "waterfront living", "infrastructure", "taxes", "tax"
}

COMPANY_KEYWORDS = {
    "marrfa", "marfa", "ceo", "founder", "owner", "team", "about us",
    "contact", "address", "phone", "email", "office", "location", "who are you"
}

# In app.py, update the PROPERTY_KEYWORDS
PROPERTY_KEYWORDS = {
    "property", "properties", "apartment", "apartments", "villa", "villas", "townhouse", "townhouses",
    "penthouse", "penthouses", "studio",
    "rent", "rental", "buy", "sale", "listing", "listings",
    "dubai", "marina", "downtown", "business bay", "palm", "jvc", "dubai hills",
    "price", "budget", "aed", "dhs", "dirham", "million", "k", "m",
    "bed", "beds", "bedroom", "bedrooms", "br",
    "ready to move", "off plan", "off-plan", "new launch",
    "investment", "invest", "roi", "high roi", "rental income", "yield",
    "tell me about", "information about", "details about",
    "show me", "find", "search"
}


def is_property_specific_query(query: str) -> bool:
    """Return True only when the user is clearly asking about ONE named project/listing.

    Important: Do NOT treat general search requests like
    'show me apartments in Dubai Marina' as a specific-property query.
    """
    q = (query or "").lower().strip()
    if not q:
        return False

    # Phrases that typically introduce a specific, named property
    property_phrases = [
        "tell me about",
        "tell me more about",
        "information about",
        "details about",
        "what is",
        "what's",
        "describe",
    ]

    # If the remainder contains any of these, it's almost certainly a FILTERED SEARCH, not a named property
    search_markers = {
        "property", "properties", "listing", "listings",
        "apartment", "apartments", "flat", "flats",
        "villa", "villas", "townhouse", "townhouses",
        "penthouse", "penthouses", "studio",
        "in", "at", "near", "within", "around",
        "under", "below", "less than", "over", "above", "between", "to",
        "budget", "aed", "dhs", "dirham", "dirhams", "million", "k", "m",
        "bed", "beds", "bedroom", "bedrooms", "br",
        "ready", "off plan", "off-plan", "new launch",
        "show me", "find", "search",
    }

    for phrase in property_phrases:
        if phrase in q:
            remainder = q.split(phrase, 1)[1].strip()
            remainder = re.sub(r"^[\s:,-]+", "", remainder)
            remainder = re.sub(r"[\?\!\.]+$", "", remainder).strip()

            if not remainder or len(remainder) < 3:
                return False

            # Very long "names" are usually search requests
            if len(remainder) > 60:
                return False

            # If remainder contains typical search/filter markers, it's not a single property name
            low = remainder
            for mk in search_markers:
                if mk in {"less than", "off plan"}:
                    if mk in low:
                        return False
                else:
                    if re.search(rf"\b{re.escape(mk)}\b", low):
                        return False

            return True

    return False



def classify_intent_simple(query_text: str) -> Dict[str, Any]:
    q = (query_text or "").lower().strip()

    if not q:
        return {"intent": "COMPANY", "method": "empty_fallback"}

    # Check for property-specific queries FIRST (before general property keywords)
    if is_property_specific_query(query_text):
        # It's asking about a specific property by name
        return {"intent": "PROPERTY", "method": "property_specific"}

    if any(g in q for g in GREETING_PATTERNS):
        return {"intent": "GREETING", "method": "keyword"}

    if any(w in q for w in POLICY_KEYWORDS):
        return {"intent": "POLICY", "method": "keyword"}

    if any(w in q for w in BLOG_KEYWORDS):
        return {"intent": "BLOG", "method": "keyword"}

    if any(w in q for w in PROPERTY_KEYWORDS):
        return {"intent": "PROPERTY", "method": "keyword"}

    if any(w in q for w in COMPANY_KEYWORDS):
        return {"intent": "COMPANY", "method": "keyword"}

    return {"intent": "COMPANY", "method": "fallback"}


# ----------------------------
# Query -> Filters (property)
# ----------------------------
PROPERTY_TYPES = {
    "villa": "Villa",
    "townhouse": "Townhouse",
    "apartment": "Apartment",
    "flat": "Apartment",
    "penthouse": "Penthouse",
    "duplex": "Duplex",
    "studio": "Studio",
    "plot": "Plot"
}

AREAS = [
    # Popular / high-signal Dubai areas (extend as needed)
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
    "mb r city",
    "mbr city",
    "meydan",
    "arjan",
    "jlt",
    "jumeirah lake towers",
    "al furjan",
    "motor city",
    "sports city",
    "dubai silicon oasis",
    "dubai land",
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
    "dubai marina jbr"
]

def _normalize(s: str) -> str:
    return (s or "").lower().strip()


def _to_aed(amount: str, unit: Optional[str]) -> int:
    n = float(amount)
    if unit == "m":
        return int(n * 1_000_000)
    if unit == "k":
        return int(n * 1_000)
    return int(n)


def parse_query_to_filters(query: str) -> Dict[str, Any]:
    """
    Best-effort parsing from natural language into Marrfa search params.

    Supported (robust):
    - budgets: under/below, above/over, between X and Y, "budget is X", "around X"
    - locations: matches AREAS list; otherwise tries "in <location>"
    - property type: apartment/villa/townhouse/penthouse/studio
    - bedrooms: studio, 1-10 bedrooms (e.g., "2 bed", "3 bedrooms")
    - desired count: "show me 3 properties" -> desired_count=3 (used by handler)
    - developer: "by Emaar", "projects by DAMAC" (filtered client-side in handler)
    """
    q_raw = (query or "").strip()
    q = _normalize(q_raw)
    if not q:
        return {}

    filters: Dict[str, Any] = {}

    # ----------------------------
    # Desired result count
    # ----------------------------
    mcount = re.search(r"\bshow\s+me\s+(\d{1,2})\b", q)
    if not mcount:
        mcount = re.search(r"\b(\d{1,2})\s*(?:properties|listings|homes|options|results)\b", q)
    if mcount:
        try:
            n = int(mcount.group(1))
            if 1 <= n <= 50:
                filters["desired_count"] = n
        except Exception:
            pass

    # ----------------------------
    # Location (area)
    # ----------------------------
    for area in AREAS:
        if area in q:
            filters["search_query"] = area
            break

    if "search_query" not in filters:
        # heuristic "in <location>" extraction
        mloc = re.search(r"\b(?:in|at|within|around|near)\s+([a-z][a-z\s\-]{2,40})", q)
        if mloc:
            loc = mloc.group(1).strip()
            # stop at common delimiters
            loc = re.split(r"\b(under|below|over|above|between|with|for|budget|around|approx|about|studio|\d+\s*bed)\b", loc)[0].strip()
            if loc:
                filters["search_query"] = loc

    # fallback if Abu Dhabi explicitly
    if "search_query" not in filters and "abu dhabi" in q:
        filters["search_query"] = "abu dhabi"

    # ----------------------------
    # Developer (client-side filter)
    # ----------------------------
    mdev = re.search(r"\b(?:by|from)\s+([a-z0-9&\.\- ]{2,40})\b", q)
    if not mdev:
        mdev = re.search(r"\bprojects?\s+by\s+([a-z0-9&\.\- ]{2,40})\b", q)
    if mdev:
        dev = mdev.group(1).strip()
        dev = re.split(r"\b(in|under|below|over|above|between|with)\b", dev)[0].strip()
        if dev and len(dev) >= 2:
            filters["developer_name"] = dev

    # ----------------------------
    # Property type
    # ----------------------------
    # include plurals & common synonyms
    type_map = {
        "villa": "Villa",
        "villas": "Villa",
        "townhouse": "Townhouse",
        "townhouses": "Townhouse",
        "apartment": "Apartment",
        "apartments": "Apartment",
        "flat": "Apartment",
        "flats": "Apartment",
        "penthouse": "Penthouse",
        "penthouses": "Penthouse",
        "studio": "Apartment",  # API uses bedrooms studio; type can remain Apartment
    }
    for k, v in type_map.items():
        if re.search(rf"\b{re.escape(k)}\b", q):
            filters["unit_types"] = [v]
            break

    # ----------------------------
    # Bedrooms
    # ----------------------------
    # NOTE: The Marrfa properties API expects a *single* bedroom label (e.g. "2 bedroom" or "Studio").
    # Passing a CSV list ("2 bedroom,2 bedrooms") can return 0 results.
    # So we pass one canonical value to the API and keep the numeric target for robust client-side filtering.
    if re.search(r"\bstudio\b", q):
        filters["unit_bedrooms"] = "Studio"
        filters["_desired_bedrooms"] = 0
    else:
        mb = re.search(r"\b(\d+)\s*(?:bed|beds|bedroom|bedrooms|br|bdrm|bdrms)\b", q)
        if mb:
            n = int(mb.group(1))
            if 1 <= n <= 10:
                filters["_desired_bedrooms"] = n
                filters["unit_bedrooms"] = f"{n} bedroom"


    # ----------------------------
    # Price parsing (AED)
    # ----------------------------
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
        mult = _word_unit_to_multiplier(unit_s)
        return int(n * mult)

    # Normalize "1 million" => "1 million" handled; also "1m", "1.5m", "500k"
    # Range: between X and Y
    cleaned = q

    # Currency other than AED -> warning (kept as before)
    currency_pattern = r"(\d+(?:\.\d+)?)\s*(usd|eur|gbp|inr|sar|qar|omr|kwd|bhd)\b"
    mcur = re.search(currency_pattern, cleaned, re.IGNORECASE)
    if mcur:
        filters["foreign_currency"] = True
        filters["amount"] = mcur.group(1)
        filters["currency"] = mcur.group(2).upper()
        return filters

    # Remove bedroom tokens when parsing money
    cleaned = re.sub(r"\b\d+\s*(bed|beds|bedroom|bedrooms|br)\b", "", cleaned)

    # Helper regex for amount + optional unit words/suffix
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

    # X to Y / X - Y
    if "unit_price_from" not in filters and "unit_price_to" not in filters:
        m = re.search(rf"\b{amt_re}\s*(?:-|to|–|—)\s*{amt_re}\b", cleaned)
        if m:
            low = _parse_amount_token(m.group(1), m.group(2) or "")
            high = _parse_amount_token(m.group(3), m.group(4) or "")
            if low is not None:
                filters["unit_price_from"] = low
            if high is not None:
                filters["unit_price_to"] = high

    # under/below/less than
    m = re.search(rf"\b(under|below|less than|upto|up to)\s+{amt_re}\b", cleaned)
    if m:
        amt = _parse_amount_token(m.group(2), m.group(3) or "")
        if amt is not None:
            filters["unit_price_to"] = amt

    # above/over/more than
    m = re.search(rf"\b(over|above|more than)\s+{amt_re}\b", cleaned)
    if m:
        amt = _parse_amount_token(m.group(2), m.group(3) or "")
        if amt is not None:
            filters["unit_price_from"] = amt

    # budget is / my budget / around / about / approx
    if "unit_price_from" not in filters and "unit_price_to" not in filters:
        m = re.search(rf"\b(budget\s*(?:is|=)|my\s+budget\s+is|around|about|approx(?:\.|imately)?|near)\s+{amt_re}\b", cleaned)
        if m:
            amt = _parse_amount_token(m.group(2), m.group(3) or "")
            if amt is not None:
                # treat as max budget (most common user expectation)
                filters["unit_price_to"] = amt

    return filters

    cleaned = re.sub(r"\b\d+\s*(bed|beds|bedroom|bedrooms|room|rooms)\b", "", q)

    m = re.search(r"between\s+(\d+(?:\.\d+)?)\s*(m|k)?\s+and\s+(\d+(?:\.\d+)?)\s*(m|k)?", cleaned)
    if m:
        low, lu, high, hu = m.groups()
        filters["unit_price_from"] = _to_aed(low, lu)
        filters["unit_price_to"] = _to_aed(high, hu)

    m = re.search(r"(\d+(?:\.\d+)?)\s*(m|k)?\s*(?:-|to|–|—)\s*(\d+(?:\.\d+)?)\s*(m|k)?", cleaned)
    if m and "unit_price_from" not in filters and "unit_price_to" not in filters:
        low, lu, high, hu = m.groups()
        filters["unit_price_from"] = _to_aed(low, lu)
        filters["unit_price_to"] = _to_aed(high, hu)

    m = re.search(r"(under|below|less than)\s+(\d+(?:\.\d+)?)\s*(m|k)?", cleaned)
    if m:
        _, amt, unit = m.groups()
        filters["unit_price_to"] = _to_aed(amt, unit)

    m = re.search(r"(over|above|more than)\s+(\d+(?:\.\d+)?)\s*(m|k)?", cleaned)
    if m:
        _, amt, unit = m.groups()
        filters["unit_price_from"] = _to_aed(amt, unit)

    # bedrooms
    if "studio" in q:
        filters["unit_bedrooms"] = "Studio"
    else:
        m = re.search(r"(\d+)\s*(bed|beds|bedroom|bedrooms|br|room|rooms)", q)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 10:
                filters["unit_bedrooms"] = f"{n} bedroom"

    # property type
    for k, v in PROPERTY_TYPES.items():
        if k in q:
            filters["unit_types"] = [v]
            break

    return filters


# ----------------------------
# Marrfa property API client
# ----------------------------
MARRFA_PROPERTIES_URL = "https://apiv2.marrfa.com/properties"
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
        resp = _session.get(MARRFA_PROPERTIES_URL, params=params, timeout=10)
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


def handle_property_query(query_text: str) -> Dict[str, Any]:
    # First check if this is a specific property query
    if is_property_specific_query(query_text):
        return handle_specific_property_query(query_text)

    # Otherwise, use the regular property search
    filters = parse_query_to_filters(query_text)

    if filters.get("foreign_currency"):
        amount = filters.get("amount")
        currency = filters.get("currency")
        return {
            "reply": (
                f"You specified {amount} {currency}. For accurate Dubai property search, "
                f"please convert to AED and search again (example: properties under 2M AED)."
            ),
            "properties": [],
            "properties_full": [],
            "total": 0,
            "filters_used": {**{k: v for k, v in filters.items() if not str(k).startswith('_')}, "intent": "PROPERTY", "currency_warning": True},
        }

    filters.setdefault("search_query", "dubai")
    filters["page"] = 1
    filters["per_page"] = min(15, int(filters.get("desired_count", 15)))

    # ----------------------------
    # API search + smart fallbacks
    # ----------------------------
    # The properties API can be strict/picky about some params (especially `unit_types` and `unit_bedrooms`).
    # We try the strict query first, then relax filters if it returns 0 results, and finally apply
    # our own robust client-side filters below.
    strict_filters = dict(filters)
    properties_min, properties_full = search_properties(strict_filters)

    # 1) If nothing found and type was specified, retry WITHOUT type (we'll filter by type locally).
    if not properties_min and strict_filters.get("unit_types"):
        relaxed = dict(strict_filters)
        relaxed.pop("unit_types", None)
        properties_min, properties_full = search_properties(relaxed)
        # keep what user asked for so client-side filter still knows intent
        filters = relaxed
        filters["unit_types"] = strict_filters.get("unit_types")

    # 2) If still nothing and bedrooms was specified, retry WITHOUT bedrooms (we'll filter by bedrooms locally).
    if not properties_min and strict_filters.get("unit_bedrooms"):
        relaxed = dict(filters)
        relaxed.pop("unit_bedrooms", None)
        properties_min, properties_full = search_properties(relaxed)
        filters = relaxed
        filters["unit_bedrooms"] = strict_filters.get("unit_bedrooms")

    # ----------------------------
    # Client-side filtering (in case API search is broad)
    # ----------------------------
    def _ci_contains(hay: str, needle_s: str) -> bool:
        return needle_s.lower() in (hay or "").lower()

    dev_filter = (filters.get("developer_name") or "").strip()
    if dev_filter:
        keep_ids = []
        for p in properties_full:
            if _ci_contains(str(p.get("developer") or ""), dev_filter):
                keep_ids.append(p.get("id"))
        properties_full = [p for p in properties_full if p.get("id") in keep_ids]
        properties_min = [p for p in properties_min if p.get("id") in keep_ids]

    # Ensure location match when we extracted a location phrase
    loc_filter = (filters.get("search_query") or "").strip()
    if loc_filter and loc_filter not in ("dubai", "abu dhabi"):
        keep_ids = []
        for p in properties_full:
            if _ci_contains(str(p.get("area") or p.get("location") or ""), loc_filter):
                keep_ids.append(p.get("id"))
        if keep_ids:
            properties_full = [p for p in properties_full if p.get("id") in keep_ids]
            properties_min = [p for p in properties_min if p.get("id") in keep_ids]

    # Ensure property type match when the user asked for a type (apartment/villa/townhouse/penthouse).
    # We do this client-side because the API's `unit_types` values can vary (e.g., "Attached Villa", etc.).
    requested_types = filters.get("unit_types") or []
    if isinstance(requested_types, str):
        requested_types = [requested_types]
    requested_types = [t.strip().lower() for t in requested_types if str(t).strip()]

    if requested_types:
        # Map our high-level intent to a set of keywords we can match in payload fields.
        type_keywords = {
            "apartment": {"apartment", "apart", "flat", "apt"},
            "villa": {"villa"},
            "townhouse": {"townhouse", "town house"},
            "penthouse": {"penthouse"},
        }

        wanted = set()
        for t in requested_types:
            wanted |= type_keywords.get(t.lower(), {t.lower()})

        keep_ids = []
        for p in properties_full:
            # Try a few likely fields from the Marrfa payload
            candidates = []
            for k in ("unit_types", "unit_type", "property_type", "type", "category", "unitType"):
                v = p.get(k) if isinstance(p, dict) else None
                if v is None:
                    continue
                if isinstance(v, list):
                    candidates.extend([str(x) for x in v])
                else:
                    candidates.append(str(v))

            hay = " ".join(candidates).lower()
            if any(kw in hay for kw in wanted):
                keep_ids.append(p.get("id"))

        if keep_ids:
            properties_full = [p for p in properties_full if p.get("id") in keep_ids]
            properties_min = [p for p in properties_min if p.get("id") in keep_ids]

    # Ensure bedrooms match when we extracted bedrooms
    desired_beds = filters.get("_desired_bedrooms", None)
    if desired_beds is not None:
        keep_ids = []
        for p in properties_full:
            # Try multiple possible fields from API payloads
            candidates = []
            for k in (
                "unit_bedrooms", "unit_bedroom", "bedrooms", "bedroom",
                "unit_bedrooms_text", "unit_bedrooms_label",
                "min_bedrooms", "max_bedrooms", "bedrooms_from", "bedrooms_to",
                "units_bedrooms", "unitBedrooms", "unitBedroomsText"
            ):
                v = p.get(k) if isinstance(p, dict) else None
                if v is not None:
                    candidates.append(v)

            hay = " ".join([str(x) for x in candidates]).lower()

            if desired_beds == 0:
                if "studio" in hay or re.search(r"\b0\b", hay):
                    keep_ids.append(p.get("id"))
            else:
                # match '2 bed', '2 bedroom', '2 bedrooms', etc.
                if re.search(rf"\b{int(desired_beds)}\s*(?:bed|beds|bedroom|bedrooms|br)\b", hay):
                    keep_ids.append(p.get("id"))
                else:
                    # sometimes APIs provide numeric min/max bedroom fields
                    try:
                        mn = int(p.get("min_bedrooms")) if p.get("min_bedrooms") is not None else None
                        mx = int(p.get("max_bedrooms")) if p.get("max_bedrooms") is not None else None
                        if mn is not None and mx is not None and mn <= int(desired_beds) <= mx:
                            keep_ids.append(p.get("id"))
                        elif mn is not None and mx is None and mn == int(desired_beds):
                            keep_ids.append(p.get("id"))
                    except Exception:
                        pass

        if keep_ids:
            properties_full = [p for p in properties_full if p.get("id") in keep_ids]
            properties_min = [p for p in properties_min if p.get("id") in keep_ids]


    total = len(properties_min)


    public_filters = {k: v for k, v in filters.items() if not str(k).startswith('_')}

    if total == 0:
        return {
            "reply": (
                "Sorry — I couldn’t find any properties that match those criteria. "
                "Try changing the budget, area, property type, or bedrooms (for example: ‘2 bed apartment in Dubai Marina under 2M AED’)."
            ),
            "properties": [],
            "properties_full": [],
            "total": 0,
            "filters_used": {**public_filters, "intent": "PROPERTY"},
        }

    loc = str(filters.get("search_query", "Dubai")).title()
    reply = f"I've found {min(int(filters.get('desired_count', 15)), total)} properties available in {loc} that match your criteria. " \
            f"Each listing includes detailed information about the property, including location, " \
            f"developer, completion year, and pricing. Please review the options below."

    return {
        "reply": reply,
        "properties": properties_min[:15],
        "properties_full": properties_full[:15],
        "total": total,
        "filters_used": {**public_filters, "intent": "PROPERTY"},
    }


def handle_specific_property_query(query_text: str) -> Dict[str, Any]:
    """Handle queries asking about a specific property by name"""
    # Extract property name from query
    property_name = extract_property_name(query_text)

    if not property_name:
        return {
            "reply": "I didn't catch the property name. Could you please specify which property you're asking about?",
            "properties": [],
            "properties_full": [],
            "total": 0,
            "filters_used": {"intent": "PROPERTY", "search_type": "specific", "property_name": property_name},
        }

    # Search with the property name as the search query
    filters = {
        "search_query": property_name,
        "page": 1,
        "per_page": 10
    }

    properties_min, properties_full = search_properties(filters)

    # If no results, try a broader search in Dubai
    if not properties_min:
        filters["search_query"] = "dubai"
        properties_min, properties_full = search_properties(filters)

        # Filter for properties whose title contains the property name
        filtered_min = []
        filtered_full = []
        for prop_min, prop_full in zip(properties_min, properties_full):
            prop_title = prop_min.get("title", "").lower()
            if property_name.lower() in prop_title:
                filtered_min.append(prop_min)
                filtered_full.append(prop_full)

        properties_min = filtered_min
        properties_full = filtered_full

    if properties_min:
        # Get the first matching property
        prop = properties_min[0]
        prop_full = properties_full[0] if properties_full else {}

        # Extract multiple images from the property data
        images = []

        # Check for gallery images
        if prop_full.get("gallery_images"):
            gallery = prop_full.get("gallery_images", [])
            if isinstance(gallery, str):
                try:
                    gallery = json.loads(gallery)
                except:
                    gallery = []

            if isinstance(gallery, list):
                for img in gallery:
                    img_url = _extract_url(img)
                    if img_url and img_url not in images:
                        images.append(img_url)
                        if len(images) >= 5:  # Limit to 5 images
                            break

        # Add cover image if not already in list
        cover_url = prop.get("cover_image")
        if cover_url and cover_url not in images and len(images) < 5:
            images.insert(0, cover_url)  # Add cover image as first

        # Add other potential image fields
        image_fields = ["image_url", "thumbnail_url", "featured_image", "property_images"]
        for field in image_fields:
            if len(images) < 5 and prop_full.get(field):
                img_url = _extract_url(prop_full.get(field))
                if img_url and img_url not in images:
                    images.append(img_url)
                    if len(images) >= 5:
                        break

        # Format basic property info
        title = prop.get("title", "Unknown property")
        location = prop.get("location", "Dubai")
        developer = prop.get("developer", "Unknown developer")
        completion = prop.get("completion_year", "N/A")
        price_from = prop.get("price_from")
        price_to = prop.get("price_to")
        currency = prop.get("currency", "AED")

        # Format price
        price_info = "Price not available"
        if price_from is not None and price_to is not None:
            price_info = f"AED {price_from:,.0f} - {price_to:,.0f}"
        elif price_from is not None:
            price_info = f"From AED {price_from:,.0f}"
        elif price_to is not None:
            price_info = f"Up to AED {price_to:,.0f}"

        # Create a descriptive paragraph about the property
        description = prop_full.get("description") or prop_full.get("short_description") or ""

        # If no description in data, create a generic one based on available info
        if not description:
            amenities = []
            if prop_full.get("has_pool"):
                amenities.append("swimming pool")
            if prop_full.get("has_gym"):
                amenities.append("state-of-the-art gym")
            if prop_full.get("has_parking"):
                amenities.append("dedicated parking")
            if prop_full.get("has_security"):
                amenities.append("24/7 security")

            amenity_text = ""
            if amenities:
                amenity_text = f" The property features {', '.join(amenities[:-1])}{' and ' + amenities[-1] if len(amenities) > 1 else amenities[0]}."

            description = f"{title} is a premium residential development located in the prestigious {location} area, developed by {developer}. This property offers modern living spaces with contemporary designs and high-quality finishes.{amenity_text} With completion scheduled for {completion}, it represents an excellent investment opportunity in one of Dubai's most sought-after neighborhoods."

        # Prepare the images data for the frontend
        images_data = images[:5]  # Ensure maximum 5 images

        reply = f"""Here are the details for **{title}**:

• **Location**: {location}
• **Developer**: {developer}
• **Completion**: {completion}
• **Price**: {price_info}

{description}

For more detailed information or to schedule a viewing, please contact our sales team."""

        return {
            "reply": reply,
            "properties": properties_min[:1],  # Just show the specific property
            "properties_full": properties_full[:1],
            "property_images": images_data,  # Add images array to response
            "total": len(properties_min),
            "filters_used": {"intent": "PROPERTY", "search_type": "specific", "property_name": property_name},
        }
    else:
        return {
            "reply": f"I couldn't find detailed information about '{property_name}'. This property might not be in our current listings or the name might be different. Would you like to see similar properties in Dubai?",
            "properties": [],
            "properties_full": [],
            "property_images": [],
            "total": 0,
            "filters_used": {"intent": "PROPERTY", "search_type": "specific", "property_name": property_name},
        }

def extract_property_name(query: str) -> str:
    """Extract property name from query"""
    q = query.lower()

    # Remove common phrases
    phrases_to_remove = [
        "tell me about", "information about", "details about",
        "what is", "where is", "show me", "property", "apartment",
        "villa", "in dubai", "please", "can you", "could you"
    ]

    for phrase in phrases_to_remove:
        q = q.replace(phrase, "")

    # Remove extra spaces and return capitalized
    name = q.strip()
    if name:
        # Capitalize first letter of each word
        return ' '.join(word.capitalize() for word in name.split())
    return ""


# ----------------------------
# SSE helpers ✅ NEW
# ----------------------------
def sse_event(data: Dict[str, Any], event: str = "message") -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "service": "marrfa-ai"}


@app.post("/chat")
def chat(req: ChatRequest):
    query = (req.query or "").strip()
    intent_obj = classify_intent_simple(query)
    intent = intent_obj["intent"]

    if intent == "GREETING":
        return JSONResponse(content={
            "reply": (
                "Welcome to Marrfa AI. I'm here to assist you with information about Marrfa's services, "
                "our leadership team, company policies, and property listings in Dubai. "
                "How may I help you today?"
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
        return JSONResponse(content={
            "reply": out["reply"],
            "route": "property",
            "intent": "PROPERTY",
            "properties": out.get("properties", []),
            "properties_full": out.get("properties_full", []),
            "total": out.get("total", 0),
            "sources": [],
            "filters_used": out.get("filters_used", {}),
        })

    rag_out = rag_answer.answer(query)
    reply = rag_out.get("answer", "") or ""
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


# ✅ NEW: Streaming endpoint (SSE) - Improved for production
@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    query = (req.query or "").strip()
    intent_obj = classify_intent_simple(query)
    intent = intent_obj["intent"]

    def gen() -> Generator[str, None, None]:
        # Send start event immediately
        yield sse_event({
            "type": "start",
            "intent": intent,
            "query": query
        }, event="start")

        # Send loading event IMMEDIATELY for all queries
        yield sse_event({
            "type": "loading",
            "message": "Processing your request..."
        }, event="loading")

        import time
        time.sleep(0.1)

        # GREETING intent - quick response
        if intent == "GREETING":
            time.sleep(0.3)  # Brief delay for realism

            final = {
                "reply": (
                    "Welcome to Marrfa AI. I'm here to assist you with information about Marrfa's services, "
                    "our leadership team, company policies, and property listings in Dubai. "
                    "How may I help you today?"
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

        # PROPERTY intent - property search (includes specific property queries)
        if intent == "PROPERTY":
            # Update loading message for property search
            loading_msg = "Searching for property details..." if is_property_specific_query(
                query) else "Searching properties..."

            yield sse_event({
                "type": "loading",
                "message": loading_msg
            }, event="loading")

            time.sleep(0.2)  # Small delay

            out = handle_property_query(query)

            # Simulate some processing time
            time.sleep(0.3)

            final = {
                "reply": out["reply"],
                "route": "property",
                "intent": "PROPERTY",
                "properties": out.get("properties", []),
                "properties_full": out.get("properties_full", []),
                "total": out.get("total", 0),
                "sources": [],
                "filters_used": out.get("filters_used", {}),
            }
            yield sse_event({"type": "final", **final}, event="final")
            yield sse_event({"type": "done"}, event="done")
            return

        # RAG intent - streaming response with real LLM
        # Update loading message for RAG
        yield sse_event({
            "type": "loading",
            "message": "Gathering information from knowledge base..."
        }, event="loading")

        time.sleep(0.2)  # Small delay for RAG retrieval

        try:
            meta, stream_gen = rag_answer.answer_stream(query)

            # Send start of content event
            yield sse_event({"type": "content_start"}, event="content_start")

            full_text = ""
            chunk_count = 0

            for chunk in stream_gen:
                if chunk:
                    full_text += chunk
                    chunk_count += 1
                    yield sse_event({
                        "type": "delta",
                        "delta": chunk
                    }, event="delta")

            # If we got no chunks, generate a fallback response
            if chunk_count == 0:
                full_text = "I couldn't retrieve information for that query. Please try rephrasing your question."

            final = {
                "reply": full_text,
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
            # Error handling
            error_msg = f"An error occurred while processing your request: {str(e)}"
            yield sse_event({
                "type": "error",
                "message": error_msg
            }, event="error")

            final = {
                "reply": error_msg,
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
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'  # Important for streaming
        }
    )
