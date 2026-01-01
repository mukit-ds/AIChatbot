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

PROPERTY_KEYWORDS = {
    "property", "properties", "apartment", "villa", "rent", "buy", "dubai", "marina",
    "downtown", "price", "bedroom", "studio", "listing", "sale"
}


def classify_intent_simple(query_text: str) -> Dict[str, Any]:
    q = (query_text or "").lower().strip()

    if not q:
        return {"intent": "COMPANY", "method": "empty_fallback"}

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
    "dubai marina", "dubai hills", "dubai hills estate", "business bay",
    "jvc", "jumeirah village circle", "jlt", "downtown", "arjan",
    "dubai south", "mbr city"
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
    q = _normalize(query)
    if not q:
        return {}

    filters: Dict[str, Any] = {}

    # area
    for area in AREAS:
        if area in q:
            filters["search_query"] = area
            break

    if "search_query" not in filters:
        if "abu dhabi" in q:
            filters["search_query"] = "abu dhabi"
        elif "sharjah" in q:
            filters["search_query"] = "sharjah"
        elif "uae" in q or "emirates" in q or "dubai" in q:
            filters["search_query"] = "dubai"

    # foreign currency detection
    currency_pattern = r"(\d+(?:\.\d+)?)\s*(usd|eur|gbp|inr|sar|qar|omr|kwd|bhd)"
    mcur = re.search(currency_pattern, q, re.IGNORECASE)
    if mcur:
        filters["foreign_currency"] = True
        filters["amount"] = mcur.group(1)
        filters["currency"] = mcur.group(2).upper()
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
            "filters_used": {**filters, "intent": "PROPERTY", "currency_warning": True},
        }

    filters.setdefault("search_query", "dubai")
    filters["page"] = 1
    filters["per_page"] = 15

    properties_min, properties_full = search_properties(filters)
    total = len(properties_min)

    if total == 0:
        return {
            "reply": (
                "I apologize, but I couldn't find any properties that match your specific criteria. "
                "I recommend adjusting your search parameters, such as expanding your budget range, "
                "considering different areas in Dubai, or exploring alternative property types. "
                "Would you like assistance with refining your search?"
            ),
            "properties": [],
            "properties_full": [],
            "total": 0,
            "filters_used": {**filters, "intent": "PROPERTY"},
        }

    loc = str(filters.get("search_query", "Dubai")).title()
    reply = f"I've found {min(15, total)} properties available in {loc} that match your criteria. " \
            f"Each listing includes detailed information about the property, including location, " \
            f"developer, completion year, and pricing. Please review the options below."

    return {
        "reply": reply,
        "properties": properties_min[:15],
        "properties_full": properties_full[:15],
        "total": total,
        "filters_used": {**filters, "intent": "PROPERTY"},
    }


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
        # start event
        yield sse_event({"type": "start", "intent": intent, "query": query}, event="start")

        # Show loading indicator
        yield sse_event({"type": "loading", "message": "Processing your request..."}, event="loading")

        # greeting (instant)
        if intent == "GREETING":
            # Simulate a brief delay for consistency
            import time
            time.sleep(0.5)

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

        # property (streaming with loading indicator)
        if intent == "PROPERTY":
            # Show property search loading message
            yield sse_event({"type": "loading", "message": "Searching for properties..."}, event="loading")

            out = handle_property_query(query)

            # Simulate a brief delay for consistency
            import time
            time.sleep(0.5)

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

        # RAG streaming with loading indicator
        yield sse_event({"type": "loading", "message": "Gathering information..."}, event="loading")

        meta, stream_gen = rag_answer.answer_stream(query)

        full_text = ""
        for chunk in stream_gen:
            full_text += chunk
            yield sse_event({"type": "delta", "delta": chunk}, event="delta")

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
        yield sse_event({"type": "done"}, event="done")

    return StreamingResponse(gen(), media_type="text/event-stream")