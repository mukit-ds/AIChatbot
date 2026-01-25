import os
import json
import uuid
import re
from typing import Any, Dict, List, Generator, Optional
from collections import defaultdict

import requests
import streamlit as st
import streamlit.components.v1 as components


# -------------------------
# Config
# -------------------------
API_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:9000").rstrip("/")
STREAM_URL = f"{API_BASE}/chat/stream"
HEALTH_URL = f"{API_BASE}/health"

# Marrfa API for full specific-property details (like streamlit_app2.py)
MARRFA_PROPERTY_API_BASE = os.environ.get("MARRFA_PROPERTY_API_BASE", "https://apiv2.marrfa.com").rstrip("/")
MARRFA_SITE_BASE = "https://www.marrfa.com"

# Images
MAX_IMAGES_TO_SHOW = 15          # ✅ total max across categories
COLS_PER_ROW = 3
MAIN_IMG_HEIGHT = 480            # ✅ bigger cover image
THUMB_IMG_HEIGHT = 190           # ✅ bigger thumbnails


# Hard blocked dev logo(s) (same idea as streamlit_app2.py)
HARD_BLOCKED_IMAGE_URLS = {
    "https://storage.googleapis.com/xdil-qda0-zofk.m2.xano.io/vault/ZZLvFZFt/GyI8f6kUS3MXO1cH4u7yT8Ibb_8/VIxZOw../313416330_5562915650429278_8926004611552043340_n.jpeg"
}
HARD_BLOCKED_FILENAMES = {
    "313416330_5562915650429278_8926004611552043340_n.jpeg"
}


# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Marrfa AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")


# -------------------------
# CSS
# -------------------------
st.markdown(
    f"""
<style>
header, footer {{ visibility: hidden; height: 0; }}
section[data-testid="stSidebar"] {{ display: none !important; }}
div[data-testid="collapsedControl"] {{ display: none !important; }}

.block-container {{ max-width: 1200px; padding-top: 26px; padding-bottom: 120px; }}

.deployment-badge {{
    position: fixed;
    top: 0;
    right: 0;
    background: {"#10b981" if ENVIRONMENT == "production" else "#f59e0b"};
    color: white;
    padding: 4px 12px;
    border-radius: 0 0 0 10px;
    font-size: 11px;
    font-weight: 600;
    z-index: 10000;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

.msg-wrap {{ display:flex; justify-content:flex-start; margin: 10px 0 14px 0; }}
.msg-bubble {{
    background: #F3F4F6;
    color: #111827;
    padding: 14px 18px;
    border-radius: 22px;
    width: min(880px, 92vw);
    box-shadow: 0 1px 0 rgba(0,0,0,0.04);
    font-size: 15px;
    line-height: 1.5;
}}
.user-bubble {{ background: #EEF2F6; }}

.loading-bubble {{
    background: #F3F4F6;
    color: #6B7280;
    padding: 14px 18px;
    border-radius: 22px;
    width: min(880px, 92vw);
    box-shadow: 0 1px 0 rgba(0,0,0,0.04);
    font-size: 15px;
    line-height: 1.5;
    display: flex;
    align-items: center;
}}
.dot {{
    height: 8px;
    width: 8px;
    background-color: #6B7280;
    border-radius: 50%;
    display: inline-block;
    margin: 0 2px;
    animation: bounce 1.4s infinite ease-in-out both;
}}
.dot:nth-child(1) {{ animation-delay: -0.32s; }}
.dot:nth-child(2) {{ animation-delay: -0.16s; }}
@keyframes bounce {{
    0%, 80%, 100% {{ transform: scale(0); }}
    40% {{ transform: scale(1.0); }}
}}

.small-muted {{
    color: #6B7280;
    font-size: 13px;
    margin-bottom: 10px;
}}

.prop-card {{
    background: white;
    border-radius: 18px;
    border: 1px solid #EEE;
    overflow: hidden;
    box-shadow: 0 2px 16px rgba(0,0,0,0.04);
    margin-bottom: 10px;
}}

.prop-body {{ padding: 14px 14px 6px 14px; }}
.prop-title {{
    font-size: 18px;
    font-weight: 800;
    margin: 0 0 8px 0;
    color: #111827;
    line-height: 1.3;
}}

.meta-box {{
    margin-top: 10px;
    background: #F3F4F6;
    border-radius: 14px;
    padding: 10px 12px;
}}
.meta-line {{
    display: flex;
    gap: 10px;
    align-items: flex-start;
    color: #374151;
    font-size: 14px;
    margin: 8px 0;
}}
.meta-icon {{ width: 18px; display: inline-block; opacity: 0.9; flex-shrink: 0; }}

.sticky-wrap{{
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255,255,255,0.95);
  backdrop-filter: blur(10px);
  border-top: 1px solid #E5E7EB;
  padding: 14px 0;
  z-index: 9999;
  box-shadow: 0 -2px 20px rgba(0,0,0,0.05);
}}
.sticky-inner{{
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 16px;
}}
.sticky-row{{
    display: flex;
    gap: 10px;
    align-items: center;
    background: white;
    border: 1px solid #D1D5DB;
    border-radius: 14px;
    padding: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}}
.sticky-inner .stTextInput > div > div > input{{
  border-radius: 12px !important;
  height: 44px !important;
  padding: 10px 12px !important;
  font-size: 16px !important;
  border: none !important;
  background: #F3F4F6 !important;
  color: #111827 !important;
}}
.send-btn button{{
  width: 44px !important;
  height: 44px !important;
  border-radius: 10px !important;
  border: 1px solid #D1D5DB !important;
  background: white !important;
  font-size: 18px !important;
  padding: 0 !important;
}}

.cat-title {{
  margin-top: 0.4rem;
  margin-bottom: 0.25rem;
  font-weight: 800;
}}

.marrfa-main-card {{
  border: 1px solid #e6e6e6;
  border-radius: 14px;
  background: #fff;
  padding: 10px;
  box-shadow: 0 1px 0 rgba(0,0,0,0.02);
  margin: 8px 0 10px 0;
}}
.marrfa-main-card img {{
  width: 100%;
  height: {MAIN_IMG_HEIGHT}px;
  object-fit: cover;
  border-radius: 12px;
  display: block;
  background: #fff;
}}

.marrfa-thumb img {{
  width: 100%;
  height: {THUMB_IMG_HEIGHT}px;
  object-fit: cover;
  border-radius: 12px;
  display: block;
  background: #fff;
  border: 1px solid #f0f0f0;
}}
</style>

<div class="deployment-badge">{ENVIRONMENT}</div>
    """,
    unsafe_allow_html=True
)


# -------------------------
# State
# -------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []
if "api_status" not in st.session_state:
    st.session_state.api_status = "unknown"
if "_scroll_to_bottom" not in st.session_state:
    st.session_state._scroll_to_bottom = False


# -------------------------
# Helpers
# -------------------------
def safe_list(x):
    return x if isinstance(x, list) else []


def escape_html(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def check_api_health() -> bool:
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        if r.status_code == 200:
            st.session_state.api_status = "healthy"
            return True
    except Exception:
        pass
    st.session_state.api_status = "unreachable"
    return False


def fmt_price(raw: Dict[str, Any]) -> str:
    cur = raw.get("price_currency") or raw.get("currency") or "AED"

    def _num(x):
        try:
            if x is None:
                return None
            if isinstance(x, (int, float)):
                return float(x)
            s = str(x).strip().replace(",", "")
            return float(s) if s else None
        except Exception:
            return None

    mn = _num(raw.get("min_price_aed") or raw.get("min_price") or raw.get("price_from") or raw.get("starting_price"))
    mx = _num(raw.get("max_price_aed") or raw.get("max_price") or raw.get("price_to") or raw.get("ending_price"))

    if mn is not None and mx is not None:
        return f"{cur} {mn:,.0f} – {mx:,.0f}"
    if mn is not None:
        return f"{cur} {mn:,.0f}+"
    if mx is not None:
        return f"Up to {cur} {mx:,.0f}"
    return "N/A"


def fmt_area(raw: Dict[str, Any]) -> str:
    unit = raw.get("area_unit") or raw.get("size_unit") or "sqft"

    def _num(x):
        try:
            if x is None:
                return None
            if isinstance(x, (int, float)):
                return float(x)
            s = str(x).strip().replace(",", "")
            return float(s) if s else None
        except Exception:
            return None

    mn = _num(raw.get("min_area") or raw.get("min_size") or raw.get("starting_area") or raw.get("from_area"))
    mx = _num(raw.get("max_area") or raw.get("max_size") or raw.get("ending_area") or raw.get("to_area"))

    if mn is not None and mx is not None:
        return f"{mn:,.0f} – {mx:,.0f} {unit}"
    if mn is not None:
        return f"{mn:,.0f}+ {unit}"
    if mx is not None:
        return f"Up to {mx:,.0f} {unit}"

    single = _num(raw.get("size") or raw.get("area_size") or raw.get("built_up_area") or raw.get("plot_area"))
    if single is not None:
        return f"{single:,.0f} {unit}"
    return "N/A"


# ----------------------------
# Specific-property image extraction (website order)
# (ported from streamlit_app2.py)
# ----------------------------
def _normalize_url(u: Optional[str]) -> str:
    return (u or "").strip()


def _should_exclude_url(url: str, blocked_urls: List[str]) -> bool:
    u = (url or "").strip()
    if not u:
        return True
    if u in HARD_BLOCKED_IMAGE_URLS:
        return True
    ul = u.lower()
    fname = ul.split("?")[0].split("#")[0].rstrip("/").split("/")[-1]
    if fname in HARD_BLOCKED_FILENAMES:
        return True
    if u in blocked_urls:
        return True

    branding_markers = ["logo", "brand", "developer_logo", "favicon", "icon"]
    if any(m in ul for m in branding_markers):
        if any(ext in ul for ext in [".svg", "favicon", ".ico", "icon"]):
            return True
        if "/logo" in ul or "logo_" in ul or "_logo" in ul:
            return True
    return False


def _pick_images_with_categories(resp_json: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Exact website order: Architecture -> Interior -> Facilities
    :contentReference[oaicite:1]{index=1}
    """
    found: List[Dict[str, str]] = []
    seen_urls = set()

    prop_data = resp_json.get("data", resp_json) if isinstance(resp_json, dict) else {}
    blocked = []
    if isinstance(prop_data, dict):
        blocked += [
            _normalize_url(prop_data.get("developer_logo")),
            _normalize_url(prop_data.get("developer_logo_url")),
            _normalize_url(prop_data.get("developerLogo")),
            _normalize_url(prop_data.get("developerLogoUrl")),
            _normalize_url((prop_data.get("developer") or {}).get("logo") if isinstance(prop_data.get("developer"), dict) else None),
        ]
    blocked_urls = [u for u in blocked if u]

    def add_url(url: Any, cat: str):
        if not isinstance(url, str):
            return
        u = url.strip()
        if not u or not (u.startswith("http") or u.startswith("//")):
            return
        if _should_exclude_url(u, blocked_urls):
            return
        if u in seen_urls:
            return
        seen_urls.add(u)
        found.append({"url": u, "category": cat})

    def extract_from_list(lst: Any, cat: str):
        if not isinstance(lst, list):
            return
        for item in lst:
            if isinstance(item, str):
                add_url(item, cat)
            elif isinstance(item, dict):
                for k in ("url", "src", "image", "image_url", "original", "large"):
                    if isinstance(item.get(k), str) and item.get(k).strip():
                        add_url(item.get(k), cat)
                        break

    if isinstance(prop_data, dict):
        extract_from_list(prop_data.get("architecture"), "Architecture")
        extract_from_list(prop_data.get("interior"), "Interior")
        extract_from_list(prop_data.get("facilities"), "Facilities")
        extract_from_list(prop_data.get("architectures"), "Architecture")
        extract_from_list(prop_data.get("interiors"), "Interior")
        extract_from_list(prop_data.get("facility"), "Facilities")

    return found


def group_images_by_category(images: List[Dict[str, str]]) -> Dict[str, List[str]]:
    grouped = defaultdict(list)
    for it in images:
        url = it.get("url")
        cat = (it.get("category") or "").strip()
        if cat in {"Architecture", "Interior", "Facilities"} and url:
            grouped[cat].append(url)

    ordered: Dict[str, List[str]] = {}
    for cat in ["Architecture", "Interior", "Facilities"]:
        if grouped.get(cat):
            ordered[cat] = grouped[cat]
    return ordered


def flatten_grouped_with_limit(grouped: Dict[str, List[str]], limit: int) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    remaining = limit
    for cat in ["Architecture", "Interior", "Facilities"]:
        if remaining <= 0:
            break
        urls = grouped.get(cat, [])
        take = urls[:remaining]
        if take:
            out[cat] = take
            remaining -= len(take)
    return out


def _pick_overview(data: Dict[str, Any]) -> Optional[str]:
    for path in [
        ("project_overview",),
        ("overview",),
        ("description",),
        ("about",),
        ("data", "project_overview"),
        ("data", "overview"),
        ("data", "description"),
    ]:
        cur = data
        ok = True
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                ok = False
                break
            cur = cur[k]
        if ok and isinstance(cur, str) and cur.strip():
            return cur.strip()
    return None


def _extract_cover_image(prop_data: Dict[str, Any], images_with_cat: List[Dict[str, str]]) -> Optional[str]:
    for k in ("cover_image_url", "coverImageUrl", "cover", "main_image", "main_image_url", "image", "image_url"):
        v = prop_data.get(k)
        if isinstance(v, str) and v.strip() and (v.startswith("http") or v.startswith("//")):
            return v.strip()
    if images_with_cat:
        return images_with_cat[0].get("url")
    return None


@st.cache_data(show_spinner=False, ttl=300)
def fetch_full_property_details(prop_id: int, timeout: int = 15) -> Dict[str, Any]:
    url = f"{MARRFA_PROPERTY_API_BASE}/properties/{prop_id}"
    try:
        resp = requests.get(url, timeout=timeout, headers={"accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()

        prop_data = data.get("data", data) if isinstance(data, dict) else {}
        prop_data = prop_data if isinstance(prop_data, dict) else {}

        images_with_cat = _pick_images_with_categories(data if isinstance(data, dict) else {})
        cover = _extract_cover_image(prop_data, images_with_cat)
        overview = _pick_overview(data if isinstance(data, dict) else {})

        return {
            "ok": True,
            "id": prop_id,
            "full_data": prop_data,
            "cover": cover,
            "images": images_with_cat,
            "overview": overview or "",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# -------------------------
# SSE stream helpers
# -------------------------
def sse_lines(resp: requests.Response) -> Generator[str, None, None]:
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        yield raw.strip()


def read_sse_events(resp: requests.Response) -> Generator[Dict[str, Any], None, None]:
    event_name = "message"
    data_buf = ""

    for line in sse_lines(resp):
        if not line:
            if data_buf:
                try:
                    payload = json.loads(data_buf)
                except Exception:
                    payload = {"type": "raw", "raw": data_buf}
                yield {"event": event_name, "data": payload}
            event_name = "message"
            data_buf = ""
            continue

        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            chunk = line.split(":", 1)[1].strip()
            data_buf += chunk

    if data_buf:
        try:
            payload = json.loads(data_buf)
        except Exception:
            payload = {"type": "raw", "raw": data_buf}
        yield {"event": event_name, "data": payload}


def stream_chat(query: str) -> Dict[str, Any]:
    if not check_api_health():
        return {"reply": "⚠️ Cannot connect to backend API.", "properties": [], "properties_full": [], "total": 0}

    payload = {"query": query, "session_id": st.session_state.session_id, "is_logged_in": False}

    typing_text = ""
    final_payload: Dict[str, Any] = {}

    try:
        with requests.post(STREAM_URL, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for ev in read_sse_events(resp):
                data = ev["data"]
                if data.get("type") == "delta":
                    typing_text += data.get("delta", "")
                elif data.get("type") == "final":
                    final_payload = data
                elif data.get("type") == "done":
                    break
    except Exception as e:
        return {"reply": f"❌ Error: {e}", "properties": [], "properties_full": [], "total": 0}

    if not final_payload:
        final_payload = {"reply": typing_text, "properties": [], "properties_full": [], "total": 0}

    return final_payload


# -------------------------
# UI render helpers
# -------------------------
def render_user(text: str):
    st.markdown(
        f"""
<div class="msg-wrap">
  <div class="msg-bubble user-bubble">{escape_html(text)}</div>
</div>
        """,
        unsafe_allow_html=True
    )


def render_assistant(text_html: str):
    st.markdown(
        f"""
<div class="msg-wrap">
  <div class="msg-bubble">{text_html}</div>
</div>
        """,
        unsafe_allow_html=True
    )



def push_user_query(q: str):
    q = (q or '').strip()
    if not q:
        return
    st.session_state.history.append({
        'id': str(uuid.uuid4())[:8],
        'q': q,
        'reply': '',
        'data': {}
    })
    st.session_state._scroll_to_bottom = True
    st.rerun()

def render_cover_image(url: str):
    st.markdown(
        f"""
<div class="marrfa-main-card">
  <a href="{escape_html(url)}" target="_blank" rel="noopener noreferrer">
    <img src="{escape_html(url)}" />
  </a>
</div>
        """,
        unsafe_allow_html=True
    )


def render_category_grid(urls: List[str], cols_per_row: int = 3):
    if not urls:
        return
    rows = (len(urls) + cols_per_row - 1) // cols_per_row
    idx = 0
    for _ in range(rows):
        cols = st.columns(cols_per_row, gap="medium")
        for c in cols:
            if idx >= len(urls):
                break
            url = urls[idx]
            with c:
                st.markdown(
                    f"""
<div class="marrfa-thumb">
  <a href="{escape_html(url)}" target="_blank" rel="noopener noreferrer">
    <img src="{escape_html(url)}" />
  </a>
</div>
                    """,
                    unsafe_allow_html=True
                )
            idx += 1


# -------------------------
# Main render (conversation)
# -------------------------
if not st.session_state.history:
    render_assistant("👋 Welcome to Marrfa AI! Ask about properties or Marrfa company info.")
else:
    for item in st.session_state.history:
        q = item.get("q", "")
        if q:
            render_user(q)

        data = item.get("data", {}) or {}
        props = safe_list(data.get("properties")) or []
        props_full = safe_list(data.get("properties_full")) or []

        filters_used = data.get("filters_used", {}) or {}
        is_specific = (filters_used.get("search_type") == "specific")

        # try to get id for specific property
        prop_id = None
        title_fallback = ""
        if is_specific:
            cand = None
            if props_full and isinstance(props_full[0], dict):
                cand = props_full[0]
            elif props and isinstance(props[0], dict):
                cand = props[0]
            if cand:
                title_fallback = cand.get("title") or cand.get("name") or ""
                try:
                    prop_id = int(cand.get("id"))
                except Exception:
                    prop_id = None

        # ✅ CHANGE 1: Do NOT show backend long paragraph for specific property (it repeats)
        if not is_specific:
            reply = item.get("reply", "") or ""
            render_assistant(escape_html(reply).replace("\n", "<br>"))

        # ✅ SPECIFIC PROPERTY VIEW (cover -> overview -> category images)
        if is_specific and prop_id:
            details = fetch_full_property_details(prop_id)
            if not details.get("ok"):
                render_assistant(f"⚠️ Could not load property details: {escape_html(details.get('error', 'Unknown error'))}")
                continue

            full_data = details.get("full_data") or {}
            cover = details.get("cover")
            overview = (details.get("overview") or "").strip()
            images = details.get("images") or []

            title = full_data.get("name") or full_data.get("title") or title_fallback or "Untitled Property"
            location = full_data.get("area") or full_data.get("location") or "Unknown Location"
            developer = full_data.get("developer") or "Unknown Developer"

            area_text = fmt_area(full_data)
            price_text = fmt_price(full_data)
            completion = (str(full_data.get("completion_year") or "")).strip()

            # assistant bubble: only ONE clean intro (no repetition)
            lines = [
                f"Here are the details for <b>{escape_html(title)}</b>:",
                f"• <b>Location</b>: {escape_html(str(location))}",
                f"• <b>Developer</b>: {escape_html(str(developer))}",
                f"• <b>Area</b>: {escape_html(str(area_text))}",
                f"• <b>Price</b>: {escape_html(str(price_text))}",
            ]
            if completion:
                lines.append(f"• <b>Completion</b>: {escape_html(completion)}")

            render_assistant("<br>".join(lines))

            # 1) Cover image first
            if isinstance(cover, str) and cover.strip():
                render_cover_image(cover)

            # 2) Description (overview) once
            if overview:
                st.markdown("### Overview")
                st.write(overview)

            # 3) Images category wise (Architecture -> Interior -> Facilities)
            st.markdown("### Images")
            grouped = group_images_by_category(images)
            limited_grouped = flatten_grouped_with_limit(grouped, MAX_IMAGES_TO_SHOW)

            for cat in ["Architecture", "Interior", "Facilities"]:
                urls = limited_grouped.get(cat, [])
                # remove cover image if repeated
                if cover:
                    urls = [u for u in urls if u != cover]
                if not urls:
                    continue

                st.markdown(f'<div class="cat-title">{cat}</div>', unsafe_allow_html=True)
                render_category_grid(urls, cols_per_row=COLS_PER_ROW)

            # "For more click here"
            property_site_url = f"{MARRFA_SITE_BASE}/propertylisting/{prop_id}"
            st.markdown(
                f'For more <a href="{escape_html(property_site_url)}" target="_blank" rel="noopener noreferrer"><b>click here</b></a>',
                unsafe_allow_html=True
            )

        # Non-specific property listing cards (optional)
        elif props:
            st.markdown(
                f'<div class="small-muted">Showing {len(props)} result{"s" if len(props) != 1 else ""}</div>',
                unsafe_allow_html=True
            )
            cols = st.columns(3, gap="large")
            for i, p in enumerate(props[:15]):
                col = cols[i % 3]
                with col:
                    title = p.get("title") or "Untitled Property"
                    location = p.get("location") or "Dubai"
                    developer = p.get("developer") or "Unknown"
                    cover = p.get("cover_image") or ""
                    listing_url = p.get("listing_url") or "#"

                    # Image: clicking the image opens the property page
                    img_inner = ""
                    if cover:
                        img_inner = f'<img src="{escape_html(cover)}" style="width:100%;height:200px;object-fit:cover;display:block;">'
                    else:
                        img_inner = '<div style="width:100%;height:200px;background:#f3f4f6;display:flex;align-items:center;justify-content:center;color:#6b7280;">No image</div>'
                    if listing_url and listing_url != "#":
                        img_html = f'<a href="{escape_html(listing_url)}" target="_blank" rel="noopener noreferrer" style="text-decoration:none;display:block;">{img_inner}</a>'
                    else:
                        img_html = img_inner

                    st.markdown(
                        f"""
<div class="prop-card">
  {img_html}
  <div class="prop-body">
    <div class="prop-title">{escape_html(title)}</div>
    <div class="meta-box">
      <div class="meta-line"><span class="meta-icon">📍</span><span>{escape_html(str(location))}</span></div>
      <div class="meta-line"><span class="meta-icon">🏗️</span><span>{escape_html(str(developer))}</span></div>
    </div>
  </div>
</div>
    """,
                        unsafe_allow_html=True
                    )

                    # View Details: ask the chatbot about this property (instead of opening a link)
                    btn_key = f"view_details_{item.get('id','msg')}_{i}"
                    if st.button("View Details", key=btn_key, use_container_width=True):
                        push_user_query(f"Tell me about {title}")


# -------------------------
# Sticky input bottom
# -------------------------
st.markdown('<div class="sticky-wrap"><div class="sticky-inner"><div class="sticky-row">', unsafe_allow_html=True)

with st.form("sticky_form", clear_on_submit=True):
    c1, c2 = st.columns([12, 1], gap="small")
    with c1:
        prompt = st.text_input(
            "Message",
            placeholder="Ask about properties, Marrfa company info, or tell me about a property by name...",
            label_visibility="collapsed",
        )
    with c2:
        st.markdown('<div class="send-btn">', unsafe_allow_html=True)
        submitted = st.form_submit_button("▶")
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted and prompt.strip():
        push_user_query(prompt.strip())

# -------------------------
# Auto-scroll to the newest message (after button-driven queries)
# -------------------------
st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)

if st.session_state.get("_scroll_to_bottom", False):
    components.html(
        """<script>
          const el = window.parent.document.getElementById("chat-bottom");
          if (el) { el.scrollIntoView({behavior: "smooth", block: "end"}); }
        </script>""",
        height=0,
    )
    st.session_state._scroll_to_bottom = False

st.markdown("</div></div></div>", unsafe_allow_html=True)


# -------------------------
# After rerun: if last message reply empty -> call backend now
# -------------------------
if st.session_state.history and st.session_state.history[-1].get("reply", "") == "":
    last = st.session_state.history[-1]
    q = last["q"]

    final = stream_chat(q)

    last["reply"] = final.get("reply", "")
    last["data"] = final

    st.session_state.history[-1] = last
    st.rerun()
